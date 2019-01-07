"""
@file: trainer_util.py
Created on 27.09.18
@project: crazyara
@author: queensgambit

Definition of the main training loop done in keras.
"""

import logging
import random
from time import time
from tqdm import tqdm_notebook
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
import mxnet as mx
from mxnet import gluon
from mxnet import nd, autograd
import datetime
from mxboard import SummaryWriter
import numpy as np


def acc_sign(y_true, y_pred):
    """
    Custom metric which is used to predict the winner of a game
    :param y_true: Ground truth value (np-array with values between -1, 0)
    :param y_pred: Predicted labels as numpy array
    :return:
    """
    return (np.sign(y_pred).flatten() == y_true).sum() / len(y_true)


def evaluate_metrics(metrics, data_iterator, net, nb_batches=None, ctx=mx.gpu()):
    """
    Runs inference of the network on a data_iterator object and evaluates the given metrics.
    The metric results are returned as a dictionary object.

    :param metrics: List of mxnet metrics which must have the
    names ['value_loss', 'policy_loss', 'value_acc_sign', 'policy_acc']
    :param data_iterator: Gluon dataiterator object
    :param net: Gluon network handle
    :param nb_batches: Number of batches to evaluate (early stopping).
     If set to None all batches of the data_iterator will be evaluated
    :param ctx: MXNET data context
    :return:
    """
    reset_metrics(metrics)
    for i, (data, value_label, policy_label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        value_label = value_label.as_in_context(ctx)
        policy_label = policy_label.as_in_context(ctx)

        [value_out, policy_out] = net(data)

        # update the metrics
        metrics["value_loss"].update(preds=value_out, labels=value_label)
        metrics["policy_loss"].update(preds=nd.SoftmaxActivation(policy_out), labels=policy_label)
        metrics["value_acc_sign"].update(preds=value_out, labels=value_label)
        metrics["policy_acc"].update(preds=nd.argmax(policy_out, axis=1), labels=policy_label)

        # stop after evaluating x batches (only recommeded to use this for the train set evaluation)
        if nb_batches is not None and i == nb_batches:
            break

    metric_values = {}
    metric_values["loss"] = 0.01 * metrics["value_loss"].get()[1] + 0.99 * metrics["policy_loss"].get()[1]

    for metric in metrics.values():
        metric_values[metric.get()[0]] = metric.get()[1]

    return metric_values


def reset_metrics(metrics):
    """
    Resets all metric entries in a dictionary object
    :param metrics:
    :return:
    """
    for metric in metrics.values():
        metric.reset()


class TrainerAgent:
    def __init__(
        self,
        net,
        val_data,
        nb_parts,
        lr_schedule,
        momentum_schedule,
        total_it,
        wd=0.0001,
        batch_steps=1000,
        k_steps_initial=0,
        cpu_count=16,
        batch_size=2048,
        normalize=True,
        export_weights=True,
        export_grad_histograms=True,
        log_metrics_to_tensorboard=True,
        ctx=mx.gpu(),
        metrics={},  # clip_gradient=60,
        use_spike_recovery=True,
        max_spikes=5,
        spike_thresh=1.5,
        seed=42,
        val_loss_factor=0.01,
        policy_loss_factor=0.99,
    ):
        # , lr_warmup_k_steps=30, lr_warmup_init=0.01):
        # patience=25, nb_lr_drops=3, nb_k_steps=200,

        self._log_metrics_to_tensorboard = log_metrics_to_tensorboard
        self._ctx = ctx
        # lr_drop_fac=0.1,
        self._metrics = metrics
        self._net = net
        self._graph_exported = False
        # self._lr = lr
        self._normalize = normalize
        # self._nb_k_steps = nb_k_steps
        # self._patience = patience
        # self._nb_lr_droups = nb_lr_drops
        self._lr_schedule = lr_schedule
        self._momentum_schedule = momentum_schedule
        self._total_it = total_it
        self._batch_size = batch_size
        self._export_grad_histograms = export_grad_histograms
        self._cpu_count = cpu_count
        # self._lr_drop_fac = lr_drop_fac
        self._k_steps_initial = k_steps_initial
        self._val_data = val_data
        self._export_weights = export_weights
        self._batch_steps = batch_steps
        self._use_spike_recovery = use_spike_recovery
        self._max_spikes = max_spikes
        self._spike_thresh = spike_thresh
        self._seed = seed
        self._val_loss_factor = val_loss_factor
        self._policy_loss_factor = policy_loss_factor

        # self._nb_lr_drops = nb_lr_drops
        # self._warmup_k_steps = lr_warmup_k_steps
        # self._lr_warmup_init = lr_warmup_init

        # define a summary writer that logs data and flushes to the file every 5 seconds
        if log_metrics_to_tensorboard is True:
            self.sw = SummaryWriter(logdir="./logs", flush_secs=5, verbose=False)

        # Define the two loss functions
        self._softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
        self._l2_loss = gluon.loss.L2Loss()

        self._trainer = gluon.Trainer(
            self._net.collect_params(),
            "nag",
            {
                "learning_rate": lr_schedule(0),
                "momentum": momentum_schedule(0),
                #'clip_gradient': clip_gradient,
                "wd": wd,
            },
        )

        # collect parameter names for logging the gradients of parameters in each epoch
        self._params = self._net.collect_params()
        self._param_names = self._params.keys()

        # define a list which describes the order of the processed batches
        self.ordering = list(range(nb_parts))

    def _log_metrics(self, metric_values, global_step, prefix="train_"):
        """
        Logs a dictionary object of metric vlaue to the console and to tensorboard if _log_metrics_to_tensorboard is set to true
        :param metric_values: Dictionary object storing the current metrics
        :param global_step: X-Position point of all metric entries
        :param prefix: Used for labelling the metrics
        :return:
        """
        for name in metric_values.keys():  # show the metric stats
            print(" - %s%s: %.4f" % (prefix, name, metric_values[name]), end="")
            # add the metrics to the tensorboard event file

            if self._log_metrics_to_tensorboard is True:
                self.sw.add_scalar(name, [prefix.replace("_", ""), metric_values[name]], global_step)

    def _process_on_data_plane_file(self, train_data, batch_proc_tmp):

        for i, (data, value_label, policy_label) in enumerate(train_data):
            data = data.as_in_context(self._ctx)
            value_label = value_label.as_in_context(self._ctx)
            policy_label = policy_label.as_in_context(self._ctx)

            # update a dummy metric to see a proper progress bar
            #  (the metrics will get evaluated at the end of 100k steps)
            # if self.batch_proc_tmp > 0:
            #    self._metrics['value_loss'].update(old_label, value_out)

            # old_label = value_label

            with autograd.record():
                [value_out, policy_out] = self._net(data)
                value_loss = self._l2_loss(value_out, value_label)
                policy_loss = self._softmax_cross_entropy(policy_out, policy_label)
                # weight the components of the combined loss
                combined_loss = self._val_loss_factor * value_loss.sum() + self._policy_loss_factor * policy_loss.sum()

                # update a dummy metric to see a proper progress bar
                self._metrics["value_loss"].update(preds=value_out, labels=value_label)

            combined_loss.backward()
            self._trainer.step(data.shape[0])
            batch_proc_tmp += 1

        return batch_proc_tmp, self._metrics["value_loss"].get()[1]

    def train(self):
        """

        :param net: Gluon network object
        :param val_data: Gluon dataloader object
        :param nb_parts: Sets how many different part files exist in the train directory
        :param lr: Initial learning rate
        :param momentum:
        :param wd:
        :param nb_k_steps: Number of steps in after which to drop the learning rate (assuming the patience counter
        early dropping hasn't activated beforehand)

        :param patience: Number of batches to wait until no progress on validation loss has been achieved.
        If the no progress has been done the learning rate is multiplied by the drop factor.
        :param nb_lr_drops: Number of time to drop the learning rate in total. This defines the end of the train loop
        :param batch_steps: Number of batches after which the validation loss is evaluated
        :param k_steps_initial: Initial starting point of the network in terms of process k batches (default 0)
        :param lr_drop_fac: Dropping factor to the learning rate to apply
        :param cpu_count: How many cpu threads on the current are available
        :param batch_size: Batch size to train the network with
        :param normalize: Weather to use data normalization after loading the data (recommend to set to True)
        :param export_weights: Sets if network checkpoints should be exported
        :param export_grad_histograms: Sets if the gradient updates of the weights should be logged to tensorboard
        :return:
        """

        # set a custom seed for reproducibility
        random.seed(self._seed)

        # define and initialize the variables which will be used
        t_s = time()

        # predefine the local variables that will be used in the training loop
        val_loss_best = None
        val_p_acc_best = None
        k_steps_best = None
        patience_cnt = 0

        epoch = 0
        # keep track on how many batches have been processed in this epoch so far
        batch_proc_tmp = 0
        # counter for thousands steps
        k_steps = self._k_steps_initial
        # calculate how many log states will be processed
        k_steps_end = self._total_it / self._batch_steps

        cur_it = 0

        # count the number of spikes that have been detected
        nb_spikes = 0
        # initialize the loss to compare with, with a very high value
        old_val_loss = 9000

        # self._lr = self._lr_warmup_init
        # logging.info('Warmup-Schedule')
        # logging.info('Initial learning rate: lr = %.5f', self._lr)
        # logging.info('=========================================')

        # set initial lr
        # self._trainer.set_learning_rate(self._lr)
        # log the current learning rate
        # self.sw.add_scalar(tag='lr', value=self._lr, global_step=k_steps)

        # create a state variable to check if the net architecture has been reported yet
        graph_exported = False

        old_label = None
        value_out = None

        # safety check to prevent eternal loop
        if len(self.ordering) == 0:
            raise Exception("You must have at least one part file in your planes-dataset directory!")

        while True:
            # reshuffle the ordering of the training game batches (shuffle works in place)
            random.shuffle(self.ordering)

            epoch += 1
            logging.info("EPOCH %d", epoch)
            logging.info("=========================")
            t_s_steps = time()

            for part_id in tqdm_notebook(self.ordering):

                # load one chunk of the dataset from memory
                s_idcs_train, x_train, yv_train, yp_train, pgn_datasets_train = load_pgn_dataset(
                    dataset_type="train", part_id=part_id, normalize=self._normalize, verbose=False
                )
                # update the train_data object
                train_dataset = gluon.data.ArrayDataset(
                    nd.array(x_train), nd.array(yv_train), nd.array(yp_train.argmax(axis=1))
                )
                train_data = gluon.data.DataLoader(
                    train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._cpu_count
                )
                # batch_proc_tmp, dummy = self._process_on_data_plane_file(train_data, batch_proc_tmp)

                for i, (data, value_label, policy_label) in enumerate(train_data):
                    data = data.as_in_context(self._ctx)
                    value_label = value_label.as_in_context(self._ctx)
                    policy_label = policy_label.as_in_context(self._ctx)

                    # update a dummy metric to see a proper progress bar
                    #  (the metrics will get evaluated at the end of 100k steps)
                    if batch_proc_tmp > 0:
                        self._metrics["value_loss"].update(old_label, value_out)

                    old_label = value_label
                    with autograd.record():
                        [value_out, policy_out] = self._net(data)
                        value_loss = self._l2_loss(value_out, value_label)
                        policy_loss = self._softmax_cross_entropy(policy_out, policy_label)
                        # weight the components of the combined loss
                        combined_loss = (
                            self._val_loss_factor * value_loss.sum() + self._policy_loss_factor * policy_loss.sum()
                        )

                        # update a dummy metric to see a proper progress bar
                        # self._metrics['value_loss'].update(preds=value_out, labels=value_label)

                    combined_loss.backward()

                    # update the learning rate
                    lr = self._lr_schedule(cur_it)
                    self._trainer.set_learning_rate(lr)

                    # update the momentum
                    momentum = self._momentum_schedule(cur_it)
                    self._trainer._optimizer.momentum = momentum

                    self._trainer.step(data.shape[0])
                    cur_it += 1
                    batch_proc_tmp += 1

                    # add the graph representation of the network to the tensorboard log file
                    if graph_exported is False and self._log_metrics_to_tensorboard is True:
                        self.sw.add_graph(self._net)
                        graph_exported = True

                    # show metrics every thousands steps
                    if batch_proc_tmp >= self._batch_steps:

                        # if k_steps < self._warmup_k_steps:
                        # update the learning rate
                        # self._lr *= k_steps * ((self._lr_first - self._lr_warmup_init) / self._warmup_k_steps) + self._lr_warmup_init #self._lr_drop_fac
                        # self._trainer.set_learning_rate(self._lr)
                        # logging.info('Learning rate update: lr = %.5f', self._lr)
                        # logging.info('=========================================')

                        # log the current learning rate

                        # update batch_proc_tmp counter by subtracting the batch_steps
                        batch_proc_tmp = batch_proc_tmp - self._batch_steps

                        # measure elapsed time
                        ms_step = ((time() - t_s_steps) / self._batch_steps) * 1000
                        # update the counters
                        k_steps += 1
                        patience_cnt += 1

                        logging.info("Step %dK/%dK - %dms/step", k_steps, k_steps_end, ms_step)
                        logging.info("-------------------------")
                        logging.debug("Iteration %d/%d", cur_it, self._total_it)
                        logging.debug("lr: %.7f - momentum: %.7f", lr, momentum)

                        train_metric_values = evaluate_metrics(
                            self._metrics, train_data, self._net, nb_batches=25, ctx=self._ctx
                        )

                        val_metric_values = evaluate_metrics(
                            self._metrics, self._val_data, self._net, nb_batches=None, ctx=self._ctx
                        )

                        # spike_detected = False
                        # spike_detected = old_val_loss * 1.5 < val_metric_values['loss']
                        # if np.isnan(val_metric_values['loss']):
                        #    spike_detected = True

                        # check for spikes
                        if self._use_spike_recovery is True and (
                            old_val_loss * self._spike_thresh < val_metric_values["loss"]
                            or np.isnan(val_metric_values["loss"])
                        ):
                            nb_spikes += 1
                            logging.warning(
                                "Spike %d/%d occurred - val_loss: %.3f",
                                nb_spikes,
                                self._max_spikes,
                                val_metric_values["loss"],
                            )
                            if nb_spikes >= self._max_spikes:

                                val_loss = val_metric_values["loss"]
                                val_p_acc = val_metric_values["policy_acc"]

                                logging.debug("The maximum number of spikes has been reached. Stop training.")
                                # finally stop training because the number of lr drops has been achieved
                                print()
                                print(
                                    "Elapsed time for training(hh:mm:ss): "
                                    + str(datetime.timedelta(seconds=round(time() - t_s)))
                                )

                                if self._log_metrics_to_tensorboard is True:
                                    self.sw.close()

                                return (k_steps, val_loss, val_p_acc), (k_steps_best, val_loss_best, val_p_acc_best)

                            logging.debug("Recover to latest checkpoint")
                            # ## Load the best model once again
                            model_path = "./weights/model-%.5f-%.3f-%04d.params" % (
                                val_loss_best,
                                val_p_acc_best,
                                k_steps_best,
                            )
                            logging.debug("load current best model:%s" % model_path)
                            self._net.load_parameters(model_path, ctx=self._ctx)
                            k_steps = k_steps_best
                            logging.debug("k_step is back at %d", k_steps_best)

                            # print the elapsed time
                            t_delta = time() - t_s_steps
                            print(" - %.ds" % t_delta)
                            t_s_steps = time()

                        else:

                            # update the val_loss_value to compare with using spike recovery
                            old_val_loss = val_metric_values["loss"]

                            # log the metric values to tensorboard
                            self._log_metrics(train_metric_values, global_step=k_steps, prefix="train_")
                            self._log_metrics(val_metric_values, global_step=k_steps, prefix="val_")

                            if self._export_grad_histograms is True:
                                grads = []
                                # logging the gradients of parameters for checking convergence
                                for i_p, name in enumerate(self._param_names):
                                    if "bn" not in name and "batch" not in name:
                                        grads.append(self._params[name].grad())
                                        self.sw.add_histogram(tag=name, values=grads[-1], global_step=k_steps, bins=20)

                            # check if a new checkpoint shall be created
                            if val_loss_best is None or val_metric_values["loss"] < val_loss_best:
                                # update val_loss_best
                                val_loss_best = val_metric_values["loss"]
                                val_p_acc_best = val_metric_values["policy_acc"]
                                k_steps_best = k_steps

                                if self._export_weights is True:
                                    prefix = "./weights/model-%.5f-%.3f" % (val_loss_best, val_p_acc_best)
                                    # the export function saves both the architecture and the weights
                                    self._net.export(prefix, epoch=k_steps_best)
                                    print()
                                    logging.info("Saved checkpoint to %s-%04d.params" % (prefix, k_steps_best))

                                # reset the patience counter
                                patience_cnt = 0

                            # print the elapsed time
                            t_delta = time() - t_s_steps
                            print(" - %.ds" % t_delta)
                            t_s_steps = time()

                            # log the samples per second metric to tensorbaord
                            self.sw.add_scalar(
                                tag="samples_per_second",
                                value={"hybrid_sync": data.shape[0] * self._batch_steps / t_delta},
                                global_step=k_steps,
                            )

                            # log the current learning rate
                            self.sw.add_scalar(tag="lr", value=self._lr_schedule(cur_it), global_step=k_steps)

                            # log the current momentum value
                            self.sw.add_scalar(
                                tag="momentum", value=self._momentum_schedule(cur_it), global_step=k_steps
                            )

                            if cur_it >= self._total_it:

                                val_loss = val_metric_values["loss"]
                                val_p_acc = val_metric_values["policy_acc"]

                                logging.debug("The number of given iterations has been reached")
                                # finally stop training because the number of lr drops has been achieved
                                print()
                                print(
                                    "Elapsed time for training(hh:mm:ss): "
                                    + str(datetime.timedelta(seconds=round(time() - t_s)))
                                )

                                if self._log_metrics_to_tensorboard is True:
                                    self.sw.close()

                                return (k_steps, val_loss, val_p_acc), (k_steps_best, val_loss_best, val_p_acc_best)

                            """
                            

                        if patience_cnt >= self._patience or k_steps == k_steps_end:

                            if cur_lr_drops == self._nb_lr_drops:

                                logging.debug('The number of given learning rate drops has been reached '
                                              '- stop training now.')
                                # finally stop training because the number of lr drops has been achieved
                                print()
                                print('Elapsed time for training(hh:mm:ss): ' + str(
                                    datetime.timedelta(seconds=round(time() - t_s))))

                                val_loss = val_metric_values['loss']
                                val_p_acc = val_metric_values['policy_acc']

                                if self._log_metrics_to_tensorboard is True:
                                    self.sw.close()

                                return (k_steps, val_loss, val_p_acc), (k_steps_best, val_loss_best, val_p_acc_best)
                                                            
                            if patience_cnt >= self._patience:
                                logging.debug('The learning rate will be dropped because the patience counter has been reached.')
                                logging.debug('The last best model checkpoint will be loaded back into memory.')
                            else:
                                logging.debug('The given number of k_steps has been reached in the current period.'
                                              'Drop the learning rate now')

                            if patience_cnt >= self._patience:
                                # ## Load the best model once again
                                model_path = "./weights/model-%.5f-%.3f-%04d.params" % (
                                val_loss_best, val_p_acc_best, k_steps_best)
                                logging.info('Revert overfiting updates')
                                logging.debug('load current best model:%s' % model_path)
                                self._net.load_parameters(model_path, ctx=self._ctx)
                                k_steps = k_steps_best
                                logging.debug('k_step is back at %d', k_steps_best)

                            # update the learning rate
                            self._lr *= self._lr_drop_fac
                            self._trainer.set_learning_rate(self._lr)
                            logging.info('Learning rate update: lr = %.7f', self._lr)
                            logging.info('=========================================')

                            # set the new k_steps end based on the new current k_steps
                            k_steps_end = k_steps + self._nb_k_steps
                            # increase the lr dropping counter
                            cur_lr_drops += 1
                            """
