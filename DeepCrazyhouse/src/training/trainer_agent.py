"""
@file: trainer_util.py
Created on 27.09.18
@project: crazyara
@author: queensgambit

Definition of the main training loop done in gluon.
"""
import datetime
import logging
import random
from time import time
import mxnet as mx
from mxnet import autograd, gluon, nd
import numpy as np
from mxboard import SummaryWriter
from tqdm import tqdm_notebook
from rtpt import RTPT
from DeepCrazyhouse.src.domain.variants.plane_policy_representation import FLAT_PLANE_IDX
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.training.trainer_agent_mxnet import prepare_policy, return_metrics_and_stop_training
from DeepCrazyhouse.configs.train_config import TrainConfig, TrainObjects
from DeepCrazyhouse.src.training.trainer_agent_mxnet import get_context


def acc_sign(y_true, y_pred):
    """
    Custom metric which is used to predict the winner of a game
    :param y_true: Ground truth value (np-array with values between -1, 0)
    :param y_pred: Predicted labels as numpy array
    :return:
    """
    denominator = len(y_true) - (y_true == 0).sum()
    if denominator != 0:
        return (np.sign(y_pred).flatten() == np.sign(y_true)).sum() / denominator
    return 0


def acc_distribution(y_true, y_pred):
    """
    Custom metrics which measures the accuracy between two distributions, in the form if both argmax() are identical
    :param y_true: Ground truth distribution
    :param y_pred: Predicted distribution
    :return:
    """
    return (y_pred.flatten() == np.argmax(y_true, axis=1)).sum() / len(y_true)


def cross_entropy(y_true, y_pred):
    """
    Cross entropy metric with support for distributions (non-sparse/non-one-hot-encoded targets).
    Adds a small epsilon(1e-12) to avoid taking log() of 0.
    :param y_true: Ground truth value, which can be non-hot-encoded
    :param y_pred: Predicted values
    :param eps: Epsilon value to avoid taking log of 0
    :return:
    """
    return -(np.sum(y_true * np.log(y_pred+1e-12), axis=1)).mean()


def evaluate_metrics(metrics, data_iterator, net, nb_batches=None, ctx=mx.gpu(), sparse_policy_label=False,
                     apply_select_policy_from_plane=True):
    """
    Runs inference of the network on a data_iterator object and evaluates the given metrics.
    The metric results are returned as a dictionary object.

    :param metrics: List of mxnet metrics which must have the
    names ['value_loss', 'policy_loss', 'value_acc_sign', 'policy_acc']
    :param data_iterator: Gluon data iterator object
    :param net: Gluon network handle
    :param nb_batches: Number of batches to evaluate (early stopping).
     If set to None all batches of the data_iterator will be evaluated
    :param ctx: MXNET data context
    :param sparse_policy_label: Should be set to true if the policy uses one-hot encoded targets
     (e.g. supervised learning)
    :param apply_select_policy_from_plane: If true, given policy label is converted to policy map index
    :return:
    """
    reset_metrics(metrics)
    for i, (data, value_label, policy_label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        value_label = value_label.as_in_context(ctx)
        policy_label = policy_label.as_in_context(ctx)
        [value_out, policy_out] = net(data)
        value_out[0][0].wait_to_read()

        # update the metrics
        metrics["value_loss"].update(preds=value_out, labels=value_label)
        metrics["policy_loss"].update(preds=nd.SoftmaxActivation(policy_out),
                                      labels=policy_label)
        metrics["value_acc_sign"].update(preds=value_out, labels=value_label)
        metrics["policy_acc"].update(preds=nd.argmax(policy_out, axis=1),
                                     labels=policy_label)
        # stop after evaluating x batches (only recommended to use this for the train set evaluation)
        if nb_batches and i == nb_batches:
            break

    metric_values = {"loss": 0.01 * metrics["value_loss"].get()[1] + 0.99 * metrics["policy_loss"].get()[1]}

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


class TrainerAgent:  # Probably needs refactoring
    """Main training loop"""

    def __init__(
        self,
        net,
        val_data,
        train_config: TrainConfig,
        train_objects: TrainObjects,
    ):
        # Too many instance attributes (29/7) - Too many arguments (24/5) - Too many local variables (25/15)
        # Too few public methods (1/2)
        self.tc = train_config
        self.to = train_objects
        if self.to.metrics is None:
            self.to.metrics = {}
        self._ctx = get_context(train_config.context, train_config.device_id)
        self._net = net
        self._graph_exported = False
        self._val_data = val_data
        # define a summary writer that logs data and flushes to the file every 5 seconds
        if self.tc.log_metrics_to_tensorboard:
            self.sum_writer = SummaryWriter(logdir=self.tc.export_dir+"logs", flush_secs=5, verbose=False)
        # Define the two loss functions
        self._softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=self.tc.sparse_policy_label)
        self._l2_loss = gluon.loss.L2Loss()
        if self.tc.optimizer_name != "nag":
            raise NotImplementedError("The requested optimizer %s Isn't supported yet." % self.tc.optimizer_name)
        self._trainer = gluon.Trainer(
            self._net.collect_params(),
            "nag",
            {
                "learning_rate": self.to.lr_schedule(0),
                "momentum": self.to.momentum_schedule(0),
                "wd": self.tc.wd,
            },
        )

        # collect parameter names for logging the gradients of parameters in each epoch
        self._params = self._net.collect_params()
        self._param_names = self._params.keys()
        self.ordering = list(range(self.tc.nb_parts))  # define a list which describes the order of the processed batches

    def _log_metrics(self, metric_values, global_step, prefix="train_"):
        """
        Logs a dictionary object of metric value to the console and to tensorboard
        if _log_metrics_to_tensorboard is set to true
        :param metric_values: Dictionary object storing the current metrics
        :param global_step: X-Position point of all metric entries
        :param prefix: Used for labelling the metrics
        :return:
        """
        for name in metric_values.keys():  # show the metric stats
            print(" - %s%s: %.4f" % (prefix, name, metric_values[name]), end="")
            # add the metrics to the tensorboard event file
            if self.tc.log_metrics_to_tensorboard:
                self.sum_writer.add_scalar(name, [prefix.replace("_", ""), metric_values[name]], global_step)

    def _process_on_data_plane_file(self, train_data, batch_proc_tmp):

        for _, (data, value_label, policy_label) in enumerate(train_data):
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
                if self.tc.select_policy_from_plane and not self.tc.is_policy_from_plane_data:
                    policy_out = policy_out[:, FLAT_PLANE_IDX]
                value_loss = self._l2_loss(value_out, value_label)
                policy_loss = self._softmax_cross_entropy(policy_out, policy_label)
                # weight the components of the combined loss
                combined_loss = self.tc.val_loss_factor * value_loss.sum() + self.tc.policy_loss_factor * policy_loss.sum()
                # update a dummy metric to see a proper progress bar
                self.to.metrics["value_loss"].update(preds=value_out, labels=value_label)

            combined_loss.backward()
            self._trainer.step(data.shape[0])
            batch_proc_tmp += 1
        return batch_proc_tmp, self.to.metrics["value_loss"].get()[1]

    def train(self, cur_it=None):  # Probably needs refactoring
        """
        Training model
        :param cur_it: Current iteration which is used for the learning rate and momentum schedule.
         If set to None it will be initialized
        """
        # Too many local variables (44/15) - Too many branches (18/12) - Too many statements (108/50)
        # set a custom seed for reproducibility
        random.seed(self.tc.seed)
        # define and initialize the variables which will be used
        t_s = time()
        # predefine the local variables that will be used in the training loop
        val_loss_best = val_p_acc_best = k_steps_best = old_label = value_out = None
        patience_cnt = epoch = batch_proc_tmp = 0  # track on how many batches have been processed in this epoch
        k_steps = self.tc.k_steps_initial  # counter for thousands steps
        # calculate how many log states will be processed
        k_steps_end = round(self.tc.total_it / self.tc.batch_steps)
        # we use k-steps instead of epochs here
        self.rtpt = RTPT(name_initials=self.tc.name_initials, experiment_name='crazyara',
                         max_iterations=k_steps_end-self.tc.k_steps_initial)
        if cur_it is None:
            cur_it = self.tc.k_steps_initial * 1000
        nb_spikes = 0  # count the number of spikes that have been detected
        # initialize the loss to compare with, with a very high value
        old_val_loss = np.inf
        graph_exported = False  # create a state variable to check if the net architecture has been reported yet

        if not self.ordering:  # safety check to prevent eternal loop
            raise Exception("You must have at least one part file in your planes-dataset directory!")

        # Start the RTPT tracking
        self.rtpt.start()

        while True:  # Too many nested blocks (7/5)
            # reshuffle the ordering of the training game batches (shuffle works in place)
            random.shuffle(self.ordering)

            epoch += 1
            logging.info("EPOCH %d", epoch)
            logging.info("=========================")
            t_s_steps = time()

            for part_id in tqdm_notebook(self.ordering):
                # load one chunk of the dataset from memory
                _, x_train, yv_train, yp_train, _, _ = load_pgn_dataset(
                    dataset_type="train", part_id=part_id, normalize=self.tc.normalize, verbose=False,
                    q_value_ratio=self.tc.q_value_ratio
                )

                yp_train = prepare_policy(y_policy=yp_train, select_policy_from_plane=self.tc.select_policy_from_plane,
                                          sparse_policy_label=self.tc.sparse_policy_label,
                                          is_policy_from_plane_data=self.tc.is_policy_from_plane_data)

                # update the train_data object
                train_dataset = gluon.data.ArrayDataset(
                    nd.array(x_train), nd.array(yv_train), nd.array(yp_train)
                )
                train_data = gluon.data.DataLoader(
                    train_dataset, batch_size=self.tc.batch_size, shuffle=True, num_workers=self.tc.cpu_count
                )

                for _, (data, value_label, policy_label) in enumerate(train_data):
                    data = data.as_in_context(self._ctx)
                    value_label = value_label.as_in_context(self._ctx)
                    policy_label = policy_label.as_in_context(self._ctx)

                    # update a dummy metric to see a proper progress bar
                    #  (the metrics will get evaluated at the end of 100k steps)
                    if batch_proc_tmp > 0:
                        self.to.metrics["value_loss"].update(old_label, value_out)

                    old_label = value_label
                    with autograd.record():
                        [value_out, policy_out] = self._net(data)
                        value_loss = self._l2_loss(value_out, value_label)
                        policy_loss = self._softmax_cross_entropy(policy_out, policy_label)
                        # weight the components of the combined loss
                        combined_loss = (
                            self.tc.val_loss_factor * value_loss + self.tc.policy_loss_factor * policy_loss
                        )
                        # update a dummy metric to see a proper progress bar
                        # self._metrics['value_loss'].update(preds=value_out, labels=value_label)

                    combined_loss.backward()
                    learning_rate = self.to.lr_schedule(cur_it)  # update the learning rate
                    self._trainer.set_learning_rate(learning_rate)
                    momentum = self.to.momentum_schedule(cur_it)  # update the momentum
                    self._trainer._optimizer.momentum = momentum
                    self._trainer.step(data.shape[0])
                    cur_it += 1
                    batch_proc_tmp += 1
                    # add the graph representation of the network to the tensorboard log file
                    if not graph_exported and self.tc.log_metrics_to_tensorboard:
                        self.sum_writer.add_graph(self._net)
                        graph_exported = True

                    if batch_proc_tmp >= self.tc.batch_steps:  # show metrics every thousands steps
                        # log the current learning rate
                        # update batch_proc_tmp counter by subtracting the batch_steps
                        batch_proc_tmp = batch_proc_tmp - self.tc.batch_steps
                        ms_step = ((time() - t_s_steps) / self.tc.batch_steps) * 1000  # measure elapsed time
                        # update the counters
                        k_steps += 1
                        patience_cnt += 1
                        logging.info("Step %dK/%dK - %dms/step", k_steps, k_steps_end, ms_step)
                        logging.info("-------------------------")
                        logging.debug("Iteration %d/%d", cur_it, self.tc.total_it)
                        logging.debug("lr: %.7f - momentum: %.7f", learning_rate, momentum)
                        train_metric_values = evaluate_metrics(
                            self.to.metrics,
                            train_data,
                            self._net,
                            nb_batches=10, #25,
                            ctx=self._ctx,
                            sparse_policy_label=self.tc.sparse_policy_label,
                            apply_select_policy_from_plane=self.tc.select_policy_from_plane and not self.tc.is_policy_from_plane_data
                        )
                        val_metric_values = evaluate_metrics(
                            self.to.metrics,
                            self._val_data,
                            self._net,
                            nb_batches=None,
                            ctx=self._ctx,
                            sparse_policy_label=self.tc.sparse_policy_label,
                            apply_select_policy_from_plane=self.tc.select_policy_from_plane and not self.tc.is_policy_from_plane_data
                        )
                        # update process title according to loss
                        self.rtpt.step(subtitle=f"loss={val_metric_values['loss']:2.2f}")
                        if self.tc.use_spike_recovery and (
                            old_val_loss * self.tc.spike_thresh < val_metric_values["loss"]
                            or np.isnan(val_metric_values["loss"])
                        ):  # check for spikes
                            nb_spikes += 1
                            logging.warning(
                                "Spike %d/%d occurred - val_loss: %.3f",
                                nb_spikes,
                                self.tc.max_spikes,
                                val_metric_values["loss"],
                            )
                            if nb_spikes >= self.tc.max_spikes:
                                val_loss = val_metric_values["loss"]
                                val_p_acc = val_metric_values["policy_acc"]
                                logging.debug("The maximum number of spikes has been reached. Stop training.")
                                # finally stop training because the number of lr drops has been achieved
                                print()
                                print(
                                    "Elapsed time for training(hh:mm:ss): "
                                    + str(datetime.timedelta(seconds=round(time() - t_s)))
                                )

                                if self.tc.log_metrics_to_tensorboard:
                                    self.sum_writer.close()
                                return return_metrics_and_stop_training(k_steps, val_metric_values, k_steps_best,
                                                                        val_loss_best, val_p_acc_best)

                            logging.debug("Recover to latest checkpoint")
                            model_path = self.tc.export_dir + "weights/model-%.5f-%.3f-%04d.params" % (
                                val_loss_best,
                                val_p_acc_best,
                                k_steps_best,
                            )  # Load the best model once again
                            logging.debug("load current best model:%s", model_path)
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

                            if self.tc.export_grad_histograms:
                                grads = []
                                # logging the gradients of parameters for checking convergence
                                for _, name in enumerate(self._param_names):
                                    if "bn" not in name and "batch" not in name and name != "policy_flat_plane_idx":
                                        grads.append(self._params[name].grad())
                                        self.sum_writer.add_histogram(
                                            tag=name, values=grads[-1], global_step=k_steps, bins=20
                                        )

                            # check if a new checkpoint shall be created
                            if val_loss_best is None or val_metric_values["loss"] < val_loss_best:
                                # update val_loss_best
                                val_loss_best = val_metric_values["loss"]
                                val_p_acc_best = val_metric_values["policy_acc"]
                                k_steps_best = k_steps

                                if self.tc.export_weights:
                                    prefix = self.tc.export_dir + "weights/model-%.5f-%.3f" \
                                             % (val_loss_best, val_p_acc_best)
                                    # the export function saves both the architecture and the weights
                                    self._net.export(prefix, epoch=k_steps_best)
                                    print()
                                    logging.info("Saved checkpoint to %s-%04d.params", prefix, k_steps_best)

                                patience_cnt = 0  # reset the patience counter
                            # print the elapsed time
                            t_delta = time() - t_s_steps
                            print(" - %.ds" % t_delta)
                            t_s_steps = time()

                            # log the samples per second metric to tensorboard
                            self.sum_writer.add_scalar(
                                tag="samples_per_second",
                                value={"hybrid_sync": data.shape[0] * self.tc.batch_steps / t_delta},
                                global_step=k_steps,
                            )

                            # log the current learning rate
                            self.sum_writer.add_scalar(tag="lr", value=self.to.lr_schedule(cur_it), global_step=k_steps)
                            # log the current momentum value
                            self.sum_writer.add_scalar(
                                tag="momentum", value=self.to.momentum_schedule(cur_it), global_step=k_steps
                            )

                            if cur_it >= self.tc.total_it:

                                val_loss = val_metric_values["loss"]
                                val_p_acc = val_metric_values["policy_acc"]
                                logging.debug("The number of given iterations has been reached")
                                # finally stop training because the number of lr drops has been achieved
                                print()
                                print(
                                    "Elapsed time for training(hh:mm:ss): "
                                    + str(datetime.timedelta(seconds=round(time() - t_s)))
                                )

                                if self.tc.log_metrics_to_tensorboard:
                                    self.sum_writer.close()

                                return return_metrics_and_stop_training(k_steps, val_metric_values, k_steps_best,
                                                                        val_loss_best, val_p_acc_best)
