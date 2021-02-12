"""
@file: trainer_agent_mxnet.py
Created on 18.05.19
@project: CrazyAra
@author: queensgambit

Definition of the main training loop done in mxnet.
"""
import os
import datetime
import logging
import random
from time import time
import numpy as np
from mxboard import SummaryWriter
from tqdm import tqdm_notebook
from rtpt import RTPT
from DeepCrazyhouse.configs.train_config import TrainConfig, TrainObjects
from DeepCrazyhouse.src.domain.variants.plane_policy_representation import FLAT_PLANE_IDX
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.domain.variants.constants import NB_LABELS_POLICY_MAP
from DeepCrazyhouse.src.training.crossentropy import *


def fill_up_batch(x, batch_size):
    """
    Fills up an array by repeating the given array to achieve the given batch_size
    :param x: Array to repeat
    :param batch_size: Given batch-size
    :return: Array with length of batch-size
    """
    return np.repeat(x, batch_size//len(x)+1, axis=0)[:batch_size]


def evaluate_metrics(metrics, data_iterator, model):
    """
    Runs inference of the network on a data_iterator object and evaluates the given metrics.
    The metric results are returned as a dictionary object.

    :param metrics: List of mxnet metrics which must have the
    names ['value_loss', 'policy_loss', 'value_acc_sign', 'policy_acc']
    :param data_iterator: Gluon data iterator object
    :param model: Gluon network handle
    :param nb_batches: Number of batches to evaluate (early stopping).
     If set to None all batches of the data_iterator will be evaluated
    :param ctx: MXNET data context
    :param select_policy_from_plane: Boolean if potential legal moves will be selected from final policy output
    :return:
    """
    reset_metrics(metrics)
    data_iterator.reset()
    metrics_res = {}

    results = model.score(data_iterator, metrics)
    for entry in results:
        name = entry[0]
        value = entry[1]
        metrics_res[name] = value

    metrics_res["loss"] = 0.01 * metrics_res["value_loss"] + 0.99 * metrics_res["policy_loss"]

    return metrics_res


def reset_metrics(metrics):
    """
    Resets all metric entries in a dictionary object
    :param metrics:
    :return:
    """
    for metric in metrics:
        metric.reset()


def add_non_sparse_cross_entropy(symbol, grad_scale_value=1.0, value_output_name="value_out_output",
                                 policy_output_name="policy_out_output"):
    """
    Adds a cross entropy loss output which support non-sparse label as targets, but distributions with value in [0,1]
    :param symbol: MXNet symbol with both a value and policy head
    :param grad_scale_value: Scaling factor for the value loss
    :param value_output_name: Output name for the value output after applying tanh activation
    :param policy_output_name: Output name for the policy output without applying softmax activation on it
    :return: MXNet symbol with adjusted policy and value loss
    """
    value_out = symbol.get_internals()[value_output_name]
    policy_out = symbol.get_internals()[policy_output_name]
    value_out = mx.sym.LinearRegressionOutput(data=value_out, name='value', grad_scale=grad_scale_value)
    policy_out = mx.sym.SoftmaxActivation(data=policy_out, name='softmax')
    policy_out = mx.symbol.Custom(data=policy_out, name='policy', op_type='CrossEntropyLoss')
    # group value_out and policy_out together
    return mx.symbol.Group([value_out, policy_out])


def remove_no_sparse_cross_entropy(symbol, grad_scale_value=1.0, value_output_name="value_out_output",
                                 policy_output_name="policy_out_output"):
    """
    Removes the last custom cross entropy loss layer to enable loading the model in the C++ API.
    :param symbol: MXNet symbol with both a value and policy head
    :param grad_scale_value: Scaling factor for the value loss
    :param value_output_name: Output name for the value output after applying tanh activation
    :param policy_output_name: Output name for the policy output without applying softmax activation on it
    :return: MXNet symbol with removed CrossEntropy-Loss in final layer
    """
    value_out = symbol.get_internals()[value_output_name]
    policy_out = symbol.get_internals()[policy_output_name]
    value_out = mx.sym.LinearRegressionOutput(data=value_out, name='value', grad_scale=grad_scale_value)
    policy_out = mx.sym.SoftmaxActivation(data=policy_out, name='softmax')
    # group value_out and policy_out together
    return mx.symbol.Group([value_out, policy_out])


def prepare_policy(y_policy, select_policy_from_plane, sparse_policy_label, is_policy_from_plane_data):
    """
    Modifies the layout of the policy vector in place according to the given definitions
    :param y_policy: Target policy vector
    :param select_policy_from_plane: If policy map representation shall be applied
    :param sparse_policy_label: True, if the labels are sparse (one-hot-encoded)
    :param is_policy_from_plane_data: True, if the policy representation is already in
     "select_policy_from_plane" representation
    :return: modified y_policy
    """
    if sparse_policy_label:
        y_policy = y_policy.argmax(axis=1)

        if select_policy_from_plane:
            y_policy[:] = FLAT_PLANE_IDX[y_policy]
    else:
        if select_policy_from_plane and not is_policy_from_plane_data:
            tmp = np.zeros((len(y_policy), NB_LABELS_POLICY_MAP), np.float32)
            tmp[:, FLAT_PLANE_IDX] = y_policy[:, :]
            y_policy = tmp
    return y_policy


def get_context(context: str, device_id: int):
    """
    Returns the computation context as an MXNet object
    :param context: Computational context either "gpu" or "cpu"
    :param device_id: Device index to use (only relevant for context=="gpu")
    :return: MXNet ctx object
    """
    if context == "gpu":
        return mx.gpu(device_id)
    else:
        return mx.cpu()


def return_metrics_and_stop_training(k_steps, val_metric_values,
                                      k_steps_best, val_loss_best, val_p_acc_best):
    return (k_steps, val_metric_values["value_loss"],
            val_metric_values["policy_loss"],
            val_metric_values["value_acc_sign"], val_metric_values["policy_acc"]), \
           (k_steps_best, val_loss_best, val_p_acc_best)


class TrainerAgentMXNET:  # Probably needs refactoring
    """Main training loop"""

    def __init__(
        self,
        model,
        symbol,
        val_iter,
        train_config: TrainConfig,
        train_objects: TrainObjects,
    ):
        # Too many instance attributes (29/7) - Too many arguments (24/5) - Too many local variables (25/15)
        # Too few public methods (1/2)
        self.tc = train_config
        self.to = train_objects
        if self.to.metrics is None:
            self.to.metrics = {}
        self._model = model
        self._symbol = symbol
        self._val_iter = val_iter
        self.x_train = self.yv_train = self.yp_train = None
        self._ctx = get_context(train_config.context, train_config.device_id)

        # define the current working directory
        if self.tc.cwd is None:
            self.tc.cwd = os.getcwd()
        # define a summary writer that logs data and flushes to the file every 5 seconds
        if self.tc.log_metrics_to_tensorboard:
            self.sum_writer = SummaryWriter(logdir="%s/logs" % self.tc.cwd, flush_secs=5, verbose=False)
        # Define the optimizer
        if self.tc.optimizer_name == "adam":
            self.optimizer = mx.optimizer.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, lazy_update=True, rescale_grad=(1.0/batch_size))
        elif self.tc.optimizer_name == "nag":
            self.optimizer = mx.optimizer.NAG(momentum=self.to.momentum_schedule(0), wd=self.tc.wd, rescale_grad=(1.0/self.tc.batch_size))
        else:
            raise Exception("%s is currently not supported as an optimizer." % self.tc.optimizer_name)
        self.ordering = list(range(self.tc.nb_parts))  # define a list which describes the order of the processed batches
        # decides if the policy indices shall be selected directly from spatial feature maps without dense layer
        self.batch_end_callbacks = [self.batch_callback]

        # few variables which are internally used
        self.val_loss_best = self.val_p_acc_best = self.k_steps_best = \
            self.old_label = self.value_out = self.t_s = None
        self.patience_cnt = self.batch_proc_tmp = None
        # calculate how many log states will be processed
        self.k_steps_end = round(self.tc.total_it / self.tc.batch_steps)
        self.k_steps = self.cur_it = self.nb_spikes = self.old_val_loss = self.continue_training = self.t_s_steps = None
        self._train_iter = self.graph_exported = self.val_metric_values = self.val_loss = self.val_p_acc = None
        # we use k-steps instead of epochs here
        self.rtpt = RTPT(name_initials=self.tc.name_initials, experiment_name='crazyara',
                         max_iterations=self.k_steps_end-self.tc.k_steps_initial)

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
                self.sum_writer.add_scalar(name, [prefix.replace("_", ""), metric_values[name]],
                                                        global_step)

    def train(self, cur_it=None):  # Probably needs refactoring
        """
        Training model
        :param cur_it: Current iteration which is used for the learning rate and momentum schedule.
         If set to None it will be initialized
        :return: return_metrics_and_stop_training()
        """
        # Too many local variables (44/15) - Too many branches (18/12) - Too many statements (108/50)
        # set a custom seed for reproducibility
        if self.tc.seed is not None:
            random.seed(self.tc.seed)
        # define and initialize the variables which will be used
        self.t_s = time()
        # track on how many batches have been processed in this epoch
        self.patience_cnt = epoch = self.batch_proc_tmp = 0
        self.k_steps = self.tc.k_steps_initial  # counter for thousands steps

        if cur_it is None:
            self.cur_it = self.tc.k_steps_initial * 1000
        else:
            self.cur_it = cur_it
        self.nb_spikes = 0  # count the number of spikes that have been detected
        # initialize the loss to compare with, with a very high value
        self.old_val_loss = 9000
        self.graph_exported = False  # create a state variable to check if the net architecture has been reported yet
        self.continue_training = True
        self.optimizer.lr = self.to.lr_schedule(self.cur_it)
        if self.tc.optimizer_name == "nag":
            self.optimizer.momentum = self.to.momentum_schedule(self.cur_it)

        if not self.ordering:  # safety check to prevent eternal loop
            raise Exception("You must have at least one part file in your planes-dataset directory!")

        # Start the RTPT tracking
        self.rtpt.start()

        while self.continue_training:  # Too many nested blocks (7/5)
            # reshuffle the ordering of the training game batches (shuffle works in place)
            random.shuffle(self.ordering)

            epoch += 1
            logging.info("EPOCH %d", epoch)
            logging.info("=========================")
            self.t_s_steps = time()
            self._model.init_optimizer(optimizer=self.optimizer)

            for part_id in tqdm_notebook(self.ordering):

                # load one chunk of the dataset from memory
                _, self.x_train, self.yv_train, self.yp_train, plys_to_end, _ = load_pgn_dataset(dataset_type="train",
                                                                                                 part_id=part_id,
                                                                                                 normalize=self.tc.normalize,
                                                                                                 verbose=False,
                                                                                                 q_value_ratio=self.tc.q_value_ratio)
                # fill_up_batch if there aren't enough games
                if len(self.yv_train) < self.tc.batch_size:
                    logging.info("filling up batch with too few samples %d" % len(self.yv_train))
                    self.x_train = fill_up_batch(self.x_train, self.tc.batch_size)
                    self.yv_train = fill_up_batch(self.yv_train, self.tc.batch_size)
                    self.yp_train = fill_up_batch(self.yp_train, self.tc.batch_size)
                    if plys_to_end is not None:
                        plys_to_end = fill_up_batch(plys_to_end, self.tc.batch_size)

                if self.tc.discount != 1:
                    self.yv_train *= self.tc.discount**plys_to_end
                self.yp_train = prepare_policy(self.yp_train, self.tc.select_policy_from_plane,
                                               self.tc.sparse_policy_label, self.tc.is_policy_from_plane_data)

                self._train_iter = mx.io.NDArrayIter({'data': self.x_train},
                                                     {'value_label': self.yv_train, 'policy_label': self.yp_train},
                                                     self.tc.batch_size,
                                                     shuffle=True)

                # avoid memory leaks by adding synchronization
                mx.nd.waitall()

                reset_metrics(self.to.metrics)
                for batch in self._train_iter:
                    self._model.forward(batch, is_train=True)  # compute predictions
                    for metric in self.to.metrics:  # update the metrics
                        self._model.update_metric(metric, batch.label)

                    self._model.backward()
                    # compute gradients
                    self._model.update()  # update parameters
                    self.batch_callback()

                    if not self.continue_training:
                        logging.info(
                            'Elapsed time for training(hh:mm:ss): ' +
                            str(datetime.timedelta(seconds=round(time() - self.t_s))))

                        return return_metrics_and_stop_training(self.k_steps, self.val_metric_values, self.k_steps_best,
                                                                self.val_loss_best, self.val_p_acc_best)

                # add the graph representation of the network to the tensorboard log file
                if not self.graph_exported and self.tc.log_metrics_to_tensorboard:
                    # self.sum_writer.add_graph(self._symbol)
                    self.graph_exported = True

    def _fill_train_metrics(self):
        """
        Fills in the training metrics
        :return:
        """
        self.train_metric_values = {}
        for metric in self.to.metrics:
            name, value = metric.get()
            self.train_metric_values[name] = value

        self.train_metric_values["loss"] = 0.01 * self.train_metric_values["value_loss"] + \
                                           0.99 * self.train_metric_values["policy_loss"]

    def recompute_eval(self):
        """
        Recomputes the score on the validataion data
        :return:
        """
        ms_step = ((time() - self.t_s_steps) / self.tc.batch_steps) * 1000
        logging.info("Step %dK/%dK - %dms/step", self.k_steps, self.k_steps_end, ms_step)
        logging.info("-------------------------")
        logging.debug("Iteration %d/%d", self.cur_it, self.tc.total_it)
        if self.tc.optimizer_name == "nag":
            logging.debug("lr: %.7f - momentum: %.7f", self.optimizer.lr, self.optimizer.momentum)
        else:
            logging.debug("lr: %.7f - momentum: -", self.optimizer.lr)

        # the metric values have already been computed during training for the train set
        self._fill_train_metrics()

        self.val_metric_values = evaluate_metrics(
            self.to.metrics,
            self._val_iter,
            self._model,
        )
        # update process title according to loss
        self.rtpt.step(subtitle=f"loss={self.val_metric_values['loss']:2.2f}")
        if self.tc.use_spike_recovery and (
                self.old_val_loss * self.tc.spike_thresh < self.val_metric_values["loss"]
                or np.isnan(self.val_metric_values["loss"])
        ):  # check for spikes
            self.handle_spike()
        else:
            self.update_eval()

    def handle_spike(self):
        """
        Handles the occurence of a spike during training, in the case validation loss increased dramatically.
        :return: self._return_metrics_and_stop_training()
        """
        self.nb_spikes += 1
        logging.warning(
            "Spike %d/%d occurred - val_loss: %.3f",
            self.nb_spikes,
            self.tc.max_spikes,
            self.val_metric_values["loss"],
        )
        if self.nb_spikes >= self.tc.max_spikes:
            val_loss = self.val_metric_values["loss"]
            val_p_acc = self.val_metric_values["policy_acc"]
            # finally stop training because the number of lr drops has been achieved

            logging.debug("The maximum number of spikes has been reached. Stop training.")
            self.continue_training = False

            if self.tc.log_metrics_to_tensorboard:
                self.sum_writer.close()
            return return_metrics_and_stop_training(self.k_steps, self.val_metric_values, self.k_steps_best,
                                                    self.val_loss_best, self.val_p_acc_best)

        logging.debug("Recover to latest checkpoint")
        # Load the best model once again
        prefix = self.tc.export_dir + "weights/model-%.5f-%.3f" % (self.val_loss_best, self.val_p_acc_best)

        logging.debug("load current best model:%s", prefix)
        self._model.load(prefix, epoch=self.k_steps_best)

        self.k_steps = self.k_steps_best
        logging.debug("k_step is back at %d", self.k_steps_best)
        # print the elapsed time
        t_delta = time() - self.t_s_steps
        print(" - %.ds" % t_delta)
        self.t_s_steps = time()

    def update_eval(self):
        """
        Updates the evaluation metrics
        :return:
        """
        # update the val_loss_value to compare with using spike recovery
        self.old_val_loss = self.val_metric_values["loss"]
        # log the metric values to tensorboard
        self._log_metrics(self.train_metric_values, global_step=self.k_steps, prefix="train_")
        self._log_metrics(self.val_metric_values, global_step=self.k_steps, prefix="val_")

        # check if a new checkpoint shall be created
        if self.val_loss_best is None or self.val_metric_values["loss"] < self.val_loss_best:
            # update val_loss_best
            self.val_loss_best = self.val_metric_values["loss"]
            self.val_p_acc_best = self.val_metric_values["policy_acc"]
            self.k_steps_best = self.k_steps

            if self.tc.export_weights:
                prefix = self.tc.export_dir + "weights/model-%.5f-%.3f" % (self.val_loss_best, self.val_p_acc_best)
                # the export function saves both the architecture and the weights
                print()
                self._model.save_checkpoint(prefix, epoch=self.k_steps_best)

            self.patience_cnt = 0  # reset the patience counter
        # print the elapsed time
        t_delta = time() - self.t_s_steps
        print(" - %.ds" % t_delta)
        self.t_s_steps = time()

        # log the samples per second metric to tensorboard
        self.sum_writer.add_scalar(
            tag="samples_per_second",
            value={"hybrid_sync": self.tc.batch_size * self.tc.batch_steps / t_delta},
            global_step=self.k_steps,
        )

        # log the current learning rate
        self.sum_writer.add_scalar(tag="lr", value=self.to.lr_schedule(self.cur_it),
                                   global_step=self.k_steps)
        if self.tc.optimizer_name == "nag":
            # log the current momentum value
            self.sum_writer.add_scalar(
                tag="momentum", value=self.to.momentum_schedule(self.cur_it),
                global_step=self.k_steps
            )

        if self.cur_it >= self.tc.total_it:

            self.continue_training = False

            self.val_loss = self.val_metric_values["loss"]
            self.val_p_acc = self.val_metric_values["policy_acc"]
            # finally stop training because the number of lr drops has been achieved
            logging.debug("The number of given iterations has been reached")

            if self.tc.log_metrics_to_tensorboard:
                self.sum_writer.close()

    def batch_callback(self):
        """
        Callback which is executed after every batch to update the momentum and learning rate
        :return:
        """

        # update the learning rate and momentum
        self.optimizer.lr = self.to.lr_schedule(self.cur_it)
        if self.tc.optimizer_name == "nag":
            self.optimizer.momentum = self.to.momentum_schedule(self.cur_it)

        self.cur_it += 1
        self.batch_proc_tmp += 1

        if self.batch_proc_tmp >= self.tc.batch_steps:  # show metrics every thousands steps
            self.batch_proc_tmp = self.batch_proc_tmp - self.tc.batch_steps
            # update the counters
            self.k_steps += 1
            self.patience_cnt += 1
            self.recompute_eval()
            self.custom_metric_eval()

    def custom_metric_eval(self):
        """
        Evaluates the model based on the validation set of different variants
        """

        if self.to.variant_metrics is None:
            return

        for part_id, variant_name in enumerate(self.to.variant_metrics):
            # load one chunk of the dataset from memory
            _, x_val, yv_val, yp_val, _, _ = load_pgn_dataset(dataset_type="val",
                                                                         part_id=part_id,
                                                                         normalize=self.tc.normalize,
                                                                         verbose=False,
                                                                         q_value_ratio=self.tc.q_value_ratio)

            if self.tc.select_policy_from_plane:
                val_iter = mx.io.NDArrayIter({'data': x_val}, {'value_label': yv_val,
                                                               'policy_label': np.array(FLAT_PLANE_IDX)[
                                                                   yp_val.argmax(axis=1)]}, self.tc.batch_size)
            else:
                val_iter = mx.io.NDArrayIter({'data': x_val},
                                             {'value_label': yv_val, 'policy_label': yp_val.argmax(axis=1)},
                                             self.tc.batch_size)

            results = self._model.score(val_iter, self.to.metrics)
            prefix = "val_"

            for entry in results:
                name = variant_name + "_" + entry[0]
                value = entry[1]
                print(" - %s%s: %.4f" % (prefix, name, value), end="")
                # add the metrics to the tensorboard event file
                if self.tc.log_metrics_to_tensorboard:
                    self.sum_writer.add_scalar(name, [prefix.replace("_", ""), value], self.k_steps)
        print()
