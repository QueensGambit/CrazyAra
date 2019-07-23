"""
@file: trainer_agent_mxnet.py
Created on 18.05.19
@project: CrazyAra
@author: queensgambit

Definition of the main training loop done in mxnet.
"""
import datetime
import logging
import random
from time import time
import mxnet as mx
import numpy as np
from mxboard import SummaryWriter
from tqdm import tqdm_notebook
from multiprocessing import Process, Lock, Queue
from DeepCrazyhouse.src.domain.crazyhouse.plane_policy_representation import FLAT_PLANE_IDX
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset


def acc_sign(y_true, y_pred):
    """
    Custom metric which is used to predict the winner of a game
    :param y_true: Ground truth value (np-array with values between -1, 0)
    :param y_pred: Predicted labels as numpy array
    :return:
    """
    return (np.sign(y_pred).flatten() == y_true).sum() / len(y_true)


def evaluate_metrics(metrics, data_iterator, model): #, nb_batches=None, ctx=mx.gpu(), select_policy_from_plane=False):
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


class TrainerAgentMXNET:  # Probably needs refactoring
    """Main training loop"""   
    # x_train = yv_train = yp_train = None

    def __init__(
        self,
        model,
        symbol,
        val_iter,
        nb_parts,
        lr_schedule,
        momentum_schedule,
        total_it,
        optimizer_name="nag",  # or "adam"
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
        metrics=None,  # clip_gradient=60,
        use_spike_recovery=True,
        max_spikes=5,
        spike_thresh=1.5,
        seed=42,
        val_loss_factor=0.01,
        policy_loss_factor=0.99,
        select_policy_from_plane=True,
    ):
        # Too many instance attributes (29/7) - Too many arguments (24/5) - Too many local variables (25/15)
        # Too few public methods (1/2)
        # , lr_warmup_k_steps=30, lr_warmup_init=0.01):
        if metrics is None:
            metrics = {}
        self._log_metrics_to_tensorboard = log_metrics_to_tensorboard
        self._ctx = ctx
        self._metrics = metrics
        self._model = model
        self._symbol = symbol
        self._graph_exported = False
        self._normalize = normalize
        self._lr_schedule = lr_schedule
        self._momentum_schedule = momentum_schedule
        self._total_it = total_it
        self._batch_size = batch_size
        self._export_grad_histograms = export_grad_histograms
        self._cpu_count = cpu_count
        self._k_steps_initial = k_steps_initial
        self._val_iter = val_iter
        self._export_weights = export_weights
        self._batch_steps = batch_steps
        self._use_spike_recovery = use_spike_recovery
        self._max_spikes = max_spikes
        self._spike_thresh = spike_thresh
        self._seed = seed
        self._val_loss_factor = val_loss_factor
        self._policy_loss_factor = policy_loss_factor
        self.x_train = self.yv_train = self.yp_train = None
        # define a summary writer that logs data and flushes to the file every 5 seconds
        if log_metrics_to_tensorboard:
            self.sum_writer = SummaryWriter(logdir="./logs", flush_secs=5, verbose=False)
        # Define the two loss functions
        self.optimizer_name = optimizer_name
        if optimizer_name == "adam":
            self.optimizer = mx.optimizer.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, lazy_update=True, rescale_grad=(1.0/batch_size))
        elif optimizer_name == "nag":
            self.optimizer = mx.optimizer.NAG(momentum=momentum_schedule(0), wd=wd, rescale_grad=(1.0/batch_size))
        else:
            raise Exception("%s is currently not supported as an optimizer." % optimizer_name)
        self.ordering = list(range(nb_parts))  # define a list which describes the order of the processed batches

        # decides if the policy indices shall be selected directly from spatial feature maps without dense layer
        self.select_policy_from_plane = select_policy_from_plane

        if self.select_policy_from_plane:
            # create a numpy instance for FLAT_PLANE_IDX
            self.FLAT_PLANE_IDX = np.array(FLAT_PLANE_IDX)

        self.batch_end_callbacks = [self.batch_callback]

        # few variables which are internally used
        self.val_loss_best = self.val_p_acc_best = self.k_steps_best = \
            self.old_label = self.value_out = self.t_s = None
        self.patience_cnt = self.batch_proc_tmp = self.k_steps_end = None
        self.k_steps = self.cur_it = self.nb_spikes = self.old_val_loss = self.continue_training = self.t_s_steps = None
        self._train_iter = self.graph_exported = self.val_metric_values = self.val_loss = self.val_p_acc = None

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
            if self._log_metrics_to_tensorboard:
                self.sum_writer.add_scalar(name, [prefix.replace("_", ""), metric_values[name]],
                                                        global_step)

    def train(self, cur_it=None):  # Probably needs refactoring
        """
        Training model
        :param cur_it: Current iteration which is used for the learning rate and momentum schedule.
         If set to None it will be initialized
        """
        # Too many local variables (44/15) - Too many branches (18/12) - Too many statements (108/50)
        # set a custom seed for reproducibility
        random.seed(self._seed)
        # define and initialize the variables which will be used
        self.t_s = time()
        # track on how many batches have been processed in this epoch
        self.patience_cnt = epoch = self.batch_proc_tmp = 0
        self.k_steps = self._k_steps_initial  # counter for thousands steps
        # calculate how many log states will be processed
        self.k_steps_end = self._total_it / self._batch_steps
        if cur_it is None:
            self.cur_it = self._k_steps_initial * 1000
        else:
            self.cur_it = cur_it
        self.nb_spikes = 0  # count the number of spikes that have been detected
        # initialize the loss to compare with, with a very high value
        self.old_val_loss = 9000
        self.graph_exported = False  # create a state variable to check if the net architecture has been reported yet
        self.continue_training = True
        self.optimizer.lr = self._lr_schedule(self.cur_it)
        if self.optimizer_name == "nag":
            self.optimizer.momentum = self._momentum_schedule(self.cur_it)

        if not self.ordering:  # safety check to prevent eternal loop
            raise Exception("You must have at least one part file in your planes-dataset directory!")

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
                _, self.x_train, self.yv_train, self.yp_train, _ = load_pgn_dataset(dataset_type="train",
                                                                                    part_id=part_id,
                                                                                    normalize=self._normalize,
                                                                                    verbose=False)
                self.yp_train = self.yp_train.argmax(axis=1)

                if self.select_policy_from_plane:
                    self.yp_train[:] = self.FLAT_PLANE_IDX[self.yp_train]

                self._train_iter = mx.io.NDArrayIter({'data': self.x_train},
                                                     {'value_label': self.yv_train, 'policy_label': self.yp_train},
                                                     self._batch_size,
                                                     shuffle=True)
                reset_metrics(self._metrics)
                for batch in self._train_iter:
                    self._model.forward(batch, is_train=True)  # compute predictions
                    for metric in self._metrics:  # update the metrics
                        self._model.update_metric(metric, batch.label)

                    self._model.backward()
                    # compute gradients
                    self._model.update()  # update parameters
                    self.batch_callback()

                    if not self.continue_training:
                        print(
                            'Elapsed time for training(hh:mm:ss): ' +
                            str(datetime.timedelta(seconds=round(time() - self.t_s))))

                        return (self.k_steps, self.val_loss, self.val_p_acc), \
                               (self.k_steps_best, self.val_loss_best, self.val_p_acc_best)

                # add the graph representation of the network to the tensorboard log file
                if not self.graph_exported and self._log_metrics_to_tensorboard:
                    self.sum_writer.add_graph(self._symbol)
                    self.graph_exported = True

    def _fill_train_metrics(self):
        """
        Fills in the training metrics
        :return:
        """
        self.train_metric_values = {}
        for metric in self._metrics:
            name, value = metric.get()
            self.train_metric_values[name] = value

        self.train_metric_values["loss"] = 0.01 * self.train_metric_values["value_loss"] + \
                                           0.99 * self.train_metric_values["policy_loss"]

    def recompute_eval(self):
        """
        Recomputes the score on the validataion data
        :return:
        """
        ms_step = ((time() - self.t_s_steps) / self._batch_steps) * 1000
        logging.info("Step %dK/%dK - %dms/step", self.k_steps, self.k_steps_end, ms_step)
        logging.info("-------------------------")
        logging.debug("Iteration %d/%d", self.cur_it, self._total_it)
        if self.optimizer_name == "nag":
            logging.debug("lr: %.7f - momentum: %.7f", self.optimizer.lr, self.optimizer.momentum)
        else:
            logging.debug("lr: %.7f - momentum: -", self.optimizer.lr)

        # the metric values have already been computed during training for the train set
        self._fill_train_metrics()

        self.val_metric_values = evaluate_metrics(
            self._metrics,
            self._val_iter,
            self._model,
        )
        if self._use_spike_recovery and (
                self.old_val_loss * self._spike_thresh < self.val_metric_values["loss"]
                or np.isnan(self.val_metric_values["loss"])
        ):  # check for spikes
            self.handle_spike()
        else:
            self.update_eval()

    def handle_spike(self):
        """
        Handles the occurence of a spike during training, in the case validation loss increased dramatically.
        :return:
        """
        self.nb_spikes += 1
        logging.warning(
            "Spike %d/%d occurred - val_loss: %.3f",
            self.nb_spikes,
            self._max_spikes,
            self.val_metric_values["loss"],
        )
        if self.nb_spikes >= self._max_spikes:
            val_loss = self.val_metric_values["loss"]
            val_p_acc = self.val_metric_values["policy_acc"]
            # finally stop training because the number of lr drops has been achieved

            logging.debug("The maximum number of spikes has been reached. Stop training.")
            self.continue_training = False

            if self._log_metrics_to_tensorboard:
                self.sum_writer.close()
            return (self.k_steps, val_loss, val_p_acc), (self.k_steps_best, self.val_loss_best,
                                                         self.val_p_acc_best)

        logging.debug("Recover to latest checkpoint")
        # Load the best model once again
        prefix = "./weights/model-%.5f-%.3f" % (self.val_loss_best, self.val_p_acc_best)

        logging.debug("load current best model:%s", prefix)
        # self._net.load_parameters(model_path, ctx=self._ctx)
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

            if self._export_weights:
                prefix = "./weights/model-%.5f-%.3f" % (self.val_loss_best, self.val_p_acc_best)
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
            value={"hybrid_sync": self._batch_size * self._batch_steps / t_delta},
            global_step=self.k_steps,
        )

        # log the current learning rate
        self.sum_writer.add_scalar(tag="lr", value=self._lr_schedule(self.cur_it),
                                   global_step=self.k_steps)
        if self.optimizer_name == "nag":
            # log the current momentum value
            self.sum_writer.add_scalar(
                tag="momentum", value=self._momentum_schedule(self.cur_it),
                global_step=self.k_steps
            )

        if self.cur_it >= self._total_it:

            self.continue_training = False

            self.val_loss = self.val_metric_values["loss"]
            self.val_p_acc = self.val_metric_values["policy_acc"]
            # finally stop training because the number of lr drops has been achieved
            logging.debug("The number of given iterations has been reached")

            if self._log_metrics_to_tensorboard:
                self.sum_writer.close()
                
    def batch_callback(self):
        """
        Callback which is executed after every batch to update the momentum and learning rate
        :return:
        """

        # update the learning rate and momentum
        self.optimizer.lr = self._lr_schedule(self.cur_it)
        if self.optimizer_name == "nag":
            self.optimizer.momentum = self._momentum_schedule(self.cur_it)

        self.cur_it += 1
        self.batch_proc_tmp += 1
    
        if self.batch_proc_tmp >= self._batch_steps:  # show metrics every thousands steps
            self.batch_proc_tmp = self.batch_proc_tmp - self._batch_steps
            # update the counters
            self.k_steps += 1
            self.patience_cnt += 1
            self.recompute_eval()
