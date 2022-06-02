"""
@file: trainer_agent_pytorch.py
Created on 31.05.22
@project: CrazyAra
@author: queensgambit

Definition of the main training loop done in pytorch.
Partially based on:
https://gitlab.com/jweil/PommerLearn/-/blob/master/pommerlearn/training/train_cnn.py
"""

import random
import logging
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from time import time
import datetime
from rtpt import RTPT
from tqdm import tqdm_notebook
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

from DeepCrazyhouse.configs.train_config import TrainConfig, TrainObjects
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.training.pytorch.metrics import Metrics
from DeepCrazyhouse.src.training.trainer_agent_mxnet import prepare_policy, return_metrics_and_stop_training


class TrainerAgentPytorch:
    """Main training loop"""

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        train_config: TrainConfig,
        train_objects: TrainObjects,
        use_rtpt: bool,
    ):
        """
        Class for training the neural network.
        :param net: The NN with loaded parameters that shall be trained.
        :param val_data: The validation data loaded with gluon DataLoader.
        :param train_config: An instance of the TrainConfig data class.
        :param train_objects: Am instance pf the TrainObject data class.
        :param use_rtpt: If True, an RTPT object will be created and modified within this class.
        """
        self.tc = train_config
        self.to = train_objects
        if self.to.metrics is None:
            self.to.metrics = {}
        self._model = model
        self._val_loader = val_loader
        self.x_train = self.yv_train = self.yp_train = None
        self._ctx = get_context(train_config.context, train_config.device_id)

        # define a summary writer that logs data and flushes to the file every 5 seconds
        if self.tc.log_metrics_to_tensorboard:
            self.sum_writer = SummaryWriter(log_dir=self.tc.export_dir+"logs", flush_secs=5)
        # Define the two loss functions
        self._cross_entropy_loss = nn.CrossEntropyLoss()
        self._l2_loss = nn.MSELoss()

        # Define the optimizer
        self.optimizer = create_optimizer(self._model, self.tc)

        self.ordering = list(range(self.tc.nb_parts))  # define a list which describes the order of the processed batches

        # decides if the policy indices shall be selected directly from spatial feature maps without dense layer
        self.batch_end_callbacks = [self.batch_callback]

        # few variables which are internally used
        self.val_loss_best = self.val_p_acc_best = self.k_steps_best = \
            self.old_label = self.value_out = self.t_s = None
        self.patience_cnt = self.batch_proc_tmp = None
        # calculate how many log states will be processed
        self.k_steps_end = round(self.tc.total_it / self.tc.batch_steps)
        if self.k_steps_end == 0:
            self.k_steps_end = 1
        self.k_steps = self.cur_it = self.nb_spikes = self.old_val_loss = self.continue_training = self.t_s_steps = None
        self._train_iter = self.graph_exported = self.val_metric_values = self.val_loss = self.val_p_acc = None
        self.val_metric_values_best = None

        self.use_rtpt = use_rtpt

        if use_rtpt:
            # we use k-steps instead of epochs here
            self.rtpt = RTPT(name_initials=self.tc.name_initials, experiment_name='crazyara',
                             max_iterations=self.k_steps_end-self.tc.k_steps_initial)

    def train(self, cur_it=None):
        """
        Training model
        :param cur_it: Current iteration which is used for the learning rate and momentum schedule.
         If set to None it will be initialized
        :return: return_metrics_and_stop_training()
        """

        self._setup_variables(cur_it)

        while self.continue_training:
            # reshuffle the ordering of the training game batches (shuffle works in place)
            random.shuffle(self.ordering)

            self.epoch += 1
            logging.info("EPOCH %d", self.epoch)
            logging.info("=========================")
            self.t_s_steps = time()
            self._model.init_optimizer(optimizer=self.optimizer)

            for part_id in tqdm_notebook(self.ordering):
                self._train_one_dataset_chunk(part_id)

    def _train_one_dataset_chunk(self, part_id):
        # load one chunk of the dataset from memory
        _, self.x_train, self.yv_train, self.yp_train, self.plys_to_end, _ = load_pgn_dataset(dataset_type="train",
                                                                                         part_id=part_id,
                                                                                         normalize=self.tc.normalize,
                                                                                         verbose=False,
                                                                                         q_value_ratio=self.tc.q_value_ratio)

        self.yp_train = prepare_policy(y_policy=self.yp_train, select_policy_from_plane=self.tc.select_policy_from_plane,
                                  sparse_policy_label=self.tc.sparse_policy_label,
                                  is_policy_from_plane_data=self.tc.is_policy_from_plane_data)

        # update the train_data object
        train_dataset = TensorDataset(torch.Tensor(self.x_train), torch.Tensor(self.yv_train),
                                      torch.Tensor(self.yp_train), torch.Tensor(self.plys_to_end))
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.tc.batch_size, num_workers=self.tc.cpu_count)

        for _, (data, value_label, policy_label, plys_label) in enumerate(train_loader):
            data = data.to(self._ctx)
            value_label = value_label.to(self._ctx)
            policy_label = policy_label.to(self._ctx)
            plys_label = plys_label.to(self._ctx)

            # update a dummy metric to see a proper progress bar
            #  (the metrics will get evaluated at the end of 100k steps)
            if self.batch_proc_tmp > 0:
                self.to.metrics["value_loss"].update(self.old_label, value_out)

            self.old_label = value_label

            value_out, policy_out = self._model(data)
            policy_out = policy_out.softmax()
            value_loss = self._l2_loss(value_out, value_label)
            policy_loss = self._cross_entropy_loss(policy_out, policy_label)
            # weight the components of the combined loss
            combined_loss = (
                    self.tc.val_loss_factor * value_loss + self.tc.policy_loss_factor * policy_loss
            )

            combined_loss.backward()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.to.lr_schedule(self.cur_it)  # update the learning rate
                param_group['momentum'] = self.to.momentum_schedule(self.cur_it)  # update the momentum

            self.optimizer.step()

            self.cur_it += 1
            self.batch_proc_tmp += 1
            # add the graph representation of the network to the tensorboard log file
            if not self.graph_exported and self.tc.log_metrics_to_tensorboard:
                self.sum_writer.add_graph(self._model)
                self.graph_exported = True

            if self.batch_proc_tmp >= self.tc.batch_steps:  # show metrics every thousands steps
                # log the current learning rate
                # update batch_proc_tmp counter by subtracting the batch_steps
                batch_proc_tmp = self.batch_proc_tmp - self.tc.batch_steps
                ms_step = ((time() - self.t_s_steps) / self.tc.batch_steps) * 1000  # measure elapsed time
                # update the counters
                self.k_steps += 1
                self.patience_cnt += 1
                logging.info("Step %dK/%dK - %dms/step", self.k_steps, self.k_steps_end, ms_step)
                logging.info("-------------------------")
                logging.debug("Iteration %d/%d", self.cur_it, self.tc.total_it)
                logging.debug("lr: %.7f - momentum: %.7f", self.to.lr_schedule(self.cur_it), self.to.momentum_schedule(self.cur_it))
                train_metric_values = evaluate_metrics(
                    self.to.metrics,
                    train_data,
                    self._model,
                    nb_batches=10,  # 25,
                    ctx=self._ctx,
                    sparse_policy_label=self.tc.sparse_policy_label,
                    apply_select_policy_from_plane=self.tc.select_policy_from_plane and not self.tc.is_policy_from_plane_data
                )
                val_metric_values = evaluate_metrics(
                    self.to.metrics,
                    self._val_data,
                    self._model,
                    nb_batches=None,
                    ctx=self._ctx,
                    sparse_policy_label=self.tc.sparse_policy_label,
                    apply_select_policy_from_plane=self.tc.select_policy_from_plane and not self.tc.is_policy_from_plane_data
                )
                if self.use_rtpt:
                    # update process title according to loss
                    self.rtpt.step(subtitle=f"loss={val_metric_values['loss']:2.2f}")
                if self.tc.use_spike_recovery and (
                        old_val_loss * self.tc.spike_thresh < val_metric_values["loss"]
                        or np.isnan(val_metric_values["loss"])
                ):  # check for spikes
                    self.nb_spikes += 1
                    logging.warning(
                        "Spike %d/%d occurred - val_loss: %.3f",
                        self.nb_spikes,
                        self.tc.max_spikes,
                        val_metric_values["loss"],
                    )
                    if self.nb_spikes >= self.tc.max_spikes:
                        self.val_loss = val_metric_values["loss"]
                        self.val_p_acc = val_metric_values["policy_acc"]
                        logging.debug("The maximum number of spikes has been reached. Stop training.")
                        # finally stop training because the number of lr drops has been achieved
                        print()
                        print(
                            "Elapsed time for training(hh:mm:ss): "
                            + str(datetime.timedelta(seconds=round(time() - self.t_s)))
                        )

                        if self.tc.log_metrics_to_tensorboard:
                            self.sum_writer.close()
                        return return_metrics_and_stop_training(self.k_steps, val_metric_values, self.k_steps_best,
                                                                self.val_metric_values_best)

                    logging.debug("Recover to latest checkpoint")
                    model_path = self.tc.export_dir + "weights/model-%.5f-%.3f-%04d.params" % (
                        self.val_loss_best,
                        self.val_p_acc_best,
                        self.k_steps_best,
                    )  # Load the best model once again
                    logging.debug("load current best model:%s", model_path)
                    self._net.load_parameters(model_path, ctx=self._ctx)
                    k_steps = self.k_steps_best
                    logging.debug("k_step is back at %d", self.k_steps_best)
                    # print the elapsed time
                    t_delta = time() - self.t_s_steps
                    print(" - %.ds" % t_delta)
                    t_s_steps = time()
                else:
                    # update the val_loss_value to compare with using spike recovery
                    old_val_loss = val_metric_values["loss"]
                    # log the metric values to tensorboard
                    self._log_metrics(train_metric_values, global_step=self.k_steps, prefix="train_")
                    self._log_metrics(val_metric_values, global_step=self.k_steps, prefix="val_")

                    if self.tc.export_grad_histograms:
                        grads = []
                        # logging the gradients of parameters for checking convergence
                        for _, name in enumerate(self._param_names):
                            if "bn" not in name and "batch" not in name and name != "policy_flat_plane_idx":
                                grads.append(self._params[name].grad())
                                self.sum_writer.add_histogram(
                                    tag=name, values=grads[-1], global_step=self.k_steps, bins=20
                                )

                    # check if a new checkpoint shall be created
                    if self.val_loss_best is None or val_metric_values["loss"] < self.val_loss_best:
                        # update val_loss_best
                        val_loss_best = val_metric_values["loss"]
                        val_p_acc_best = val_metric_values["policy_acc"]
                        val_metric_values_best = val_metric_values
                        k_steps_best = self.k_steps

                        if self.tc.export_weights:
                            prefix = self.tc.export_dir + "weights/model-%.5f-%.3f" \
                                     % (val_loss_best, val_p_acc_best)
                            # the export function saves both the architecture and the weights
                            export_as_script_module(self._model, batch_size, data, "./weights")
                            print()
                            logging.info("Saved checkpoint to %s-%04d.params", prefix, k_steps_best)

                        patience_cnt = 0  # reset the patience counter
                    # print the elapsed time
                    t_delta = time() - self.t_s_steps
                    print(" - %.ds" % t_delta)
                    t_s_steps = time()

                    # log the samples per second metric to tensorboard
                    self.sum_writer.add_scalar(
                        tag="samples_per_second",
                        scalar_value={"hybrid_sync": data.shape[0] * self.tc.batch_steps / t_delta},
                        global_step=self.k_steps,
                    )

                    # log the current learning rate
                    self.sum_writer.add_scalar(tag="lr", scalar_value=self.to.lr_schedule(self.cur_it), global_step=self.k_steps)
                    # log the current momentum value
                    self.sum_writer.add_scalar(
                        tag="momentum", scalar_value=self.to.momentum_schedule(self.cur_it), global_step=k_steps
                    )

                    if self.cur_it >= self.tc.total_it:

                        val_loss = val_metric_values["loss"]
                        val_p_acc = val_metric_values["policy_acc"]
                        logging.debug("The number of given iterations has been reached")
                        # finally stop training because the number of lr drops has been achieved
                        print()
                        print(
                            "Elapsed time for training(hh:mm:ss): "
                            + str(datetime.timedelta(seconds=round(time() - self.t_s)))
                        )

                        if self.tc.log_metrics_to_tensorboard:
                            self.sum_writer.close()

                        return return_metrics_and_stop_training(k_steps, val_metric_values, self.k_steps_best,
                                                                self.val_metric_values_best)


    # make sure to empty cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    def _setup_variables(self, cur_it):
        if self.tc.seed is not None:
            random.seed(self.tc.seed)
        # define and initialize the variables which will be used
        self.t_s = time()
        # track on how many batches have been processed in this epoch
        self.patience_cnt = self.epoch = self.batch_proc_tmp = 0
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
        if self.use_rtpt:
            # Start the RTPT tracking
            self.rtpt.start()

        self.train_metrics = Metrics()


def create_optimizer(model: nn.Module, train_config: TrainConfig):
    if train_config.optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=train_config.max_lr, momentum=train_config.max_momentum,
                               weight_decay=train_config.wd)
    raise Exception(f"Selected optimizer {train_config.optimizer_name} is not supported.")


def get_context(context: str, device_id: int):
    """
    Returns the computation context as  Pytorch device object.
    :param context: Computational context either "gpu" or "cpu"
    :param device_id: Device index to use (only relevant for context=="gpu")
    :return: Pytorch device object
    """
    if context == "gpu":
        if torch.cuda.is_available():
            return torch.cuda.device(device_id)
        logging.info("No cuda device available. Fallback to CPU")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def export_model(model, batch_sizes, input_shape, dir=Path('.'), torch_cpu=True, torch_cuda=True, onnx=True,
                 verbose=False):
    """
    Exports the model in ONNX and Torch Script Module.

    :param model: Pytorch model
    :param batch_sizes: List of batch sizes to use for export
    :param input_shape: Input shape of the model
    :param dir: The base path for all models
    :param torch_cpu: Whether to export as script module with cpu inputs
    :param torch_cuda: Whether to export as script module with cuda inputs
    :param onnx: Whether to export as onnx
    :param verbose: Print debug information
    """

    if dir.exists():
        # make sure that all the content is deleted first so we don't run into strange caching issues
        dir.rm_dir(dir, keep_empty_dir=False)

    dir.mkdir(parents=True, exist_ok=False)

    onnx_dir = dir / "onnx"
    if torch_cpu:
        onnx_dir.mkdir(parents=True, exist_ok=False)

    cpu_dir = dir / "torch_cpu"
    if torch_cpu:
        cpu_dir.mkdir(parents=True, exist_ok=False)

    torch_cuda = torch_cuda and torch.cuda.is_available()
    cuda_dir = dir / "torch_cuda"
    if torch_cuda:
        cuda_dir.mkdir(parents=True, exist_ok=False)

    model = model.eval()

    for batch_size in batch_sizes:
        dummy_input = torch.ones(batch_size, input_shape[0], input_shape[1], input_shape[2], dtype=torch.float)

        if model.is_stateful:
            dummy_input = model.flatten(dummy_input, model.get_init_state_bf_flat(batch_size, "cpu"))
            model.set_input_options(sequence_length=None, has_state_input=True)
        else:
            dummy_input = model.flatten(dummy_input, None)
            model.set_input_options(sequence_length=None, has_state_input=False)

        if onnx:
            dummy_input = dummy_input.cpu()
            model = model.cpu()
            export_to_onnx(model, batch_size, dummy_input, onnx_dir)

        if torch_cpu:
            dummy_input = dummy_input.cpu()
            model = model.cpu()
            export_as_script_module(model, batch_size, dummy_input, cpu_dir)

        if torch_cuda:
            dummy_input = dummy_input.cuda()
            model = model.cuda()
            export_as_script_module(model, batch_size, dummy_input, cuda_dir)

        if verbose:
            print("Input shape: ")
            print(dummy_input.shape)
            print("Output shape: ")
            for i, e in enumerate(model(dummy_input)):
                print(f"{i}: {e.shape}")


def export_to_onnx(model, batch_size, dummy_input, dir) -> None:
    """
    Exports the model to ONNX format to allow later import in TensorRT.

    :param model: Pytorch model
    :param batch_size: The batch size of the input
    :param dummy_input: Dummy input which defines the input shape for the model
    :return:
    """
    if model.is_stateful:
        input_names = ["data"]
        output_names = ["value_out", "policy_out", "auxiliary_out"]
    else:
        input_names = ["data"]
        output_names = ["value_out", "policy_out"]

    torch.onnx.export(model, dummy_input, str(dir / Path(f"model-bsize-{batch_size}.onnx")), input_names=input_names,
                      output_names=output_names)


def export_as_script_module(model, batch_size, dummy_input, dir) -> None:
    """
    Exports the model to a Torch Script Module to allow later import in C++.

    :param model: Pytorch model
    :param batch_size: The batch size of the input
    :param dummy_input: Dummy input which defines the input shape for the model
    :return:
    """

    # generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, dummy_input)

    # serialize script module to file
    traced_script_module.save(str(dir / Path(f"model-bsize-{batch_size}.pt")))


def reset_metrics(metrics):
    """
    Resets all metric entries in a dictionary object
    :param metrics:
    :return:
    """
    for metric in metrics.values():
        metric.reset()


def evaluate_metrics(metrics, data_iterator, model, nb_batches, ctx, sparse_policy_label=False,
                     apply_select_policy_from_plane=True):
    """
    Runs inference of the network on a data_iterator object and evaluates the given metrics.
    The metric results are returned as a dictionary object.

    :param metrics: List of mxnet metrics which must have the
    names ['value_loss', 'policy_loss', 'value_acc_sign', 'policy_acc']
    :param data_iterator: Pytorch data iterator object
    :param model: Pytorch model handle
    :param nb_batches: Number of batches to evaluate (early stopping).
     If set to None all batches of the data_iterator will be evaluated
    :param ctx: Pytorch data context
    :param sparse_policy_label: Should be set to true if the policy uses one-hot encoded targets
     (e.g. supervised learning)
    :param apply_select_policy_from_plane: If true, given policy label is converted to policy map index
    :return:
    """
    reset_metrics(metrics)
    for i, (data, value_label, policy_label) in enumerate(data_iterator):
        data = data.to(ctx)
        value_label = value_label.to(ctx)
        policy_label = policy_label.to(ctx)

        value_out, policy_out = model(data)

        # update the metrics
        metrics["value_loss"].update(preds=value_out, labels=value_label)
        metrics["policy_loss"].update(preds=policy_out.softmax(),
                                      labels=policy_label)
        metrics["value_acc_sign"].update(preds=value_out, labels=value_label)
        metrics["policy_acc"].update(preds=policy_out.argmax(axis=1),
                                     labels=policy_label)
        # stop after evaluating x batches (only recommended to use this for the train set evaluation)
        if nb_batches and i == nb_batches:
            break

    metric_values = {"loss": 0.01 * metrics["value_loss"].get()[1] + 0.99 * metrics["policy_loss"].get()[1]}

    for metric in metrics.values():
        metric_values[metric.get()[0]] = metric.get()[1]
    return metric_values
