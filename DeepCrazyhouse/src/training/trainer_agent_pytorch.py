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
from pathlib import Path
import torch
import torch.nn as nn
from time import time
from rtpt import RTPT
from tqdm import tqdm_notebook
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

from DeepCrazyhouse.configs.train_config import TrainConfig, TrainObjects
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.training.pytorch.metrics import Metrics


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
        train_dataset = TensorDataset(torch.Tensor(self.x_train), torch.Tensor(self.yv_train),
                                      torch.Tensor(self.yp_train), torch.Tensor(self.plys_to_end))
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.tc.batch_size)
        # training
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            x_train, yv_train, yp_train, plys_to_end = batch

            x_train = x_train.to(self._ctx)
            yv_train = yv_train.to(self._ctx)
            yp_train = yp_train.to(self._ctx)
            plys_to_end = plys_to_end.to(self._ctx)

            combined_loss = self.train_metrics.update(self._model, policy_loss, value_loss, self.tc.val_loss_factor, x_train, yv_train, yp_train)

            combined_loss.backward()
            lr, momentum = 0, 0
            for param_group in self.optimizer.param_groups:
                lr = lr_schedule(local_step)
                param_group['lr'] = lr

                momentum = momentum_schedule(local_step)
                param_group['momentum'] = momentum

            self.optimizer.step()

            if (batches_until_eval is not None and (batch_idx + 1) % batches_until_eval == 0) \
                    or (batch_idx + 1) == len(train_loader):
                msg = f' epoch: {epoch}, batch index: {batch_idx + 1}, train value loss: {m_train.value_loss():5f},'\
                      f' train policy loss: {m_train.policy_loss():5f}, train policy acc: {m_train.policy_acc():5f}'

                # log last lr and momentum
                self.sum_writer.add_scalar('Hyperparameter/Learning Rate', lr, global_step)
                self.sum_writer.add_scalar('Hyperparameter/Momentum', momentum, global_step)

                # log aggregated train stats
                m_train.log_to_tensorboard(writer_train, global_step)
                m_train.reset()

                if val_loader is not None:
                    model.eval()
                    m_val = get_val_loss(model, value_loss_ratio, value_loss, policy_loss, device, val_loader)

                    m_val.log_to_tensorboard(writer_val, global_step)
                    msg += f', val value loss: {m_val.value_loss():5f}, val policy loss: {m_val.policy_loss():5f},'\
                           f' val policy acc: {m_val.policy_acc():5f}'

                print(msg)

            global_step += 1
            local_step += 1
            progress.update(1)

    progress.close()

    self.sum_writer.close()
    if val_loader is not None:
        writer_val.close()

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

