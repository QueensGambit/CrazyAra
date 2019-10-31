"""
@file: rl_loop.py
Created on 12.10.19
@project: crazy_ara
@author: queensgambit

Main reinforcement learning for generating games and train the neural network.
"""

from subprocess import PIPE, Popen
import datetime
import time
from numcodecs import Blosc
import zarr
import os
import logging
import mxnet as mx
import numpy as np
import glob
from multiprocessing import cpu_count

from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.runtime.color_logger import enable_color_logging
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.src.training.trainer_agent_mxnet import TrainerAgentMXNET, adjust_loss_weighting, prepare_policy
from DeepCrazyhouse.src.training.trainer_agent import acc_sign, cross_entropy
from DeepCrazyhouse.src.training.lr_schedules.lr_schedules import ConstantSchedule,\
    MomentumSchedule, OneCycleSchedule, LinearWarmUp


def read_output(proc, last_line=b"readyok\n"):
    """
    Reads the output of a process pip until the given last line has been reached.
    :param proc: Process to be read
    :param last_line Content when to stop reading (e.g. b'\n', b'', b"readyok\n")
    :return:
    """
    while True:
        line = proc.stdout.readline()
        # error = proc.stderr.readline()
        print(line)
        # print(error)
        if line == last_line:
            break


def compress_zarr_dataset(data, export_dir, compression='lz4', clevel=5, start_idx=0, end_idx=0):
    """
    Loads in a zarr data set and exports it with a given compression type and level
    :param data: Zarr data set which will be compressed
    :param export_dir: Export directory for the compressed data set
    :param compression: Compression type
    :param clevel: Compression level
    :param end_idx: If end_idx != 0 the data set will be exported to the specified index,
    excluding the sample at end_idx (e.g. end_idx = len(x) will export it fully)
    :return:
    """
    # include current timestamp in dataset export file
    timestmp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")
    timestmp_dir = export_dir + timestmp + "/"
    # create a directory of the current timestmp
    if not os.path.exists(timestmp_dir):
        os.makedirs(timestmp_dir)

    zarr_path = timestmp_dir + timestmp + ".zip"
    compressor = Blosc(cname=compression, clevel=clevel, shuffle=Blosc.SHUFFLE)

    # open a dataset file and create arrays
    store = zarr.ZipStore(zarr_path, mode="w")
    zarr_file = zarr.group(store=store, overwrite=True)

    for key in data.keys():
        if end_idx == 0:
            x = data[key]
        else:
            x = data[key][start_idx:end_idx]

        array_shape = list(x.shape)
        array_shape[0] = 128
        # export array
        zarr_file.create_dataset(
            name=key,
            data=x,
            shape=x.shape,
            dtype=type(x.flatten()[0]),
            chunks=array_shape,
            synchronizer=zarr.ThreadSynchronizer(),
            compression=compressor,
        )
    store.close()
    logging.debug("dataset was exported to: %s", zarr_path)


def create_dir(directory):
    """
    Creates a given director in case it doesn't exists already
    :param directory: Directory path
    :return:
    """

    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info("Created directory %s" % directory)


class RLLoop:
    """
    This class uses the C++ binary to generate games and updates the network from the newly acquired games
    """

    def __init__(self, crazyara_binary_dir="./", nb_games_to_update=1024, nb_arena_games=100):
        """
        Constructor
        :param crazyara_binary_dir: Directory to the C++ binary
        :param nb_games_to_update: Number of games to generate before the neural network will be updated
        :param nb_arena_games: Number of games which will be generated in a tournament setting in order to determine
        if the updated NN weights are stronger than the old one and by how much.
        """

        self.crazyara_binary_dir = crazyara_binary_dir
        self.proc = None
        self.nb_games_to_update = nb_games_to_update
        if nb_arena_games % 2 == 1:
            raise Exception("The number of tournament games should be an even number to avoid giving one player more"
                            "games as white.")
        self.nb_arena_games = nb_arena_games

        self.cwd = os.getcwd() + '/'
        logging.info("Current working directory %s" % self.cwd)

        # directories for training
        create_dir("logs")
        create_dir("weights")

    def initialize(self):
        """
        Initializes the CrazyAra binary and loads the neural network weights
        :return:
        """
        # initialize
        self.proc = Popen([self.crazyara_binary_dir+"CrazyAra"],
                          stdin=PIPE,
                          stdout=PIPE,
                          stderr=PIPE,
                          shell=False)

        # CrazyAra header
        read_output(self.proc, b'\n')
        read_output(self.proc, b'\n')

        # self.proc.stdin.write(b'setoption name Model_Directory value %s\n' % bytes(self.crazyara_binary_dir+"model/",
        #                                                                            'utf-8'))
        set_uci_param(self.proc, "Nodes", 800)
        set_uci_param(self.proc, "Centi_Temperature", 10)
        set_uci_param(self.proc, "Temperature_Moves", 7)

        # load network
        self.proc.stdin.write(b"isready\n")
        self.proc.stdin.flush()
        read_output(self.proc, b"readyok\n")

    def generate_games(self):
        """
        Requests the binary to generate X number of games
        :return:
        """
        self.proc.stdin.write(b"selfplay %d\n" % self.nb_games_to_update)
        self.proc.stdin.flush()
        read_output(self.proc, b"readyok\n")

    def compress_dataset(self, validation_size=0.1):
        """
        Loads the uncompressed data file, select all sample until the index specified in "startIdx.txt",
        compresses it and exports it
        :param validation_size: Floating value that should be in [0.0, 1.0] and defines the proportion of the data set that is
         used as validation data
        :return:
        """
        with open("startIdx.txt", "r") as file:
            end_idx = int(file.readline())
        end_idx_train = int(end_idx * (1-validation_size))
        data = zarr.load(self.crazyara_binary_dir + "data.zarr")
        compress_zarr_dataset(data, "./export/train/", start_idx=0, end_idx=end_idx_train)
        compress_zarr_dataset(data, "./export/val/", start_idx=end_idx_train+1, end_idx=end_idx)

    def update_network(self):
        """
        Updates the neural network with the newly acquired games from the replay memory
        :return:
        """

        main_config["planes_train_dir"] = self.cwd + "export/"

        # set the context on CPU, switch to GPU if there is one available (strongly recommended for training)
        ctx = mx.gpu(0)
        # set a specific seed value for reproducibility
        seed = 7  # 42

        export_weights = True
        log_metrics_to_tensorboard = True
        export_grad_histograms = True
        #  div factor is a constant which can be used to reduce the batch size and learning rate respectively
        div_factor = 1 #4
        # use a value smaller 1 if you encounter memory allocation errors

        # batch_steps = 1000 means for example that every 1000 batches the validation set gets processed
        batch_steps = 1 * div_factor  # this defines how often a new checkpoint will be saved and the metrics evaluated
        # k_steps_initial defines how many steps have been trained before
        # (k_steps_initial != 0 if you continue training from a checkpoint)
        k_steps_initial = 0  # 498
        cur_it = k_steps_initial * batch_steps  # iteration counter used for the momentum and learning rate schedule
        # these are the weights to continue training with
        # symbol_file = 'model_init-symbol.json' #model-1.19246-0.603-symbol.json'
        # params_file = 'model_init-0000.params' #model-1.19246-0.603-0223.params'
        symbol_file = 'model-1.19246-0.603-symbol.json'
        params_file = 'model-1.19246-0.603-0223.params'

        batch_size = int(
            1024 / div_factor)  # 1024 # the batch_size needed to be reduced to 1024 in order to fit in the GPU 1080Ti
        # 4096 was originally used in the paper -> works slower for current GPU
        # 2048 was used in the paper Mastering the game of Go without human knowledge and fits in GPU memory
        # typically if you half the batch_size, you should double the lr

        # optimization parameters
        optimizer_name = "nag"
        max_lr = 0.1 / div_factor #0.35 / div_factor
        min_lr = 0.00001 / div_factor #0.2 / div_factor  # 0.00001
        max_momentum = 0.95
        min_momentum = 0.8
        # loads a previous checkpoint if the loss increased significantly
        use_spike_recovery = True
        # stop training as soon as max_spikes has been reached
        max_spikes = 20
        # define spike threshold when the detection will be triggered
        spike_thresh = 1.5
        # weight decay
        wd = 1e-4
        # dropout_rate = 0  # 0.2
        # weight the value loss a lot lower than the policy loss in order to prevent overfitting
        val_loss_factor = 1  # 0.01
        policy_loss_factor = 1  # 0.99
        discount = 1.0

        normalize = True  # define whether to normalize input data to [0,1]
        nb_epochs = 3  # 7 # define how many epochs the network will be trained

        select_policy_from_plane = True  # Boolean if potential legal moves will be selected from final policy output
        # Boolean if the policy target is one-hot encoded (sparse=True) or a target distribution (sparse=False)
        sparse_policy_label = False
        # use_mxnet_style = True  # Decide between mxnet and gluon style for training
        # Fixing the random seed
        mx.random.seed(seed)

        _, x, y_value, y_policy, _, _ = load_pgn_dataset(dataset_type="train",
                                                                    part_id=0,
                                                                    normalize=normalize,
                                                                    verbose=False)

        y_policy = prepare_policy(y_policy, select_policy_from_plane, sparse_policy_label)

        symbol = mx.sym.load("model/" + symbol_file)
        symbol = adjust_loss_weighting(symbol, val_loss_factor, policy_loss_factor,
                                       "value_tanh0_output", "flatten0_output")
                                        # "value_out_output", "policy_out_output")

        nb_parts = len(glob.glob(main_config['planes_train_dir'] + '**/*'))
        nb_it_per_epoch = (len(x) * nb_parts) // batch_size  # calculate how many iterations per epoch exist
        # one iteration is defined by passing 1 batch and doing backprop
        total_it = int(nb_it_per_epoch * nb_epochs)

        # lr_schedule = ConstantSchedule(min_lr)
        lr_schedule = OneCycleSchedule(start_lr=max_lr / 8, max_lr=max_lr, cycle_length=total_it * .3,
                                       cooldown_length=total_it * .6, finish_lr=min_lr)
        lr_schedule = LinearWarmUp(lr_schedule, start_lr=min_lr, length=total_it / 30)
        # plot_schedule(lr_schedule, iterations=total_it)
        momentum_schedule = MomentumSchedule(lr_schedule, min_lr, max_lr, min_momentum, max_momentum)

        if select_policy_from_plane:
            val_iter = mx.io.NDArrayIter({'data': x}, {'value_label': y_value,
                                                       'policy_label': y_policy}, batch_size)
        else:
            val_iter = mx.io.NDArrayIter({'data': x},
                                         {'value_label': y_value, 'policy_label': y_policy}, batch_size)

        nb_parts = len(glob.glob(main_config['planes_train_dir'] + '**/*'))
        nb_it_per_epoch = (len(x) * nb_parts) // batch_size  # calculate how many iterations per epoch exist
        # one iteration is defined by passing 1 batch and doing backprop
        total_it = int(nb_it_per_epoch * nb_epochs)
        CPU_COUNT = cpu_count()

        input_shape = x[0].shape
        model = mx.mod.Module(symbol=symbol, context=ctx, label_names=['value_label', 'policy_label'])
        mx.viz.print_summary(
            symbol,
            shape={'data': (1, input_shape[0], input_shape[1], input_shape[2])},
        )
        model.bind(for_training=True,
                   data_shapes=[('data', (batch_size, input_shape[0], input_shape[1], input_shape[2]))],
                   label_shapes=val_iter.provide_label)
        model.load_params("model/" + params_file)

        metrics = [
            mx.metric.MSE(name='value_loss', output_names=['value_output'], label_names=['value_label']),
            mx.metric.create(acc_sign, name='value_acc_sign', output_names=['value_output'],
                             label_names=['value_label']),
            mx.metric.Accuracy(axis=1, name='policy_acc', output_names=['policy_output'],
                               label_names=['policy_label'])
        ]

        if sparse_policy_label:
            # the default cross entropy only supports sparse lables
            metrics.append(mx.metric.CrossEntropy(name='policy_loss', output_names=['policy_output'],
                             label_names=['policy_label']))
        else:
            metrics.append(mx.metric.create(cross_entropy, name='policy_loss', output_names=['policy_output'],
                             label_names=['policy_label']))

        train_agent = TrainerAgentMXNET(model, symbol, val_iter, nb_parts, lr_schedule, momentum_schedule, total_it,
                                        optimizer_name, wd=wd, batch_steps=batch_steps,
                                        k_steps_initial=k_steps_initial, cpu_count=CPU_COUNT - 2, batch_size=batch_size,
                                        normalize=normalize, export_weights=export_weights,
                                        export_grad_histograms=export_grad_histograms,
                                        log_metrics_to_tensorboard=log_metrics_to_tensorboard, ctx=ctx, metrics=metrics,
                                        use_spike_recovery=use_spike_recovery, max_spikes=max_spikes,
                                        spike_thresh=spike_thresh, seed=seed,
                                        val_loss_factor=val_loss_factor, policy_loss_factor=policy_loss_factor,
                                        select_policy_from_plane=select_policy_from_plane, discount=discount,
                                        sparse_policy_label=sparse_policy_label)
        train_agent.train(cur_it)

    def compare_new_weights(self):
        """
        Compares the old nn-weights with the newly acquired one and returns the win-rate
        :return:
        """
        self.proc.stdin.write(b"arena %d\n" % self.nb_arena_games)
        self.proc.stdin.flush()
        read_output(self.proc, b"readyok\n")

    def create_new_contender(self):
        """
        Moves neural network architecture definition and .params file into the "model_contender" directory
        :return:
        """
        nn_files = os.listdir(self.cwd + "weights/")

        for nn_file in nn_files:
            nn_file_origin = self.cwd + "weights/" + nn_file
            nn_file_destination = self.cwd + "model_contender/" + nn_file
            os.rename(nn_file_origin, nn_file_destination)
            logging.debug("moved %s into %s" % (nn_file, nn_file_destination))


def set_uci_param(proc, name, value):
    """
    Sets the value for a given UCI-parameter in the binary.
    :param proc: Process to set the parameter for
    :param name: Name of the UCI-parameter
    :param value: Value for the UCI-parameter
    :return:
    """
    if isinstance(value, int):
        proc.stdin.write(b"setoption name %b value %d\n" % (bytes(name, encoding="ascii"), value))
    else:
        raise NotImplementedError(f"To set uci-parameters of type {type(value)} has not been implemented yet.")
    proc.stdin.flush()


if __name__ == "__main__":
    enable_color_logging()
    rl_loop = RLLoop(crazyara_binary_dir="./",
                     nb_games_to_update=200,
                     nb_arena_games=50)
    # rl_loop.initialize()
    # rl_loop.generate_games()
    # rl_loop.compress_dataset(validation_size=0.1)
    rl_loop.create_new_contender()
    # rl_loop.update_network()
    rl_loop.compare_new_weights()
