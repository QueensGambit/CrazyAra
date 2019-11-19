"""
@file: rl_loop.py
Created on 12.10.19
@project: crazy_ara
@author: queensgambit

Main reinforcement learning for generating games and train the neural network.
"""

import os
import sys
import time
import glob
import zarr
import logging
import datetime
import argparse
import mxnet as mx
from numcodecs import Blosc
from subprocess import PIPE, Popen
from multiprocessing import cpu_count

sys.path.append("../../../")
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.configs.train_config import train_config
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.training.trainer_agent import acc_sign, cross_entropy
from DeepCrazyhouse.src.training.trainer_agent_mxnet import TrainerAgentMXNET, add_non_sparse_cross_entropy, prepare_policy
from DeepCrazyhouse.src.training.lr_schedules.lr_schedules import MomentumSchedule, OneCycleSchedule, LinearWarmUp


def read_output(proc, last_line=b"readyok\n"):
    """
    Reads the output of a process pip until the given last line has been reached.
    :param proc: Process to be read
    :param last_line Content when to stop reading (e.g. b'\n', b'', b"readyok\n")
    :return:
    """
    while True:
        line = proc.stdout.readline()
        print(line)
        if line == b'':
            error = proc.stderr.readline()
            logging.error(error)
            raise Exception("The process raised an error!")
        if line == last_line:
            break


def compress_zarr_dataset(data, file_path: str, compression='lz4', clevel=5, start_idx=0, end_idx=0):
    """
    Loads in a zarr data set and exports it with a given compression type and level
    :param data: Zarr data set which will be compressed
    :param file_path: File name path where the data will be exported (e.g. "./export/data.zip")
    :param compression: Compression type
    :param clevel: Compression level
    :param end_idx: If end_idx != 0 the data set will be exported to the specified index,
    excluding the sample at end_idx (e.g. end_idx = len(x) will export it fully)
    :return: timestmp_dir: directory where the files have been exported
    """
    compressor = Blosc(cname=compression, clevel=clevel, shuffle=Blosc.SHUFFLE)

    # open a dataset file and create arrays
    store = zarr.ZipStore(file_path, mode="w")
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
    logging.debug("dataset was exported to: %s", file_path)

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

    def __init__(self, args, nb_games_to_update=1024, nb_arena_games=100):
        """
        Constructor
        :param args: Command line arguments, see parse_args() for details.
        :param nb_games_to_update: Number of games to generate before the neural network will be updated
        :param nb_arena_games: Number of games which will be generated in a tournament setting in order to determine
        if the updated NN weights are stronger than the old one and by how much.
        """

        self.crazyara_binary_dir = args.crazyara_binary_dir
        self.proc = None
        self.nb_games_to_update = nb_games_to_update
        if nb_arena_games % 2 == 1:
            raise Exception("The number of tournament games should be an even number to avoid giving one player more"
                            "games as white.")
        self.nb_arena_games = nb_arena_games
        self.args = args
        self.device_name = "%s_%d" % (args.context, args.device_id)

        self.export_dir = self.crazyara_binary_dir + "export/train/"

        # change working directory (otherwise the .zip files would be generated at the .py location)
        os.chdir(self.crazyara_binary_dir)

        # directories for training
        create_dir(self.crazyara_binary_dir+"logs")
        create_dir(self.crazyara_binary_dir+"weights")

    def _get_current_model_arch_file(self):
        """
        Returns the filenames of the current active model architecture (.json) file
        """
        return glob.glob(self.crazyara_binary_dir + "model/*.json")[0]

    def _get_current_model_weight_file(self):
        """
        Returns the filenames of the current active model weight (.params) file
        :return:
        """
        return glob.glob(self.crazyara_binary_dir + "model/*.params")[0]

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

        self.proc.stdin.write(b'setoption name Model_Directory value %s\n' % bytes(self.crazyara_binary_dir+"model/",
                                                                                   'utf-8'))
        set_uci_param(self.proc, "Context", self.args.context)
        set_uci_param(self.proc, "Device_ID", self.args.device_id)
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

    def create_export_dir(self):
        # include current timestamp in dataset export file
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")
        time_stamp_dir = self.export_dir + time_stamp + "-" + self.device_name + "/"
        # create a directory of the current time_stamp
        if not os.path.exists(time_stamp_dir):
            os.makedirs(time_stamp_dir)

        return time_stamp_dir, time_stamp

    def _move_game_data_to_export_dir(self, export_dir: str):
        """
        Moves the generated games saved in .pgn format and the number how many games have been generated to the given export
         directory
        :param export_dir: Export directory for the newly generated data

        :return:
        """
        file_names = ["games_" + self.device_name + ".pgn",
                      "gameIdx_" + self.device_name + ".txt"]
        for file_name in file_names:
            os.rename(file_name, export_dir + file_name)

    def compress_dataset(self):
        """
        Loads the uncompressed data file, select all sample until the index specified in "startIdx.txt",
        compresses it and exports it
        :return:
        """
        data = zarr.load(self.crazyara_binary_dir + "data_" + self.device_name + ".zarr")

        export_dir, time_stamp = self.create_export_dir()
        zarr_path = export_dir + time_stamp + ".zip"
        compress_zarr_dataset(data, zarr_path, start_idx=0)
        self._move_game_data_to_export_dir(export_dir)

    def update_network(self):
        """
        Updates the neural network with the newly acquired games from the replay memory
        :return:
        """

        # main_config["planes_train_dir"] = self.cwd + "export/"

        # set the context on CPU, switch to GPU if there is one available (strongly recommended for training)
        ctx = mx.gpu(train_config["device_id"]) if train_config["context"] == "gpu"else mx.cpu()
        # set a specific seed value for reproducibility

        # Fixing the random seed
        mx.random.seed(train_config["seed"])

        nb_parts = len(glob.glob(main_config["planes_train_dir"] + '**/*.zip'))
        logging.info("number parts: %d" % nb_parts)

        if nb_parts <= 0:
            raise Exception('No .zip files for training available. Check the path in main_config["planes_train_dir"]:'
                            ' %s' % main_config["planes_train_dir"])

        _, x_val, y_val_value, y_val_policy, _, _ = load_pgn_dataset(dataset_type="val",
                                                                     part_id=0,
                                                                     normalize=train_config["normalize"],
                                                                     verbose=False,
                                                                     q_value_ratio=train_config["q_value_ratio"])

        y_val_policy = prepare_policy(y_val_policy, train_config["select_policy_from_plane"],
                                      train_config["sparse_policy_label"])

        symbol = mx.sym.load(self._get_current_model_arch_file())
        symbol = add_non_sparse_cross_entropy(symbol, train_config["val_loss_factor"],
                                              "value_tanh0_output", "flatten0_output")
                                        # "value_out_output", "policy_out_output")

        # calculate how many iterations per epoch exist
        nb_it_per_epoch = (len(x_val) * nb_parts) // train_config["batch_size"]
        # one iteration is defined by passing 1 batch and doing backprop
        total_it = int(nb_it_per_epoch * train_config["nb_epochs"])

        # lr_schedule = ConstantSchedule(min_lr)
        lr_schedule = OneCycleSchedule(start_lr=train_config["max_lr"] / 8, max_lr=train_config["max_lr"],
                                       cycle_length=total_it * .3,
                                       cooldown_length=total_it * .6, finish_lr=train_config["min_lr"])
        lr_schedule = LinearWarmUp(lr_schedule, start_lr=train_config["min_lr"], length=total_it / 30)
        # plot_schedule(lr_schedule, iterations=total_it)
        momentum_schedule = MomentumSchedule(lr_schedule, train_config["min_lr"], train_config["max_lr"],
                                             train_config["min_momentum"], train_config["max_momentum"])

        if train_config["select_policy_from_plane"]:
            val_iter = mx.io.NDArrayIter({'data': x_val}, {'value_label': y_val_value,
                                                           'policy_label': y_val_policy}, train_config["batch_size"])
        else:
            val_iter = mx.io.NDArrayIter({'data': x_val},
                                         {'value_label': y_val_value, 'policy_label': y_val_policy},
                                         train_config["batch_size"])

        # calculate how many iterations per epoch exist
        nb_it_per_epoch = (len(x_val) * nb_parts) // train_config["batch_size"]
        # one iteration is defined by passing 1 batch and doing backprop
        total_it = int(nb_it_per_epoch * train_config["nb_epochs"])
        CPU_COUNT = cpu_count()

        input_shape = x_val[0].shape
        model = mx.mod.Module(symbol=symbol, context=ctx, label_names=['value_label', 'policy_label'])
        # mx.viz.print_summary(
        #     symbol,
        #     shape={'data': (1, input_shape[0], input_shape[1], input_shape[2])},
        # )
        model.bind(for_training=True,
                   data_shapes=[('data', (train_config["batch_size"], input_shape[0], input_shape[1], input_shape[2]))],
                   label_shapes=val_iter.provide_label)
        model.load_params(self._get_current_model_weight_file())

        metrics = [
            mx.metric.MSE(name='value_loss', output_names=['value_output'], label_names=['value_label']),
            mx.metric.create(acc_sign, name='value_acc_sign', output_names=['value_output'],
                             label_names=['value_label']),
            mx.metric.Accuracy(axis=1, name='policy_acc', output_names=['policy_output'],
                               label_names=['policy_label'])
        ]

        if train_config["sparse_policy_label"]:
            # the default cross entropy only supports sparse lables
            metrics.append(mx.metric.CrossEntropy(name='policy_loss', output_names=['policy_output'],
                             label_names=['policy_label']))
        else:
            metrics.append(mx.metric.create(cross_entropy, name='policy_loss', output_names=['policy_output'],
                             label_names=['policy_label']))

        logging.info("Perfomance pre training")
        print(model.score(val_iter, metrics))

        train_agent = TrainerAgentMXNET(model, symbol, val_iter, nb_parts, lr_schedule, momentum_schedule, total_it,
                                        train_config["optimizer_name"], wd=train_config["wd"],
                                        batch_steps=train_config["batch_steps"],
                                        k_steps_initial=train_config["k_steps_initial"], cpu_count=CPU_COUNT - 2,
                                        batch_size=train_config["batch_size"], normalize=train_config["normalize"],
                                        export_weights=train_config["export_weights"],
                                        export_grad_histograms=train_config["export_grad_histograms"],
                                        log_metrics_to_tensorboard=train_config["log_metrics_to_tensorboard"], ctx=ctx,
                                        metrics=metrics, use_spike_recovery=train_config["use_spike_recovery"],
                                        max_spikes=train_config["max_spikes"],
                                        spike_thresh=train_config["spike_thresh"],
                                        seed=train_config["seed"], val_loss_factor=train_config["val_loss_factor"],
                                        policy_loss_factor=train_config["policy_loss_factor"],
                                        select_policy_from_plane=train_config["select_policy_from_plane"],
                                        discount=train_config["discount"],
                                        sparse_policy_label=train_config["sparse_policy_label"],
                                        q_value_ratio=train_config["q_value_ratio"],
                                        cwd=self.crazyara_binary_dir)
        # iteration counter used for the momentum and learning rate schedule
        cur_it = train_config["k_steps_initial"] * train_config["batch_steps"]
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
        nn_files = os.listdir(self.crazyara_binary_dir + "weights/")

        for nn_file in nn_files:
            nn_file_origin = self.crazyara_binary_dir + "weights/" + nn_file
            nn_file_destination = self.crazyara_binary_dir + "model_contender/" + nn_file
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


def parse_args(cmd_args: list):
    """
    Parses command-line argument and returns them as a dictionary object
    :param cmd_args: Command-line arguments (sys.argv[1:])
    :return: Parsed arguments as dictionary object
    """
    parser = argparse.ArgumentParser(description='Reinforcement learning loop')

    parser.add_argument("--crazyara-binary-dir", type=str, default="data/RL/",
                        help="directory where the CrazyAra executable is located and where the selfplay data will be "
                             "stored")
    parser.add_argument('--context', type=str, default="gpu",
                        help='Computational device context to use. Possible values ["cpu", "gpu"]. (default: gpu)')
    parser.add_argument("--device_id", type=int, default=0,
                        help="GPU index to use for selfplay generation and/or network training. (default: 0)")
    parser.add_argument("--trainer", default=False, action="store_true",
                        help="The given GPU index is used for training the neural network."
                             " The gpu trainer will stop generating games and update the network as soon as enough"
                             " training samples have been acquired.  (default: False)")
    parser.add_argument('--export-no-log', default=False, action="store_true",
                        help="By default the log messages are stored in {context}_{device}.log."
                             " If this parameter is enabled no log messages will be stored")

    args = parser.parse_args(cmd_args)

    if not os.path.exists(args.crazyara_binary_dir):
        raise Exception("Your given args.crazyara_binary_dir: %s does not exist. Make sure to define a valid directory")

    if args.crazyara_binary_dir[-1] != '/':
        args.crazyara_binary_dir += '/'

    if args.context not in ["cpu", "gpu"]:
        raise ValueError('Given value: %s for context is invalid. It must be in ["cpu", "gpu"].' % args.context)

    return args


def enable_logging(logging_lvl=logging.DEBUG, log_filename=None):
    """
    Enables logging for a given level
    :param logging_lvl: Specifies logging level (e.g. logging.DEBUG, logging.INFO...)
    :param log_filename: Will export all log message into given logfile (append mode) if not None
    :return:
    """
    root = logging.getLogger()
    root.setLevel(logging_lvl)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_lvl)

    formatting_method = "%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s"
    formatter = logging.Formatter(formatting_method, "%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_filename is not None:
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)


def get_log_filename(args):
    """
    Returns the file name for the log messages
    :param args: Command line arguments
    :return: Filename: str or None
    """
    if args.export_no_log is False:
        return "%s_%d.log" % (args.context, args.device_id)
    return None


def main():
    """
    Main function which is executed on start-up
    :return:
    """
    args = parse_args(sys.argv[1:])

    enable_logging(logging.DEBUG, get_log_filename(args))
    logging.info("Command line parameters:")
    logging.info(str(args))

    rl_loop = RLLoop(args,
                     nb_games_to_update=0,
                     nb_arena_games=50)
    rl_loop.initialize()

    while True:
        rl_loop.generate_games()
        rl_loop.compress_dataset()

    # rl_loop.update_network()
    # rl_loop.create_new_contender()
    # rl_loop.compare_new_weights()


if __name__ == "__main__":
    main()
