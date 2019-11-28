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
import numpy as np
import logging
import datetime
import argparse
from numcodecs import Blosc
from subprocess import PIPE, Popen
from multiprocessing import Process, Queue

sys.path.append("../../../")
from DeepCrazyhouse.configs.train_config import train_config
from engine.src.rl.rl_training import update_network


def read_output(proc, last_line=b"readyok\n", check_error=False):
    """
    Reads the output of a process pip until the given last line has been reached.
    :param proc: Process to be read
    :param last_line Content when to stop reading (e.g. b'\n', b'', b"readyok\n")
    :param check_error: Listens to stdout for errors
    :return:
    """
    while True:
        line = proc.stdout.readline()
        # print(line)
        if check_error and line == b'':
            error = proc.stderr.readline()
            if error != b'':
                logging.error(error)
        if line == last_line:
            break


def compress_zarr_dataset(data, file_path, compression='lz4', clevel=5, start_idx=0, end_idx=0):
    """
    Loads in a zarr data set and exports it with a given compression type and level
    :param data: Zarr data set which will be compressed
    :param file_path: File name path where the data will be exported (e.g. "./export/data.zip")
    :param compression: Compression type
    :param clevel: Compression level
    :param end_idx: If end_idx != 0 the data set will be exported to the specified index,
    excluding the sample at end_idx (e.g. end_idx = len(x) will export it fully)
    :return: True if a NaN value was detected
    """
    compressor = Blosc(cname=compression, clevel=clevel, shuffle=Blosc.SHUFFLE)

    # open a dataset file and create arrays
    store = zarr.ZipStore(file_path, mode="w")
    zarr_file = zarr.group(store=store, overwrite=True)

    nan_detected = False
    for key in data.keys():
        if end_idx == 0:
            x = data[key]
        else:
            x = data[key][start_idx:end_idx]

        if np.isnan(x).any():
            nan_detected = True

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
    logging.info("dataset was exported to: %s", file_path)
    return nan_detected


def create_dir(directory):
    """
    Creates a given director in case it doesn't exists already
    :param directory: Directory path
    :return:
    """

    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info("Created directory %s" % directory)


def move_all_files(from_dir, to_dir):
    """
    Moves all files from a given directory to a destination directory
    :param from_dir: Origin directory where the files are located
    :param to_dir: Destinataion directory where all files (including subdirectories directories) will be moved
    :return:
    """
    file_names = os.listdir(from_dir)

    for file_name in file_names:
        os.rename(from_dir + file_name, to_dir + file_name)


class RLLoop:
    """
    This class uses the C++ binary to generate games and updates the network from the newly acquired games
    """

    def __init__(self, args, nb_games_to_update=1024, nb_arena_games=100, lr_reduction=0.0001, k_steps=0):
        """
        Constructor
        :param args: Command line arguments, see parse_args() for details.
        :param nb_games_to_update: Number of games to generate before the neural network will be updated
        :param nb_arena_games: Number of games which will be generated in a tournament setting in order to determine
        :param lr_reduction: Learning rate reduction of maximum learning rate after a single NN update
        if the updated NN weights are stronger than the old one and by how much.
        :param k_steps: Amount of total batch-updates for the NN so far (sets the tensorboard offset properly)
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

        self.export_dir_gen_data = self.crazyara_binary_dir + "export/new_data/"
        self.train_dir = self.crazyara_binary_dir + "export/train/"
        self.val_dir = self.crazyara_binary_dir + "export/val/"
        self.weight_dir = self.crazyara_binary_dir+"weights/"
        self.train_dir_archive = self.crazyara_binary_dir + "export/archive/train/"
        self.val_dir_archive = self.crazyara_binary_dir + "export/archive/val/"
        self.model_dir = self.crazyara_binary_dir + "model/"
        self.model_contender_dir = self.crazyara_binary_dir + "model_contender/"
        self.model_dir_archive = self.crazyara_binary_dir + "export/archive/model/"

        # change working directory (otherwise the .zip files would be generated at the .py location)
        os.chdir(self.crazyara_binary_dir)

        self._create_directories()
        self.model_name = ""  # will be set in initialize()
        self.nn_update_index = args.nn_update_idx
        self.max_lr = train_config["max_lr"]
        self.lr_reduction = lr_reduction

        self.k_steps = k_steps

    def _create_directories(self):
        """
        Creates directories in the crazyara_binary_path which will be used during RL
        :return:
        """
        # directories for training
        create_dir(self.crazyara_binary_dir+"logs")
        create_dir(self.weight_dir)
        create_dir(self.crazyara_binary_dir+"export")
        create_dir(self.export_dir_gen_data)
        create_dir(self.train_dir)
        create_dir(self.val_dir)
        create_dir(self.crazyara_binary_dir+"export/archive")
        create_dir(self.train_dir_archive)
        create_dir(self.val_dir_archive)
        create_dir(self.crazyara_binary_dir+"export/archive/model")
        create_dir(self.model_contender_dir)
        create_dir(self.model_dir_archive)

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
        model_params = glob.glob(self.crazyara_binary_dir + "model/*.params")
        if len(model_params) == 0:
            logging.warning("No model found in model directory")
            return ""
        return model_params[0]

    def _set_uci_options(self, is_arena=False):
        """
        Defines custom UCI options
        :param is_arena: Applies setting for the arena comparison
        :return:
        """
        self.proc.stdin.write(b'setoption name Model_Directory value %s\n' % bytes(self.crazyara_binary_dir+"model/",
                                                                                   'utf-8'))
        set_uci_param(self.proc, "Context", self.args.context)
        set_uci_param(self.proc, "Device_ID", self.args.device_id)
        if is_arena is True:
            set_uci_param(self.proc, "Centi_Temperature", 60)
        else:
            set_uci_param(self.proc, "Centi_Temperature", 80)

    def _read_output_arena(self):
        """
        Reads the output for arena matches and waits for the key-words "keep" or "replace"
        :return: True - If current NN generator should be replaced
                 False - If current NN generator should be kept
        """
        while True:
            line = self.proc.stdout.readline()
            if line == b"keep\n":
                return False
            if line == b"replace\n":
                return True

    def _stop_process(self):
        """
        Stops the current process by sending SIGTERM to all process groups
        :return:
        """
        # os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        self.proc.kill()
        # sleep for 1 sec to ensure the process exited
        time.sleep(1)

    def initialize(self, is_arena=False):
        """
        Initializes the CrazyAra binary and loads the neural network weights
        is_arena: Signals that UCI option should be set for arena comparision
        :return:
        """
        # initialize
        self.model_name = self._get_current_model_weight_file()
        self.proc = Popen([self.crazyara_binary_dir+"CrazyAra"],
                          stdin=PIPE,
                          stdout=PIPE,
                          stderr=PIPE,
                          shell=False)
        self._set_uci_options(is_arena=is_arena)

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
        read_output(self.proc, b"readyok\n", check_error=True)

    def create_export_dir(self):
        # include current timestamp in dataset export file
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")
        time_stamp_dir = self.export_dir_gen_data + time_stamp + "-" + self.device_name + "/"
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

    def _get_number_generated_files(self) -> int:
        """
        Returns the amount of file that have been newly generated
        :return:
        """
        return len(glob.glob(self.export_dir_gen_data + "**/*.zip"))

    def _move_previous_files_into_archive(self):
        """
        Moves previous training files into the archive directory
        :return:
        """
        move_all_files(self.train_dir, self.train_dir_archive)
        move_all_files(self.val_dir, self.val_dir_archive)
        move_all_files(self.model_contender_dir, self.model_dir_archive)

    def _move_generated_data_to_train_val(self):
        """
        Moves the generated samples, games (pgn format) and the number how many games have been generated to the given
         training and validation directory
        :return:
        """
        file_names = os.listdir(self.export_dir_gen_data)

        # move the last file into the validation directory
        os.rename(self.export_dir_gen_data + file_names[-1], self.val_dir + file_names[-1])

        # move the rest into the training directory
        for file_name in file_names[:-1]:
            os.rename(self.export_dir_gen_data + file_name, self.train_dir + file_name)

    def _include_data_from_replay_memory_into_training(self, nb_files=8, fraction_for_selection=0.05):
        """
        :param nb_files: Number of files to include from replay memory into training
        :param fraction_for_selection: Proportion for selecting files from the replay memory
        :return:
        """
        file_names = os.listdir(self.train_dir_archive)

        # invert ordering (most recent files are on top)
        file_names = file_names[::-1]

        if len(file_names) < nb_files:
            logging.info("Not enough replay memory available. Only current data will be used")
            return

        thresh_idx = max(int(len(file_names) * fraction_for_selection), nb_files)

        indices = np.arange(0, thresh_idx)
        np.random.shuffle(indices)

        # cap the index list
        indices = indices[:nb_files]

        # move selected files into train dir
        for index in list(indices):
            os.rename(self.train_dir_archive + file_names[index], self.train_dir + file_names[index])

    def _remove_temporary_checkpoints(self):
        """
        Removes all checkpoint files in the weight/ directory
        :return:
        """
        file_list = glob.glob(os.path.join(self.weight_dir, "model-*"))
        for file in file_list:
            os.remove(file)

    def _prepare_data_for_training(self):
        """
        Moves the newly generated files into the training directory
        :return:
        """
        self._move_previous_files_into_archive()
        self._move_generated_data_to_train_val()
        self._remove_temporary_checkpoints()
        self._include_data_from_replay_memory_into_training(5, 0.05)

    def compress_dataset(self):
        """
        Loads the uncompressed data file, select all sample until the index specified in "startIdx.txt",
        compresses it and exports it
        :return:
        """
        data = zarr.load(self.crazyara_binary_dir + "data_" + self.device_name + ".zarr")

        export_dir, time_stamp = self.create_export_dir()
        zarr_path = export_dir + time_stamp + ".zip"
        nan_detected = compress_zarr_dataset(data, zarr_path, start_idx=0)
        if nan_detected is True:
            logging.error("NaN value detected in file %s.zip" % time_stamp)
            new_export_dir = self.crazyara_binary_dir + time_stamp
            os.rename(export_dir, new_export_dir)
            export_dir = new_export_dir
        self._move_game_data_to_export_dir(export_dir)

    def compare_new_weights(self):
        """
        Compares the old nn-weights with the newly acquired one and returns the win-rate
        :return:
        """
        self.proc.stdin.write(b"arena %d\n" % self.nb_arena_games)
        self.proc.stdin.flush()
        return self._read_output_arena()

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

    def check_for_new_model(self):
        """
        Checks if the current neural network generator has been updated and restarts the executable if this is the case
        :return:
        """
        model_name = self._get_current_model_weight_file()
        if model_name != "" and model_name != self.model_name:
            logging.info("Loading new model: %s" % model_name)
            self._stop_process()
            self.initialize()

    def _replace_model_generator_weights(self):
        """
        Moves the previous model into archive directory and the model_contender into the model directory
        :return:
        """
        move_all_files(self.model_dir, self.model_dir_archive)
        move_all_files(self.model_contender_dir, self.model_dir)

    def check_for_enough_train_data(self, number_files_to_update):
        """
        Checks if enough training games have been generated to trigger training a new network
        :param number_files_to_update: Number of newly generated files needed to trigger a new NN update
        :return:
        """
        if self._get_number_generated_files() >= number_files_to_update:
            self._stop_process()
            self._prepare_data_for_training()
            # start training using a process to ensure memory clearing afterwards
            queue = Queue()  # start a subprocess to be memory efficient
            process = Process(target=update_network, args=(queue, self.nn_update_index, self.k_steps,
                                                           self.max_lr, self._get_current_model_arch_file(),
                                                           self._get_current_model_weight_file(),
                                                           self.crazyara_binary_dir))
            logging.info("start training")
            process.start()
            self.k_steps = queue.get() + 1
            process.join()  # this blocks until the process terminates

            if self.max_lr > train_config["min_lr"]:
                self.max_lr = max(self.max_lr - self.lr_reduction, train_config["min_lr"] * 10)

            self.nn_update_index += 1
            self.initialize()
            did_contender_win = self.compare_new_weights()
            if did_contender_win is True:
                logging.info("Replacing current generator with contender")
                self._replace_model_generator_weights()
            else:
                logging.info("Keep current generator")
            self._stop_process()
            self.initialize()


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
    elif isinstance(value, str):
        proc.stdin.write(b"setoption name %b value %b\n" % (bytes(name, encoding="ascii"),
                                                            bytes(value, encoding="ascii")))
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

    parser.add_argument("--crazyara-binary-dir", type=str, default="/data/RL/",
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
    parser.add_argument('--nn-update-idx', type=int, default=0,
                        help="Index of how many NN updates have been done so far."
                             " This will be used to label the NN weights (default: 0)")
    parser.add_argument("--nn-update-files", type=int, default=10,
                        help="How many new generated training files are needed to apply an update to the NN")
    parser.add_argument("--arena-games", type=int, default=100,
                        help="How many arena games will be done to judge the quality of the new network")

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
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        root.addHandler(fh)


def get_log_filename(args):
    """
    Returns the file name for the log messages
    :param args: Command line arguments
    :return: Filename: str or None
    """
    if args.export_no_log is False:
        return "%s/%s_%d.log" % (args.crazyara_binary_dir, args.context, args.device_id)
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
                     nb_arena_games=args.arena_games,
                     lr_reduction=0.0001)
    rl_loop.initialize()

    while True:
        if args.trainer:
            rl_loop.check_for_enough_train_data(args.nn_update_files)
        else:
            rl_loop.check_for_new_model()

        rl_loop.generate_games()
        rl_loop.compress_dataset()

if __name__ == "__main__":
    main()
