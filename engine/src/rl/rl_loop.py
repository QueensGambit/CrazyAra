"""
@file: rl_loop.py
Created on 12.10.19
@project: crazy_ara
@author: queensgambit

Main reinforcement learning for generating games and train the neural network.
"""

import os
import sys
import logging
import argparse
from rtpt import RTPT
import dataclasses
from multiprocessing import Process, Queue

assert os.getcwd().endswith(f'engine/src/rl'), f'Please change working directory'
sys.path.append("../../../")

from engine.src.rl.rl_utils import enable_logging, get_log_filename, get_current_binary_name, \
    extract_nn_update_idx_from_binary_name, change_binary_name
from engine.src.rl.binaryio import BinaryIO
from engine.src.rl.fileio import FileIO
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.configs.train_config import rl_train_config
from DeepCrazyhouse.configs.rl_config import RLConfig, UCIConfigArena, UCIConfig
from engine.src.rl.rl_training import update_network


class RLLoop:
    """
    This class uses the C++ binary to generate games and updates the network from the newly acquired games
    """

    def __init__(self, args, rl_config, nb_arena_games=100, lr_reduction=0.0001, k_steps=0):
        """
        Constructor
        :param args: Command line arguments, see parse_args() for details.
        :param nb_arena_games: Number of games which will be generated in a tournament setting in order to determine
        :param lr_reduction: Learning rate reduction of maximum learning rate after a single NN update
        if the updated NN weights are stronger than the old one and by how much.
        :param k_steps: Amount of total batch-updates for the NN so far (sets the tensorboard offset properly and will
        be written to the new model filenames to track how many iterations the model has trained in total)
        """
        self.args = args

        self.tc = rl_train_config()
        self.rl_config = rl_config

        self.file_io = FileIO(orig_binary_name=self.rl_config.binary_name, binary_dir=self.rl_config.binary_dir,
                              uci_variant=self.rl_config.uci_variant)
        self.binary_io = None

        if nb_arena_games % 2 == 1:
            raise IOError(f'Number of games should be even to avoid giving a player an advantage')
        self.nb_arena_games = nb_arena_games
        self.lr_reduction = lr_reduction
        self.tc.k_steps = k_steps
        self.device_name = f'{args.context}_{args.device_id}'
        self.model_name = ""  # will be set in initialize()
        self.did_contender_win = False

        # change working directory (otherwise binary would generate .zip files at .py location)
        os.chdir(self.file_io.binary_dir)
        self.tc.cwd = self.file_io.binary_dir

        # The original binary name in TrainConfig will always stay the same & be a substring of the updated name
        self.current_binary_name = get_current_binary_name(self.file_io.binary_dir, self.rl_config.binary_name)

        self.nn_update_index = args.nn_update_idx
        if not args.trainer:  # only trainer gpu needs the update index as cmd line argument
            self.nn_update_index = extract_nn_update_idx_from_binary_name(self.current_binary_name)
        self.last_nn_update_index = self.nn_update_index + self.rl_config.nb_nn_updates

        # Continuously update the process name
        self.rtpt = RTPT(name_initials=self.tc.name_initials,
                         experiment_name=f'{self.rl_config.binary_name}_{self.rl_config.uci_variant}',
                         max_iterations=self.rl_config.nb_nn_updates, moving_avg_window_size=1)
        self.rtpt.start()

    def initialize(self, is_arena=False):
        """
        Initializes the CrazyAra binary and loads the neural network weights
        is_arena: Signals that UCI option should be set for arena comparison
        :return:
        """
        self.model_name = self.file_io.get_current_model_tar_file()
        self.binary_io = BinaryIO(binary_path=self.file_io.binary_dir+self.current_binary_name)
        self.binary_io.set_uci_options(self.rl_config.uci_variant, self.args.context, self.args.device_id,
                                       self.rl_config.precision, self.file_io.model_dir,
                                       self.file_io.model_contender_dir, is_arena)
        self.binary_io.load_network()

    def check_for_new_model(self):
        """
        Checks if the current neural network generator has been updated and restarts the executable if this is the case
        :return:
        """

        new_binary_name = get_current_binary_name(self.file_io.binary_dir, self.rl_config.binary_name)
        if new_binary_name != self.current_binary_name:
            self.current_binary_name = new_binary_name
            # when binary name changes, also epoch changes
            self.nn_update_index = extract_nn_update_idx_from_binary_name(self.current_binary_name)

            # If a new model is available, the binary name has also changed
            model_name = self.file_io.get_current_model_tar_file()
            if model_name != "" and model_name != self.model_name:
                logging.info("Loading new model: %s" % model_name)

            self.binary_io.stop_process()
            self.rtpt.step()
            self.initialize()

    def check_for_enough_train_data(self, number_files_to_update):
        """
        Checks if enough training games have been generated to trigger training a new network
        :param number_files_to_update: Number of newly generated files needed to trigger a new NN update
        :return: True, if enough training data was available and a training run has been executed.
        """
        if self.file_io.get_number_generated_files() >= number_files_to_update:
            self.binary_io.stop_process()
            self.file_io.prepare_data_for_training(self.rl_config.rm_nb_files, self.rl_config.rm_fraction_for_selection,
                                                   self.did_contender_win)
            # start training using a process to ensure memory clearing afterwards
            self.tc.device_id = self.args.device_id

            nb_train_iterations = 1 if not self.file_io.is_moe else self.file_io.number_phases

            if self.file_io.use_moe_staged_learning:
                # train one model on full data
                self._update_network_wrapper(phase_idx=None)

            load_model_filename = None
            if self.file_io.use_moe_staged_learning:
                load_model_filename = self.file_io.get_current_model_tar_file(phase="phaseNone",
                                                                              base_dir=self.file_io.model_contender_dir)

            # use reverse ordering so that the default model (0) updates as last
            # and other GPUs only update their model if the full training has finished.
            for phase_idx in reversed(range(nb_train_iterations)):
                self._update_network_wrapper(phase_idx, load_model_filename)

            if self.tc.max_lr > self.tc.min_lr:
                self.tc.max_lr = max(self.tc.max_lr - self.lr_reduction, self.tc.min_lr * 10)

            self.file_io.move_training_logs(self.nn_update_index)

            self.nn_update_index += 1

            self.initialize()
            logging.info(f'Start arena tournament ({self.nb_arena_games} rounds)')
            self.did_contender_win = self.binary_io.compare_new_weights(self.nb_arena_games)
            if self.did_contender_win is True:
                logging.info("REPLACING current generator with contender")
                self.file_io.replace_current_model_with_contender()
            else:
                logging.info("KEEPING current generator")

            self.file_io.remove_intermediate_weight_files()

            self.binary_io.stop_process()
            self.rtpt.step()  # BUG: process changes its name 1 iteration too late, fix?
            self.current_binary_name = change_binary_name(self.file_io.binary_dir, self.current_binary_name,
                                                          self.rtpt._get_title(), self.nn_update_index)
            self.initialize()

    def _update_network_wrapper(self, phase_idx, load_model_filename=None):
        """
        Creates a process for update_network.
        :param phase_idx: Phase index
        :param load_model_filename: The neural model to load for training.
        """
        phase = ""
        if self.file_io.is_moe:
            phase = f"phase{phase_idx}/"

        suffix_train = phase
        suffix_val = phase
        if phase_idx is None:  # staged learning
            suffix_train = "**/"
            suffix_val = f"phase{self.file_io.number_phases // 2}/"

        main_config["planes_train_dir"] = self.file_io.binary_dir + f"export/train/{self.file_io.uci_variant}/" + suffix_train
        main_config["planes_val_dir"] = self.file_io.binary_dir + f"export/val/{self.file_io.uci_variant}/" + suffix_val

        if load_model_filename is None:
            load_model_filename = self.file_io.get_current_model_tar_file(phase)
        model_export_dir = self.file_io.model_contender_dir + phase

        queue = Queue()  # start a subprocess to be memory efficient
        process = Process(target=update_network, args=(queue, self.nn_update_index,
                                                       load_model_filename,
                                                       not self.args.no_onnx_export,
                                                       main_config, self.tc,
                                                       model_export_dir))
        logging.info("Start Training")
        if self.file_io.is_moe:
            logging.info(f"Phase {phase_idx}")
        process.start()
        self.tc.k_steps = queue.get() + 1
        process.join()  # this blocks until the process terminates


def parse_args(cmd_args: list):
    """
    Parses command-line argument and returns them as a dictionary object
    :param cmd_args: Command-line arguments (sys.argv[1:])
    :return: Parsed arguments as dictionary object
    """
    parser = argparse.ArgumentParser(description='Reinforcement learning loop')

    parser.add_argument('--context', type=str, default="gpu",
                        help='Computational device context to use. Possible values ["cpu", "gpu"]. (default: gpu)')
    parser.add_argument("--device-id", type=int, default=0,
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
    parser.add_argument('--no-onnx-export', default=False, action="store_true",
                        help="By default the networks will be converted to ONNX to allow TensorRT inference."
                             " If this parameter is enabled no conversion will be done")

    args = parser.parse_args(cmd_args)

    if args.context not in ["cpu", "gpu"]:
        raise ValueError('Given value: %s for context is invalid. It must be in ["cpu", "gpu"].' % args.context)

    return args


def main():
    """
    Main function which is executed on start-up. If you train on multiple GPUs, start the
    trainer GPU before the generating GPUs to get correct epoch counting and process/binary naming.
    :return:
    """
    args = parse_args(sys.argv[1:])
    rl_config = RLConfig()
    uci_config = UCIConfig()

    if not os.path.exists(rl_config.binary_dir):
        raise Exception(f'Your given binary_dir: {rl_config.binary_dir} does not exist. '
                        f'Make sure to define a valid directory')
    if rl_config.binary_dir[-1] != '/':
        rl_config.binary_dir += '/'

    enable_logging(logging.DEBUG, get_log_filename(args, rl_config))

    rl_loop = RLLoop(args, rl_config, nb_arena_games=rl_config.arena_games, lr_reduction=0)
    if args.trainer:
        rl_loop.current_binary_name = change_binary_name(rl_loop.file_io.binary_dir, rl_loop.current_binary_name,
                                                         rl_loop.rtpt._get_title(), rl_loop.nn_update_index)
    rl_loop.initialize()

    logging.info(f'--------------- CONFIG SETTINGS ---------------')
    for key, value in sorted(vars(args).items()):
        logging.info(f'CMD line args:      {key} = {value}')
    for key, value in sorted(dataclasses.asdict(rl_loop.tc).items()):
        logging.info(f'Train Config:       {key} = {value}')
    for key, value in sorted(dataclasses.asdict(rl_config).items()):
        logging.info(f'RL Options:         {key} = {value}')
    for key, value in rl_loop.binary_io.get_uci_options().items():
        logging.info(f'UCI Options:        {key} = {value}')
    for key, value in sorted(dataclasses.asdict(UCIConfigArena()).items()):
        logging.info(f'UCI Options Arena:  {key} = {value}')
    logging.info(f'-----------------------------------------------')

    while True:
        if args.trainer:
            rl_loop.check_for_enough_train_data(rl_config.nn_update_files)

        else:
            rl_loop.check_for_new_model()

        if rl_loop.nn_update_index >= rl_loop.last_nn_update_index:
            logging.info(f'{rl_loop.rl_config.nb_nn_updates} NN updates reached, shutting down')
            break

        rl_loop.binary_io.generate_games()
        if uci_config.Number_Parallel_Games > 1:
            rl_loop.file_io.combine_dataset_and_files(rl_loop.device_name, uci_config.Number_Parallel_Games)
        else:
            rl_loop.file_io.compress_dataset(rl_loop.device_name)


if __name__ == "__main__":
    main()

