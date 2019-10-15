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

from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.runtime.color_logger import enable_color_logging
from DeepCrazyhouse.configs.main_config import main_config


def read_output(proc):
    """
    Reads the output of a process pip until a '\n' or "readyok\n" has been reached
    :param proc: Process to be read
    :return:
    """
    while True:
        line = proc.stdout.readline()
        # error = proc.stderr.readline()
        print(line)
        # print(error)
        if line == b'\n' or line == b"readyok\n" or line == b'':
            break


def compress_zarr_dataset(data, export_dir, compression='lz4', clevel=5):
    """
    Loads in a zarr data set and exports it with a given compression type and level
    :param data:
    :param export_dir:
    :param compression:
    :param clevel:
    :return:
    """
    # include current timestamp in dataset export file
    timestmp = datetime.datetime.fromtimestamp(time()).strftime("%Y-%m-%d-%H-%M-%S")
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
        x = data[key]
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


class RLLoop:
    """
    This class uses the C++ binary to generate games and updates the network from the newly acquired games
    """

    def __init__(self, crazyara_binary_dir='./', nb_games_to_update=1024):
        """
        Constructor
        :param crazyara_binary_dir: Directory to the C++ binary
        :param nb_games_to_update: Number of games to generate before the neural network will be updated
        """

        self.crazyara_binary_dir = crazyara_binary_dir
        self.proc = Popen([crazyara_binary_dir+'CrazyAra'],
                          stdin=PIPE,
                          stdout=PIPE,
                          stderr=PIPE,
                          shell=False)
        self.nb_games_to_update = nb_games_to_update

    def initialize(self):
        """
        Initializes the CrazyAra binary and loads the neural network weights
        :return:
        """
        # initialize

        # CrazyAra header
        time.sleep(0.1)
        read_output(self.proc)
        time.sleep(0.1)
        read_output(self.proc)
        time.sleep(1)

        # self.proc.stdin.write(b'setoption name Model_Directory value %s\n' % bytes(self.crazyara_binary_dir+"model/",
        #                                                                            'utf-8'))
        # self.proc.stdin.flush()
        # time.sleep(0.1)
        # read_output(self.proc)
        # time.sleep(0.1)
        set_uci_param(self.proc, "Nodes", 300)

        # read_output(self.proc)
        # time.sleep(1)

        # load network
        self.proc.stdin.write(b'isready\n')
        self.proc.stdin.flush()
        # time.sleep(1)
        read_output(self.proc)

    def generate_games(self):
        """
        Requests the binary to generate X number of games
        :return:
        """
        self.proc.stdin.write(b"selfplay %d\n" % self.nb_games_to_update)
        self.proc.stdin.flush()
        read_output(self.proc)


def set_uci_param(proc, name, value):
    """
    Sets the value for a given UCI-parameter in the binary.
    :param proc: Process to set the parameter for
    :param name: Name of the UCI-parameter
    :param value: Value for the UCI-parameter
    :return:
    """
    if type(value) == int:
        proc.stdin.write(b'setoption name %b value %d\n' % (bytes(name, encoding="ascii"), 300))
    else:
        raise NotImplementedError(f"To set uci-parameters of type {type(value)} has not been implemented yet.")
    proc.stdin.flush()


def update_network():
    """
    Updates the neural network with the newly acquired games from the replay memory
    :return:
    """
    cwd = os.getcwd() + '/'
    logging.info("Current working directory %s" % cwd)
    data = zarr.load('/home/queensgambit/Desktop/CrazyAra/engine/build-CrazyAra-Release/data.zarr')
    compress_zarr_dataset(data, './export/')
    main_config['planes_train_dir'] = cwd + "export/"
    starting_idx, x, y_value, y_policy, _, _ = load_pgn_dataset()
    print('len(starting_idx):', len(starting_idx))


if __name__ == "__main__":
    enable_color_logging()
    rl_loop = RLLoop(crazyara_binary_dir="./",
                     nb_games_to_update=2)
    rl_loop.initialize()
    rl_loop.generate_games()
    # update_network()
