"""
@file: rl_loop.py
Created on 12.10.19
@project: crazy_ara
@author: queensgambit

Main reinforcement learning for generating games and train the neural network.
"""

from subprocess import PIPE, Popen
import datetime
from time import time
from numcodecs import Blosc
import zarr
import os
import logging


def read_output(proc):
    """
    Reads the output of a process pip until a '\n' or "readyok\n" has been reached
    :param proc: Process to be read
    :return:
    """
    while True:
        line = proc.stdout.readline()
        print(line)
        if line == b'\n' or line == b"readyok\n":
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

    def __init__(self, crazyara_binary_path='./CrazyAra', nb_games_to_update=1024):
        """
        Constructor
        :param crazyara_binary_path: Filepath to the C++ binary
        :param nb_games_to_update: Number of games to generate before the neural network will be updated
        """

        self.proc = Popen([crazyara_binary_path],
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
        read_output(self.proc)
        read_output(self.proc)

        # load network
        self.proc.stdin.write(b'isready\n')
        self.proc.stdin.flush()
        read_output(self.proc)

    def generate_games(self):
        """
        Requests the binary to generate X number of games
        :return:
        """
        self.proc.stdin.write(b"selfplay %d\n" % self.nb_games_to_update)
        self.proc.stdin.flush()
        read_output(self.proc)


def __main__():
    rl_loop = RLLoop()
    rl_loop.initialize()
    rl_loop.generate_games()
