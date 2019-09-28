"""
@file: dataset_loader.py
Created on 22.10.18
@project: crazy_ara_refactor
@author: queensgambit

Please describe what the content of this file is about
"""
import glob
import logging
import numpy as np
import zarr
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.src.domain.util import get_numpy_arrays, MATRIX_NORMALIZER


def _load_dataset_file(dataset_filepath):
    """
    Loads a single dataset file give by its path
    :param dataset_filepath: path where the file is located
    :return:starting_idx: [int] - List of indices where ech game starts
            x: nd.array - Numpy array which contains the game positions
            y_value: nd.array - Numpy array which describes the winner for each board position
            y_policy: nd.array - Numpy array which describes the policy distribution for each board state
                                 (in case of a pgn dataset the move is one hot encoded)
            plys_to_end - array of how many plys to the end of the game for each position.
             This can be used to apply discounting
    """
    return get_numpy_arrays(zarr.group(store=zarr.ZipStore(dataset_filepath, mode="r")))


def load_pgn_dataset(
    dataset_type="train", part_id=0, print_statistics=False, print_parameters=False, verbose=True, normalize=False
):  # Too many arguments (6/5)
    """
    Loads one part of the pgn dataset in form of planes / multidimensional numpy array.
    It reads all files which are located either in the main_config['test_dir'] or main_config['test_dir']

    :param dataset_type: either ['train', 'test', 'mate_in_one']
    :param part_id: Decides which part of the data set will be loaded
    :param print_statistics: Decides whether to print file statistics
    :param print_parameters: Decide whether to print the parameters with which the dataset was generated
    :param verbose: True if the log message shall be shown
    :param normalize: True if the inputs shall be normalized to 0-1
    ! Note this only supported for hist-length=1 at the moment
    :return: numpy-arrays:
            starting_idx - defines the index where each game starts
            x - the board representation for all games
            y_value - the game outcome (-1,0,1) for each board position
            y_policy - the movement policy for the next_move played
            pgn_datasets - the dataset file handle (you can use .tree() to show the file structure)
    """

    if dataset_type == "train":
        zarr_filepaths = glob.glob(main_config["planes_train_dir"] + "**/*")
    elif dataset_type == "val":
        zarr_filepaths = glob.glob(main_config["planes_val_dir"] + "**/*")
    elif dataset_type == "test":
        zarr_filepaths = glob.glob(main_config["planes_test_dir"] + "**/*")
    elif dataset_type == "mate_in_one":
        zarr_filepaths = glob.glob(main_config["planes_mate_in_one_dir"] + "**/*")
    else:
        raise Exception(
            'Invalid dataset type "%s" given. It must be either "train", "val", "test" or "mate_in_one"' % dataset_type
        )

    if len(zarr_filepaths) < part_id + 1:
        raise Exception("There aren't enough parts available in the given directory for partid=" + str(part_id))

    # load the zarr-files
    pgn_datasets = zarr_filepaths
    if verbose:
        logging.debug("loading: %s...", pgn_datasets[part_id])
        logging.debug("")

    pgn_dataset = zarr.group(store=zarr.ZipStore(pgn_datasets[part_id], mode="r"))
    starting_idx, x, y_value, y_policy, plys_to_end = get_numpy_arrays(pgn_dataset)  # Get the data

    if print_statistics:
        logging.info("STATISTICS:")
        for member in pgn_dataset["statistics"]:
            print(member, list(pgn_dataset["statistics"][member]))

    if print_parameters:
        logging.info("PARAMETERS:")
        for member in pgn_dataset["parameters"]:
            print(member, list(pgn_dataset["parameters"][member]))

    if normalize:
        x = x.astype(np.float32)
        # the y-vectors need to be casted as well in order to be accepted by the network
        y_value = y_value.astype(np.float32)
        y_policy = y_policy.astype(np.float32)
        # apply rescaling using a predefined scaling constant (this makes use of vectorized operations)
        x *= MATRIX_NORMALIZER
    return starting_idx, x, y_value, y_policy, plys_to_end, pgn_dataset
