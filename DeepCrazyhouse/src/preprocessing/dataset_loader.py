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
from DeepCrazyhouse.src.domain.util import get_numpy_arrays, get_x_y_and_indices
from DeepCrazyhouse.src.domain.variants.input_representation import MATRIX_NORMALIZER


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
    dataset_type="train", part_id=0, verbose=True, normalize=False, q_value_ratio=0,
):
    """
    Loads one part of the pgn dataset in form of planes / multidimensional numpy array.
    It reads all files which are located either in the main_config['test_dir'] or main_config['test_dir']

    :param dataset_type: either ['train', 'test', 'mate_in_one']
    :param part_id: Decides which part of the data set will be loaded
    :param verbose: True if the log message shall be shown
    :param normalize: True if the inputs shall be normalized to 0-1
    ! Note this only supported for hist-length=1 at the moment
    :param q_value_ratio: Ratio for mixing the value return with the corresponding q-value
    For a ratio of 0 no q-value information will be used. Value must be in [0, 1]
    :return: numpy-arrays:
            start_indices - defines the index where each game starts
            x - the board representation for all games
            y_value - the game outcome (-1,0,1) for each board position
            y_policy - the movement policy for the next_move played
            plys_to_end - array of how many plys to the end of the game for each position.
             This can be used to apply discounting
            pgn_datasets - the dataset file handle (you can use .tree() to show the file structure)
    """

    if dataset_type == "train":
        zarr_filepaths = glob.glob(main_config["planes_train_dir"] + "**/*.zip")
    elif dataset_type == "val":
        zarr_filepaths = glob.glob(main_config["planes_val_dir"] + "**/*.zip")
    elif dataset_type == "test":
        zarr_filepaths = glob.glob(main_config["planes_test_dir"] + "**/*.zip")
    elif dataset_type == "mate_in_one":
        zarr_filepaths = glob.glob(main_config["planes_mate_in_one_dir"] + "**/*.zip")
    else:
        raise Exception(
            'Invalid dataset type "%s" given. It must be either "train", "val", "test" or "mate_in_one"' % dataset_type
        )

    if len(zarr_filepaths) < part_id + 1:
        raise Exception("There aren't enough parts available (%d parts) in the given directory for partid=%d"
                        % (len(zarr_filepaths), part_id))

    # load the zarr-files
    pgn_datasets = zarr_filepaths
    if verbose:
        logging.debug("loading: %s ...", pgn_datasets[part_id])
        logging.debug("")

    pgn_dataset = zarr.group(store=zarr.ZipStore(pgn_datasets[part_id], mode="r"))
    start_indices, x, y_value, y_policy, plys_to_end, y_best_move_q, eval_init, eval_search = get_numpy_arrays(pgn_dataset)  # Get the data

    if verbose:
        logging.info("STATISTICS:")
        try:
            for member in pgn_dataset["statistics"]:
                print(member, list(pgn_dataset["statistics"][member]))
        except KeyError:
            logging.warning("no statistics found")

        logging.info("PARAMETERS:")
        try:
            for member in pgn_dataset["parameters"]:
                print(member, list(pgn_dataset["parameters"][member]))
        except KeyError:
            logging.warning("no parameters found")

    if q_value_ratio != 0:
        y_value = (1-q_value_ratio) * y_value + q_value_ratio * y_best_move_q

    if normalize:
        x = x.astype(np.float32)
        # the y-vectors need to be casted as well in order to be accepted by the network
        y_value = y_value.astype(np.float32)
        y_policy = y_policy.astype(np.float32)
        # apply rescaling using a predefined scaling constant (this makes use of vectorized operations)
        x *= MATRIX_NORMALIZER
    return start_indices, x, y_value, y_policy, plys_to_end, eval_init, eval_search, pgn_dataset


def load_xiangqi_dataset(dataset_type="train", part_id=0, verbose=True, normalize=False):
    """
    Loads one part of the preprocessed data set of xiangqi games, originally given as csv.

    :parram dataset_type: either ['train', 'test', 'val']
    :param part_id: Decides which part of the data set will be loaded
    :param verbose: True if the log message shall be shown
    :param normalize: True if the inputs shall be normalized to 0-1
    :return: numpy-arrays:
        start_indices - defines the index where each game starts
        x - the board representation for all games
        y_value - the game outcome (-1,0,1) for each board position
        y_policy - the movement policy for the next_move played
        dataset - the dataset file handle (you can use .tree() to show the file structure)
    """
    if dataset_type == "train":
        zarr_filepaths = glob.glob(main_config["planes_train_dir"] + "**/*.zip")
    elif dataset_type == "val":
        zarr_filepaths = glob.glob(main_config["planes_val_dir"] + "**/*.zip")
    elif dataset_type == "test":
        zarr_filepaths = glob.glob(main_config["planes_test_dir"] + "**/*.zip")
    else:
        raise Exception(
            'Invalid dataset type "%s" given. It must be either "train", "val" or "test"' % dataset_type
        )

    if part_id >= len(zarr_filepaths):
        raise Exception("There aren't enough parts available (%d parts) in the given directory for part_id=%d"
                        % (len(zarr_filepaths), part_id))

    # load zarr-files
    datasets = zarr_filepaths
    if verbose:
        logging.debug("loading: %s...\n", datasets[part_id])

    dataset = zarr.group(store=zarr.ZipStore(datasets[part_id], mode="r"))
    start_indices, x, y_value, y_policy  = get_x_y_and_indices(dataset)

    if verbose:
        logging.info("STATISTICS:")
        try:
            for member in dataset["statistics"]:
                if member in ["avg_elo", "avg_elo_red", "avg_elo_black", "num_red_wins", "num_black_wins", "num_draws"]:
                    print(member, list(dataset["statistics"][member]))
        except KeyError:
            logging.warning("no statistics found")

        logging.info("PARAMETERS:")
        try:
            for member in dataset["parameters"]:
                print(member, list(dataset["parameters"][member]))
        except KeyError:
            logging.warning("no parameters found")

    if normalize:
        x = x.astype(np.float32)
        y_value = y_value.astype(np.float32)
        y_policy = y_policy.astype(np.float32)

        x *= MATRIX_NORMALIZER
    return start_indices, x, y_value, y_policy, dataset
