"""
@file: dataset_loader.py
Created on 22.10.18
@project: crazy_ara_refactor
@author: queensgambit

Please describe what the content of this file is about
"""
import glob
import logging
import chess
import numpy as np
import zarr
from DeepCrazyhouse.configs.main_config_sample import main_config
from DeepCrazyhouse.src.domain.crazyhouse.constants import (
    CHANNEL_MAPPING_CONST,
    CHANNEL_MAPPING_POS,
    MAX_NB_MOVES,
    MAX_NB_NO_PROGRESS,
    MAX_NB_PRISONERS,
    NB_CHANNELS_POS,
    POCKETS_SIZE_PIECE_TYPE,
)
from DeepCrazyhouse.src.domain.util import get_numpy_arrays


def _load_dataset_file(dataset_filepath):
    """
    Loads a single dataset file give by its path
    :param dataset_filepath: path where the file is located
    :return:
    """
    store = zarr.ZipStore(dataset_filepath, mode="r")
    pgn_dataset = zarr.group(store=store)

    s_idcs, x, yv, yp = get_numpy_arrays(pgn_dataset)

    return s_idcs, x, yv, yp


def load_pgn_dataset(
    dataset_type="train", part_id=0, print_statistics=False, print_parameters=False, verbose=True, normalize=False
):
    """
    Loads one part of the pgn dataset in form of planes / multidimensional numpy array.
    It reads all files which are located either in the main_config['test_dir'] or main_config['test_dir']

    :param config_path: Define where your config file is located
    :param dataset_type: either ['train', 'test', 'mate_in_one']
    :param part_id: Decides which part of the data set will be loaded
    :param print_statistics: Decides whether to print file statistics
    :param print_parameters: Decide whether to print the parameters with which the dataset was generated
    :param verbose: True if the log message shall be shown
    :param normalize: True if the inputs shall be normalized to 0-1 ! Note this only supported for hist-length=1 at the moment
    :return: numpy-arrays:
            s_idcs - defines the index where each game starts
            x - the board representation for all games
            yv - the game outcome (-1,0,1) for each board position
            yp - the movement policy for the next_move played
            pgn_datasets - the dataset file handle (you can use .tree() to show the file structure)
    """

    if dataset_type == "train":
        zarr_filepaths = glob.glob(main_config["planes_train_dir"] + "**/*")
    elif dataset_type == "val":
        zarr_filepaths = glob.glob(main_config["planes_test_dir"] + "**/*")
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
    if verbose is True:
        logging.debug("loading: %s...", zarr_filepaths[part_id])
        logging.debug("")

    store = zarr.ZipStore(zarr_filepaths[part_id], mode="r")
    pgn_dataset = zarr.group(store=store)

    # Get the data
    s_idcs, x, yv, yp = get_numpy_arrays(pgn_dataset)

    if print_statistics is True:
        logging.info("STATISTICS:")
        for member in pgn_dataset["statistics"]:
            print(member, list(pgn_dataset["statistics"][member]))

    if print_parameters is True:
        logging.info("PARAMETERS:")
        for member in pgn_dataset["parameters"]:
            print(member, list(pgn_dataset["parameters"][member]))

    if normalize is True:
        x = x.astype(np.float32)

        # the y-vectors need to be casted as well in order to be accepted by the network
        yv = yv.astype(np.float32)
        yp = yp.astype(np.float32)

        # !TODO replace this by function normalize_input_planes()
        mat_pos = x[:, :NB_CHANNELS_POS, :, :]
        mat_const = x[:, NB_CHANNELS_POS:, :, :]

        # iterate over all pieces except the king
        for p_type in chess.PIECE_TYPES[:-1]:
            # p_type -1 because p_type starts with 1
            ch = CHANNEL_MAPPING_POS["prisoners"] + p_type - 1

            mat_pos[:, ch, :, :] /= MAX_NB_PRISONERS
            # the prison for black begins 5 channels later
            mat_pos[:, ch + POCKETS_SIZE_PIECE_TYPE, :, :] /= MAX_NB_PRISONERS

        ### Total Move Count
        # 500 was set as the max number of total moves
        mat_const[:, CHANNEL_MAPPING_CONST["total_mv_cnt"], :, :] /= MAX_NB_MOVES
        ### No progress count
        #  after 40 moves of no progress the 40 moves rule for draw applies
        mat_const[:, CHANNEL_MAPPING_CONST["no_progress_cnt"], :, :] /= MAX_NB_NO_PROGRESS

    return s_idcs, x, yv, yp, pgn_dataset
