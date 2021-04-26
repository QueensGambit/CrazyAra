"""
@file: util.py
Created on 09.06.18
@project: DeepCrazyhouse
@author: queensgambit

Utility functions which are use by the converter scripts
"""

import copy
import numpy as np
from DeepCrazyhouse.src.domain.variants.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    MODE,
    MODE_LICHESS,
    MODE_CRAZYHOUSE,
    MODE_XIANGQI,
    LABELS_XIANGQI,
    CHANNEL_MAPPING_CONST,
    CHANNEL_MAPPING_POS,
    MAX_NB_MOVES,
    MAX_NB_NO_PROGRESS,
    MAX_NB_PRISONERS,
    NB_CHANNELS_TOTAL,
    NB_CHANNELS_POS,
    POCKETS_SIZE_PIECE_TYPE,
    chess,
)


# file lookup for vertically mirrored xiangqi boards
mirrored_files_lookup = {'a': 'i', 'b': 'h', 'c': 'g', 'd': 'f', 'e': 'e', 'f': 'd', 'g': 'c', 'h': 'b', 'i': 'a'}


def get_row_col(position, mirror=False):
    """
    Maps a value [0,63] to its row and column index

    :param position: Position id which is an integer [0,63]
    :param mirror: Returns the indices for the mirrored board
    :return: Row and columns index

    """
    # returns the column and row index of a given position
    row = position // 8
    col = position % 8

    if mirror:
        row = 7 - row

    return row, col


def get_board_position_index(row, col, mirror=False):
    """
    Maps a row and column index to the integer value [0, 63].

    :param row: Row index of the square
    :param col: Column index of the square
    :param mirror: Returns integer value for a mirrored board
    :return:
    """
    if mirror:
        row = 7 - row

    return (row * 8) + col


def mirror_field_index(row, col):
    """
    Mirrors a given row and column index
    :param row: Row index starting at 0
    :param col: Column index starting at 0
    :return:
    """
    return 7 - row, col


def show_promask(bin_mask):
    """
    # gives an ascii representation of a binary mask

    :param bin_mask: Binary which are used by python-chess
    :return: nothing
    """
    for idx, char in enumerate(bin_mask):
        print(char, end=" ")
        if idx % 8 == 7:
            print()


def stack_data(data):
    """
    Prepares the training data by concatenating the list to a numpy matrix

    :param data: list of np.arrays which can be x_train, x_test, y_train or y_test for example
    :return: the prepared data
    """
    return np.concatenate(data, axis=0).astype(np.int16)


def get_dic_sorted_by_key(dic):
    """
    Returns the values of a dictionary based on its sorted keys ordering

    :param dic: dictionary with typically an integer as key type
    :return: list: sorted values based on key ordering
    """

    return [dic[key] for key in sorted(dic)]


def get_numpy_arrays(pgn_dataset):
    """
    Loads the content of the dataset file into numpy arrays

    :param pgn_dataset: dataset file handle
    :return: numpy-arrays:
            starting_idx - defines the index where each game starts
            x - the board representation for all games
            y_value - the game outcome (-1,0,1) for each board position
            y_policy - the movement policy for the next_move played
            plys_to_end - array of how many plys to the end of the game for each position.
             This can be used to apply discounting
            y_best_move_q - Q-value for the position of the selected move
             (this information is only available for generated data during selfplay)
    """
    # Get the data
    start_indices = np.array(pgn_dataset["start_indices"])
    x = np.array(pgn_dataset["x"])
    y_value = np.array(pgn_dataset["y_value"])
    try:
        y_policy = np.array(pgn_dataset["y_policy_prediction_risev2_27"])
    except Exception:
        y_policy = np.array(pgn_dataset["y_policy"])

    possible_entries = ["plys_to_end", "y_best_move_q"]
    entries = [None] * 2
    for idx, entry in enumerate(possible_entries):
        try:
            entries[idx] = np.array(pgn_dataset[entry])
        except KeyError:
            pass
    return start_indices, x, y_value, y_policy, entries[0], entries[1]


def get_x_y_and_indices(dataset):
    """
    Loads the content of the given dataset into numpy arrays.

    :param dataset: dataset file handle
    :return: numpy-arrays:
        starting_idx - defines the index where each game starts
        x - the board representation for all games
        y_value - the game outcome (-1,0,1) for each board position
        y_policy - the movement policy for the next_move played
    """
    start_indices = np.array(dataset["start_indices"])
    x = np.array(dataset["x"])
    y_value = np.array(dataset["y_value"])
    y_policy = np.array(dataset["y_policy"])
    return start_indices, x, y_value, y_policy


def normalize_input_planes(x):
    """
    Normalizes input planes to range [0,1]. Works in place / meaning the input parameter x is manipulated
    :param x: Input planes representation
    :return: The normalized planes
    """

    # convert the input planes to float32 assuming that the datatype is int
    if x.dtype != np.float32:
        x = x.astype(np.float32)

    mat_pos = x[:NB_CHANNELS_POS, :, :]
    mat_const = x[NB_CHANNELS_POS:, :, :]

    # iterate over all pieces except the king, (because the king can't be in a pocket)
    if MODE == MODE_CRAZYHOUSE or MODE == MODE_LICHESS:
        for p_type in chess.PIECE_TYPES[:-1]:
            # p_type -1 because p_type starts with 1
            channel = CHANNEL_MAPPING_POS["prisoners"] + p_type - 1
            mat_pos[channel, :, :] /= MAX_NB_PRISONERS
            # the prison for black begins 5 channels later
            mat_pos[channel + POCKETS_SIZE_PIECE_TYPE, :, :] /= MAX_NB_PRISONERS
    # xiangqi has 7 piece types (king/general is excluded as prisoner)
    elif MODE == MODE_XIANGQI:
        for p_type in range(6):
            channel = CHANNEL_MAPPING_POS["prisoners"] + p_type
            mat_pos[channel, :, :] /= MAX_NB_PRISONERS
            # the prison for opponent begins 6 channels later
            mat_pos[channel + POCKETS_SIZE_PIECE_TYPE, :, :] /= MAX_NB_PRISONERS

    # Total Move Count
    # 500 was set as the max number of total moves
    mat_const[CHANNEL_MAPPING_CONST["total_mv_cnt"], :, :] /= MAX_NB_MOVES
    # No progress count
    # after 40 moves of no progress the 40 moves rule for draw applies
    if MODE != MODE_XIANGQI:
        mat_const[CHANNEL_MAPPING_CONST["no_progress_cnt"], :, :] /= MAX_NB_NO_PROGRESS

    return x


# use a constant matrix for normalization to allow broad cast operations
# in policy version 2, the king promotion moves were added to support antichess, this deprecates older nets
MATRIX_NORMALIZER = normalize_input_planes(np.ones((NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH)))


def augment(x, y_policy):
    """
    Augments a given set of planes and their corresponding policy targets.
    The returned planes are vertically mirrored. The returned policy targets
    are adjusted, so that they correspond to the new planes.
    Works in-place.
    :param x: Input planes
    :param y_policy: Policy targets
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] = np.fliplr(x[i][j])

        idx_mv = np.where(y_policy[i] == 1)[0][0]
        y_policy[i][idx_mv] = 0
        ucci = LABELS_XIANGQI[idx_mv]

        from_square_aug = mirrored_files_lookup[ucci[0]] + ucci[1]
        to_square_aug = mirrored_files_lookup[ucci[2]] + ucci[3]
        ucci_aug = from_square_aug + to_square_aug

        idx_mv_aug = LABELS_XIANGQI.index(ucci_aug)
        y_policy[i][idx_mv_aug] = 1


def customize_input_planes(x):
    """
    Reverts normalization back to integer values. Works in place.
    :param x: Input Planes Representation
    :return: The customized planes (converted back to integer)
    """
    mat_pos = x[:NB_CHANNELS_POS, :, :]
    mat_const = x[NB_CHANNELS_POS:, :, :]

    # iterate over all pieces except the king
    for p_type in chess.PIECE_TYPES[:-1]:
        # p_type -1 because p_type starts with 1
        channel = CHANNEL_MAPPING_POS["prisoners"] + p_type - 1
        mat_pos[channel, :, :] /= MAX_NB_PRISONERS
        # the prison for black begins 5 channels later
        mat_pos[channel + POCKETS_SIZE_PIECE_TYPE, :, :] /= MAX_NB_PRISONERS

    # Total Move Count
    # 500 was set as the max number of total moves
    mat_const[CHANNEL_MAPPING_CONST["total_mv_cnt"], :, :] *= MAX_NB_MOVES
    # apply rounding before converting to integer
    # No progress count
    # after 40 moves of no progress the 40 moves rule for draw applies
    mat_const[CHANNEL_MAPPING_CONST["no_progress_cnt"], :, :] *= MAX_NB_NO_PROGRESS
    np.round(x, decimals=0, out=x)
    return x


def multi_axis_by_vec(mat, vec, axis=0):
    # https://stackoverflow.com/questions/30031828/multiply-numpy-ndarray-with-1d-array-along-a-given-axis
    """
    Multiplies a matrix by a given vector element-wise along a given axis
    :param axis: The axis
    :param mat: Numpy matrix to perform the operation on
    :param vec: Numpy array which is a single vector (must have same dim as desired axis)
    :return Element-wise multiplied matrix across axis
    """
    # Create an array which would be used to reshape 1D array, b to have
    # singleton dimensions except for the given axis where we would put -1
    # signifying to use the entire length of elements along that axis
    dim_array = np.ones((1, mat.ndim), np.int16).ravel()
    dim_array[axis] = -1
    # Reshape b with dim_array and perform element-wise multiplication with
    # broadcasting along the singleton dimensions for the final output
    return mat * vec.reshape(dim_array)


def get_check_move_mask(board, legal_moves):
    """
    Returns a binary mask indicating the checking moves marked with True [Caution: Not performant]
    :param board: Python chess both
    :param legal_moves: list of legal moves
    :return: check_mask: np-boolean array marking the checking moves
            nb_checks: Number of possible checks
    """

    check_move_mask = np.zeros(len(legal_moves))
    nb_checks = 0

    for idx, move in enumerate(legal_moves):
        board_tmp = copy.deepcopy(board)
        board_tmp.push(move)
        if board_tmp.is_check():
            check_move_mask[idx] = 1
            nb_checks += 1
    return np.logical_and(check_move_mask, True), nb_checks


def get_check_moves(board, legal_moves):
    """
    Returns all possible checking moves in a list [Caution: Not performant]
    :param board: Python chess both
    :param legal_moves: list of legal moves
    :return: np-boolean array marking the checking moves
    """

    for move in legal_moves:
        board_tmp = copy.deepcopy(board)
        board_tmp.push(move)
        if board_tmp.is_check():
            yield move


def get_check_move_indices(board, legal_moves):
    """
    Returns all possible checking moves in a list [Caution: Not performant]
    :param board: Python chess both
    :param legal_moves: list of legal moves
    :return: check_move_idces: np-boolean array marking the checking moves
            nb_checks: Number of possible checks
    """

    check_move_idces = []
    nb_checks = 0
    for idx, move in enumerate(legal_moves):
        board_tmp = copy.deepcopy(board)
        board_tmp.push(move)
        if board_tmp.is_check():
            check_move_idces.append(idx)
            nb_checks += 1
    return check_move_idces, nb_checks
