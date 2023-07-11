"""
@file: input_representation
Created on 26.09.19
@project: CrazyAra
@author: queensgambit

Input representation for all available lichess chess variants board states (including  crazyhouse)
which is passed to the neural network
"""

import numpy as np
import DeepCrazyhouse.src.domain.variants.classical_chess.v2.input_representation as chess_v2
import DeepCrazyhouse.src.domain.variants.classical_chess.v3.input_representation as chess_v3
import DeepCrazyhouse.src.domain.variants.crazyhouse.v2.input_representation as crazyhouse_v2
import DeepCrazyhouse.src.domain.variants.crazyhouse.v3.input_representation as crazyhouse_v3
from DeepCrazyhouse.src.domain.variants.default_input_representation import default_board_to_planes,\
    default_normalize_input_planes, default_planes_to_board
from DeepCrazyhouse.src.domain.variants.constants import (
    MODES,
    VERSION,
    MODE_CRAZYHOUSE,
    MODE_CHESS,
    NB_LAST_MOVES,
    chess)
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.src.domain.variants.constants import MODE


def board_to_planes(board, board_occ=0, normalize=True, mode=MODE_CRAZYHOUSE, last_moves=None):
    """
    Gets the plane representation of a given board state.
    (No history of past board positions is used.)

    :param board: Board handle (Python-chess object)
    :param board_occ: Sets how often the board state has occurred before (by default 0)
    :param normalize: True if the inputs shall be normalized to the range [0.-1.]
    :param mode: 0 - MODE_CRAZYHOUSE: Crazyhouse only specification.
                 (Visit variants.crazyhouse.input_representation for detailed documentation)
                 1 - MODE_LICHESS: Specification for all supported variants on lichess.org
                 (Visit variants.lichess.input_representation for detailed documentation)
                 2 - MODE_CHESS: Specification for chess only with chess960 support
                 (Visit variants.chess.input_representation for detailed documentation)
    :param last_moves: List of last moves. It is assumed that the most recent move is the first entry !
    :return: planes - the plane representation of the current board state
    """
    if mode == MODE_CHESS and VERSION == 2:
        return chess_v2.board_to_planes(board, normalize, last_moves)
    if mode == MODE_CHESS and VERSION == 3:
        return chess_v3.board_to_planes(board, board_occ, normalize, last_moves)
    if mode == MODE_CRAZYHOUSE and VERSION == 2:
        return crazyhouse_v2.board_to_planes(board, board_occ, normalize, last_moves)
    if mode == MODE_CRAZYHOUSE and VERSION == 3:
        return crazyhouse_v3.board_to_planes(board, board_occ, normalize, last_moves)

    return default_board_to_planes(board, board_occ, last_moves, mode, normalize)


def planes_to_board(planes, normalized_input=False, mode=MODE_CRAZYHOUSE):
    """
    Converts a board in plane representation to the python chess board representation
    see get_planes_of_board() for input encoding description

    :param planes: Input plane representation
    :param normalized_input: True if the input has been normalized to range[0., 1.]
    :param mode: 0 - MODE_CRAZYHOUSE: Crazyhouse only specification.
                 (Visit variants.crazyhouse.input_representation for detailed documentation)
                 1 - MODE_LICHESS: Specification for all supported variants on lichess.org
                 (Visit variants.lichess.input_representation for detailed documentation)
                 2 - MODE_CHESS: Specification for chess only with chess960 support
                 (Visit variants.chess.input_representation for detailed documentation)
    :return: python chess board object
    """
    if mode not in MODES:
        raise ValueError(f"Given {mode} is not {MODES}.")

    if mode == MODE_CHESS and VERSION == 2:
        return chess_v2.planes_to_board(planes)
    if mode == MODE_CHESS and VERSION == 3:
        return chess_v3.planes_to_board(planes, normalized_input)
    if mode == MODE_CRAZYHOUSE and VERSION == 2:
        return crazyhouse_v2.planes_to_board(planes, normalized_input)
    if mode == MODE_CRAZYHOUSE and VERSION == 3:
        return crazyhouse_v3.planes_to_board(planes, normalized_input)

    return default_planes_to_board(planes, normalized_input, mode)


def normalize_input_planes(x):
    """
    Normalizes input planes to range [0,1]. Works in place / meaning the input parameter x is manipulated
    :param x: Input planes representation
    :return: The normalized planes
    """

    # convert the input planes to float32 assuming that the datatype is int
    if x.dtype != np.float32:
        x = x.astype(np.float32)

    if MODE == MODE_CHESS and VERSION == 2:
        return chess_v2.normalize_input_planes(x)
    if MODE == MODE_CHESS and VERSION == 3:
        return chess_v3.normalize_input_planes(x)

    return default_normalize_input_planes(x)


def get_planes_statistics(board: chess.Board, normalize: bool, last_moves_uci: list, board_occ=0):
    """
    Returns a dictionary for statistics of the plane which can be used for Unit-Testing.
    e.g get_planes_statistics(board, False, last_moves=[chess.Move.from_uci("d7d5")])
    :param board: Chess board object
    :param normalize: Decides if the planes should be normalized
    :param last_moves_uci: Last moves in UCI notation. Chronologically ordered, meaning first move is first entry and
    most recent move is last entry.
    :param board_occ: Gives information on how often this position has occurred already.
    """
    last_moves = []
    for uci_move in last_moves_uci[::-1]:
        last_moves.append(chess.Move.from_uci(uci_move))
    if len(last_moves) < NB_LAST_MOVES:
        for _ in range(NB_LAST_MOVES-len(last_moves)):
            last_moves.append(None)
    last_moves = last_moves[:NB_LAST_MOVES]

    planes = board_to_planes(board, board_occ=board_occ, normalize=normalize, mode=main_config['mode'],
                             last_moves=last_moves)
    planes = planes.flatten()
    stats = {}
    stats['sum'] = planes.sum()
    stats['argMax'] = planes.argmax()
    stats['maxNum'] = planes.max()
    stats['key'] = 0
    for i in range(len(planes)):
        stats['key'] += i * planes[i]
    return stats
