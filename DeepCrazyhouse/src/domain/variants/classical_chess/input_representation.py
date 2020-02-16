"""
@file: input_representation.py
Created on 13.02.20
@project: CrazyAra
@author: queensgambit

Input representation for the chess board state which is passed to the neural network
(folder is name "classical_chess" to avoid name clash with python chess library)
"""

import DeepCrazyhouse.src.domain.variants.input_representation as variants
from DeepCrazyhouse.src.domain.variants.constants import MODE_CHESS


def board_to_planes(board, board_occ=0, normalize=True):
    """
    Gets the plane representation of a given board state.
    (No history of past board positions is used.)

    ## Chess:

    Feature | Planes

    --- | ---

    P1 piece | 6 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)

    P2 piece | 6 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)

    Repetitions | 2 (two planes (full zeros/ones) indicating how often the board positions has occurred)

    En-passant square | 1 (Binary map indicating the square where en-passant capture is possible)

    ---
    15 planes

    * * *

    Colour | 1 (all zeros for black and all ones for white)

    Total move count | 1 (integer value setting the move count (uci notation))

    P1 castling | 2 (One if castling is possible, else zero)

    P2 castling | 2 (One if castling is possible, else zero)

    No-progress count | 1 (Setting the no progress counter as integer values, (described by uci halfmoves format)

    ---
    7 planes

    The history list of the past 7 board states have been removed
    The total number of planes is calculated as follows:

    * * *

    is960 = | 1 (boolean, 1 when active)

    ---
    1 plane

    # --------------
    15 + 7 + 1
    Total: 23 planes

    :param board: Board handle (Python-chess object)
    :param board_occ: Sets how often the board state has occurred before (by default 0)
    :param normalize: True if the inputs shall be normalized to the range [0.-1.]
    :return: planes - the plane representation of the current board state
    """

    # return the plane representation of the given board
    return variants.board_to_planes(board, board_occ, normalize, mode=MODE_CHESS)


def planes_to_board(planes, normalized_input=False):
    """
    Converts a board in plane representation to the python chess board representation
    see get_planes_of_board() for input encoding description

    :param planes: Input plane representation
    :param normalized_input: True if the input has been normalized to range[0., 1.]
    :return: python chess board object
    """

    # used as wrapper function
    return variants.planes_to_board(planes, normalized_input, mode=MODE_CHESS)
