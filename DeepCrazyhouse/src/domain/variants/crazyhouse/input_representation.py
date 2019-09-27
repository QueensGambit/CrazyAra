"""
@file: input_representation.py
Created on 20.06.18
@project: jc39qevo-deep-learning-project
@author: queensgambit

Input representation for the Crazyhouse board state which is passed to the neural network
"""

import DeepCrazyhouse.src.domain.variants.input_representation as variants


def board_to_planes(board, board_occ=0, normalize=True):
    """
    Gets the plane representation of a given board state.
    (No history of past board positions is used.)

    ## Crazyhouse:

    Feature | Planes

    --- | ---

    P1 piece | 6 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)

    P2 piece | 6 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)

    Repetitions | 2 (two planes (full zeros/ones) indicating how often the board positions has occurred)

    P1 prisoner count | 5 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN) (excluding the KING)

    P2 prisoner count | 5 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN) (excluding the KING)

    P1 Promoted Pawns Mask | 1 (binary map indicating the pieces which have been promoted)

    P2 Promoted Pawns Mask | 1 (binary map indicating the pieces which have been promoted)

    En-passant square | 1 (Binary map indicating the square where en-passant capture is possible)

    ---
    27 planes

    * * *

    Colour | 1 (all zeros for black and all ones for white)

    Total move count | 1 (integer value setting the move count (uci notation))

    P1 castling | 2 (One if castling is possible, else zero)

    P2 castling | 2 (One if castling is possible, else zero)

    No-progress count | 1 (Setting the no progress counter as integer values, (described by uci halfmoves format)

    # --------------
    7 planes

    The history list of the past 7 board states have been removed
    The total number of planes is calculated as follows:

    27 + 7
    Total: 34 planes

    :param board: Board handle (Python-chess object)
    :param board_occ: Sets how often the board state has occurred before (by default 0)
    :param normalize: True if the inputs shall be normalized to the range [0.-1.]
    :return: planes - the plane representation of the current board state
    """

    # return the plane representation of the given board
    return variants.board_to_planes(board, board_occ, normalize, crazyhouse_only=True)


def planes_to_board(planes, normalized_input=False):
    """
    Converts a board in plane representation to the python chess board representation
    see get_planes_of_board() for input encoding description

    :param planes: Input plane representation
    :param normalized_input: True if the input has been normalized to range[0., 1.]
    :return: python chess board object
    """

    # used as wrapper function
    return variants.planes_to_board(planes, normalized_input, crazyhouse_only=True)
