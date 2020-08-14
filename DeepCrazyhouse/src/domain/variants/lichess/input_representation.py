"""
@file: input_representation.py
Created on 13.02.20
@project: CrazyAra
@author: queensgambit

Input representation for which is compatible to all available lichess variants and passed to the neural network
"""

import DeepCrazyhouse.src.domain.variants.input_representation as variants
from DeepCrazyhouse.src.domain.variants.constants import MODE_LICHESS


def board_to_planes(board, board_occ=0, normalize=True):
    """
    Gets the plane representation of a given board state.
    (No history of past board positions is used.)

    ## Chess Variants:

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

    P1 remaining-checks | 2 (only needed for the 3check variant, after 3 checks by one player the game ends)

    P2 remaining-checks | 2 (only needed for the 3check variant, after 3 checks by one player the game ends)

    ---
    11 planes

    * * *

    is960 = | 1 (boolean, 1 when active)

    Variants indicator each variant gets a whole channel assigned. All variants are one-hot encoded

    1 - "chess" | 1
    2 - "crazyhouse" | 1
    3 - "kingofthehill" | 1
    4 - "3check" | 1
    5 - "giveaway" | 1
    6 - "atomic" | 1
    7 - "horde" | 1
    8 - "racingkings" | 1

    ---
    9 planes

    # --------------

    * * *

    Last 8 moves | 16 (indicated by origin and destination square, the most recent move is described by first 2 planes)
    -> added since version 2

    ---
    16 planes

    The total number of planes is calculated as follows:

    27 + 11 + 9 + 16
    Total: 63 planes (version 2)
    Total: 47 planes (version 1)

    :param board: Board handle (Python-chess object)
    :param board_occ: Sets how often the board state has occurred before (by default 0)
    :param normalize: True if the inputs shall be normalized to the range [0.-1.]
    :param mode: 0 - MODE_CRAZYHOUSE: Crazyhouse only specification.
                 (Visit variants.crazyhouse.input_representation for detailed documentation)
                 1 - MODE_LICHESS: Specification for all supported variants on lichess.org
                 (Visit variants.lichess.input_representation for detailed documentation)
                 2 - MODE_CHESS: Specification for chess only with chess960 support
                 (Visit variants.chess.input_representation for detailed documentation)
    :return: planes - the plane representation of the current board state
    """

    # return the plane representation of the given board
    return variants.board_to_planes(board, board_occ, normalize, mode=MODE_LICHESS)
