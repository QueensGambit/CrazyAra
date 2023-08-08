"""
@file: input_representation.py
Created on 03.07.23
@project: CrazyAra
@author: queensgambit

Input representation 2.0 for crazyhouse based on the input representation 1.0 of chess.
"""

import chess
from chess.variant import CrazyhouseBoard

from DeepCrazyhouse.src.domain.variants.classical_chess.v3.input_representation import set_no_progress_counter
from DeepCrazyhouse.src.domain.variants.default_input_representation import default_board_to_planes,\
    default_normalize_input_planes
from DeepCrazyhouse.src.domain.variants.constants import MODE_CRAZYHOUSE, MAX_NB_MOVES, MAX_NB_PRISONERS, \
    MAX_NB_NO_PROGRESS
from DeepCrazyhouse.src.domain.variants.classical_chess.v2.input_representation import set_ep_square, \
    set_castling_rights
from DeepCrazyhouse.src.domain.variants.default_input_representation import set_pieces_on_board, \
    set_pocket_pieces_to_board

NORMALIZE_POCKETS = MAX_NB_PRISONERS  # at maximum, you can have only 16 pawns (your own and the ones of the opponent)
NORMALIZE_50_MOVE_RULE = MAX_NB_NO_PROGRESS
# These constant describe the starting channel for the corresponding info
CHANNEL_POCKETS = 14
CHANNEL_PROMO = 24
CHANNEL_EN_PASSANT = 26
CHANNEL_COLOR = 27
CHANNEL_MV_CNT = 28
CHANNEL_CASTLING = 29
CHANNEL_NO_PROGRESS = 33
CHANNEL_IS_960 = 50


def board_to_planes(board: chess.Board, board_occ, normalize=True, last_moves=None):
    """
    Returns the plane representation 2.0 of a given board state for crazyhouse.
    This representation is based on the chess representation 1.0 and adds missing additional information for crazyhouse.

    ## Crazyhouse

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

    ---
    7 planes

    * * *

    is960 = | 1 (boolean, 1 when active)

    ---

    * * *

    Last 8 moves | 16 (indicated by origin and destination square, the most recent move is described by first 2 planes)

    ---
    16 planes


    The total number of planes is calculated as follows:
    # --------------
    27 + 7 + 1 + 16
    = 39 + 12
    Total: 51 planes

    """
    planes = default_board_to_planes(board, board_occ, last_moves, MODE_CRAZYHOUSE, normalize)
    return planes


def planes_to_board(planes, normalized_input):
    """
    Converts a board in plane representation to the python chess board representation
    see get_planes_of_board() for input encoding description
    ! Board is always returned with WHITE to move and move number and no progress counter = 0 !

    :param planes: Input plane representation
    :param normalized_input: Defines if the inputs are normalized to [0,1]
    :return: python chess board object
    """
    is960 = planes[CHANNEL_IS_960, 0, 0] == 1
    board = CrazyhouseBoard(chess960=is960)
    board.clear()

    set_pieces_on_board(board, planes, check_for_promo=True, promo_channel=CHANNEL_PROMO)
    set_pocket_pieces_to_board(board, CHANNEL_POCKETS, planes, normalized_input, NORMALIZE_POCKETS)
    set_ep_square(board, CHANNEL_EN_PASSANT, planes)
    set_castling_rights(board, CHANNEL_CASTLING, planes, is960)
    set_no_progress_counter(board, CHANNEL_NO_PROGRESS, planes, normalized_input, NORMALIZE_50_MOVE_RULE)
    set_total_move_count_to_board(board, planes, normalized_input, CHANNEL_MV_CNT, MAX_NB_MOVES)

    board = mirror_board_depending_on_color(board, planes, CHANNEL_COLOR)

    return board


def normalize_input_planes(planes):
    """
    Normalizes input planes to range [0,1]. Works in place / meaning the input parameter x is manipulated
    :param planes: Input planes representation
    :return: The normalized planes
    """
    return default_normalize_input_planes(planes)


def set_total_move_count_to_board(board, planes, normalized_input, channel_mv_cnt, max_nb_moves):
    """"Sets the move counter of the board object depending on the planes object."""
    total_mv_cnt = planes[channel_mv_cnt, 0, 0]

    if normalized_input is True:
        total_mv_cnt *= max_nb_moves
        total_mv_cnt = int(round(total_mv_cnt))
    board.fullmove_number = total_mv_cnt


def mirror_board_depending_on_color(board, planes, channel_color):
    """Mirrors the board depending on the color information in the planes object."""
    if planes[channel_color, 0, 0] == 1:
        board.board_turn = chess.WHITE
    else:
        board = board.mirror()
        board.board_turn = chess.BLACK
    return board
