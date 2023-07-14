"""
@file: input_representation.py
Created on 03.07.23
@project: CrazyAra
@author: queensgambit

Input representation 3.0 for crazyhouse based on the input representation 3.0 of chess.
"""
import chess
from chess.variant import CrazyhouseBoard
import DeepCrazyhouse.src.domain.variants.classical_chess.v3.input_representation as chess_v3
from DeepCrazyhouse.src.domain.variants.classical_chess.v2.input_representation import set_ep_square, \
    set_castling_rights
from DeepCrazyhouse.src.domain.variants.default_input_representation import _set_crazyhouse_info, set_pieces_on_board, \
    set_pocket_pieces_to_board

NORMALIZE_POCKETS = 32  # at maximum, you can have only 16 pawns (your own and the ones of the opponent)
NORMALIZE_50_MOVE_RULE = 40  # we choose the older value that was used for crazyhouse
# These constant describe the starting channel for the corresponding info
CHANNEL_POCKETS = 52
CHANNEL_PROMO = 62


def board_to_planes(board: chess.Board, board_occ, normalize=True, last_moves=None):
    """
    Returns the plane representation 3.0 of a given board state for crazyhouse.
    This representation is based on the chess representation 3.0 and adds missing additional information for crazyhouse.

    ## Crazyhouse

    Feature | Planes

    Chess inputs planes features as in chess v3.0 | 52

    ---
    52 planes

    * * *

    P1 prisoner count | 5 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN) (excluding the KING)

    P2 prisoner count | 5 (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN) (excluding the KING)

    P1 Promoted Pawns Mask | 1 (binary map indicating the pieces which have been promoted)

    P2 Promoted Pawns Mask | 1 (binary map indicating the pieces which have been promoted)

    ---
    12 planes

    The total number of planes is calculated as follows:
    # --------------
    52 + 12
    Total: 64 planes

    """
    planes = chess_v3.board_to_planes(board, board_occ, normalize, last_moves, NORMALIZE_50_MOVE_RULE)
    _set_crazyhouse_info(board, planes, normalize,
                         channel_prisoners=CHANNEL_POCKETS,
                         max_nb_prisoners=NORMALIZE_POCKETS,
                         channel_promo=CHANNEL_PROMO)
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
    is960 = planes[chess_v3.CHANNEL_IS_960, 0, 0] == 1
    board = CrazyhouseBoard(chess960=is960)
    board.clear()

    set_pieces_on_board(board, planes, check_for_promo=True, promo_channel=CHANNEL_PROMO)
    set_ep_square(board, chess_v3.CHANNEL_EN_PASSANT, planes)
    set_castling_rights(board, chess_v3.CHANNEL_CASTLING, planes, is960)
    set_pocket_pieces_to_board(board, CHANNEL_POCKETS, planes, normalized_input, NORMALIZE_POCKETS)
    chess_v3.set_no_progress_counter(board, chess_v3.CHANNEL_NO_PROGRESS, planes, normalized_input,
                                     NORMALIZE_50_MOVE_RULE)

    return board


def normalize_input_planes(planes):
    """
    Normalizes input planes to range [0,1]. Works in place / meaning the input parameter x is manipulated
    :param planes: Input planes representation
    :return: The normalized planes
    """
    chess_v3.normalize_input_planes(planes, NORMALIZE_50_MOVE_RULE)
    channel = CHANNEL_POCKETS
    for _ in range(2):  # iterate for player 1 and player 2
        for _ in chess.PIECE_TYPES[:-1]:
            planes[channel, :, :] /= NORMALIZE_POCKETS
            channel += 1

    return planes
