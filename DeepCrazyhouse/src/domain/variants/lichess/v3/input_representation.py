"""
@file: input_representation.py
Created on 17.08.23
@project: CrazyAra
@author: queensgambit

Input representation v3 which is compatible to all available lichess variants and passed to the neural network.
"""
import numpy as np
import chess
from DeepCrazyhouse.src.domain.variants.default_input_representation import default_board_to_planes, default_planes_to_board
from DeepCrazyhouse.src.domain.variants.constants import MODE_LICHESS, NB_CHANNELS_FX, BOARD_HEIGHT, BOARD_WIDTH
from DeepCrazyhouse.src.domain.variants.classical_chess.v3.input_representation import set_additional_custom_features

NORMALIZE_POCKETS = 16
NORMALIZE_PIECE_NUMBER = 8
NORMALIZE_50_MOVE_RULE = 50

NB_PLAYERS = 2
CHANNEL_POCKETS = 14
CHANNEL_COLOR_INFO = 27
CHANNEL_TOTAL_MOVE_COUNTER = 28
CHANNEL_NO_PROGRESS = 33
CHANNEL_CUSTOM_FEATURES = 63
#CHANNEL_MATERIAL_DIFF = 66
#CHANNEL_MATERIAL_COUNT = 74

CHANNEL_PIECE_MASK = 37 + 26
CHANNEL_CHECKERBOARD = 39 + 26
CHANNEL_MATERIAL_DIFF = 40 + 26
CHANNEL_OPP_BISHOPS = 46 + 26
CHANNEL_CHECKERS = 47 + 26
CHANNEL_MATERIAL_COUNT = 48 + 26


def board_to_planes(board, board_occ=0, normalize=True, last_moves=None):
    """
    Gets the plane representation of a given board state.
    (No history of past board positions is used.)

    ## Chess Variants:

    Feature | Planes

    Chess variant input planes features as in lichess v2.0 | 63
    ---
    63 planes

    (but set color info and total move counter planes to 0)

    ###

    Additional features as in chess inputs v3.0

    P1 pieces | 1 | A grouped mask of all WHITE pieces |
    P2 pieces | 1 | A grouped mask of all BLACK pieces |
    Checkerboard | 1 | A chess board pattern |
    P1 Material Diff | 6 | (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING), normalized with 8
    Opposite Color Bishops | 1 | Indicates if they are only two bishops and the bishops are opposite color |
    Checkers | 1 | Indicates all pieces giving check |
    P1 Material Count | 6 | (pieces are ordered: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING), normalized with 8 |
    ---
    17 planes

    Total : 63 + 17 = 80 planes

    :param board: Board handle (Python-chess object)
    :param board_occ: Sets how often the board state has occurred before (by default 0)
    :param normalize: True if the inputs shall be normalized to the range [0.-1.]
    ;param last_moves: List of last moves played
    :return: planes - the plane representation of the current board state
    """

    # return the plane representation of the given board
    planes = default_board_to_planes(board, board_occ, normalize=False, mode=MODE_LICHESS, last_moves=last_moves)
    # set color info and total move counter to 0
    planes[CHANNEL_COLOR_INFO, :, :] = 0
    planes[CHANNEL_TOTAL_MOVE_COUNTER, :, :] = 0

    # mirror all bitboard entries for the black player
    mirror = board.turn == chess.BLACK and board.uci_variant != "racingkings"

    planes_fx = np.zeros((NB_CHANNELS_FX, BOARD_HEIGHT, BOARD_WIDTH))
    planes = np.concatenate((planes, planes_fx), axis=0)

    set_additional_custom_features(planes, board, CHANNEL_CUSTOM_FEATURES, mirror, normalize=False, include_king=True,
                                   channel_piece_mask=CHANNEL_PIECE_MASK, channel_checkerboard=CHANNEL_CHECKERBOARD,
                                   channel_material_diff=CHANNEL_MATERIAL_DIFF, channel_opp_bishops=CHANNEL_OPP_BISHOPS,
                                   channel_checkers=CHANNEL_CHECKERS, channel_material_count=CHANNEL_MATERIAL_COUNT)
    if normalize:
        normalize_input_planes(planes)

    return planes


def normalize_input_planes(planes):
    """
    Normalizes input planes to range [0,1]. Works in place / meaning the input parameter x is manipulated
    :param planes: Input planes representation
    :return: The normalized planes
    """
    channel = CHANNEL_POCKETS
    for _ in range(NB_PLAYERS):
        for _ in chess.PIECE_TYPES[:-1]:  # exclude the king for the pocket pieces
            planes[channel, :, :] /= NORMALIZE_POCKETS
            channel += 1
    channel = CHANNEL_MATERIAL_DIFF
    for _ in chess.PIECE_TYPES:
        planes[channel, :, :] /= NORMALIZE_PIECE_NUMBER
        channel += 1
    planes[CHANNEL_NO_PROGRESS, :, :] /= NORMALIZE_50_MOVE_RULE
    channel = CHANNEL_MATERIAL_COUNT
    for _ in chess.PIECE_TYPES:
        planes[channel, :, :] /= NORMALIZE_PIECE_NUMBER
        channel += 1

    return planes


def planes_to_board(planes, normalized_input):
    """
    Converts a board in plane representation to the python chess board representation
    see get_planes_of_board() for input encoding description
    ! Board is always returned with WHITE to move and move number = 0 !

    :param planes: Input plane representation
    :param normalized_input: Defines if the inputs are normalized to [0,1]
    :return: python chess board object
    """
    return default_planes_to_board(planes, normalized_input)
