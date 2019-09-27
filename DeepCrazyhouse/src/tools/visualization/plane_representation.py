"""
@file: plane_representation
Created on 24.09.18
@project: crazy_ara_refactor
@author: queensgambit

Please describe what the content of this file is about
"""
import numpy as np
from DeepCrazyhouse.src.domain.util import multi_axis_by_vec
from DeepCrazyhouse.src.domain.variants.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CHANNEL_MAPPING_CONST,
    NB_CHANNELS_POS,
    PIECES,
    PIECES_VALUE,
    chess,
)


# create vector which scales the piece values according to their crazyhouse value
# (used in get_x_vis())
SCALE_VEC = np.zeros(len(chess.PIECE_TYPES))
for i, p_char in enumerate(PIECES[: len(chess.PIECE_TYPES)]):
    SCALE_VEC[i] = PIECES_VALUE[p_char] * 2


def get_plane_vis(mat, normalize=False):
    """
    Returns the board representation of all piece positions in a form
    which is plot-able using a diverging colormap
    The pieces are defined by their subjective board value (defined in util.py/pieces_value).
    :param mat: Input plane representation of a single data sample
    :param normalize: True if the outputs should be normalized to [0,1]
    :return: 8x8 numpy array which represents the piece positions
    """
    color_bit = int(mat[CHANNEL_MAPPING_CONST["color"] + NB_CHANNELS_POS][0][0])
    if color_bit not in (0, 1):
        raise Exception("Invalid setting of color bit: ", color_bit)

    sign_bit = -1 if chess.COLOR_NAMES[color_bit] == "black" else 1
    x_vis = np.zeros((BOARD_HEIGHT, BOARD_WIDTH))
    x_vis += sign_bit * multi_axis_by_vec(mat[: len(chess.PIECE_TYPES)], SCALE_VEC, axis=0).max(axis=0)
    x_vis += -sign_bit * multi_axis_by_vec(
        mat[len(chess.PIECE_TYPES) : 2 * len(chess.PIECE_TYPES)], SCALE_VEC, axis=0
    ).max(axis=0)
    # in real life the board isn't flipped but rotated instead
    x_vis = np.flipud(x_vis)

    if normalize:
        sc_max = SCALE_VEC.max()
        x_vis += sc_max
        x_vis /= 2 * sc_max

    return x_vis
