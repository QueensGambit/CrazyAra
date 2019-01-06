"""
@file: plane_representation
Created on 24.09.18
@project: crazy_ara_refactor
@author: queensgambit

Please describe what the content of this file is about
"""

from DeepCrazyhouse.src.domain.crazyhouse.input_representation import *

# create vector which scales the piece values according to their crazyhouse value
# (used in get_x_vis())
scale_vec = np.zeros(len(chess.PIECE_TYPES))


def fill_scale_vec():
    global scale_vec
    for i, p_char in enumerate(PIECES[: len(chess.PIECE_TYPES)]):
        scale_vec[i] = PIECES_VALUE[p_char] * 2


fill_scale_vec()


def get_plane_vis(mat, normalize=False):
    """
    Returns the board representation of all piece positions in a form
    which is plot-able using a diverging colormap
    The pieces are defined by their subjective board value (defined in util.py/pieces_value).
    :param mat: Input plane representation of a single data sample
    :param normalize: True if the outputs should be normalized to [0,1]
    :return: 8x8 numpy array which represents the piece positions
    """
    color_ch = CHANNEL_MAPPING_CONST["color"] + NB_CHANNELS_POS

    color_bit = int(mat[color_ch][0][0])
    if color_bit != 0 and color_bit != 1:
        raise Exception("Invalid setting of color bit: ", color_bit)

    sign_bit = -1 if chess.COLOR_NAMES[color_bit] == "black" else 1
    x_vis = np.zeros((BOARD_HEIGHT, BOARD_WIDTH))
    x_vis += sign_bit * mult_axis_by_vec(mat[: len(chess.PIECE_TYPES)], scale_vec, axis=0).max(axis=0)
    x_vis += -sign_bit * mult_axis_by_vec(
        mat[len(chess.PIECE_TYPES) : 2 * len(chess.PIECE_TYPES)], scale_vec, axis=0
    ).max(axis=0)

    # in real life the board isn't flipped but rotated instead
    x_vis = np.flipud(x_vis)

    if normalize is True:
        sc_max = scale_vec.max()
        x_vis += sc_max
        x_vis /= 2 * sc_max

    return x_vis
