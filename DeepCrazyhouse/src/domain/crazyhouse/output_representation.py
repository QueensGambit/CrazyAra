"""
@file: output_presentation
Created on 24.09.18
@project: crazy_ara_refactor
@author: queensgambit

Provides all methods to convert a move to policy representation and back
Loads all needed constants for the Crazyhouse game internally.
"""
import chess.variant
import numpy as np
from DeepCrazyhouse.src.domain.crazyhouse.constants import (
    LABELS,
    LABELS_MIRRORED,
    MV_LOOKUP,
    MV_LOOKUP_MIRRORED,
    NB_LABELS,
)


def move_to_policy(move, is_white_to_move=True):
    """
    Returns a numpy vector with the bit set to 1 a the according index (one hot encoding)

    :param move Python chess obj. defining a move
    :param is_white_to_move: Define the current player turn
    :return: Policy numpy vector in boolean format
    """

    if is_white_to_move is True:
        mv_idx = MV_LOOKUP[move.uci()]
    else:
        mv_idx = MV_LOOKUP_MIRRORED[move.uci()]

    policy_vec = np.zeros(NB_LABELS, dtype=np.bool)
    # set the bit to 1 at the according move index
    policy_vec[mv_idx] = 1

    return policy_vec


def policy_to_move(policy_vec_clean, is_white_to_move=True):
    """
    Returns a python-chess move object based on the given move index

    :param policy_vec_clean: Numpy array which represents the moves
    :param is_white_to_move: Define the current player turn
    :return: sinlge move - Python chess move object
    """

    mv_idx = np.argmax(policy_vec_clean)

    # ensure that the provided mv_idx is legal
    assert 0 <= mv_idx < NB_LABELS

    if is_white_to_move is True:
        mv_uci = LABELS[mv_idx]
    else:
        mv_uci = LABELS_MIRRORED[mv_idx]

    move = chess.Move.from_uci(mv_uci)

    return move


def policy_to_best_move(board: chess.variant.CrazyhouseBoard, policy_vec, normalize=True):
    """
    Returns a python-chess move object of the best legal move according to the policy vector
    The policy vec can also be one hot encoded.

    :param board: Current CrazyhouseBoard state
    :param policy_vec: 1 Dimensional array representing all possible Crazyhouse moves
    :param normalize: normalizes the policy vector in the cleaning step
    :return: best_mv - Single move object representing the highest activation in the probability vector
             prob - Probability value for the selected move
    """

    policy_vec_clean, nb_legal_moves = set_illegal_moves_to_zero(board, policy_vec, normalize)

    mv_idx = np.argmax(policy_vec_clean)
    prob = np.max(policy_vec_clean)

    if board.turn is chess.WHITE:
        mv_uci = LABELS[mv_idx]
    else:
        mv_uci = LABELS_MIRRORED[mv_idx]

    best_mv = chess.Move.from_uci(mv_uci)

    return best_mv, prob


def policy_to_moves(board: chess.variant.CrazyhouseBoard, policy_vec, normalize=True):
    """
    Returns a list of the legal moves from best to worst based on the given policy and their corresponding probability

    :param board: Current CrazyhouseBoard state (don't mirror the board for the black player)
    :param policy_vec: 1 Dimensional array representing all possible Crazyhouse moves
    :param normalize: normalizes the policy vector in the cleaning step
    :return: selected_moves - List of all python chess move objects from best to worst
            move_probabilites - Corresponding probabilities ordered descending
    """

    policy_vec_clean, nb_legal_moves = set_illegal_moves_to_zero(board, policy_vec, normalize)

    sorted_move_value_indices = np.argsort(policy_vec_clean)[::-1]
    move_probabilites = np.sort(policy_vec_clean)[::-1]

    selected_moves = []

    # iterate over all legal moves and add them to the list from best to worst
    for i in range(nb_legal_moves):
        mv_idx = sorted_move_value_indices[i]

        if board.turn is chess.WHITE:
            mv_uci = LABELS[mv_idx]
        else:
            mv_uci = LABELS_MIRRORED[mv_idx]

        mv = chess.Move.from_uci(mv_uci)
        selected_moves.append(mv)

    return selected_moves, move_probabilites


def set_illegal_moves_to_zero(board: chess.variant.CrazyhouseBoard, policy_vec, normalize=True):
    """
    :param board:
    :param policy_vec:
    :param normalize: normalizes the confidence that they will sum up to 1. again
    :return: policy_vec_out - cleaned probability vector by masking illegal moves to 0
             nb_legal_moves - number of legal moves in the given position
    """

    legal_moves = list(board.legal_moves)

    policy_vec_out = np.zeros(NB_LABELS, dtype=np.float32)

    nb_legal_moves = len(legal_moves)

    # check if the game is already over
    if nb_legal_moves == 0:
        raise Exception("No legal move is available in the current position.")

    # fast routine if only 1 move is available
    if nb_legal_moves == 1:

        mv = legal_moves[0]

        # get the according label index for the selected move
        if board.turn is chess.WHITE:
            idx = MV_LOOKUP[mv.uci()]
        else:
            idx = MV_LOOKUP_MIRRORED[mv.uci()]
        policy_vec_out[idx] = 1

        return policy_vec_out, 1

    # iterate over all legal move and get the move probabilities
    for mv in legal_moves:

        # get the according label index for the selected move
        if board.turn is chess.WHITE:
            idx = MV_LOOKUP[mv.uci()]
        else:
            idx = MV_LOOKUP_MIRRORED[mv.uci()]

        policy_vec_out[idx] = policy_vec[idx]

    # make sure that the probabilities sum up to 1. again
    if normalize is True:
        policy_vec_out /= policy_vec_out.sum()

    return policy_vec_out, nb_legal_moves


def get_probs_of_move_list(policy_vec: np.ndarray, mv_list: [chess.Move], is_white_to_move, normalize=True):
    """
    Returns an array in which entry relates to the probability for the given move list.
    Its assumed that the moves in the move list are legal and shouldn't be mirrored.
    :param policy_vec: Policy vector from the neural net prediction
    :param mv_list: List of legal moves for a specific board position
    :param is_white_to_move: Determine if it's white's or black's turn to move
    :param normalize: True, if the probability should be normalized
    :return: p_vec_small - A numpy vector which stores the probabilities for the given move list
    """

    # allocate sufficient memory
    p_vec_small = np.zeros(len(mv_list), np.float32)

    for i, mv in enumerate(mv_list):

        if is_white_to_move is True:
            # find the according index in the vector
            idx = MV_LOOKUP[mv.uci()]
        else:
            # use the mirrored look-up table instead
            idx = MV_LOOKUP_MIRRORED[mv.uci()]

        # set the right prob value
        p_vec_small[i] = policy_vec[idx]

    if normalize is True:
        p_vec_small /= sum(p_vec_small)

    return p_vec_small


def value_to_centipawn(value):
    """
    Converts a value in A0-notation to roughly a centi-pawn loss
    :param value: floating value from [-1.,1.]
    :return:
    """

    if np.absolute(value) >= 1.0:
        # return a constant if the given value is 1 (otherwise log will result in infinity)
        return np.sign(value) * 9999
    # use logaritmic scaling with basis 1.1 as a pseudo centipawn conversion
    return -(np.sign(value) * np.log(1.0 - np.absolute(value)) / np.log(1.2)) * 100


if __name__ == "__main__":
    print(MV_LOOKUP)
