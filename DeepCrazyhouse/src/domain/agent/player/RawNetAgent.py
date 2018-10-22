"""
@file: RawNetAgent.py
Created on 14.10.18
@project: crazy_ara_refactor
@author: queensgambit

The raw network uses the the single network prediction for it's evaluation.
No mcts search is being done.
"""

from DeepCrazyhouse.src.domain.agent.player._Agent import _Agent
from DeepCrazyhouse.src.domain.abstract_cls._GameState import _GameState
from DeepCrazyhouse.src.domain.agent.NeuralNetAPI import NeuralNetAPI
from DeepCrazyhouse.src.domain.crazyhouse.output_representation import get_probs_of_move_list, value_to_centipawn
from time import time


class RawNetAgent(_Agent):

    def __init__(self, net: NeuralNetAPI, temperature=0., clip_quantil=0., verbose=True):
        super().__init__(temperature, clip_quantil, verbose)
        self._net = net

    def evaluate_board_state(self, state: _GameState, verbose=True):
        """

        :param state:
        :return:
        """
        t_start_eval = time()
        pred_value, pred_policy = self._net.predict_single(state.get_state_planes())

        if state.is_white_to_move() is False:
            pred_value *= -1

        legal_moves = list(state.get_legal_moves())
        p_vec_small = get_probs_of_move_list(pred_policy, legal_moves, state.is_white_to_move())

        if verbose is True:
            # use the move with the highest probability as the best move for logging
            instinct_move = legal_moves[p_vec_small.argmax()]

            # show the best calculated line
            print('info score cp %d depth %d nodes %d time %d pv %s' % (
            value_to_centipawn(pred_value), 1, 1, (time() - t_start_eval) * 1000, instinct_move.uci()))

        return pred_value, legal_moves, p_vec_small
