"""
@file: RawNetAgent.py
Created on 14.10.18
@project: crazy_ara_refactor
@author: queensgambit

The raw network uses the the single network prediction for it's evaluation.
No mcts search is being done.
"""

from time import time
from deep_crazy_house.src.domain.agent.player._Agent import _Agent
from deep_crazy_house.src.domain.abstract_cls._GameState import _GameState
from deep_crazy_house.src.domain.agent.NeuralNetAPI import NeuralNetAPI
from deep_crazy_house.src.domain.crazyhouse.output_representation import get_probs_of_move_list, value_to_centipawn


class RawNetAgent(_Agent):
    def __init__(self, net: NeuralNetAPI, temperature=0.0, temperature_moves=4, verbose=True):
        super().__init__(temperature, temperature_moves, verbose)
        self._net = net

    def evaluate_board_state(self, state: _GameState):
        """

        :param state:
        :return:
        """

        t_start_eval = time()
        pred_value, pred_policy = self._net.predict_single(state.get_state_planes())

        legal_moves = list(state.get_legal_moves())

        p_vec_small = get_probs_of_move_list(pred_policy, legal_moves, state.is_white_to_move())

        # use the move with the highest probability as the best move for logging
        instinct_move = legal_moves[p_vec_small.argmax()]

        # define the remaining return variables
        time_e = time() - t_start_eval
        cp = value_to_centipawn(pred_value)
        depth = 1
        nodes = 1
        time_elapsed_s = time_e * 1000
        nps = nodes / time_e
        pv = instinct_move.uci()

        return pred_value, legal_moves, p_vec_small, cp, depth, nodes, time_elapsed_s, nps, pv
