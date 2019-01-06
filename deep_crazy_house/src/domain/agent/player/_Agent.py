"""
@file: _PlayerAgent
Created on 14.10.18
@project: crazy_ara_refactor
@author: queensgambit

Please describe what the content of this file is about
"""

from copy import deepcopy
import numpy as np
from deep_crazy_house.src.domain.abstract_cls._GameState import _GameState


class _Agent:
    """
    The greedy agent always performs the first legal move with the highest move probability
    """

    def __init__(self, temperature=0, temperature_moves=4, verbose=True):
        self.temperature = temperature
        self.temperature_current = temperature
        self.temperature_moves = temperature_moves
        # self.p_vec_small = None
        self.verbose = verbose

    def evaluate_board_state(self, state: _GameState):
        raise NotImplementedError

    def perform_action(self, state: _GameState):

        # the first step is to call you policy agent to evaluate the given position
        value, legal_moves, p_vec_small, cp, depth, nodes, time_elapsed_s, nps, pv = self.evaluate_board_state(state)

        if len(legal_moves) != len(p_vec_small):
            raise Exception("Legal move list %s is uncompatible to policy vector %s" % (legal_moves, p_vec_small))

        if state.get_fullmove_number() <= self.temperature_moves:
            self.temperature_current = self.temperature
        else:
            self.temperature_current = 0

        if len(legal_moves) == 1:
            selected_move = legal_moves[0]
            confidence = 1.0
            idx = 0
        else:
            if self.temperature_current <= 0.01:
                idx = p_vec_small.argmax()
            else:
                p_vec_small = self._apply_temperature_to_policy(p_vec_small)
                idx = np.random.choice(range(len(legal_moves)), p=p_vec_small)

            selected_move = legal_moves[idx]
            confidence = p_vec_small[idx]

            if value > 0:
                # check for draw and decline if value is greater 0
                state_future = deepcopy(state)
                state_future.apply_move(selected_move)
                if state_future.get_pythonchess_board().can_claim_threefold_repetition() is True:
                    p_vec_small[idx] = 0
                    idx = p_vec_small.argmax()
                    selected_move = legal_moves[idx]
                    confidence = p_vec_small[idx]

        return value, selected_move, confidence, idx, cp, depth, nodes, time_elapsed_s, nps, pv

    def _apply_temperature_to_policy(self, p_vec_small):
        """

        :return:
        """
        # treat very small temperature value as a deterministic policy
        if self.temperature_current <= 0.01:
            p_vec_one_hot = np.zeros_like(p_vec_small)
            p_vec_one_hot[np.argmax(p_vec_small)] = 1.0
            p_vec_small = p_vec_one_hot
        else:
            # apply exponential scaling
            p_vec_small = p_vec_small ** (1 / self.temperature_current)
            # renormalize the values to probabilities again
            p_vec_small /= p_vec_small.sum()

        return p_vec_small
