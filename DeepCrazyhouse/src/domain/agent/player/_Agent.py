"""
@file: _PlayerAgent
Created on 14.10.18
@project: crazy_ara_refactor
@author: queensgambit

Please describe what the content of this file is about
"""

from DeepCrazyhouse.src.domain.abstract_cls._GameState import _GameState
import numpy as np
from copy import deepcopy


class _Agent:
    """
    The greedy agent always performs the first legal move with the highest move probability
    """

    def __init__(self, temperature=0., clip_quantil=0., verbose=True):
        self.temperature = temperature
        self.p_vec_small = None
        self.clip_quantil = clip_quantil
        self.verbose = verbose

    def evaluate_board_state(self, state: _GameState):
        raise NotImplementedError

    def perform_action(self, state: _GameState):

        # the first step is to call you policy agent to evaluate the given position
        value, legal_moves, self.p_vec_small = self.evaluate_board_state(state)

        if len(legal_moves) != len(self.p_vec_small):
            raise Exception('Legal move list %s is uncompatible to policy vector %s' % (legal_moves, self.p_vec_small))

        if len(legal_moves) == 1:
            selected_move = legal_moves[0]
            confidence = 1.
            idx = 0
        else:
            if self.temperature <= 0.01:
                idx = self.p_vec_small.argmax()
            else:
                self._apply_temperature_to_policy()
                self._apply_quantil_clipping()
                idx = np.random.choice(range(len(legal_moves)), p=self.p_vec_small)

            selected_move = legal_moves[idx]
            confidence = self.p_vec_small[idx]

        return value, selected_move, confidence, idx

    def _apply_quantil_clipping(self):
        """

        :param p_vec_small:
        :param clip_quantil:
        :return:
        """

        if self.clip_quantil > 0:
            # remove the lower percentage values in order to avoid strange blunders for moves with low confidence
            p_vec_small_clipped = deepcopy(self.p_vec_small)

            # get the sorted indices in ascending order
            idx_order = np.argsort(self.p_vec_small)
            # create a quantil tank which measures how much quantil power is left
            quantil_tank = self.clip_quantil

            # iterate over the indices (ascending) and apply the quantil clipping to it
            for idx in idx_order:
                if quantil_tank >= p_vec_small_clipped[idx]:
                    # remove the prob from the quantil tank
                    quantil_tank -= p_vec_small_clipped[idx]
                    # clip the index to 0
                    p_vec_small_clipped[idx] = 0
                else:
                    # the target prob is greate than the current quantil tank
                    p_vec_small_clipped[idx] -= quantil_tank
                    # stop the for loop
                    break

            # renormalize the policy
            p_vec_small_clipped /= p_vec_small_clipped.sum()

            # apply the changes
            self.p_vec_small = p_vec_small_clipped

    def _apply_temperature_to_policy(self):
        """

        :return:
        """
        # treat very small temperature value as a deterministic policy
        if self.temperature <= 0.01:
            p_vec_one_hot = np.zeros_like(self.p_vec_small)
            p_vec_one_hot[np.argmax(self.p_vec_small)] = 1.
            self.p_vec_small = p_vec_one_hot
        else:
            # apply exponential scaling
            self.p_vec_small = np.power(self.p_vec_small, 1/self.temperature)
            # renormalize the values to probabilities again
            self.p_vec_small /= self.p_vec_small.sum()
