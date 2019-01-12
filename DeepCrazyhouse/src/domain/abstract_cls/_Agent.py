"""
@file: _PlayerAgent
Created on 14.10.18
@project: crazy_ara_refactor
@author: queensgambit

Abstract class for defining a playing agent.
"""
from copy import deepcopy
import numpy as np
from DeepCrazyhouse.src.domain.abstract_cls._GameState import _GameState


class _Agent:

    def __init__(self, temperature=0, temperature_moves=4, verbose=True):
        self.temperature = temperature
        self.temperature_current = temperature
        self.temperature_moves = temperature_moves
        self.verbose = verbose

    def evaluate_board_state(self, state: _GameState):
        raise NotImplementedError

    def perform_action(self, state: _GameState):
        """
        Retuns a selected move given a game state by calling evaluate_board_state(state) in order to get a probability
        distribution.
        :param state: Game state object for a board position
        :return:
        value - Value prediction in the current players view from [-1,1]: -1 -> 100% lost, +1 100% won
        selected_move - Python chess move object of the selected move
        confidence - Probability value for the selected move in the probability distribution
        idx - Integer index of the move which was retuned
        cp - Centi pawn evaluation which is converted from the value prediction in currents player view
        depth - Depth which was reached after the search
        nodes - Number of nodes which have been evaluated in the search
        time_elapsed_s - Elapsed time in seconds for the full search
        nps - Nodes per second metric
        pv - Calculated best line for both players
        """
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
        Applies temperature rescaling to the policy distribution by enhancing higher probability values.
        A temperature below 0.01 relates to one hot encoding.
        :param p_vec_small: Probability distribution for all legal moves in the position
        :return: p_vec_small - Probability distribution after applying temperature scaling to it
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
