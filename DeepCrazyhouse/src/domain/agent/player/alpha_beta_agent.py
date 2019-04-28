"""
@file: minimax_agent.py
Created on 24.03.19
@project: CrazyAra
@author: queensgambit

Classical negamax search with alpha beta pruning.
For more details see: https://en.wikipedia.org/wiki/Negamax
"""
import math
import logging
import copy
from time import time
import numpy as np


from DeepCrazyhouse.src.domain.abstract_cls.abs_agent import AbsAgent
from DeepCrazyhouse.src.domain.abstract_cls.abs_game_state import AbsGameState
from DeepCrazyhouse.src.domain.agent.neural_net_api import NeuralNetAPI
from DeepCrazyhouse.src.domain.crazyhouse.output_representation import value_to_centipawn, get_probs_of_move_list
from DeepCrazyhouse.src.domain.util import get_check_move_indices


class AlphaBetaAgent(AbsAgent):
    """
    Alpha beta agent which has the option to clip moves to make the search tractable for NN engines
    """

    def __init__(self, net: NeuralNetAPI, depth=5, nb_candidate_moves=7, include_check_moves=False):
        """
        Constructor
        :param net: Neural network inference service
        :param depth: Depth of the search tree from which all evaluations will be based on
        :param nb_candidate_moves: Number of moves to consider at each depth during search which are clipped according
        to the neural network policy
        :param include_check_moves: Defines if checking moves shall always be considered
        """
        AbsAgent.__init__(self)
        self.t_start_eval = None
        self.net = net
        self.nodes = 0
        self.depth = depth
        self.nb_candidate_moves = nb_candidate_moves
        self.best_moves = [None] * depth
        self.sel_mv_idx = [None] * depth
        self.include_check_moves = include_check_moves

    def negamax(self, state, depth, alpha=-math.inf, beta=math.inf, color=1, all_moves=1):
        """
        Evaluates all nodes at a given depth and back-propagates their values to their respective parent nodes.
        In order to keep the number nof nodes manageable for neural network evaluation
        :param all_moves: All possible moves
        :param state: Game state object
        :param depth: Number of depth to reach during search
        :param alpha: Current alpha value which is used for pruning
        :param beta: Current beta value which is used for pruning
        :param color: Integer color value 1 for white, -1 for black
        :return: best_value - Best value for the current player until search depth
        """

        if state.is_won():  # check for draw is neglected for now due to bad runtime
            return -1

        [value, policy_vec] = self.net.predict_single(state.get_state_planes())  # start a brand new tree

        if depth == 0:
            return value  # the value is always returned in the view of the current player

        best_value = -math.inf  # initialization

        legal_moves = state.get_legal_moves()
        p_vec_small = get_probs_of_move_list(policy_vec, state.get_legal_moves(), state.is_white_to_move())

        if all_moves > 0:
            mv_idces = list(np.argsort(p_vec_small)[::-1])
        else:
            mv_idces = list(np.argsort(p_vec_small)[::-1][: self.nb_candidate_moves])

        if self.include_check_moves:
            check_idces, _ = get_check_move_indices(state.get_pythonchess_board(), state.get_legal_moves())
            mv_idces += check_idces

        for mv_idx in mv_idces:  # each child of position
            if p_vec_small[mv_idx] > 0.1:
                mv = legal_moves[mv_idx]
                state_child = copy.deepcopy(state)
                state_child.apply_move(mv)
                value = -self.negamax(state_child, depth - 1, -beta, -alpha, -color, all_moves - 1)
                if value > best_value:
                    self.best_moves[-depth] = mv
                    self.sel_mv_idx[-depth] = mv_idx
                    best_value = value
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        return best_value

    def evaluate_board_state(self, state: AbsGameState) -> tuple:
        """
        Evaluates a given board position according to alpha beta search
        :param state: Game state object
        :return:
        """
        self.t_start_eval = time()
        value = self.negamax(
            state, depth=self.depth, alpha=-math.inf, beta=math.inf, color=1 if state.board.turn else -1
        )

        legal_moves = state.get_legal_moves()
        policy = np.zeros(len(legal_moves))
        policy[self.sel_mv_idx[0]] = 1
        centipawn = value_to_centipawn(value)
        # depth = 1
        nodes = self.nodes
        time_e = time() - self.t_start_eval  # In uci the depth is given using half-moves notation also called plies
        time_elapsed_s = time_e * 1000
        nps = nodes / time_e
        pv = self.best_moves[0].uci()

        logging.info(f"{self.best_moves}")
        logging.info(f"Value: {value}, Centipawn: {centipawn}")
        return value, legal_moves, policy, centipawn, self.depth, nodes, time_elapsed_s, nps, pv
