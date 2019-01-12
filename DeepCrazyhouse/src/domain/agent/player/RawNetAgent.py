"""
@file: RawNetAgent.py
Created on 14.10.18
@project: crazy_ara_refactor
@author: queensgambit

The raw network uses the the single network prediction for it's evaluation.
No mcts search is being done.
"""
from time import time
from DeepCrazyhouse.src.domain.abstract_cls._GameState import _GameState
from DeepCrazyhouse.src.domain.agent.NeuralNetAPI import NeuralNetAPI
from DeepCrazyhouse.src.domain.abstract_cls._Agent import _Agent
from DeepCrazyhouse.src.domain.crazyhouse.output_representation import get_probs_of_move_list, value_to_centipawn


class RawNetAgent(_Agent):
    def __init__(self, net: NeuralNetAPI, temperature=0.0, temperature_moves=4, verbose=True):
        super().__init__(temperature, temperature_moves, verbose)
        self._net = net

    def evaluate_board_state(self, state: _GameState):
        """
        The greedy agent always performs the first legal move with the highest move probability

        :param state: Gamestate object
        :return:
        value - Value prediction in the current players view from [-1,1]: -1 -> 100% lost, +1 100% won
        selected_move - Python chess move object of the selected move
        confidence - Probability value for the selected move in the probability distribution
        idx - Integer index of the move which was returned
        centipawn - Centi pawn evaluation which is converted from the value prediction in currents player view
        depth - Depth which was reached after the search
        nodes - Number of nodes which have been evaluated in the search
        time_elapsed_s - Elapsed time in seconds for the full search
        nps - Nodes per second metric
        pv - Calculated best line for both players
        """

        t_start_eval = time()
        pred_value, pred_policy = self._net.predict_single(state.get_state_planes())

        legal_moves = list(state.get_legal_moves())

        p_vec_small = get_probs_of_move_list(pred_policy, legal_moves, state.is_white_to_move())

        # use the move with the highest probability as the best move for logging
        instinct_move = legal_moves[p_vec_small.argmax()]

        # define the remaining return variables
        time_e = time() - t_start_eval
        centipawn = value_to_centipawn(pred_value)
        depth = 1
        nodes = 1
        time_elapsed_s = time_e * 1000
        nps = nodes / time_e
        pv = instinct_move.uci()

        return pred_value, legal_moves, p_vec_small, centipawn, depth, nodes, time_elapsed_s, nps, pv
