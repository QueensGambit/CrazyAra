"""
@file: MCTSAgent.py
Created on 10.10.18
@project: crazy_ara_refactor
@author: queensgambit

Please describe what the content of this file is about
"""

import numpy as np
from DeepCrazyhouse.src.domain.crazyhouse.output_representation import get_probs_of_move_list, value_to_centipawn
from DeepCrazyhouse.src.domain.agent.NeuralNetAPI import NeuralNetAPI
from copy import deepcopy
from multiprocessing import Barrier, Pipe
import logging
from DeepCrazyhouse.src.domain.agent.player.util.NetPredService import NetPredService
from DeepCrazyhouse.src.domain.agent.player.util.Node import Node
from concurrent.futures import ThreadPoolExecutor
from time import time
from DeepCrazyhouse.src.domain.agent.player._Agent import _Agent
from DeepCrazyhouse.src.domain.crazyhouse.GameState import GameState


class MCTSAgent(_Agent):

    def __init__(self, net: NeuralNetAPI, virtual_loss=3, threads=8, cpuct=1, nb_playouts_empty_pockets=256, nb_playouts_filled_pockets=256, dirichlet_alpha=0.2, dirichlet_epsilon=.25,
                 temperature=0., clip_quantil=0., verbose=True, max_search_depth=15, nb_playouts_update=256, max_search_time_s=300):

        super().__init__(temperature, clip_quantil, verbose)

        # the root node contains all references to its child nodes
        self.root_node = None

        # stores the links for all nodes
        self.max_depth = 10

        # stores a lookup for all possible board states after the opposite player played its move
        self.node_lookup = {}

        # get the network reference
        self.net = net

        self.virtual_loss = virtual_loss
        self.cpuct_init = cpuct
        self.cpuct = cpuct
        self.max_search_depth = max_search_depth
        self.nb_playouts_update = nb_playouts_update
        self.max_search_time_s = max_search_time_s
        self.nb_workers = threads

        # create pip endings for itself and the prediction service
        self.my_pipe_endings = []
        pip_endings_external = []
        for i in range(threads):
            ending1, ending2 = Pipe()
            self.my_pipe_endings.append(ending1)
            pip_endings_external.append(ending2)

        self.net_pred_service = NetPredService(pip_endings_external, self.net)

        self.nb_playouts_empty_pockets = nb_playouts_empty_pockets
        self.nb_playouts_filled_pockets = nb_playouts_filled_pockets

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def evaluate_board_state(self, state_in: GameState):
        """
        Analyzes the current board state

        :param state:
        :param nb_playouts:
        :return:
        """

        # store the time at which the search started
        t_start_eval = time()

        state = deepcopy(state_in)

        # test about using a decaying cpuct value
        """
        if self.cpuct_decay != 0:
            # update the cpuct number accordingly
            sub = (state_in.get_pythonchess_board().fullmove_number-1) * self.cpuct_decay
            self.cpuct = max(self.cpuct_init - sub, self.cpuct_min)
        """

        # check if the net prediction service has already been started
        if self.net_pred_service.running is False:
            # start the prediction daemon thread
            self.net_pred_service.start()

        # receive a list of all possible legal move in the current board position
        legal_moves = list(state.get_legal_moves())

        # store what depth has been reached at maximum in the current search tree
        # default is 1, in case only 1 move is available
        max_depth_reached = 1

        # consistency check
        if len(legal_moves) == 0:
            raise Exception('The given board state has no legal move available')

        # check for fast way out
        if len(legal_moves) == 1:

            p_vec_small = np.array([1], np.float32)

            board_fen = state.get_pythonchess_board().fen()

            # check first if the the current tree can be reused
            if board_fen in self.node_lookup:
                self.root_node = self.node_lookup[board_fen]
                logging.debug('Reuse the search tree. Number of nodes in search tree: %d', self.root_node.n_sum)
            else:
                logging.debug("The given board position wasn't found in the search tree.")
                logging.debug("Starting a brand new search tree...")

                # create a new root node
                self.root_node = Node(p_vec_small, legal_moves, str(state.get_legal_moves()))

                # check a child node if it doesn't exists already
                if self.root_node.child_nodes[0] is None:
                    state_child = deepcopy(state_in)
                    state_child.apply_move(legal_moves[0])

                    legal_moves_child = list(state_child.get_legal_moves())

                    # start a brand new prediction for the child
                    state_planes = state_child.get_state_planes()
                    [_, policy_vec] = self.net.predict_single(state_planes)

                    # extract a sparse policy vector with normalized probabilities
                    p_vec_small_child = get_probs_of_move_list(policy_vec, legal_moves_child, state_child.is_white_to_move())

                    # create a new child node
                    child_node = Node(p_vec_small_child, legal_moves_child, str(state_child.get_legal_moves()))

                    # connect the child to the root
                    self.root_node.child_nodes[0] = child_node

        else:
            board_fen = state.get_board_fen()

            # check first if the the current tree can be reused
            if board_fen in self.node_lookup:
                self.root_node = self.node_lookup[board_fen]
                logging.debug('Reuse the search tree. Number of nodes in search tree: %d', self.root_node.nb_total_expanded_child_nodes)
            else:
                logging.debug("The given board position wasn't found in the search tree.")
                logging.debug("Starting a brand new search tree...")
                # start a brand new tree
                state_planes = state.get_state_planes()
                [_, policy_vec] = self.net.predict_single(state_planes)

                # extract a sparse policy vector with normalized probabilities
                p_vec_small = get_probs_of_move_list(policy_vec, legal_moves, state.is_white_to_move())

                # create a new root node
                self.root_node = Node(p_vec_small, legal_moves, str(state.get_legal_moves()))

            # clear the look up table
            self.node_lookup = {}

            # apply dirichlet noise to the prior probabilities in order to ensure that every move can possibly be visited
            self.root_node.apply_dirichlet_noise_to_prior_policy(epsilon=self.dirichlet_epsilon, alpha=self.dirichlet_alpha)

            futures = []

            # set the number of playouts accordingly
            if state_in.are_pocket_empty() is True:
                nb_playouts = self.nb_playouts_empty_pockets
            else:
                nb_playouts = self.nb_playouts_filled_pockets

            t_s = time()
            t_elapsed = 0
            cur_playouts = 0

            while max_depth_reached < self.max_search_depth and cur_playouts < nb_playouts and t_elapsed < self.max_search_time_s:

                # start searching
                with ThreadPoolExecutor(max_workers=self.nb_workers) as executor:
                    for i in range(self.nb_playouts_update):
                        # calculate the thread id based on the current playout
                        futures.append(executor.submit(self._run_single_playout, state=deepcopy(state), parent_node=self.root_node, depth=1, mv_list=[])) #, id=thread_id)

                modulo = len(futures) // 10
                cur_playouts += self.nb_playouts_update

                for i, f in enumerate(futures):
                    cur_value, cur_depth, mv_list = f.result()

                    if cur_depth > max_depth_reached:
                        max_depth_reached = cur_depth

                    if i % modulo == 0:

                        if state_in.is_white_to_move() is False:
                            cur_value *= -1

                        if self.verbose is True:
                            str_moves = self._mv_list_to_str(mv_list)
                            logging.debug('Update: %d' % cur_depth)
                            print('info score cp %d depth %d nodes %d pv%s' % (
                                value_to_centipawn(cur_value), cur_depth, self.root_node.n_sum, str_moves))

                # update the current search time
                t_elapsed = time() - t_start_eval
                print('info nps %d time %d' % ((self.nb_playouts_update / t_elapsed), t_elapsed * 1000))

            # receive the policy vector based on the MCTS search
            p_vec_small = self.root_node.get_mcts_policy()

        # store the current root in the lookup table
        self.node_lookup[state.get_board_fen()] = self.root_node

        # select the q value which would score the highest value
        value = self.root_node.q.max()

        if self.verbose is True:
            lst_best_moves, _ = self.get_calclated_line()

            str_moves = self._mv_list_to_str(lst_best_moves)

            if state_in.is_white_to_move() is False:
                value *= -1

            # show the best calculated line
            print('info score cp %d depth %d nodes %d time %d pv%s' % (
            value_to_centipawn(value), max_depth_reached, self.root_node.n_sum, (time() - t_start_eval) * 1000, str_moves))

        if len(legal_moves) != len(p_vec_small):
            print('Legal move list %s with length %s is uncompatible to policy vector %s with shape %s for board state %s' % (legal_moves, len(legal_moves), p_vec_small, p_vec_small.shape, state_in))
            self.node_lookup = {}
            # restart the search TODO: Fix this error
            """
                raise Exception('Legal move list %s with length %s is uncompatible to policy vector %s with shape %s for board state %s' % (legal_moves, len(legal_moves), p_vec_small, p_vec_small.shape, state_in))
                    Exception: Legal move list [Move.from_uci('e4h7'), Move.from_uci('e4g6'), Move.from_uci('e4f5'), Move.from_uci('c4a6'), Move.from_uci('c4b5'), Move.from_uci('c4b3'), Move.from_uci('f3g5'), Move.from_uci('f3e5'), Move.from_uci('f3h4'), Move.from_uci('f3d4'), Move.from_uci('f3d2'), Move.from_uci('f3e1'), Move.from_uci('g1h1'), Move.from_uci('f1e1'), Move.from_uci('d1e2'), Move.from_uci('d1d2'), Move.from_uci('d1e1'), Move.from_uci('d1c1'), Move.from_uci('d1b1'), Move.from_uci('a1c1'), Move.from_uci('a1b1'), Move.from_uci('d3d4'), Move.from_uci('h2h3'), Move.from_uci('g2g3'), Move.from_uci('c2c3'), Move.from_uci('b2b3'), Move.from_uci('a2a3'), Move.from_uci('h2h4'), Move.from_uci('b2b4'), Move.from_uci('a2a4'), Move.from_uci('N@b1'), Move.from_uci('N@c1'), Move.from_uci('N@e1'), Move.from_uci('N@h1'), Move.from_uci('N@d2'), Move.from_uci('N@e2'), Move.from_uci('N@a3'), Move.from_uci('N@b3'), Move.from_uci('N@c3'), Move.from_uci('N@e3'), Move.from_uci('N@g3'), Move.from_uci('N@h3'), Move.from_uci('N@a4'), Move.from_uci('N@b4'), Move.from_uci('N@d4'), Move.from_uci('N@f4'), Move.from_uci('N@h4'), Move.from_uci('N@b5'), Move.from_uci('N@f5'), Move.from_uci('N@g5'), Move.from_uci('N@h5'), Move.from_uci('N@a6'), Move.from_uci('N@b6'), Move.from_uci('N@c6'), Move.from_uci('N@e6'), Move.from_uci('N@g6'), Move.from_uci('N@d7'), Move.from_uci('N@e7'), Move.from_uci('N@h7'), Move.from_uci('N@b8'), Move.from_uci('N@c8'), Move.from_uci('N@d8'), Move.from_uci('N@e8'), Move.from_uci('N@h8')] with length 64 is uncompatible to policy vector [0.71529347 0.00194482 0.00194482 0.00389555 0.00194482 0.00194482
                     0.00389942 0.00389942 0.00389941 0.0038994  0.0019448  0.0038994
                     0.0019448  0.00389941 0.00389941 0.00194482 0.00585401 0.00194482
                     0.00194482 0.00389941 0.00389942 0.00194482 0.00194482 0.00389942
                     0.00389942 0.00389941 0.00585341 0.00194482 0.00585396 0.00389942
                     0.00389941 0.00389941 0.00389941 0.00389941 0.00194482 0.00585401
                     0.00585401 0.00194482 0.00585399 0.00780859 0.00389942 0.00389941
                     0.00585401 0.00976319 0.00780829 0.00585215 0.00389942 0.00389942
                     0.00194482 0.00194482 0.02735228 0.00389942 0.005854   0.00389939
                     0.00389924 0.00389942 0.00194482 0.00389942 0.00585398 0.00389942
                     0.0038994  0.0038994  0.00585398 0.00194482 0.00389942 0.00389942
                     0.00389942 0.00389942] with shape (68,) for board state r4rk1/ppp2pp1/3p1q1p/n1bPp3/2B1B1b1/3P1N2/PPP2PPP/R2Q1RK1[Nn] w - - 2 13
             """
            return self.evaluate_board_state(state_in)

        return value, legal_moves, p_vec_small

    def perform_action(self, state: GameState, verbose=True):

        value, selected_move, confidence, selected_child_idx = super().perform_action(state)

        # apply the selected mve on the current board state in order to create a lookup table for future board states
        state.apply_move(selected_move)

        # select the q value for the child which leads to the best calculated line
        value = self.root_node.q[selected_child_idx]

        # invert the value to the view of the white player if needed
        if state.is_white_to_move() is False:
            value = -value

        # select the next node
        node = self.root_node.child_nodes[selected_child_idx]

        # store the reference links for all possible child future child to the node lookup table
        for idx, mv in enumerate(state.get_legal_moves()):
            state_future = deepcopy(state)
            state_future.apply_move(mv)

            #try:
            # store the current child node with it's board fen as the hash-key if the child node has already been expanded
            if node.child_nodes[idx] is not None:
                self.node_lookup[state_future.get_board_fen()] = node.child_nodes[idx]

        return value, selected_move, confidence, selected_child_idx

    def _run_single_playout(self, state: GameState, parent_node: Node, depth=1, mv_list=[]): #, pipe_id):
        """
        This function works recursively until a terminal node is reached

        :param state:
        :param parent_node:
        :return:
        """

        # select a legal move on the chess board
        node, move, child_idx = self._select_node(parent_node)

        #logging.debug('selected move: %s' % move)

        #logging.error('Selected Move: %s' % move)
        if move is None:
            raise Exception("Illegal tree setup. A 'None' move was selected which souldn't be possible")

        # update the visit counts to this node
        # temporarily reduce the attraction of this node by applying a virtual loss /
        # the effect of virtual loss will be undone if the playout is over
        parent_node.apply_virtual_loss_to_child(child_idx, self.virtual_loss)

        # apply the selected move on the board
        state.apply_move(move)

        # append the selected move to the move list
        mv_list.append(move)

        board_fen = None
        if node is None:
            board_fen = state.get_board_fen()

            # iterate over all links if they exist in the look-up table
            if board_fen in self.node_lookup:
                # get the node from the look-up list
                node = self.node_lookup[board_fen]

                with parent_node.lock:
                    # setup a new connection from the parent to the child
                    parent_node.child_nodes[child_idx] = node

                # get the value from the leaf node (the current function is called recursively)
                value, depth, mv_list = self._run_single_playout(state, node, depth + 1, mv_list)

                # revert the virtual loss and apply the predicted value by the network to the node
                parent_node.revert_virtual_loss_and_update(child_idx, self.virtual_loss, -value)

                # we invert the value prediction for the parent of the above node layer because the player's turn is flipped every turn
                return -value, depth, mv_list

        # check if the current player has won the game
        # (we don't need to check for is_lost() because the game is already over if the current player checkmated his opponent)
        if state.is_won() is True:
            value = -1

        # check if you can claim a draw - its assumed that the draw is always claimed
        if state.is_draw() is True:
            value = 0.

        elif node is None:
            # expand and evaluate the new board state (the node wasn't found in the look-up table)
            # its value will be backpropagated through the tree and flipped after every layer

            # receive a free available pipe
            my_pipe = self.my_pipe_endings.pop()
            my_pipe.send(state.get_state_planes())
            # this pipe waits for the predictions of the network inference service
            [value, policy_vec] = my_pipe.recv()
            # put the used pipe back into the list
            self.my_pipe_endings.append(my_pipe)

            # get the current legal move of its board state
            legal_moves = list(state.get_legal_moves())

            if len(legal_moves) < 1:
                raise Exception('No legal move is available for state: %s' % state)

            # extract a sparse policy vector with normalized probabilities
            try:
                p_vec_small = get_probs_of_move_list(policy_vec, legal_moves,
                                                     is_white_to_move=state.is_white_to_move(), normalize=True)

            except KeyError:
                raise Exception('Key Error for state: %s' % state)

            # create a new node
            new_node = Node(p_vec_small, legal_moves, str(state.get_legal_moves()))

            # test of adding dirichlet noise to a new node
            #new_node.apply_dirichlet_noise_to_prior_policy(epsilon=self.dirichlet_epsilon/8, alpha=self.dirichlet_alpha)

            # include a reference to the new node in the look-up table
            self.node_lookup[board_fen] = new_node

            with parent_node.lock:
                # add the new node to its parent
                parent_node.child_nodes[child_idx] = new_node

            # check if the new node has a mate_in_one connection (if yes overwrite the network prediction)
            if new_node.mate_child_idx is not None:
                value = 1
        else:
            # get the value from the leaf node (the current function is called recursively)
            value, depth, mv_list = self._run_single_playout(state, node, depth+1, mv_list)

        # revert the virtual loss and apply the predicted value by the network to the node
        parent_node.revert_virtual_loss_and_update(child_idx, self.virtual_loss, -value)

        # we invert the value prediction for the parent of the above node layer because the player's turn is flipped every turn
        return -value, depth, mv_list

    def _select_node(self, parent_node: Node):
        """
        Selects the best child node from a given parent node based on the q and u value

        :param parent_node:
        :return: node - Reference to the node object which has been selected
                        If this node hasn't been expanded yet, None will be returned
                move - The move which leads to the selected child node from the given parent node on forward
                node_idx - Integer idx value indicating the index for the selected child of the parent node
        """

        # check first if there's an immediate mate in one move possible
        if parent_node.mate_child_idx is not None:
            child_idx = parent_node.mate_child_idx
        else:
            # find the move according to the q- and u-values for each move

            # calculate the current u values
            # it's not worth to save the u values as a node attribute because u is updated every time n_sum changes
            u = self.cpuct * parent_node.p * (np.sqrt(parent_node.n_sum) / (1 + parent_node.n))
            child_idx = (parent_node.q + u).argmax()

        node = parent_node.child_nodes[child_idx]
        # now receive the according move based on the calculated move index
        move = parent_node.legal_moves[child_idx]

        return node, move, child_idx

    def _select_node_based_on_visit(self, parent_node: Node):

        #child_idx = parent_node.n.argmax()

        child_idx = parent_node.get_mcts_policy().argmax()

        nb_visits = parent_node.n[child_idx]
        move = parent_node.legal_moves[child_idx]

        return parent_node.child_nodes[child_idx], move, nb_visits

    def show_next_pred_line(self):
        best_moves = []
        # start at the root node
        node = self.root_node

        while node is not None:
            # go deep through the tree by always selecting the best move for both players
            node, move, child_idx = self._select_node(node)
            best_moves.append(move)
        return best_moves

    def get_calclated_line(self):

        if self.root_node is None:
            logging.warning('You must run an evaluation first in order to get the calculated line')

        lst_best_moves = []
        lst_nb_visits = []
        # start at the root node
        node = self.root_node

        while node is not None:
            # go deep through the tree by always selecting the best move for both players
            node, move, nb_visits = self._select_node_based_on_visit(node)
            lst_best_moves.append(move)
            lst_nb_visits.append(nb_visits)
        return lst_best_moves, lst_nb_visits

    def _mv_list_to_str(self, lst_moves):
        """
        Converts a given list of chess moves to a single string seperated by spaces.
        :param lst_moves: List chess.Moves objects
        :return: String representing each move in the list
        """
        str_moves = ""
        for mv in lst_moves:
            str_moves += " " + mv.uci()
        return str_moves
