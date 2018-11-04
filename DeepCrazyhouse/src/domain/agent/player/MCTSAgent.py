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

import cProfile, pstats, io


def profile(fnc):
    """
    A decorator that uses cProfile to profile a function
    :param fnc: Function handle to decorate.
    :return:
    """

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


class MCTSAgent(_Agent):

    def __init__(self, net: NeuralNetAPI, threads=16, batch_size=8, playouts_empty_pockets=256,
                 playouts_filled_pockets=512, cpuct=1, dirichlet_epsilon=.25,
                 dirichlet_alpha=0.2, max_search_depth=15, temperature=0., clip_quantil=0.,
                 q_value_weight=0., virtual_loss=3, verbose=True, min_movetime=100):
        """
        Constructor of the MCTSAgent.
        The MCTSAgent runs playouts/simulations in the search tree and updates the node statistics.
        The final move is chosen according to the visit count of each direct child node.
        One playout is defined as expanding one new node in the tree. In the case of chess this means evaluating a new board position.

        If the evaluation for one move takes too long on your hardware you can decrease the value for:
         nb_playouts_empty_pockets and nb_playouts_filled_pockets.

        For more details and the mathematical equations please take a look at src/domain/agent/README.md as well as the
        official DeepMind-papers.

        :param net: NeuralNetAPI handle which is used to communicate with the neural network
        :param threads: Number of threads to evaluate the nodes in parallel
        :param playouts_empty_pockets: Number of playouts/simulations which will be done if the Crazyhouse-Pockets of
                                        both players are empty.
        :param playouts_filled_pockets: Number of playouts/simulations which will be done if at least one player has a
                                        piece in their pocket. The number of legal-moves is higher when drop
                                        moves are available.
        :param cpuct: CPUCT-value which weights the balance between the policy/action and value term.
                     The playstyle depends strongly on this value.
        :param dirichlet_epsilon: Weigh value for the dirichlet noise. If 0. -> no noise. If 1. -> complete noise.
                                The dirichlet noise ensures that unlikely nodes can be explored
        :param dirichlet_alpha: Alpha parameter of the dirichlet noise which is applied to the prior policy for the
                                current root node: https://en.wikipedia.org/wiki/Dirichlet_process
        :param max_search_depth: Maximum search depth to reach in the current search tree. If the depth has been reached
                                the evaluation stops.
        :param temperature: The temperature parameters is an exponential scaling factor which is applied to the
                            posterior policy. Afterwards the chosen move to play is sampled from this policy.
                            Range: [0.,1.]:
                            If 0. -> Deterministic policy. The move is chosen with the highest probability
                            If 1. -> Pure random sampling policy. The move is sampled from the posterior without any
                                    scaling being applied.
        :param clip_quantil: A quantil clipping parameter with range [0., 1.]. All cummulated low percentages for moves
                            are set to 0. This makes sure that very unlikely moves (blunders) are clipped after
                            the exponential scaling.
        :param: q_value_weight: Float indicating how the number of visits and the q-values should be mixed.
                                Expected to be in range [0.,1.]
        :param virtual_loss: An artificial loss term which is applied to each node which is currently being visited.
                             This term make it look like that the current visit of this node led to +X losses where X
                             is the virtual loss. This prevents that every thread will evaluate the same node.
        :param verbose: Defines weather to print out info messages for the current calculated line
        :param min_movetime: Minimum time in milliseconds to search for the best move
        """

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
        self.nb_workers = threads
        self.batch_size = batch_size

        # create pip endings for itself and the prediction service
        self.my_pipe_endings = []
        pip_endings_external = []
        for i in range(threads):
            ending1, ending2 = Pipe()
            self.my_pipe_endings.append(ending1)
            pip_endings_external.append(ending2)

        self.net_pred_service = NetPredService(pip_endings_external, self.net, batch_size)

        self.nb_playouts_empty_pockets = playouts_empty_pockets
        self.nb_playouts_filled_pockets = playouts_filled_pockets

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        self.movetime_ms = min_movetime
        self.q_value_weight = q_value_weight

    def evaluate_board_state(self, state_in: GameState):
        """
        Analyzes the current board state

        :param state_in: Actual game state to evaluate for the MCTS
        :return:
        """

        # store the time at which the search started
        t_start_eval = time()

        state = deepcopy(state_in)

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

            # set value 0 as a dummy value
            value = 0
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
                self.root_node = Node(value, p_vec_small, legal_moves, str(state.get_legal_moves()))

                # check a child node if it doesn't exists already
                if self.root_node.child_nodes[0] is None:
                    state_child = deepcopy(state_in)
                    state_child.apply_move(legal_moves[0])

                    # initialize is_leaf by default to false
                    is_leaf = False

                    # check if the current player has won the game
                    # (we don't need to check for is_lost() because the game is already over
                    #  if the current player checkmated his opponent)
                    if state.is_won() is True:
                        value = -1
                        is_leaf = True
                        legal_moves_child = []
                        p_vec_small_child = None

                    # check if you can claim a draw - its assumed that the draw is always claimed
                    elif state.is_draw() is True:
                        value = 0
                        is_leaf = True
                        legal_moves_child = []
                        p_vec_small_child = None

                    else:
                        legal_moves_child = list(state_child.get_legal_moves())

                        # start a brand new prediction for the child
                        state_planes = state_child.get_state_planes()
                        [value, policy_vec] = self.net.predict_single(state_planes)

                        # extract a sparse policy vector with normalized probabilities
                        p_vec_small_child = get_probs_of_move_list(policy_vec, legal_moves_child, state_child.is_white_to_move())

                    # create a new child node
                    child_node = Node(value, p_vec_small_child, legal_moves_child, str(state_child.get_legal_moves()), is_leaf)

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

                # initialize is_leaf by default to false
                is_leaf = False

                # start a brand new tree
                state_planes = state.get_state_planes()
                [value, policy_vec] = self.net.predict_single(state_planes)

                # extract a sparse policy vector with normalized probabilities
                p_vec_small = get_probs_of_move_list(policy_vec, legal_moves, state.is_white_to_move())

                # create a new root node
                self.root_node = Node(value, p_vec_small, legal_moves, str(state.get_legal_moves()), is_leaf)

            # clear the look up table
            self.node_lookup = {}

            # apply dirichlet noise to the prior probabilities in order to ensure
            #  that every move can possibly be visited
            self.root_node.apply_dirichlet_noise_to_prior_policy(epsilon=self.dirichlet_epsilon, alpha=self.dirichlet_alpha)

            futures = []

            # set the number of playouts accordingly
            if state_in.are_pocket_empty() is True:
                nb_playouts = self.nb_playouts_empty_pockets
            else:
                nb_playouts = self.nb_playouts_filled_pockets

            t_elapsed = 0
            cur_playouts = 0
            old_time = time()

            while max_depth_reached < self.max_search_depth and\
                       cur_playouts < nb_playouts and\
                     t_elapsed*1000 < self.movetime_ms: #and np.abs(self.root_node.q.mean()) < 0.99:

                # start searching
                with ThreadPoolExecutor(max_workers=self.nb_workers) as executor:
                    for i in range(self.nb_workers):
                        # calculate the thread id based on the current playout
                        futures.append(executor.submit(self._run_single_playout, state=deepcopy(state),
                                                       parent_node=self.root_node, depth=1, mv_list=[]))

                cur_playouts += self.nb_workers
                time_show_info = time() - old_time

                # store the mean of all value predictions in this variable
                #mean_value = 0

                for i, f in enumerate(futures):
                    cur_value, cur_depth, mv_list = f.result()

                    # sum up all values
                    #mean_value += cur_value

                    if cur_depth > max_depth_reached:
                        max_depth_reached = cur_depth

                    # Print every second if verbose is true
                    if self.verbose and time_show_info > 1:
                        str_moves = self._mv_list_to_str(mv_list)
                        logging.debug('Update: %d' % cur_depth)
                        print('info score cp %d depth %d nodes %d pv%s' % (
                            value_to_centipawn(cur_value), cur_depth, self.root_node.n_sum, str_moves))
                        old_time = time()

                # update the current search time
                t_elapsed = time() - t_start_eval
                if self.verbose and time_show_info > 1:
                    print('info nps %d time %d' % ((self.root_node.n_sum / t_elapsed), t_elapsed * 1000))

            # receive the policy vector based on the MCTS search
            p_vec_small = self.root_node.get_mcts_policy(self.q_value_weight)
            print('info string move overhead is %dms' % (t_elapsed*1000 - self.movetime_ms))

        # store the current root in the lookup table
        self.node_lookup[state.get_board_fen()] = self.root_node

        # select the q value which would score the highest value

        #value = self.root_node.q.max()

        # select the q-value according to the mcts best child value
        best_child_idx = self.root_node.get_mcts_policy(self.q_value_weight).argmax()
        value = self.root_node.q[best_child_idx]

        lst_best_moves, _ = self.get_calculated_line()

        str_moves = self._mv_list_to_str(lst_best_moves)

        # show the best calculated line
        time_e = time() - t_start_eval
        node_searched = self.root_node.n_sum
        print('info score cp %d depth %d nodes %d time %d nps %d pv%s' % (
            value_to_centipawn(value), max_depth_reached, node_searched, time_e*1000, node_searched/max(1, time_e), str_moves))

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

        # select the next node
        node = self.root_node.child_nodes[selected_child_idx]

        # store the reference links for all possible child future child to the node lookup table
        for idx, mv in enumerate(state.get_legal_moves()):
            state_future = deepcopy(state)
            state_future.apply_move(mv)

            # store the current child node with it's board fen as the hash-key if the child node has already been expanded
            if node is not None and idx < node.nb_direct_child_nodes and node.child_nodes[idx] is not None:
                self.node_lookup[state_future.get_board_fen()] = node.child_nodes[idx]

        return value, selected_move, confidence, selected_child_idx

    #@profile
    def _run_single_playout(self, state: GameState, parent_node: Node, depth=1, mv_list=[]): #, pipe_id):
        """
        This function works recursively until a terminal node is reached

        :param state: Current game-state for the evaluation. This state differs between the treads
        :param parent_node: Current parent-node of the selected node. In the first  expansion this is the root node.
        :param depth: Current depth for the evaluation. Depth is increased by 1 for every recusive call
        :param mv_list: List of moves which have been taken in the current path. For each selected child node this list
                        is expanded by one move recursively.
        :return: -value: The inverse value prediction of the current board state. The flipping by -1 each turn is needed
                        because the point of view changes each half-move
                depth: Current depth reach by this evaluation
                mv_list: List of moves which have been selected
        """

        # select a legal move on the chess board
        node, move, child_idx = self._select_node(parent_node)

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

        if node is None:

            # get the board-fen which is used as an identifier for the board positions in the look-up table
            board_fen = state.get_board_fen()

            # check if the addressed fen exist in the look-up table
            if board_fen in self.node_lookup:
                # get the node from the look-up list
                node = self.node_lookup[board_fen]

                with parent_node.lock:
                    # setup a new connection from the parent to the child
                    parent_node.child_nodes[child_idx] = node

                # get the prior value from the leaf node which has already been expanded
                #value = node.v

                # get the value from the leaf node (the current function is called recursively)
                value, depth, mv_list = self._run_single_playout(state, node, depth+1, mv_list)

            else:
                # expand and evaluate the new board state (the node wasn't found in the look-up table)
                # its value will be backpropagated through the tree and flipped after every layer

                # receive a free available pipe
                my_pipe = self.my_pipe_endings.pop()
                my_pipe.send(state.get_state_planes())
                # this pipe waits for the predictions of the network inference service
                [value, policy_vec] = my_pipe.recv()
                # put the used pipe back into the list
                self.my_pipe_endings.append(my_pipe)

                # initialize is_leaf by default to false
                is_leaf = False

                # check if the current player has won the game
                # (we don't need to check for is_lost() because the game is already over
                #  if the current player checkmated his opponent)
                if state.is_won() is True:
                    value = -1
                    is_leaf = True
                    legal_moves = []
                    p_vec_small = None

                # check if you can claim a draw - its assumed that the draw is always claimed
                elif state.is_draw() is True:
                    value = 0
                    is_leaf = True
                    legal_moves = []
                    p_vec_small = None
                else:
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
                #new_node = Node(value, p_vec_small, legal_moves, str(state.get_legal_moves()), is_leaf)
                new_node = Node(value, p_vec_small, legal_moves, '', is_leaf)

                #if is_leaf is False:
                #    # test of adding dirichlet noise to a new node
                #    new_node.apply_dirichlet_noise_to_prior_policy(epsilon=self.dirichlet_epsilon/4, alpha=self.dirichlet_alpha)

                # include a reference to the new node in the look-up table
                self.node_lookup[board_fen] = new_node

                with parent_node.lock:
                    # add the new node to its parent
                    parent_node.child_nodes[child_idx] = new_node

                # check if the new node has a mate_in_one connection (if yes overwrite the network prediction)
                if new_node.mate_child_idx is not None:
                    value = 1

        # check if we have reached a leaf node
        elif node.is_leaf is True:
            value = node.v
            # receive a free available pipe
            my_pipe = self.my_pipe_endings.pop()
            my_pipe.send(state.get_state_planes())
            # this pipe waits for the predictions of the network inference service
            [_, _] = my_pipe.recv()
            # put the used pipe back into the list
            self.my_pipe_endings.append(my_pipe)

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

    def _select_node_based_on_mcts_policy(self, parent_node: Node):
        """
        Selects the next node based on the mcts policy which is used to predict the final best move.

        :param parent_node: Node from which to select the next child.
        :return:
        """
        child_idx = parent_node.get_mcts_policy(self.q_value_weight).argmax()

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

    def get_calculated_line(self):
        """
        Prints out the best search line estimated for both players on the given board state.
        :return:
        """

        if self.root_node is None:
            logging.warning('You must run an evaluation first in order to get the calculated line')

        lst_best_moves = []
        lst_nb_visits = []
        # start at the root node
        node = self.root_node

        while node is not None and node.is_leaf is False:
            # go deep through the tree by always selecting the best move for both players
            node, move, nb_visits = self._select_node_based_on_mcts_policy(node)
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

    def update_movetime(self, time_ms_per_move):
        """
        Update move time allocation.
        :param time_ms_per_move:  Sets self.movetime_ms to this value
        :return:
        """
        self.movetime_ms = time_ms_per_move
