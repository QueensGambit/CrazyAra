"""
@file: MCTSAgent.py
Created on 10.10.18
@project: crazy_ara_refactor
@author: queensgambit

The MCTSAgent runs playouts/simulations in the search tree and updates the node statistics.
The final move is chosen according to the visit count of each direct child node.
One playout is defined as expanding one new node in the tree. In the case of chess this means evaluating a new board position.

If the evaluation for one move takes too long on your hardware you can decrease the value for:
 nb_playouts_empty_pockets and nb_playouts_filled_pockets.

For more details and the mathematical equations please take a look at src/domain/agent/README.md as well as the
official DeepMind-papers.
"""

import numpy as np

from DeepCrazyhouse.src.domain.crazyhouse.output_representation import get_probs_of_move_list, value_to_centipawn
from DeepCrazyhouse.src.domain.agent.NeuralNetAPI import NeuralNetAPI
from copy import deepcopy
from multiprocessing import Pipe
import logging
from DeepCrazyhouse.src.domain.agent.player.util.NetPredService import NetPredService
from DeepCrazyhouse.src.domain.agent.player.util.Node import Node
from concurrent.futures import ThreadPoolExecutor
from time import time
from DeepCrazyhouse.src.domain.agent.player._Agent import _Agent
from DeepCrazyhouse.src.domain.crazyhouse.GameState import GameState
import collections
from DeepCrazyhouse.src.domain.crazyhouse.constants import NB_CHANNELS_FULL, BOARD_WIDTH, BOARD_HEIGHT, NB_LABELS

import cProfile, pstats, io
DTYPE = np.float


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

    def __init__(self, nets: [NeuralNetAPI], threads=16, batch_size=8, playouts_empty_pockets=256,
                 playouts_filled_pockets=512, cpuct=1, dirichlet_epsilon=.25,
                 dirichlet_alpha=0.2, max_search_depth=15, temperature=0., temperature_moves=4,
                 q_value_weight=0., virtual_loss=3, verbose=True, min_movetime=100, check_mate_in_one=False,
                 use_pruning=True, use_oscillating_cpuct=True):
        """
        Constructor of the MCTSAgent.

        :param net: NeuralNetAPI handle which is used to communicate with the neural network
        :param threads: Number of threads to evaluate the nodes in parallel
        :param batch_size: Fixed batch_size which is used in the network prediction service.
                           The batch_size coordinates the prediction flow for the network-prediction service.
                           Using a mxnet executor object which uses a fixed batch_size is faster than accepting
                           arbitrary batch_sizes.
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
        :param temperature_moves: Number of fullmoves in which the temperature parameter will be applied.
                                  Otherwise the temperature will be set to 0 for deterministic play.
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
        :param check_mate_in_one: Decide whether to check for every leaf node if a there is a mate in one move then
                                  create a mate in one short cut which prioritzes this move. Currently by default this
                                  option is disabled because it takes costs too much nps regarding its benefit.
        :param enable_timeout: Decides weather to enable a timout if a batch didn't occur under 1 second for the
                               NetPredService.
        """

        super().__init__(temperature, temperature_moves, verbose)

        # the root node contains all references to its child nodes
        self.root_node = None

        # stores the links for all nodes
        self.max_depth = 10

        # stores a lookup for all possible board states after the opposite player played its move
        self.node_lookup = {}

        # get the network reference
        self.nets = nets

        self.virtual_loss = virtual_loss
        self.cpuct_init = cpuct

        if cpuct < 0.01 or cpuct > 10:
            raise Exception('You might have confused centi-cpuct with cpuct.'
                            'The requested cpuct is beyond reasonable range: cpuct should be around > 0.01 and < 10.')

        self.cpuct = cpuct
        self.max_search_depth = max_search_depth
        self.threads = threads

        # check for possible issues when giving an illegal batch_size and number of threads combination
        if batch_size > threads:
            raise Exception('info string The given batch_size %d is higher than the number of threads %d. '
                            'The maximum legal batch_size is the same as the number of threads (here: %d) '
                            % (batch_size, threads, threads))

        if threads % batch_size != 0:
            raise Exception('You requested an illegal combination of threads %d and batch_size %d.'
                            ' The batch_size must be a divisor of the number of threads' % (threads, batch_size))

        self.batch_size = batch_size

        # create pip endings for itself and the prediction service
        self.my_pipe_endings = []
        pip_endings_external = []
        for i in range(threads):
            ending1, ending2 = Pipe()
            self.my_pipe_endings.append(ending1)
            pip_endings_external.append(ending2)

        self.nb_playouts_empty_pockets = playouts_empty_pockets
        self.nb_playouts_filled_pockets = playouts_filled_pockets

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        self.movetime_ms = min_movetime
        self.q_value_weight = q_value_weight

        self.check_mate_in_one = check_mate_in_one

        # temporary variables
        # time counter
        self.t_start_eval = None
        # number of nodes before the evaluate_board_state() call are stored here to measure the nps correctly
        self.total_nodes_pre_search = None

        # allocate shared memory for communicating with the network prediction service
        self.batch_state_planes = np.zeros((self.threads, NB_CHANNELS_FULL, BOARD_HEIGHT, BOARD_WIDTH), DTYPE)
        self.batch_value_results = np.zeros(self.threads, DTYPE)
        self.batch_policy_results = np.zeros((self.threads, NB_LABELS), DTYPE)

        # initialize the NetworkPredictionService and give the pointers to the shared memory
        self.net_pred_services = []
        nb_pipes = self.threads // len(nets)

        # create multiple gpu-access points
        for i, net in enumerate(nets):
            net_pred_service = NetPredService(pip_endings_external[i*nb_pipes:(i+1)*nb_pipes], net, batch_size, self.batch_state_planes,
                                                self.batch_value_results, self.batch_policy_results)
            self.net_pred_services.append(net_pred_service)

        self.transposition_table = collections.Counter()
        self.send_batches = False
        self.root_node_prior_policy = None

        self.use_pruning = use_pruning
        self.use_oscillating_cpuct = use_oscillating_cpuct

    def evaluate_board_state(self, state: GameState):
        """
        Analyzes the current board state. This is the main method which get called by the uci interface or analysis
        request.

        :param state_in: Actual game state to evaluate for the MCTS
        :return:
        """

        # store the time at which the search started
        self.t_start_eval = time()

        # check if the net prediction service has already been started
        if self.net_pred_services[0].running is False:
            # start the prediction daemon thread
            for net_pred_service in self.net_pred_services:
                net_pred_service.start()

        # receive a list of all possible legal move in the current board position
        legal_moves = state.get_legal_moves()

        # consistency check
        if len(legal_moves) == 0:
            raise Exception('The given board state has no legal move available')

        # check first if the the current tree can be reused
        key = (state.get_transposition_key(), state.get_halfmove_counter)

        if self.use_pruning is False and key in self.node_lookup:
            self.root_node = self.node_lookup[key]
            logging.debug('Reuse the search tree. Number of nodes in search tree: %d',
                          self.root_node.nb_total_expanded_child_nodes)
            self.total_nodes_pre_search = deepcopy(self.root_node.n_sum)

            # reset potential good nodes for the root
            self.root_node.q[self.root_node.q < 1.1] = 0

        else:
            logging.debug("Starting a brand new search tree...")
            self.root_node = None
            self.total_nodes_pre_search = 0

        # check for fast way out
        if len(legal_moves) == 1:

            # if there's only a single legal move you only must go 1 depth
            max_depth_reached = 1

            if self.root_node is None:
                # conduct all necessary steps for fastest way out
                self._expand_root_node_single_move(state, legal_moves)
        else:

            if self.root_node is None:
                # run a single expansion on the root node
                self._expand_root_node_multiple_moves(state, legal_moves)

            # conduct the mcts-search based on the given settings
            max_depth_reached = self._run_mcts_search(state)

            t_elapsed = time() - self.t_start_eval
            print('info string move overhead is %dms' % (t_elapsed*1000 - self.movetime_ms))

        # receive the policy vector based on the MCTS search
        p_vec_small = self.root_node.get_mcts_policy(self.q_value_weight)

        if self.use_pruning is False:
            # store the current root in the lookup table
            self.node_lookup[key] = self.root_node

        # select the q-value according to the mcts best child value
        best_child_idx = p_vec_small.argmax()
        value = self.root_node.q[best_child_idx]

        lst_best_moves, _ = self.get_calculated_line()
        str_moves = self._mv_list_to_str(lst_best_moves)

        # show the best calculated line
        node_searched = int(self.root_node.n_sum - self.total_nodes_pre_search)
        # In uci the depth is given using half-moves notation also called plies
        time_e = time() - self.t_start_eval

        if len(legal_moves) != len(p_vec_small):
            raise Exception('Legal move list %s with length %s is uncompatible to policy vector %s'
                             ' with shape %s for board state %s and nodes legal move list: %s' %
                             (legal_moves, len(legal_moves),
                                p_vec_small, p_vec_small.shape, state,
                                self.root_node.legal_moves))

        # define the remaining return variables
        cp = value_to_centipawn(value)
        depth = max_depth_reached
        nodes = node_searched
        time_elapsed_s = time_e*1000
        nps = node_searched/time_e
        pv = str_moves

        return value, legal_moves, p_vec_small, cp, depth, nodes, time_elapsed_s, nps, pv

    def _expand_root_node_multiple_moves(self, state, legal_moves):
        """
        Checks if the current root node can be found in the look-up table.
        Otherwise run a single inference of the neural network for this board state

        :param state: Current game state
        :param legal_moves: Available moves
        :return:
        """

        # initialize is_leaf by default to false
        is_leaf = False

        # start a brand new tree
        state_planes = state.get_state_planes()
        [value, policy_vec] = self.nets[0].predict_single(state_planes)

        # extract a sparse policy vector with normalized probabilities
        p_vec_small = get_probs_of_move_list(policy_vec, legal_moves, state.is_white_to_move())

        if self.check_mate_in_one is True:
            str_legal_moves = str(state.get_legal_moves())
        else:
            str_legal_moves = ''

        # create a new root node
        self.root_node = Node(value, p_vec_small, legal_moves, str_legal_moves, is_leaf, clip_low_visit=False)

    def _expand_root_node_single_move(self, state, legal_moves):
        """
        Expands the current root in the case if there's only a single move available.
        The neural network search can be omitted in this case.

        :param state: Request games state
        :param legal_moves: Available moves
        :return:
        """

        # set value 0 as a dummy value
        value = 0
        p_vec_small = np.array([1], np.float32)

        # create a new root node
        self.root_node = Node(value, p_vec_small, legal_moves, str(state.get_legal_moves()))

        # check a child node if it doesn't exists already
        if self.root_node.child_nodes[0] is None:
            state_child = deepcopy(state)
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
            elif self.can_claim_threefold_repetition(state.get_transposition_key(), [0]) or\
                    state.get_pythonchess_board().can_claim_fifty_moves() is True:
                value = 0
                is_leaf = True
                legal_moves_child = []
                p_vec_small_child = None

            else:
                legal_moves_child = state_child.get_legal_moves()

                # start a brand new prediction for the child
                state_planes = state_child.get_state_planes()
                [value, policy_vec] = self.nets[0].predict_single(state_planes)

                # extract a sparse policy vector with normalized probabilities
                p_vec_small_child = get_probs_of_move_list(policy_vec, legal_moves_child,
                                                           state_child.is_white_to_move())

            # create a new child node
            child_node = Node(value, p_vec_small_child, legal_moves_child, str(state_child.get_legal_moves()),
                              is_leaf)

            # connect the child to the root
            self.root_node.child_nodes[0] = child_node

    def _run_mcts_search(self, state):
        """
        Runs a new or continues the mcts on the current search tree.

        :param state: Input state given by the user
        :return: max_depth_reached (int) - The longest search path length after the whole search
        """

        # clear the look up table
        self.node_lookup = {}

        # safe the prior policy of the root node
        self.root_node_prior_policy = deepcopy(self.root_node.p)

        # apply dirichlet noise to the prior probabilities in order to ensure
        #  that every move can possibly be visited
        self.root_node.apply_dirichlet_noise_to_prior_policy(epsilon=self.dirichlet_epsilon,
                                                             alpha=self.dirichlet_alpha)

        # iterate through all children and add dirichlet if there exists any
        for child_node in self.root_node.child_nodes:
            if child_node is not None:
                # test of adding dirichlet noise to a new node
                child_node.apply_dirichlet_noise_to_prior_policy(epsilon=self.dirichlet_epsilon * .1,
                                                               alpha=self.dirichlet_alpha)

        # store what depth has been reached at maximum in the current search tree
        # default is 1, in case only 1 move is available
        max_depth_reached = 1

        futures = []

        # set the number of playouts accordingly
        if state.are_pocket_empty() is True:
            nb_playouts = self.nb_playouts_empty_pockets
        else:
            nb_playouts = self.nb_playouts_filled_pockets
            self.temperature_current = 0

        t_elapsed = 0
        cur_playouts = 0
        old_time = time()

        cpuct_init = self.cpuct

        decline = True

        while max_depth_reached < self.max_search_depth and \
                cur_playouts < nb_playouts and \
                t_elapsed * 1000 < self.movetime_ms:  # and np.abs(self.root_node.q.mean()) < 0.99:

            if self.use_oscillating_cpuct is True:
                # Test about decreasing CPUCT value
                if decline is True:
                    self.cpuct -= 0.01
                else:
                    self.cpuct += 0.01
                if self.cpuct < cpuct_init * .5:
                    decline = False
                elif self.cpuct > cpuct_init:
                    decline = True

            # start searching
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                for i in range(self.threads):
                    # calculate the thread id based on the current playout
                    futures.append(executor.submit(self._run_single_playout, state=state,
                                                   parent_node=self.root_node, pipe_id=i, depth=1, chosen_nodes=[]))

            cur_playouts += self.threads
            time_show_info = time() - old_time

            for i, f in enumerate(futures):
                cur_value, cur_depth, chosen_nodes = f.result()

                if cur_depth > max_depth_reached:
                    max_depth_reached = cur_depth

                # Print the explored line of the last line for every x seconds if verbose is true
                if self.verbose and time_show_info > 0.5 and i == len(futures) - 1:
                    mv_list = self._create_mv_list(chosen_nodes)
                    str_moves = self._mv_list_to_str(mv_list)
                    print('info score cp %d depth %d nodes %d pv%s' % (
                        value_to_centipawn(cur_value), cur_depth, self.root_node.n_sum, str_moves))
                    logging.debug('Update info')
                    old_time = time()

            # update the current search time
            t_elapsed = time() - self.t_start_eval
            if self.verbose and time_show_info > 1:
                node_searched = int(self.root_node.n_sum - self.total_nodes_pre_search)
                print('info nps %d time %d' % (int((node_searched / t_elapsed)), t_elapsed * 1000))

        self.cpuct = cpuct_init

        return max_depth_reached

    def perform_action(self, state_in: GameState, verbose=True):
        """
        Return a value, best move with according to the mcts search.
        This method is used when using the mcts agent as a player.

        :param state: Requested games state
        :param verbose: Boolean if debug messages shall be shown
        :return: value - Board value prediction
                selected_move - Python chess move object according to mcts
                confidence - Confidence for selecting this move
                selected_child_idx - Child index which correspond to the selected child
        """

        # create a deepcopy of the state in order not to change the given input parameter
        state = deepcopy(state_in)

        return super().perform_action(state)

    def _run_single_playout(self, state: GameState, parent_node: Node, pipe_id=0, depth=1, chosen_nodes=[]):
        """
        This function works recursively until a leaf or terminal node is reached.
        It ends by backpropagating the value of the new expanded node or by propagating the value of a terminal state.

        :param state_: Current game-state for the evaluation. This state differs between the treads
        :param parent_node: Current parent-node of the selected node. In the first  expansion this is the root node.
        :param depth: Current depth for the evaluation. Depth is increased by 1 for every recusive call
        :param chosen_nodes: List of moves which have been taken in the current path. For each selected child node this list
                        is expanded by one move recursively.
        :param chosen_nodes: List of all nodes that this thread has explored with respect to the root node
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

        if depth == 1:
            state = GameState(deepcopy(state.get_pythonchess_board()))

        # apply the selected move on the board
        state.apply_move(move)

        # append the selected move to the move list
        # append the chosen child idx to the chosen_nodes list
        chosen_nodes.append(child_idx)

        if node is None:

            # get the transposition-key which is used as an identifier for the board positions in the look-up table
            transposition_key = state.get_transposition_key()

            # check if the addressed fen exist in the look-up table
            # note: It's important to use also the halfmove-counter here, otherwise the system can create an infinite
            # feed-back-loop
            key = (transposition_key, state.get_halfmove_counter())

            # expand and evaluate the new board state (the node wasn't found in the look-up table)
            # its value will be backpropagated through the tree and flipped after every layer
            # receive a free available pipe
            my_pipe = self.my_pipe_endings[pipe_id]

            if self.send_batches is True:
                my_pipe.send(state.get_state_planes())
                # this pipe waits for the predictions of the network inference service
                [value, policy_vec] = my_pipe.recv()
            else:
                state_planes = state.get_state_planes()
                self.batch_state_planes[pipe_id] = state_planes

                my_pipe.send(pipe_id)

                result_channel = my_pipe.recv()

                value = np.array(self.batch_value_results[result_channel])
                policy_vec = np.array(self.batch_policy_results[result_channel])

            # initialize is_leaf by default to false
            is_leaf = False


            # check if the current player has won the game
            # (we don't need to check for is_lost() because the game is already over
            #  if the current player checkmated his opponent)
            is_won = False
            is_check = False

            if state.is_check() is True:
                is_check = True
                if state.is_won() is True:
                    is_won = True

            if is_won is True:
                    value = -1
                    is_leaf = True
                    legal_moves = []
                    p_vec_small = None
                    # establish a mate in one connection in order to stop exploring different alternatives
                    parent_node.mate_child_idx = child_idx

            # get the value from the leaf node (the current function is called recursively)
            # check if you can claim a draw - its assumed that the draw is always claimed
            elif self.can_claim_threefold_repetition(transposition_key, chosen_nodes) or \
                    state.get_pythonchess_board().can_claim_fifty_moves() is True:
                value = 0
                is_leaf = True
                legal_moves = []
                p_vec_small = None
            else:
                # get the current legal move of its board state
                legal_moves = state.get_legal_moves()

                if len(legal_moves) < 1:
                    raise Exception('No legal move is available for state: %s' % state)

                # extract a sparse policy vector with normalized probabilities
                try:
                    p_vec_small = get_probs_of_move_list(policy_vec, legal_moves,
                                                         is_white_to_move=state.is_white_to_move(), normalize=True)

                except KeyError:
                    raise Exception('Key Error for state: %s' % state)

            # convert all legal moves to a string if the option check_mate_in_one was enabled
            if self.check_mate_in_one is True:
                str_legal_moves = str(state.get_legal_moves())
            else:
                str_legal_moves = ''

            # clip the visit nodes for all nodes in the search tree except the director opp. move
            clip_low_visit = self.use_pruning and depth != 1

            # create a new node
            new_node = Node(value, p_vec_small, legal_moves, str_legal_moves, is_leaf, transposition_key, clip_low_visit)

            if depth == 1:

                # disable uncertain moves from being visited by giving them a very bad score
                if is_leaf is False:
                    if self.root_node_prior_policy[child_idx] < 1e-3 and value*-1 < self.root_node.v:
                        with parent_node.lock:
                            value = 99

                if value < 0:  # and state.are_pocket_empty(): #and pipe_id == 0:
                    # test of adding dirichlet noise to a new node
                    new_node.apply_dirichlet_noise_to_prior_policy(epsilon=self.dirichlet_epsilon*.02,
                                                                   alpha=self.dirichlet_alpha)

            if self.use_pruning is False:
                # include a reference to the new node in the look-up table
                self.node_lookup[key] = new_node

            with parent_node.lock:
                # add the new node to its parent
                parent_node.child_nodes[child_idx] = new_node

        # check if we have reached a leaf node
        elif node.is_leaf is True:
            value = node.v

        else:
            # get the value from the leaf node (the current function is called recursively)
            value, depth, chosen_nodes = self._run_single_playout(state, node, pipe_id, depth + 1, chosen_nodes)

        # revert the virtual loss and apply the predicted value by the network to the node
        parent_node.revert_virtual_loss_and_update(child_idx, self.virtual_loss, -value)

        # we invert the value prediction for the parent of the above node layer because the player's turn is flipped every turn
        return -value, depth, chosen_nodes

    def can_claim_threefold_repetition(self, transposition_key, chosen_nodes):

        search_occurrence_counter = 0

        node = self.root_node.child_nodes[chosen_nodes[0]]

        for node_idx in chosen_nodes[1:-1]:
            if node.transposition_key == transposition_key:
                search_occurrence_counter += 1
            node = node.child_nodes[node_idx]
            if node is None:
                break

        return self.transposition_table[transposition_key] + search_occurrence_counter >= 2

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

    def _select_node_based_on_mcts_policy(self, parent_node: Node, clip_bad_visit_nodes=False):
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
            clip_bad_visit_nodes = node == self.root_node  # and mv_number > 1
            node, move, nb_visits = self._select_node_based_on_mcts_policy(node, clip_bad_visit_nodes=clip_bad_visit_nodes)
            lst_best_moves.append(move)
            lst_nb_visits.append(nb_visits)
        return lst_best_moves, lst_nb_visits

    def _mv_list_to_str(self, lst_moves):
        """
        Converts a given list of chess moves to a single string seperated by spaces.
        :param lst_moves: List chess.Moves objects
        :return: String representing each move in the list
        """
        str_moves = lst_moves[0].uci()

        for mv in lst_moves[1:]:
            str_moves += " " + mv.uci()

        return str_moves

    def _create_mv_list(self, lst_chosen_nodes: [int]):
        """
        Creates a movement list given the child node indices from the root node onwards.
        :param lst_chosen_nodes: List of chosen nodes
        :return: mv_list - List of python chess moves
        """
        #str_moves = ""
        mv_list = []
        node = self.root_node #.child_nodes[lst_chosen_nodes[0]]

        for child_idx in lst_chosen_nodes:
            mv = node.legal_moves[child_idx]
            node = node.child_nodes[child_idx]
            mv_list.append(mv)
        return mv_list

    def update_movetime(self, time_ms_per_move):
        """
        Update move time allocation.
        :param time_ms_per_move:  Sets self.movetime_ms to this value
        :return:
        """
        self.movetime_ms = time_ms_per_move
