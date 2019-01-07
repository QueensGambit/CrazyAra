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
import collections
import cProfile
import io
import logging
import math
import pstats
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from multiprocessing import Pipe
from time import time
import numpy as np
from DeepCrazyhouse.src.domain.agent.NeuralNetAPI import NeuralNetAPI
from DeepCrazyhouse.src.domain.agent.player._Agent import _Agent
from DeepCrazyhouse.src.domain.agent.player.util.NetPredService import NetPredService
from DeepCrazyhouse.src.domain.agent.player.util.Node import Node
from DeepCrazyhouse.src.domain.crazyhouse.constants import BOARD_HEIGHT, BOARD_WIDTH, NB_CHANNELS_FULL, NB_LABELS
from DeepCrazyhouse.src.domain.crazyhouse.GameState import GameState
from DeepCrazyhouse.src.domain.crazyhouse.output_representation import get_probs_of_move_list, value_to_centipawn


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
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


class MCTSAgent(_Agent):
    def __init__(
        self,
        nets: [NeuralNetAPI],
        threads=16,
        batch_size=8,
        playouts_empty_pockets=256,
        playouts_filled_pockets=512,
        cpuct=1,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.2,
        max_search_depth=15,
        temperature=0.0,
        temperature_moves=4,
        q_value_weight=0.0,
        virtual_loss=3,
        verbose=True,
        min_movetime=100,
        check_mate_in_one=False,
        use_pruning=True,
        use_oscillating_cpuct=True,
        use_time_management=True,
        opening_guard_moves=0,
    ):
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
        :param use_time_management: If set to true the mcts will spent less time on "obvious" moves an allocate a time
                                    buffer for more critical moves.
        :param opening_guard_moves: Number of moves for which the exploration is limited (only recommended for . Moves which have a prior
                                    probability < 5% are clipped and not evaluated. If 0 no clipping will be done in the
                                    opening.
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
            raise Exception(
                "You might have confused centi-cpuct with cpuct."
                "The requested cpuct is beyond reasonable range: cpuct should be around > 0.01 and < 10."
            )

        self.cpuct = cpuct
        self.max_search_depth = max_search_depth
        self.threads = threads

        # check for possible issues when giving an illegal batch_size and number of threads combination
        if batch_size > threads:
            raise Exception(
                "info string The given batch_size %d is higher than the number of threads %d. "
                "The maximum legal batch_size is the same as the number of threads (here: %d) "
                % (batch_size, threads, threads)
            )

        if threads % batch_size != 0:
            raise Exception(
                "You requested an illegal combination of threads %d and batch_size %d."
                " The batch_size must be a divisor of the number of threads" % (threads, batch_size)
            )

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
            net_pred_service = NetPredService(
                pip_endings_external[i * nb_pipes : (i + 1) * nb_pipes],
                net,
                batch_size,
                self.batch_state_planes,
                self.batch_value_results,
                self.batch_policy_results,
            )
            self.net_pred_services.append(net_pred_service)

        self.transposition_table = collections.Counter()
        self.send_batches = False
        self.root_node_prior_policy = None

        self.use_pruning = use_pruning
        self.use_oscillating_cpuct = use_oscillating_cpuct
        self.time_buffer_ms = 0
        self.use_time_management = use_time_management
        self.opening_guard_moves = opening_guard_moves

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
        if not legal_moves:
            raise Exception("The given board state has no legal move available")

        # check first if the the current tree can be reused
        key = state.get_transposition_key() + (state.get_fullmove_number(),)

        if self.use_pruning is False and key in self.node_lookup:
            # if key in self.node_lookup:

            self.root_node = self.node_lookup[key]

            logging.debug(
                "Reuse the search tree. Number of nodes in search tree: %d",
                self.root_node.nb_total_expanded_child_nodes,
            )
            self.total_nodes_pre_search = deepcopy(self.root_node.n_sum)

            # reset potential good nodes for the root
            # self.root_node.q[self.root_node.q < 1.1] = 0
            self.root_node.q[self.root_node.q < 0] = self.root_node.q.max() - 0.25

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

            # opening guard
            if state.get_fullmove_number() <= self.opening_guard_moves:  # 100: #7: #10:
                self.root_node.q[self.root_node.p < 5e-2] = -9999
            # elif len(legal_moves) > 50:
            #    self.root_node.q[self.root_node.p < 1e-3] = -9999

            # conduct the mcts-search based on the given settings
            max_depth_reached = self._run_mcts_search(state)

            t_elapsed = time() - self.t_start_eval
            print("info string move overhead is %dms" % (t_elapsed * 1000 - self.movetime_ms))

        # xth_n_max = self.get_xth_max(10)
        # print('xth_n-max: ', xth_n_max)
        # receive the policy vector based on the MCTS search
        p_vec_small = self.root_node.get_mcts_policy(self.q_value_weight)  # , xth_n_max=xth_n_max, is_root=True)

        # experimental
        """
        orig_q = np.array(self.root_node.q)
        #indices = self.root_node.n.max() > clip_fac
        candidate_child = p_vec_small.argmax() #self.get_2nd_max()
        latest, indices = self.get_last_q_values(candidate_child)


        # ensure that the q value for the end node are properly set
        #if len(indices) > 0:
        #self.root_node.w[indices] += (self.root_node.n[indices]/1) * latest[indices]
        #self.root_node.q[indices] = self.root_node.w[indices] / (self.root_node.n[indices] + (self.root_node.n[indices]/1))
        if True: #self.root_node.q[candidate_child] < 0: # and latest[candidate_child] + 0.5 < self.root_node.q[candidate_child]:
            #self.root_node.q[candidate_child] = (self.root_node.q[candidate_child] + latest[candidate_child])
            #self.root_node.q[latest[self.root_node.thresh_idcs_root] < self.root_node.q[self.root_node.thresh_idcs_root]] = -1
            #print('q - shape', self.root_node.q.shape)
            #print('latest  - shape', latest.shape)
            #print('thresh - shape', self.root_node.thresh_idcs_root.shape)
            sel_indices = latest < self.root_node.q
            sel_indices[np.invert(self.root_node.thresh_idcs_root)] = False


            #print('sel indices -shape', len(sel_indices))
            self.root_node.q[sel_indices] = (latest[sel_indices] + self.root_node.q[sel_indices]) / 2 #-1

            sel_indices = np.invert(sel_indices)
            sel_indices[self.root_node.thresh_idcs_root] = False
            self.root_node.q[sel_indices] = (latest[sel_indices] + self.root_node.q[sel_indices]) / 2

            #prior_child = self.root_node.p.argmax()
            #if latest[prior_child] > self.root_node.q[prior_child]:
            #    self.root_node.q[prior_child] = (latest[prior_child] + self.root_node.q[prior_child]) / 2 #latest[prior_child]

            #self.root_node.q[indices] = self.root_node.q[indices] * (latest[indices] + 1)
            #self.root_node.q[indices] += latest[indices]

            # receive the policy vector based on the MCTS search
            p_vec_small = self.root_node.get_mcts_policy(self.q_value_weight) #, xth_n_max=xth_n_max, is_root=True)
        """

        # max_n = self.root_node.n.max()
        # latest[self.root_node.n < max_n / 2] = -1
        # latest += 1
        # latest /= sum(latest)
        # if latest.max() > 0:
        #    p_vec_small[latest < 0] = 0

        # p_vec_small = p_vec_small + latest
        # p_vec_small[p_vec_small < 0] = 0
        # p_vec_small[p_vec_small > 1] = 1

        # p_vec_small /= sum(p_vec_small)

        # if self.use_pruning is False:
        # store the current root in the lookup table
        self.node_lookup[key] = self.root_node

        # select the q-value according to the mcts best child value
        best_child_idx = p_vec_small.argmax()
        value = self.root_node.q[best_child_idx]
        # value = orig_q[best_child_idx]

        lst_best_moves, _ = self.get_calculated_line()
        str_moves = self._mv_list_to_str(lst_best_moves)

        # show the best calculated line
        node_searched = int(self.root_node.n_sum - self.total_nodes_pre_search)
        # In uci the depth is given using half-moves notation also called plies
        time_e = time() - self.t_start_eval

        if len(legal_moves) != len(p_vec_small):
            raise Exception(
                "Legal move list %s with length %s is uncompatible to policy vector %s"
                " with shape %s for board state %s and nodes legal move list: %s"
                % (legal_moves, len(legal_moves), p_vec_small, p_vec_small.shape, state, self.root_node.legal_moves)
            )

        # define the remaining return variables
        cp = value_to_centipawn(value)
        depth = max_depth_reached
        nodes = node_searched
        time_elapsed_s = time_e * 1000
        nps = node_searched / time_e
        pv = str_moves

        # print out the score as a debug message if verbose it set to true
        # the file crazyara.py will print the chosen line to the std output
        if self.verbose is True:
            score = "score cp %d depth %d nodes %d time %d nps %d pv %s" % (cp, depth, nodes, time_elapsed_s, nps, pv)
            logging.info("info string %s" % score)

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
            str_legal_moves = ""

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

        # request the value prediction for the current position
        state_planes = state.get_state_planes()
        [value, _] = self.nets[0].predict_single(state_planes)
        # we can create the move probability vector without the NN this time
        p_vec_small = np.array([1], np.float32)

        # create a new root node
        self.root_node = Node(value, p_vec_small, legal_moves, str(state.get_legal_moves()), clip_low_visit=False)

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
            elif (
                self.can_claim_threefold_repetition(state.get_transposition_key(), [0])
                or state.get_pythonchess_board().can_claim_fifty_moves() is True
            ):
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
                p_vec_small_child = get_probs_of_move_list(
                    policy_vec, legal_moves_child, state_child.is_white_to_move()
                )

            # create a new child node
            child_node = Node(value, p_vec_small_child, legal_moves_child, str(state_child.get_legal_moves()), is_leaf)

            # connect the child to the root
            self.root_node.child_nodes[0] = child_node

            # assign the value of the root node as the q-value for the child
            # here we must invert the invert the value because it's the value prediction of the next state
            self.root_node.q[0] = -value

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
        self.root_node.apply_dirichlet_noise_to_prior_policy(epsilon=self.dirichlet_epsilon, alpha=self.dirichlet_alpha)

        # store what depth has been reached at maximum in the current search tree
        # default is 1, in case only 1 move is available
        max_depth_reached = 1

        futures = []

        # set the number of playouts accordingly
        if state.are_pocket_empty() is True:
            nb_playouts = self.nb_playouts_empty_pockets
        else:
            nb_playouts = self.nb_playouts_filled_pockets
            # self.temperature_current = 0

        # if self.root_node.v > 0.65:
        #    fac = 0.1
        # else:
        #    fac = 0.02 #2
        # iterate through all children and add dirichlet if there exists any
        for child_node in self.root_node.child_nodes:
            if child_node is not None:
                # add dirichlet noise to a the child nodes of the root node
                child_node.apply_dirichlet_noise_to_prior_policy(
                    epsilon=self.dirichlet_epsilon * 0.05, alpha=self.dirichlet_alpha  # 02,
                )
                # child_node.q[child_node.q < 0] = child_node.q.max() - 0.25

        t_elapsed_ms = 0

        cur_playouts = 0
        old_time = time()

        cpuct_init = self.cpuct

        decline = True

        move_step = self.movetime_ms / 2
        move_update = move_step
        move_update_2 = self.movetime_ms * 0.9

        # nb_playouts_update_step = 4000
        # nb_playouts_update = 4000

        # self.hard_clipping = True

        if self.use_time_management is True:
            time_checked = False
            time_checked_early = False
        else:
            time_checked = True
            time_checked_early = True

        consistent_check = False  # False
        consistent_check_playouts = 2048

        while (
            max_depth_reached < self.max_search_depth and cur_playouts < nb_playouts and t_elapsed_ms < self.movetime_ms
        ):  # and np.abs(self.root_node.q.mean()) < 0.99:

            if self.use_oscillating_cpuct is True:
                # Test about decreasing CPUCT value
                if decline is True:
                    self.cpuct -= 0.01
                else:
                    self.cpuct += 0.01
                if self.cpuct < cpuct_init * 0.5:
                    decline = False
                elif self.cpuct > cpuct_init:
                    decline = True

            """
            if cur_playouts >= nb_playouts_update:
                #print('UPDATE')
                self.root_node.apply_dirichlet_noise_to_prior_policy(epsilon=self.dirichlet_epsilon,
                                                                     alpha=self.dirichlet_alpha)
                # iterate through all children and add dirichlet if there exists any
                for child_node in self.root_node.child_nodes:
                    if child_node is not None:
                        # test of adding dirichlet noise to a new node
                        child_node.apply_dirichlet_noise_to_prior_policy(epsilon=self.dirichlet_epsilon * fac,
                                                                         alpha=self.dirichlet_alpha)
                nb_playouts_update += nb_playouts_update_step
                #move_update += move_step
            """
            # start searching

            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                for i in range(self.threads):
                    # calculate the thread id based on the current playout
                    futures.append(
                        executor.submit(
                            self._run_single_playout,
                            state=state,
                            parent_node=self.root_node,
                            pipe_id=i,
                            depth=1,
                            chosen_nodes=[],
                        )
                    )

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
                    print(
                        "info score cp %d depth %d nodes %d pv %s"
                        % (value_to_centipawn(cur_value), cur_depth, self.root_node.n_sum, str_moves)
                    )
                    logging.debug("Update info")
                    old_time = time()

            # update the current search time
            t_elapsed = time() - self.t_start_eval
            t_elapsed_ms = t_elapsed * 1000
            if time_show_info > 1:
                node_searched = int(self.root_node.n_sum - self.total_nodes_pre_search)
                print("info nps %d time %d" % (int((node_searched / t_elapsed)), t_elapsed_ms))

            if time_checked_early is False and t_elapsed_ms > move_update:
                # node, _, _, child_idx = self._select_node_based_on_mcts_policy(self.root_node)
                if self.root_node.p.max() > 0.9 and self.root_node.p.argmax() == self.root_node.q.argmax():
                    self.time_buffer_ms += (self.movetime_ms - t_elapsed_ms) * 0.9
                    print("info early break up")
                    break
                else:
                    time_checked_early = True

            if (
                consistent_check is False
                and cur_playouts > consistent_check_playouts
                and self.root_node_prior_policy.max()
                > np.partition(self.root_node_prior_policy.flatten(), -2)[-2] + 0.3
            ):
                print("Consistency check")
                if self.root_node.get_mcts_policy(self.q_value_weight).argmax() == self.root_node_prior_policy.argmax():
                    self.time_buffer_ms += (self.movetime_ms - t_elapsed_ms) * 0.9
                    print("info early break up")
                    break
                else:
                    consistent_check = True

            if (
                self.time_buffer_ms > 2500
                and time_checked is False
                and t_elapsed_ms > move_update_2
                and self.root_node.q[self.root_node.n.argmax()] < self.root_node.v + 0.01
            ):
                print("info increase time")
                time_checked = True
                # for child_node in self.root_node.child_nodes:
                #    if child_node is not None:
                #        # test of adding dirichlet noise to a new node
                #        child_node.apply_dirichlet_noise_to_prior_policy(epsilon=self.dirichlet_epsilon * .5,
                #                                                         alpha=self.dirichlet_alpha)
                time_boni = self.time_buffer_ms / 4
                # increase the movetime
                self.time_buffer_ms -= time_boni
                self.movetime_ms += (time_boni) * 0.75
                self.root_node.v = self.root_node.q[self.root_node.n.argmax()]
                if self.time_buffer_ms < 0:
                    self.movetime_ms += self.time_buffer_ms
                    self.time_buffer_ms = 0
                    # if self.root_node.q[child_idx] < 0:
                    #    self.hard_clipping = False
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
            key = transposition_key + (state.get_fullmove_number(),)
            use_tran_table = True

            node_varified = False
            if use_tran_table is True and key in self.node_lookup:
                # if self.check_for_duplicate(transposition_key, chosen_nodes) is False:
                # get the node from the look-up list
                node = self.node_lookup[key]

                # make sure that you don't connect to a node with lower visits
                if node.n_sum > parent_node.n_sum:
                    node_varified = True

            if node_varified is True:

                with parent_node.lock:
                    # setup a new connection from the parent to the child
                    parent_node.child_nodes[child_idx] = node

                # logging.debug('found key: %s' % state.get_board_fen())
                # get the prior value from the leaf node which has already been expanded
                value = node.v

                # receive a free available pipe
                # my_pipe = self.my_pipe_endings[pipe_id]
                # my_pipe.send(state.get_state_planes())
                # this pipe waits for the predictions of the network inference service
                # [_, _] = my_pipe.recv()

                # get the value from the leaf node (the current function is called recursively)
                # value, depth, mv_list = self._run_single_playout(state, node, pipe_id, depth+1, mv_list)
            else:
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
                # is_check = False

                if state.is_check() is True:
                    # enhance checking nodes
                    # if depth == 1:
                    #    parent_node.p.mean()
                    #    with parent_node.lock:
                    #        if parent_node.p[child_idx] < 0.1:
                    #            parent_node.p[child_idx] = 0.1
                    # is_check = True
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
                elif (
                    self.can_claim_threefold_repetition(transposition_key, chosen_nodes)
                    or state.get_pythonchess_board().can_claim_fifty_moves() is True
                ):
                    # raise Exception('Threefold!')
                    value = 0
                    is_leaf = True
                    legal_moves = []
                    p_vec_small = None
                else:
                    # get the current legal move of its board state
                    legal_moves = state.get_legal_moves()

                    if not legal_moves:
                        raise Exception("No legal move is available for state: %s" % state)

                    # extract a sparse policy vector with normalized probabilities
                    try:
                        p_vec_small = get_probs_of_move_list(
                            policy_vec, legal_moves, is_white_to_move=state.is_white_to_move(), normalize=True
                        )

                    except KeyError:
                        raise Exception("Key Error for state: %s" % state)

                # if state.get_board_fen() == 'r1b3k1/ppq2pP1/3n1Ppp/4Q2N/4B3/P1P3bP/2P1nPPr/4rB1K/PRPNp w - - 0 36':
                #    print('found it! > is won %d' % is_won)

                # convert all legal moves to a string if the option check_mate_in_one was enabled
                if self.check_mate_in_one is True:
                    str_legal_moves = str(state.get_legal_moves())
                else:
                    str_legal_moves = ""

                # clip the visit nodes for all nodes in the search tree except the director opp. move
                clip_low_visit = self.use_pruning and depth != 1

                # create a new node
                new_node = Node(
                    value, p_vec_small, legal_moves, str_legal_moves, is_leaf, transposition_key, clip_low_visit
                )

                if depth == 1:

                    # disable uncertain moves from being visited by giving them a very bad score
                    if is_leaf is False and self.use_pruning is True:
                        if self.root_node_prior_policy[child_idx] < 1e-3 and value * -1 < self.root_node.v:
                            with parent_node.lock:
                                value = 99

                    if parent_node.v > 0.65:  # and state.are_pocket_empty(): #and pipe_id == 0:
                        # test of adding dirichlet noise to a new node
                        fac = 0.25
                        if len(parent_node.legal_moves) < 20:
                            fac *= 5
                        new_node.apply_dirichlet_noise_to_prior_policy(
                            epsilon=self.dirichlet_epsilon * fac, alpha=self.dirichlet_alpha
                        )

                    if value < 0:  # and state.are_pocket_empty(): #and pipe_id == 0:
                        # test of adding dirichlet noise to a new node
                        new_node.apply_dirichlet_noise_to_prior_policy(
                            epsilon=self.dirichlet_epsilon * 0.02, alpha=self.dirichlet_alpha
                        )

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

    def check_for_duplicate(self, transposition_key, chosen_nodes):

        node = self.root_node.child_nodes[chosen_nodes[0]]

        # iterate over all accessed nodes during the current search of the thread and check for same transposition key
        for node_idx in chosen_nodes[1:-1]:
            if node.transposition_key == transposition_key:
                # print('DUPLICATE CHECK = TRUE! ')
                return True
            node = node.child_nodes[node_idx]
            if node is None:
                break

        return False

    def can_claim_threefold_repetition(self, transposition_key, chosen_nodes):
        """
        Checks if a three fold repetition event can be claimed in the current search path.
        This method makes use of the class transposition table and checks for board occurrences in the local search path
        of the current thread as well.

        :param transposition_key: Transposition key which defines the board state by all it's pieces and pocket state.
                                  The move counter is disregarded.
        :param chosen_nodes: List of integer indices which correspond to the child node indices chosen from the
                            root node downwards.
        :return: True, if threefold repetition can be claimed, else False
        """

        # set the number of occurrences by default to 0
        search_occurrence_counter = 0

        node = self.root_node.child_nodes[chosen_nodes[0]]

        # iterate over all accessed nodes during the current search of the thread and check for same transposition key
        for node_idx in chosen_nodes[1:-1]:
            if node.transposition_key == transposition_key:
                search_occurrence_counter += 1
            node = node.child_nodes[node_idx]
            if node is None:
                break

        # use all occurrences in the class transposition table as well as the locally found equalities
        return (self.transposition_table[transposition_key] + search_occurrence_counter) >= 2

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

            if self.use_oscillating_cpuct is False:
                pb_c_base = 19652
                pb_c_init = self.cpuct

                cpuct = math.log((parent_node.n_sum + pb_c_base + 1) / pb_c_base) + pb_c_init
            else:
                cpuct = self.cpuct

            # calculate the current u values
            # it's not worth to save the u values as a node attribute because u is updated every time n_sum changes
            u = cpuct * parent_node.p * (np.sqrt(parent_node.n_sum) / (1 + parent_node.n))

            # if depth == 1 and self.hard_clipping is True and self.use_pruning is True: # and id <= (self.threads//2+1): #and id % 2 != 0:
            #    u[parent_node.thresh_idcs_root] = -9999 #1
            # if self.use_pruning is True and depth >= 2:  # and depth >= 2:  and id % 3 != 0:
            #    u[parent_node.thresh_idcs] = -9999

            child_idx = (parent_node.q + u).argmax()

        node = parent_node.child_nodes[child_idx]
        # now receive the according move based on the calculated move index
        move = parent_node.legal_moves[child_idx]

        return node, move, child_idx

    def _select_node_based_on_mcts_policy(self, parent_node: Node, is_root=False):
        """
        Selects the next node based on the mcts policy which is used to predict the final best move.

        :param parent_node: Node from which to select the next child.
        :return:
        """

        child_idx = parent_node.get_mcts_policy(self.q_value_weight, is_root=is_root).argmax()

        nb_visits = parent_node.n[child_idx]
        move = parent_node.legal_moves[child_idx]

        return parent_node.child_nodes[child_idx], move, nb_visits, child_idx

    def show_next_pred_line(self):
        best_moves = []
        # start at the root node
        node = self.root_node

        while node is not None:
            # go deep through the tree by always selecting the best move for both players
            node, move, child_idx = self._select_node(node)
            best_moves.append(move)
        return best_moves

    def get_2nd_max(self):
        n_child = self.root_node.n.argmax()
        n_max = self.root_node.n[n_child]
        self.root_node.n[n_child] = 0

        second_max = self.root_node.n.max()
        self.root_node.n[n_child] = n_max

        return second_max

    def get_xth_max(self, xth_node):
        if len(self.root_node.n) < xth_node:
            return self.root_node.n.min()
        else:
            return np.sort(self.root_node.n)[-xth_node]

    def get_last_q_values(self, second_max=0, clip_fac=0.25):
        """
        Returns the values of the last node in the caluclated lines according to the mcts search for the most
         visited nodes
        :return:
        """

        q_future = np.zeros(self.root_node.nb_direct_child_nodes)

        indices = []
        for i in range(self.root_node.nb_direct_child_nodes):
            if self.root_node.n[i] >= self.root_node.n.max() * 0.33:  # i == second_max: # #second_max:
                node = self.root_node.child_nodes[i]
                print(self.root_node.legal_moves[i].uci(), end=" ")
                turn = 1
                final_node = node
                move = self.root_node.legal_moves[i]

                while node is not None and node.is_leaf is False and node.n_sum > 3:
                    final_node = node
                    print(move.uci() + " ", end="")
                    node, move, _, _ = self._select_node_based_on_mcts_policy(node)
                    turn *= -1

                if final_node is not None:
                    q_future[i] = final_node.v
                    indices.append(i)
                    q_future[i] *= turn
                print(q_future[i])

        return q_future, indices

    def get_calculated_line(self):
        """
        Prints out the best search line estimated for both players on the given board state.
        :return:
        """

        if self.root_node is None:
            logging.warning("You must run an evaluation first in order to get the calculated line")

        lst_best_moves = []
        lst_nb_visits = []
        # start at the root node
        node = self.root_node
        is_root = True

        while node is not None and node.is_leaf is False:
            # go deep through the tree by always selecting the best move for both players
            node, move, nb_visits, _ = self._select_node_based_on_mcts_policy(node, is_root)
            is_root = False
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
        mv_list = []
        node = self.root_node

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

    def set_max_search_depth(self, max_search_depth: int):
        """
        Assigns a new maximum search depth for the next search
        :param max_search_depth: Specifier of the search depth
        :return:
        """
        self.max_search_depth = max_search_depth

    def update_tranposition_table(self, transposition_key):
        """

        :param transposition_key: (gamestate.get_transposition_key(),)
        :return:
        """

        self.transposition_table.update(transposition_key)
