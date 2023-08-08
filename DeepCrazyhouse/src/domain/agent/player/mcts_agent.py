"""
@file: MCTSAgent.py
Created on 10.10.18
@project: crazy_ara_refactor
@author: queensgambit

The MCTSAgent runs playouts/simulations in the search tree and updates the node statistics.
The final move is chosen according to the visit count of each direct child node.
One playout is defined as expanding one new node in the tree.
In the case of chess this means evaluating a new board position.
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

from DeepCrazyhouse.src.domain.agent.neural_net_api import NeuralNetAPI
from DeepCrazyhouse.src.domain.abstract_cls.abs_agent import AbsAgent
from DeepCrazyhouse.src.domain.agent.player.util.net_pred_service import NetPredService
from DeepCrazyhouse.src.domain.agent.player.util.node import Node
from DeepCrazyhouse.src.domain.variants.constants import BOARD_HEIGHT, BOARD_WIDTH, NB_CHANNELS_TOTAL, NB_LABELS
from DeepCrazyhouse.src.domain.variants.game_state import GameState
from DeepCrazyhouse.src.domain.variants.output_representation import get_probs_of_move_list, value_to_centipawn
from DeepCrazyhouse.src.domain.util import get_check_move_mask

DTYPE = np.float


def profile(fnc):
    """
    A decorator that uses cProfile to profile a function
    :param fnc: Function handle to decorate.
    :return:
    """

    def inner(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        retval = fnc(*args, **kwargs)
        profiler.disable()
        string_buffer = io.StringIO()
        profile_stats = pstats.Stats(profiler, stream=string_buffer).sort_stats("cumulative")
        profile_stats.print_stats()
        print(string_buffer.getvalue())
        return retval

    return inner


class MCTSAgent(AbsAgent):  # Too many instance attributes (31/7)
    """This class runs simulations in the tree and updates the node statistics smartly"""

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
        enhance_checks=False,
        enhance_captures=False,
        use_future_q_values=False,
        use_pruning=True,
        use_time_management=True,
        use_transposition_table=True,
        opening_guard_moves=0,
        u_init_divisor=1,
    ):  # Too many arguments (21/5) - Too many local variables (29/15)
        """
        Constructor of the MCTSAgent.
        :param nets: NeuralNetAPI handle which is used to communicate with the neural network
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
                     The play style depends strongly on this value.
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
        :param temperature_moves: Number of full moves in which the temperature parameter will be applied.
                                  Otherwise the temperature will be set to 0 for deterministic play.
        :param: q_value_weight: Float indicating how the number of visits and the q-values should be mixed.
                                Expected to be in range [0.,1.]
        :param virtual_loss: An artificial loss term which is applied to each node which is currently being visited.
                             This term make it look like that the current visit of this node led to +X losses where X
                             is the virtual loss. This prevents that every thread will evaluate the same node.
        :param verbose: Defines weather to print out info messages for the current calculated line
        :param min_movetime: Minimum time in milliseconds to search for the best move
        :param enhance_checks: Decide whether to increase the probability for checking moves below 10% by 10%.
                               This lowers the chance of missing forced mates and possible direct mate threats.
                               Currently it is only applied to the root node and its direct child node due to runtime
                               costs.
        :param enhance_captures: Decide whether to increase the probability for capture moves below 10% by 5%.
                               This lowers the chance of missing captures.
                               Currently it is only applied to the root node and its direct child node due to runtime
                               costs.
        :param use_time_management: If set to true the mcts will spent less time on "obvious" moves an allocate a time
                                    buffer for more critical moves.
        :param use_transposition_table: Stores a transposition table for all nodes to modify the tree structure for
                                        transpositions. Enables reaching higher depth with same number of nodes.
        :param opening_guard_moves: Number of moves for which the exploration is limited
                                    (only recommended for . Moves which have a prior probability < 5%)
                                    are clipped and not evaluated.
                                    If 0 no clipping will be done in the
                                    opening.
        :param use_future_q_values: If set True, the q-values of the most visited child nodes will be updated by taking
                                    the minimum of both the current and future q-values.
        :param u_init_divisor: Division factor for calculating the u-value in select_node(). Default value is 1.0 to
                                avoid division by 0. Values smaller 1.0 increases the chance of exploring each node at
                                least once. This value must be greater 0.
        """

        super().__init__(temperature, temperature_moves, verbose)
        self.root_node = None  # the root node contains all references to its child nodes
        self.max_depth = 10  # stores the links for all nodes
        self.node_lookup = {}  # stores a lookup for all possible board states after the opposite player played its move
        self.nets = nets  # get the network reference
        self.virtual_loss = virtual_loss

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
        self.my_pipe_endings = []  # create pip endings for itself and the prediction service
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
        self.enhance_checks = enhance_checks
        self.enhance_captures = enhance_captures

        # temporary variables
        # time counter - n° of nodes stored to measure the nps - priority policy for the root node
        self.t_start_eval = self.total_nodes_pre_search = self.root_node_prior_policy = None
        # allocate shared memory for communicating with the network prediction service
        self.batch_state_planes = np.zeros((self.threads, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH), DTYPE)
        self.batch_value_results = np.zeros(self.threads, DTYPE)
        self.batch_policy_results = np.zeros((self.threads, NB_LABELS), DTYPE)
        # initialize the NetworkPredictionService and give the pointers to the shared memory
        self.net_pred_services = []
        nb_pipes = self.threads // len(nets)

        for i, net in enumerate(nets):  # create multiple gpu-access points
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
        self.use_pruning = use_pruning
        self.time_buffer_ms = 0
        self.use_time_management = use_time_management
        if self.use_pruning:  # pruning is incompatible with transposition usage
            self.use_transposition_table = False
        else:
            self.use_transposition_table = use_transposition_table
        self.opening_guard_moves = opening_guard_moves
        self.use_future_q_values = use_future_q_values
        if u_init_divisor <= 0 or u_init_divisor > 1:
            raise Exception("The value for the u-value initial divisor must be in (0,1]")
        self.u_init_divisor = u_init_divisor

    def evaluate_board_state(self, state: GameState):  # Probably is better to be refactored
        """
        Analyzes the current board state. This is the main method which get called by the uci interface or analysis
        request.
        :param state: Actual game state to evaluate for the MCTS
        :return:
        """
        # Too many local variables (28/15) - Too many branches (25/12) - Too many statements (75/50)
        self.t_start_eval = time()  # store the time at which the search started

        if not self.net_pred_services[0].running:  # check if the net prediction service has already been started
            for net_pred_service in self.net_pred_services:  # start the prediction daemon thread
                net_pred_service.start()

        legal_moves = state.get_legal_moves()  # list of all possible legal move in the current board position

        if not legal_moves:  # consistency check
            raise Exception("The given board state has no legal move available")

        key = state.get_transposition_key() + (
            state.get_fullmove_number(),
        )  # check first if the the current tree can be reused

        if not self.use_pruning and key in self.node_lookup:
            chess_board = state.get_pythonchess_board()
            self.root_node = self.node_lookup[key]  # if key in self.node_lookup:
            if self.enhance_captures:
                self._enhance_captures(chess_board, legal_moves, self.root_node.policy_prob)
                # enhance checks for all direct child nodes
                for child_node in self.root_node.child_nodes:
                    if child_node:
                        self._enhance_captures(child_node.board, child_node.legal_moves, child_node.policy_prob)

            if self.enhance_checks:
                self._enhance_checks(chess_board, legal_moves, self.root_node.policy_prob)
                # enhance checks for all direct child nodes
                for child_node in self.root_node.child_nodes:
                    if child_node:
                        self._enhance_checks(child_node.board, child_node.legal_moves, child_node.policy_prob)

            logging.debug(
                "Reuse the search tree. Number of nodes in search tree: %d",
                self.root_node.nb_total_expanded_child_nodes,
            )
            self.total_nodes_pre_search = deepcopy(self.root_node.n_sum)
        else:
            logging.debug("Starting a brand new search tree...")
            self.root_node = None
            self.total_nodes_pre_search = 0

        if len(legal_moves) == 1:  # check for fast way out
            max_depth_reached = 1  # if there's only a single legal move you only must go 1 depth

            if self.root_node is None:
                # conduct all necessary steps for fastest way out
                self._expand_root_node_single_move(state, legal_moves)

            # increase the move time buffer
            # subtract half a second as a constant for possible delay
            self.time_buffer_ms += max(self.movetime_ms - 500, 0)
        else:
            if self.root_node is None:
                self._expand_root_node_multiple_moves(state, legal_moves)  # run a single expansion on the root node
            # opening guard
            if state.get_fullmove_number() <= self.opening_guard_moves:  # 100: #7: #10:
                self.root_node.q_value[self.root_node.policy_prob < 5e-2] = -9999
            # elif len(legal_moves) > 50:
            #    self.root_node.q_value[self.root_node.policy_prob < 1e-3] = -9999
            # conduct the mcts-search based on the given settings
            max_depth_reached = self._run_mcts_search(state)
            t_elapsed = time() - self.t_start_eval
            print("info string move overhead is %dms" % (t_elapsed * 1000 - self.movetime_ms))

        # receive the policy vector based on the MCTS search
        p_vec_small = self.root_node.get_mcts_policy(self.q_value_weight)  # , xth_n_max=xth_n_max, is_root=True)

        if self.use_future_q_values:
            # use q-future value to update the q-values of direct child nodes
            q_future, indices = self.get_last_q_values(min_nb_visits=5, max_depth=5) #25)
            # self.root_node.q_value = 0.5 * self.root_node.q_value + 0.5 * q_future
            # TODO: make this matrix vector form
            if max_depth_reached >= 5:
                for idx in indices:
                    self.root_node.q_value[idx] = min(self.root_node.q_value[idx], q_future[idx])
                p_vec_small = self.root_node.get_mcts_policy(self.q_value_weight)

        # if self.use_pruning is False:
        self.node_lookup[key] = self.root_node  # store the current root in the lookup table
        best_child_idx = p_vec_small.argmax()  # select the q-value according to the mcts best child value
        value = self.root_node.q_value[best_child_idx]
        # value = orig_q[best_child_idx]
        lst_best_moves, _ = self.get_calculated_line()
        str_moves = self._mv_list_to_str(lst_best_moves)
        node_searched = int(self.root_node.n_sum - self.total_nodes_pre_search)  # show the best calculated line
        time_e = time() - self.t_start_eval  # In uci the depth is given using half-moves notation also called plies

        if len(legal_moves) != len(p_vec_small):
            raise Exception(
                "Legal move list %s with length %s is incompatible to policy vector %s"
                " with shape %s for board state %s and nodes legal move list: %s"
                % (legal_moves, len(legal_moves), p_vec_small, p_vec_small.shape, state, self.root_node.legal_moves)
            )

        # define the remaining return variables
        centipawns = value_to_centipawn(value)
        depth = max_depth_reached
        nodes = node_searched
        time_elapsed_s = time_e * 1000

        # avoid division by 0
        if time_e > 0.0:
            nps = node_searched / time_e
        else:
            # return a high constant in otherwise
            nps = 999999999

        pv = str_moves
        if self.verbose:
            score = "score cp %d depth %d nodes %d time %d nps %d pv %s" % (
                centipawns,
                depth,
                nodes,
                time_elapsed_s,
                nps,
                pv,
            )
            logging.info("info string %s", score)
        return value, legal_moves, p_vec_small, centipawns, depth, nodes, time_elapsed_s, nps, pv

    @staticmethod
    def _enhance_checks(chess_board, legal_moves, policy_prob):
        """
        Increases the probability by 10% for checking moves lower than 10% in policy_prob
        :param chess_board: Board state
        :param legal_moves: List of legal moves in the position
        :param policy_prob: Numpy probability vector for each move. Note this variable will be modified.
        :return:
        """
        check_mask, nb_checks = get_check_move_mask(chess_board, legal_moves)

        if nb_checks > 0:
            # increase chances of checking
            policy_prob[np.logical_and(check_mask, policy_prob < 0.1)] += 0.1
            # normalize back to 1.0
            if policy_prob is not None:
                policy_prob /= policy_prob.sum()

    @staticmethod
    def _enhance_captures(chess_board, legal_moves, policy_prob):
        """
        Increases the probability by 5% for capturing moves lower than 10% in policy_prob
        :param chess_board: Board state
        :param legal_moves: List of legal moves in the position
        :param policy_prob: Numpy probability vector for each move. Note this variable will be modified.
        :return:
        """
        for capture_move in chess_board.generate_legal_captures():
            index = legal_moves.index(capture_move)
            if policy_prob[index] < 0.04:
                policy_prob[index] += 0.04
        if policy_prob is not None:
            policy_prob /= policy_prob.sum()

    def _expand_root_node_multiple_moves(self, state, legal_moves):
        """
        Checks if the current root node can be found in the look-up table.
        Otherwise run a single inference of the neural network for this board state
        :param state: Current game state
        :param legal_moves: Available moves
        :return:
        """

        is_leaf = False  # initialize is_leaf by default to false
        [value, policy_vec] = self.nets[0].predict_single(state.get_state_planes())  # start a brand new tree
        # extract a sparse policy vector with normalized probabilities
        p_vec_small = get_probs_of_move_list(policy_vec, legal_moves, state.mirror_policy())
        chess_board = state.get_pythonchess_board()
        if self.enhance_captures:
            self._enhance_captures(chess_board, legal_moves, p_vec_small)

        if self.enhance_checks:
            self._enhance_checks(chess_board, legal_moves, p_vec_small)

        # create a new root node
        self.root_node = Node(chess_board, value, p_vec_small, legal_moves, is_leaf, clip_low_visit=False)

    def _expand_root_node_single_move(self, state, legal_moves):
        """
        Expands the current root in the case if there's only a single move available.
        The neural network search can be omitted in this case.
        :param state: Request games state
        :param legal_moves: Available moves
        :return:
        """
        # request the value prediction for the current position
        [value, _] = self.nets[0].predict_single(state.get_state_planes())
        p_vec_small = np.array([1], np.float32)  # we can create the move probability vector without the NN this time

        # create a new root node
        self.root_node = Node(state.get_pythonchess_board(), value, p_vec_small, legal_moves, clip_low_visit=False)

        if self.root_node.child_nodes[0] is None:  # check a child node if it doesn't exists already
            state_child = deepcopy(state)
            state_child.apply_move(legal_moves[0])
            is_leaf = False  # initialize is_leaf by default to false
            # we don't need to check for is_lost() because the game is already over
            if state.is_loss():  # check if the current player has won the game
                value = -1
                is_leaf = True
                legal_moves_child = []
                p_vec_small_child = None
            elif state.board.uci_variant == "giveaway" and state.is_win():
                # giveaway chess is a variant in which you win on your own turn
                value = +1
                is_leaf = True
                legal_moves_child = []
                p_vec_small_child = None
            # check if you can claim a draw - it's assumed that the draw is always claimed
            elif (
                self.can_claim_threefold_repetition(state.get_transposition_key(), [0])
                or state.get_pythonchess_board().can_claim_fifty_moves()
            ):
                value = 0
                is_leaf = True
                legal_moves_child = []
                p_vec_small_child = None
            else:
                legal_moves_child = state_child.get_legal_moves()
                # start a brand new prediction for the child
                [value, policy_vec] = self.nets[0].predict_single(state_child.get_state_planes())
                # extract a sparse policy vector with normalized probabilities
                p_vec_small_child = get_probs_of_move_list(
                    policy_vec, legal_moves_child, state_child.mirror_policy()
                )

            # create a new child node
            child_node = Node(state.get_pythonchess_board(), value, p_vec_small_child, legal_moves_child, is_leaf)
            self.root_node.child_nodes[0] = child_node  # connect the child to the root
            # assign the value of the root node as the q-value for the child
            # here we must invert the invert the value because it's the value prediction of the next state
            self.root_node.q_value[0] = -value

    def _run_mcts_search(self, state):
        """
        Runs a new or continues the mcts on the current search tree.
        :param state: Input state given by the user
        :return: max_depth_reached (int) - The longest search path length after the whole search
        """

        self.node_lookup = {}  # clear the look up table
        self.root_node_prior_policy = deepcopy(self.root_node.policy_prob)  # safe the prior policy of the root node
        # apply dirichlet noise to the prior probabilities in order to ensure
        #  that every move can possibly be visited
        self.root_node.apply_dirichlet_noise_to_prior_policy(epsilon=self.dirichlet_epsilon, alpha=self.dirichlet_alpha)
        # store what depth has been reached at maximum in the current search tree
        max_depth_reached = 1  # default is 1, in case only 1 move is available
        futures = []

        if state.are_pocket_empty():  # set the number of playouts accordingly
            nb_playouts = self.nb_playouts_empty_pockets
        else:
            nb_playouts = self.nb_playouts_filled_pockets

        t_elapsed_ms = cur_playouts = 0
        old_time = time()
        cpuct_init = self.cpuct

        if self.use_time_management:
            time_checked = time_checked_early = False
        else:
            time_checked = time_checked_early = True

        while (
            max_depth_reached < self.max_search_depth and cur_playouts < nb_playouts and t_elapsed_ms < self.movetime_ms
        ):  # and np.abs(self.root_node.q_value.mean()) < 0.99:

            # start searching
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                for i in range(self.threads):
                    # calculate the thread id based on the current playout
                    futures.append(
                        executor.submit(
                            self._run_single_playout, parent_node=self.root_node, pipe_id=i, depth=1, chosen_nodes=[]
                        )
                    )

            cur_playouts += self.threads
            time_show_info = time() - old_time

            for i, future in enumerate(futures):
                cur_value, cur_depth, chosen_nodes = future.result()

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

            t_elapsed = time() - self.t_start_eval  # update the current search time
            t_elapsed_ms = t_elapsed * 1000
            if time_show_info > 1:
                node_searched = int(self.root_node.n_sum - self.total_nodes_pre_search)
                print("info nps %d time %d" % (int((node_searched / t_elapsed)), t_elapsed_ms))

            if not time_checked_early and t_elapsed_ms > self.movetime_ms / 2:
                if (
                    self.root_node.policy_prob.max() > 0.9
                    and self.root_node.policy_prob.argmax() == self.root_node.q_value.argmax()
                ):
                    self.time_buffer_ms += (self.movetime_ms - t_elapsed_ms) * 0.9
                    print("info early break up")
                    break
                else:
                    time_checked_early = True

            if (
                self.time_buffer_ms > 2500
                and not time_checked
                and t_elapsed_ms > self.movetime_ms * 0.9
                and self.root_node.q_value[self.root_node.child_number_visits.argmax()]
                < self.root_node.initial_value + 0.01
            ):
                print("info increase time")
                time_checked = True
                time_bonus = self.time_buffer_ms / 4
                self.time_buffer_ms -= time_bonus  # increase the movetime
                self.movetime_ms += time_bonus * 0.75
                self.root_node.initial_value = self.root_node.q_value[self.root_node.child_number_visits.argmax()]

                if self.time_buffer_ms < 0:
                    self.movetime_ms += self.time_buffer_ms
                    self.time_buffer_ms = 0
        self.cpuct = cpuct_init
        return max_depth_reached

    def perform_action(self, state_in: GameState):
        """
        Return a value, best move with according to the mcts search.
        This method is used when using the mcts agent as a player.
        :param state_in: Requested games state
        :return: value - Board value prediction
                selected_move - Python chess move object according to mcts
                confidence - Confidence for selecting this move
                selected_child_idx - Child index which correspond to the selected child
        """
        # create a deepcopy of the state in order not to change the given input parameter
        return super().perform_action(deepcopy(state_in))

    def _run_single_playout(self, parent_node: Node, pipe_id=0, depth=1, chosen_nodes=None):
        """
        This function works recursively until a leaf or terminal node is reached.
        It ends by back-propagating the value of the new expanded node or by propagating the value of a terminal state.

        :param state: Current game-state for the evaluation. This state differs between the treads
        :param parent_node: Current parent-node of the selected node. In the first  expansion this is the root node.
        :param depth: Current depth for the evaluation. Depth is increased by 1 for every recursive call
        :param chosen_nodes: List of moves which have been taken in the current path.
                        For each selected child node this list is expanded by one move recursively.
        :param chosen_nodes: List of all nodes that this thread has explored with respect to the root node
        :return: -value: The inverse value prediction of the current board state. The flipping by -1 each turn is needed
                        because the point of view changes each half-move
                depth: Current depth reach by this evaluation
                mv_list: List of moves which have been selected
        """
        # Probably is better to be refactored
        # Too many arguments (6/5) - Too many local variables (27/15) - Too many branches (28/12) -
        # Too many statements (86/50)
        if chosen_nodes is None:  # select a legal move on the chess board
            chosen_nodes = []
        node, move, child_idx = self._select_node(parent_node)

        if move is None:
            raise Exception("Illegal tree setup. A 'None' move was selected which shouldn't be possible")
        # update the visit counts to this node
        # temporarily reduce the attraction of this node by applying a virtual loss /
        # the effect of virtual loss will be undone if the playout is over
        parent_node.apply_virtual_loss_to_child(child_idx, self.virtual_loss)

        # append the selected move to the move list
        chosen_nodes.append(child_idx)  # append the chosen child idx to the chosen_nodes list

        if node is None:
            state = GameState(deepcopy(parent_node.board))  # get the board from the parent node
            state.apply_move(move)  # apply the selected move on the board

            # get the transposition-key which is used as an identifier for the board positions in the look-up table
            transposition_key = state.get_transposition_key()
            # check if the addressed fen exist in the look-up table
            # note: It's important to use also the halfmove-counter here, otherwise the system can create an infinite
            # feed-back-loop
            key = transposition_key + (state.get_fullmove_number(),)

            if self.use_transposition_table and key in self.node_lookup:

                node = self.node_lookup[key]  # get the node from the look-up list

                # get the prior value from the leaf node which has already been expanded
                value = node.initial_value

                # clip the visit nodes for all nodes in the search tree except the director opp. move
                clip_low_visit = self.use_pruning

                new_node = Node(
                    node.board,
                    value,
                    node.policy_prob,
                    node.legal_moves,
                    node.is_leaf,
                    key,
                    clip_low_visit,
                )  # create a new node

                with parent_node.lock:
                    parent_node.child_nodes[child_idx] = new_node  # add the new node to its parent

            else:
                # expand and evaluate the new board state (the node wasn't found in the look-up table)
                # its value will be back-propagated through the tree and flipped after every layer
                my_pipe = self.my_pipe_endings[pipe_id]  # receive a free available pipe

                if self.send_batches:
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

                is_leaf = is_won = False  # initialize is_leaf by default to false and check if the game is won
                # check if the current player has won the game
                # (we don't need to check for is_lost() because the game is already over
                #  if the current player checkmated his opponent)
                if state.is_check():
                    if state.is_loss():
                        is_won = True

                # needed for e.g. atomic because the king explodes and is not in check mate anymore
                if state.is_variant_loss():
                    is_won = True

                if is_won:
                    value = -1
                    is_leaf = True
                    legal_moves = []
                    p_vec_small = None
                    # establish a mate in one connection in order to stop exploring different alternatives
                    parent_node.set_check_mate_node_idx(child_idx)
                # get the value from the leaf node (the current function is called recursively)
                # check if you can claim a draw - it's assumed that the draw is always claimed
                elif (
                    self.can_claim_threefold_repetition(transposition_key, chosen_nodes)
                    or state.get_pythonchess_board().can_claim_fifty_moves() is True
                ):
                    value = 0
                    is_leaf = True
                    legal_moves = []
                    p_vec_small = None
                else:
                    legal_moves = state.get_legal_moves()  # get the current legal move of its board state

                    if not legal_moves:
                        # stalemate occurred which is very rare for crazyhouse
                        if state.uci_variant == "giveaway":
                            value = 1
                        else:
                            value = 0
                        is_leaf = True
                        legal_moves = []
                        p_vec_small = None
                        # raise Exception("No legal move is available for state: %s" % state)
                    else:
                        try:  # extract a sparse policy vector with normalized probabilities
                            p_vec_small = get_probs_of_move_list(
                                policy_vec, legal_moves, mirror_policy=state.mirror_policy(), normalize=True
                            )
                        except KeyError:
                            raise Exception("Key Error for state: %s" % state)

                # clip the visit nodes for all nodes in the search tree except the director opp. move
                clip_low_visit = self.use_pruning and depth != 1  # and depth > 4
                new_node = Node(
                    state.get_pythonchess_board(),
                    value,
                    p_vec_small,
                    legal_moves,
                    is_leaf,
                    transposition_key,
                    clip_low_visit,
                )  # create a new node

                if depth == 1:
                    # disable uncertain moves from being visited by giving them a very bad score
                    if not is_leaf and self.use_pruning:
                        if self.root_node_prior_policy[child_idx] < 1e-3 and value * -1 < self.root_node.initial_value:
                            with parent_node.lock:
                                value = 99

                    # for performance reasons only apply check enhancement on depth 1 for now
                    chess_board = state.get_pythonchess_board()
                    if self.enhance_checks:
                        self._enhance_checks(chess_board, legal_moves, p_vec_small)

                    if self.enhance_captures:
                        self._enhance_captures(chess_board, legal_moves, p_vec_small)

                if not self.use_pruning:
                    self.node_lookup[key] = new_node  # include a reference to the new node in the look-up table

                with parent_node.lock:
                    parent_node.child_nodes[child_idx] = new_node  # add the new node to its parent
        elif node.is_leaf:  # check if we have reached a leaf node
            value = node.initial_value
        else:
            # get the value from the leaf node (the current function is called recursively)
            value, depth, chosen_nodes = self._run_single_playout(node, pipe_id, depth + 1, chosen_nodes)
        # revert the virtual loss and apply the predicted value by the network to the node
        parent_node.revert_virtual_loss_and_update(child_idx, self.virtual_loss, -value)
        # invert the value prediction for the parent of the above node layer because the player's changes every turn
        return -value, depth, chosen_nodes

    def check_for_duplicate(self, transposition_key, chosen_nodes):
        """

        :param transposition_key: Transposition key which defines the board state by all its pieces and pocket state.
                                  The move counter is disregarded.
        :param chosen_nodes: List of moves which have been taken in the current path.
        :return:
        """
        node = self.root_node.child_nodes[chosen_nodes[0]]
        # iterate over all accessed nodes during the current search of the thread and check for same transposition key
        for node_idx in chosen_nodes[1:-1]:
            if node.transposition_key == transposition_key:
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
        :param transposition_key: Transposition key which defines the board state by all its pieces and pocket state.
                                  The move counter is disregarded.
        :param chosen_nodes: List of integer indices which correspond to the child node indices chosen from the
                            root node downwards.
        :return: True, if threefold repetition can be claimed, else False
        """

        search_occurrence_counter = 0  # set the number of occurrences by default to 0
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

        if parent_node.check_mate_node:
            child_idx = parent_node.check_mate_node
        else:
            # find the move according to the q- and u-values for each move
            # pb_c_base = 19652
            # pb_c_init = self.cpuct
            cpuct = math.log((parent_node.n_sum + 19652 + 1) / 19652) + self.cpuct

            # pb_u_base = 19652 / 10
            # pb_u_init = 1
            # pb_u_low = self.u_init_divisor
            # u_init = np.exp((-parent_node.n_sum + 1965 + 1) / 1965) / np.exp(1) * (1 - self.u_init_divisor) + self.u_init_divisor

            # calculate the current u values
            # it's not worth to save the u values as a node attribute because u is updated every time n_sum changes
            u_value = (
                cpuct
                * parent_node.policy_prob
                * (np.sqrt(parent_node.n_sum) / (self.u_init_divisor + parent_node.child_number_visits))
            )

            # if parent_node.n_sum % 10 == 0:
            #     prob = parent_node.q_value + u_value
            #     child_idx = prob.argmax()
            #     prob[child_idx] = 0
            #     child_idx = prob.argmax()
            #     # child_idx = np.random.randint(parent_node.nb_direct_child_nodes)
            # else:
            child_idx = (parent_node.q_value + u_value).argmax()

        return parent_node.child_nodes[child_idx], parent_node.legal_moves[child_idx], child_idx

    def _select_node_based_on_mcts_policy(self, parent_node: Node):
        """
        Selects the next node based on the mcts policy which is used to predict the final best move.
        :param parent_node: Node from which to select the next child.
        :return:
        """

        child_idx = parent_node.get_mcts_policy(self.q_value_weight).argmax()
        nb_visits = parent_node.child_number_visits[child_idx]
        return parent_node.child_nodes[child_idx], parent_node.legal_moves[child_idx], nb_visits, child_idx

    def show_next_pred_line(self):
        """ It returns the predicted best moves for both players"""
        best_moves = []
        node = self.root_node  # start at the root node

        while node:
            # go deep through the tree by always selecting the best move for both players
            node, move, _ = self._select_node(node)
            best_moves.append(move)
        return best_moves

    def get_2nd_max(self) -> int:
        """
        Returns the number of visits of the 2nd most visited direct child node
        :return: Integer value of number of visits
        """
        n_child = self.root_node.child_number_visits.argmax()
        n_max = self.root_node.child_number_visits[n_child]
        self.root_node.child_number_visits[n_child] = 0
        second_max = self.root_node.child_number_visits.max()
        self.root_node.child_number_visits[n_child] = n_max
        return second_max

    def get_xth_max(self, xth_node):
        """
        Returns the number of visits of the X most visited direct child node
        ;:param xth_node: Index number for the number of visits. 1 ist the most visited child
        :return: Integer value of number of visits
        """
        if len(self.root_node.child_number_visits) < xth_node:
            return self.root_node.child_number_visits.min()
        return np.sort(self.root_node.child_number_visits)[-xth_node]

    def get_last_q_values(self, min_nb_visits=5, max_depth=25):
        """
        Returns the values of the last node in the calculated lines according to the mcts search for the most
         visited nodes
        :param max_depth : maximum depth to reach for evaluating the q-values.
                 This avoids that very deep q-values are assigned to the original q-value which might have very
                 low actual correspondence
        :param min_nb_visits: Integer defining how deep the tree will be traversed to return the final q-value
        :return: q_future - q-values for the most visited nodes when going deeper in the tree
                indices - indices of the evaluated child nodes

        """

        q_future = np.zeros(self.root_node.nb_direct_child_nodes)
        indices = []

        for idx in range(self.root_node.nb_direct_child_nodes):
            depth = 1
            if self.root_node.child_number_visits[idx] >= self.root_node.child_number_visits.max() * 0.33:
                node = self.root_node.child_nodes[idx]
                final_node = self.root_node
                move = self.root_node.legal_moves[idx]
                child_idx = idx

                while node and not node.is_leaf and node.n_sum >= min_nb_visits and depth <= max_depth:
                    final_node = node
                    print(move.uci() + " ", end="")
                    print(str(node.initial_value) + " ", end="")
                    node, move, _, child_idx = self._select_node_based_on_mcts_policy(node)
                    depth += 1

                if final_node:
                    q_future[idx] = final_node.q_value[child_idx]
                    indices.append(idx)
                    # invert the value prediction for an odd depth number
                    if depth % 2 == 0:
                        q_future[idx] *= -1

                print(q_future[idx])

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
        node = self.root_node  # start at the root node

        while node and not node.is_leaf:
            # go deep through the tree by always selecting the best move for both players
            node, move, nb_visits, _ = self._select_node_based_on_mcts_policy(node)
            lst_best_moves.append(move)
            lst_nb_visits.append(nb_visits)
        return lst_best_moves, lst_nb_visits

    @staticmethod
    def _mv_list_to_str(lst_moves):
        """
        Converts a given list of chess moves to a single string separated by spaces.
        :param lst_moves: List chess.Moves objects
        :return: String representing each move in the list
        """
        str_moves = lst_moves[0].uci()

        for move in lst_moves[1:]:
            str_moves += " " + move.uci()
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
            mv_list.append(node.legal_moves[child_idx])
            node = node.child_nodes[child_idx]
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

    def update_transposition_table(self, transposition_key):
        """
        :param transposition_key: (gamestate.get_transposition_key(),)
        :return:
        """

        self.transposition_table.update(transposition_key)
