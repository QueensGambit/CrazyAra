"""
@file: Node.py
Created on 13.10.18
@project: crazy_ara_refactor
@author: queensgambit

Helper class which stores the statistics of all nodes and in the search tree.
"""
from copy import deepcopy
from threading import Lock
import chess
import numpy as np


QSIZE = 100


class Node:  # Too many instance attributes (14/7)
    """Helper class for nodes stats in the search tree."""

    def __init__(
        self,
        board: chess.BaseBoard,
        value,
        p_vec_small: np.ndarray,
        legal_moves: [chess.Move],
        str_legal_moves: str,
        is_leaf=False,
        transposition_key=None,
        clip_low_visit=True,
    ):  # Too many arguments (8/5)

        self.lock = Lock()  # lock object for this node to protect its member variables
        self.board = board  # python-chess board object representing the current position
        self.initial_value = value  # store the initial value prediction of the current board position

        if is_leaf:
            self.nb_direct_child_nodes = 0
        else:
            # specify the number of direct child nodes from this node
            self.nb_direct_child_nodes = len(p_vec_small)  # np.array(len(p_vec_small))

        self.policy_prob = p_vec_small  # prior probability selecting each child, which is estimated by the NN
        self.legal_moves = legal_moves  # possible legal moves from this node on which represents the edges
        # stores the number of all direct children and all grand children which have already been expanded
        self.nb_total_expanded_child_nodes = np.array(0)
        self.child_number_visits = np.zeros(self.nb_direct_child_nodes)  # visit count of all its child nodes
        # total action value estimated by MCTS for each child node
        self.action_value = np.zeros(self.nb_direct_child_nodes)
        # self.w = np.ones(self.nb_direct_child_nodes) * -0.01 #1
        # q: combined action value which is calculated by the averaging over all action values
        # u: exploration metric for each child node
        # (the q and u values are stacked into 1 list in order to speed-up the argmax() operation
        #self.q_value = np.zeros(self.nb_direct_child_nodes)
        self.q_value = np.ones(self.nb_direct_child_nodes) * -1

        if not is_leaf:
            if clip_low_visit:
                self.q_value[p_vec_small < 1e-3] = -9999
            # else:
            #    self.thresh_idcs_root = p_vec_small < 5e-2

        # number of total visits to this node
        self.n_sum = 1  # we initialize with 1 because if the node was created it must have been visited

        # check if there's a possible mate on the board if yes create a quick link to the mate move
        mate_mv_idx_str = str_legal_moves.find("#")
        if mate_mv_idx_str != -1:  # -1 means that no mate move has been found
            # find the according index of the move in the legal_moves generator list
            # and make a quick reference path to a child node which leads to mate
            self.mate_child_idx = str_legal_moves[:mate_mv_idx_str].count(",")  # Count the ',' its the move index
        else:
            self.mate_child_idx = None  # If no direct mate move is possible so set the reference to None

        # stores the number of all possible expandable child nodes
        # self.nb_expandable_child_nodes = np.array(self.nb_direct_child_nodes)
        # list of all child nodes which are described by each board position
        # the children are ordered in the same way as the legal_move generator output
        self.child_nodes = [None] * int(self.nb_direct_child_nodes)
        # determine if the node is a leaf node this avoids checking for state.is_draw() or .state.is_won()
        self.is_leaf = is_leaf
        # store a unique identifier for the board state excluding the move counter for this node
        self.transposition_key = transposition_key

    def get_mcts_policy(self, q_value_weight=0.65, clip_low_visit_nodes=True):  # , is_root=False, xth_n_max=0
        """
        Calculates the finetuned policies based on the MCTS search.
        These policies should be better than the initial policy predicted by a the raw network.
        The policy values are ordered in the same way as list(board.legal_moves)
        :param: q_value_weight: Float indicating how the number of visits and the q-values should be mixed.
                                Expected to be in range [0.,1.]
        :param: q_value_weight: Boolean indicating if for the final selected move also the q-values should be
                                taken into account. By default use the average of the q-value and the visit count.
        :return: Pruned policy vector based on the MCTS search
        """

        if clip_low_visit_nodes and q_value_weight > 0:
            visit = deepcopy(self.child_number_visits)
            value = deepcopy((self.q_value + 1))
            max_visits = visit.max()
            if max_visits > 0:
                # normalize to sum of 1
                value[visit < max_visits * 0.33] = 0  # mask out nodes that haven't been visited much
                value[value < 0] = 0
                # re-normalize to 1
                visit /= visit.sum()
                if value.max() > 0:
                    # make sure not to divide by 0
                    value /= value.sum()

                policy = (1 - q_value_weight) * visit + q_value_weight * value
                return policy / sum(policy)
            return visit

        if q_value_weight > 0:
            # disable the q values if there's at least one child which wasn't explored
            if None in self.child_nodes:
                q_value_weight = 0
            # we add +1 to the q values to avoid negative values, then the q values are normalized to [0,1] before
            # the q_value_weight is applied.
            policy = (self.child_number_visits / self.n_sum) * (1 - q_value_weight) + (
                (self.q_value + 1) * 0.5
            ) * q_value_weight
            return policy

        if max(self.child_number_visits) == 1:
            policy = self.child_number_visits + 0.05 * self.policy_prob
        else:
            policy = self.child_number_visits - 0.05 * self.policy_prob
        return policy / sum(policy)

    def apply_dirichlet_noise_to_prior_policy(self, epsilon=0.25, alpha=0.15):
        """
        Promote exploration from the root node child nodes by adding dirichlet noise
        This ensures that every can be possibly be explored in the distant future
        :param epsilon: Percentage amount of the dirichlet noise to apply to the priors
        :param alpha: Dirichlet strength -
         This is a hyper-parameter which depends on the typical amount of moves in the current game type
        :return:
        """

        if not self.is_leaf:
            with self.lock:
                dirichlet_noise = np.random.dirichlet([alpha] * self.nb_direct_child_nodes)
                self.policy_prob = (1 - epsilon) * self.policy_prob + epsilon * dirichlet_noise

    def apply_virtual_loss_to_child(self, child_idx, virtual_loss):
        """
        Apply virtual loss to the child nodes
        :param child_idx: Where the child node starts
        :param virtual_loss: Specify the virtual loss value
        :return:
        """
        # update the stats of the parent node
        with self.lock:
            # update the visit counts to this node
            # temporarily reduce the attraction of this node by applying a virtual loss /
            # the effect of virtual loss will be undone if the playout is over
            # virtual increase the number of visits
            self.n_sum += virtual_loss
            self.child_number_visits[child_idx] += virtual_loss
            # make it look like if one has lost X games from this node forward where X is the virtual loss value
            self.action_value[child_idx] -= virtual_loss
            self.q_value[child_idx] = self.action_value[child_idx] / self.child_number_visits[child_idx]

    def revert_virtual_loss_and_update(self, child_idx, virtual_loss, value):
        """
        Revert the virtual loss effect and apply the backpropagated value of its child node
        :param child_idx:  Where the child node starts
        :param virtual_loss: Specify the virtual loss value
        :param value:  Specify the backpropagated value
        :return:
        """
        with self.lock:
            self.n_sum -= virtual_loss - 1
            self.child_number_visits[child_idx] -= virtual_loss - 1
            self.action_value[child_idx] += virtual_loss + value
            self.q_value[child_idx] = self.action_value[child_idx] / self.child_number_visits[child_idx]
