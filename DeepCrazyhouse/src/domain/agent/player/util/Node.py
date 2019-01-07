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


class Node:
    def __init__(
        self,
        value,
        p_vec_small: np.ndarray,
        legal_moves: [chess.Move],
        str_legal_moves: str,
        is_leaf=False,
        transposition_key=None,
        clip_low_visit=True,
    ):

        # lock object for this node to protect its member variables
        self.lock = Lock()

        # store the initial value prediction of the current board position
        self.v = value

        if is_leaf is True:
            self.nb_direct_child_nodes = 0
        else:
            # specify the number of direct child nodes from this node
            self.nb_direct_child_nodes = np.array(len(p_vec_small))

        # prior probability selecting each child, which is estimated by the neural network
        self.p = p_vec_small
        # possible legal moves from this node on which represents the edges
        self.legal_moves = legal_moves

        # stores the number of all direct children and all grand children which have already been expanded
        self.nb_total_expanded_child_nodes = np.array(0)

        # visit count of all its child nodes
        self.n = np.zeros(self.nb_direct_child_nodes)
        # total action value estimated by MCTS for each child node
        self.w = np.zeros(self.nb_direct_child_nodes)
        # self.w = np.ones(self.nb_direct_child_nodes) * -0.01 #1

        # q: combined action value which is calculated by the averaging over all action values
        # u: exploration metric for each child node
        # (the q and u values are stacked into 1 list in order to speed-up the argmax() operation

        # self.q = np.zeros(self.nb_direct_child_nodes)
        self.q = np.ones(self.nb_direct_child_nodes) * -1

        if is_leaf is False:
            if clip_low_visit is True:
                self.q[p_vec_small < 1e-3] = -9999
            # else:
            #    self.thresh_idcs_root = p_vec_small < 5e-2

        # number of total visits to this node
        # we initialize with 1 because if the node was created it must have been visited
        self.n_sum = 1

        # check if there's a possible mate on the board if yes create a quick link to the mate move
        mate_mv_idx_str = str_legal_moves.find("#")
        if mate_mv_idx_str != -1:
            # -1 means that no mate move has been found
            # find the according index of the move in the legal_moves generator list
            # here we count the ',' which represent the move index
            mate_mv_idx = str_legal_moves[:mate_mv_idx_str].count(",")
            # quick reference path to a child node which leads to mate
            self.mate_child_idx = mate_mv_idx
        else:
            # no direct mate move is possible so set the reference to None
            self.mate_child_idx = None

        # stores the number of all possible expandable child nodes
        # self.nb_expandable_child_nodes = np.array(self.nb_direct_child_nodes)

        # list of all child nodes which are described by each board position
        # the children are ordered in the same way as the legal_move generator output
        self.child_nodes = [None] * int(self.nb_direct_child_nodes)

        # determine if the node is a leaf node this avoids checking for state.is_draw() or .state.is_won()
        self.is_leaf = is_leaf

        # store a unique identifier for the board state excluding the move counter for this node
        self.transposition_key = transposition_key

        # self.replay_buffer = deque([0] * 512)
        # self.q_freash = np.zeros(self.nb_direct_child_nodes)
        # self.w_freash = np.zeros(self.nb_direct_child_nodes)

    def get_mcts_policy(self, q_value_weight=0.65, clip_low_visit_nodes=True, is_root=False, xth_n_max=0):
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

        if clip_low_visit_nodes is True and q_value_weight > 0:

            # q_value_weight -= 1e-4 * self.n_sum
            # q_value_weight = max(q_value_weight, 0.01)

            visit = deepcopy(self.n)
            value = deepcopy((self.q + 1))

            # values_confident = self.p[]
            if visit.max() > 0:

                max_visits = visit.max()

                # if is_root is True:
                #    if self.n_sum > 2000 and self.thresh_idcs_root.max() == 1:
                #        if visit[self.thresh_idcs_root].max() < visit[np.invert(self.thresh_idcs_root)].max() * 1.5:
                #            visit[self.thresh_idcs_root] = 0
                #            #print('clipped nodes')

                # mask out nodes that haven't been visited much
                # thresh_idces = visit < max(max_visits * 0.33, xth_n_max)
                thresh_idces = visit < max_visits * 0.33  # , xth_n_max)

                # normalize to sum of 1
                value[thresh_idces] = 0
                value[value < 0] = 0
                # visit[thresh_idces] = 0

                # renormalize ot 1
                visit /= visit.sum()

                # value *= self.p
                value /= value.sum()

                # use prior policy
                # init_p = deepcopy(self.p)
                # init_p[value < value.max() * 0.2] = 0
                # visit += self.p

                policy = (1 - q_value_weight) * visit + q_value_weight * value

                # if is_root is True:
                #    indices = (self.q < self.v) * self.thresh_idcs_root
                #    policy[indices] = 0

                return policy / sum(policy)
            else:
                return visit

        elif q_value_weight > 0:
            # disable the q values if there's at least one child which wasn't explored
            if None in self.child_nodes:
                q_value_weight = 0

            # we add +1 to the q values to avoid negative values, then the q values are normalized to [0,1] before
            # the q_value_weight is applied.
            policy = (self.n / self.n_sum) * (1 - q_value_weight) + ((self.q + 1) * 0.5) * q_value_weight
            return policy
        else:
            if max(self.n) == 1:
                policy = self.n + 0.05 * self.p
            else:
                policy = self.n - 0.05 * self.p

            return policy / sum(policy)

    def apply_dirichlet_noise_to_prior_policy(self, epsilon=0.25, alpha=0.15):
        """
        # Promote exploration from the root node child nodes by adding dirichlet noise
        # This ensures that every can be possibly be explored in the distant future
        :param epsilon: Percentage amount of the dirichlet noise to apply to the priors
        :param alpha: Dirichlet strength - This is a hyperparameter which depends on the typical amount of moves in the current game type
        :return:
        """

        if self.is_leaf is False:
            with self.lock:
                dirichlet_noise = np.random.dirichlet([alpha] * self.nb_direct_child_nodes)
                self.p = (1 - epsilon) * self.p + epsilon * dirichlet_noise

    def apply_virtual_loss_to_child(self, child_idx, virtual_loss):

        # update the stats of the parent node
        with self.lock:
            # update the visit counts to this node
            # temporarily reduce the attraction of this node by applying a virtual loss /
            # the effect of virtual loss will be undone if the playout is over
            # virtual increase the number of visits
            self.n_sum += virtual_loss
            self.n[child_idx] += virtual_loss
            # make it look like if one has lost X games from this node forward where X is the virtual loss value
            self.w[child_idx] -= virtual_loss
            self.q[child_idx] = self.w[child_idx] / self.n[child_idx]

            # use queue
            # self.q[child_idx] = self.w[child_idx] / min(self.n[child_idx], QSIZE)

    def revert_virtual_loss_and_update(self, child_idx, virtual_loss, value):
        # revert the virtual loss effect and apply the backpropagated value of its child node
        with self.lock:

            self.n_sum -= virtual_loss - 1
            # factor = max(self.n[child_idx] // 1000, 1)
            # fac = (self.n[child_idx]+1) ** 0.2

            self.n[child_idx] -= virtual_loss - 1

            self.w[child_idx] += virtual_loss + value

            self.q[child_idx] = self.w[child_idx] / self.n[child_idx]

            # self.nb_total_expanded_child_nodes += 1
            # self.nb_expandable_child_nodes += self.nb_direct_child_nodes

            # last_value = self.child_nodes[child_idx].replay_buffer.popleft()
            # self.child_nodes[child_idx].replay_buffer.append(value)

            # use queue
            # self.w_freash[child_idx] += virtual_loss + value - last_value
            # self.q_freash[child_idx] = self.w[child_idx] / QSIZE # min(self.n[child_idx], QSIZE)
