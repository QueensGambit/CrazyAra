"""
@file: Node.py
Created on 13.10.18
@project: crazy_ara_refactor
@author: queensgambit

Please describe what the content of this file is about
"""

from threading import Lock
import chess
import numpy as np
import logging


class Node:
    """
    Helper Class which stores the statistics of all child nodes of this node in the search tree
    """

    def __init__(self, p_vec_small: np.ndarray, legal_moves: [chess.Move], str_legal_moves: str):

        # lock object for this node to protect its member variables
        self.lock = Lock()

        # store the initial value prediction of the current board position
        #self.v = v
        # specify the number of direct child nodes from this node
        self.nb_direct_child_nodes = np.array(len(p_vec_small)) #, np.uint32)
        # prior probability selecting each child, which is estimated by the neural network
        self.p = p_vec_small #np.zeros(self.nb_direct_child_nodes, np.float32)
        # possible legal moves from this node on which represents the edges
        self.legal_moves = legal_moves

        # stores the number of all direct children and all grand children which have already been expanded
        self.nb_total_expanded_child_nodes = np.array(0) #, np.uint32)

        # visit count of all its child nodes
        self.n = np.zeros(self.nb_direct_child_nodes) #, np.int32)
        # total action value estimated by MCTS for each child node
        self.w = np.zeros(self.nb_direct_child_nodes) #, np.float32)
        # q: combined action value which is calculated by the averaging over all action values
        # u: exploration metric for each child node
        # (the q and u values are stacked into 1 list in order to speed-up the argmax() operation

        self.q = np.zeros(self.nb_direct_child_nodes) #, np.float32)
        #self.q_u = np.stack((q, u))

        #np.concatenate((q, u))

        # number of total visits to this node
        # we initialize with 1 because if the node was created it must have been visited
        self.n_sum = np.array(1) #, #np.int32)

        # check if there's a possible mate on the board if yes create a quick link to the mate move
        mate_mv_idx_str = str_legal_moves.find('#')
        #logging.debug('legal_moves: %s' % str(str_legal_moves))
        #logging.debug('mate_mv_idx_str: %d' % mate_mv_idx_str)
        if mate_mv_idx_str != -1:
            # -1 means that no mate move has been found
            # find the according index of the move in the legal_moves generator list
            # here we count the ',' which represent the move index
            mate_mv_idx = str_legal_moves[:mate_mv_idx_str].count(',')
            # quick reference path to a child node which leads to mate
            self.mate_child_idx = mate_mv_idx #legal_moves[mate_mv_idx]
            # overwrite the number of direct child nodes to 1
            #self.nb_direct_child_nodes = np.array(1) #, np.uint32)
            #logging.debug('set mate in one connection')
        else:

            # no direct mate move is possible so set the reference to None
            self.mate_child_idx = None

        # stores the number of all possible expandable child nodes
        self.nb_expandable_child_nodes = np.array(self.nb_direct_child_nodes) #, np.uint32)

        assert self.nb_direct_child_nodes > 0

        # list of all child nodes which are described by each board position
        # the children are ordered in the same way as the legal_move generator output
        self.child_nodes = [None] * int(self.nb_direct_child_nodes)

    ''' TODO: Delete
    def update_u_for_child(self, child_idx, cpuct):
        """
        Updates the u parameter via the formula given in the AlphaZero paper for a given child index

        :param child_idx: Child index to update
        :param cpuct: cpuct constant to apply (cpuct manages the amount of exploration)
        :return:
        """
        self.q_u[child_idx] = cpuct * self.p[child_idx] * (np.sqrt(self.n_sum) / (1 + self.n[child_idx]))
    '''

    def get_mcts_policy(self):
        """
        Calculates the finetuned policies based on the MCTS search.
        These policies should be better than the initial policy predicted by a the raw network.
        THe policy values are ordered in the same way as list(board.legal_moves)

        :return: Pruned policy vector based on the MCTS search
        """

        if max(self.n) == 1:
            policy = (self.n + 0.05 * self.p) #/ self.n_sum
        else:
            policy = (self.n - 0.05 * self.p) #/ self.n_sum
        return policy / sum(policy)
        #return self.n / self.n_sum

    def apply_dirichlet_noise_to_prior_policy(self, epsilon=0.25, alpha=0.15):
        """
        # Promote exploration from the root node child nodes by adding dirichlet noise
        # This ensures that every can be possibly be explored in the distant future

        :param epsilon: Percentage amount of the dirichlet noise to apply to the priors
        :param alpha: Dirichlet strength - This is a hyperparameter which depends on the typical amount of moves in the current game type
        :return:
        """

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
            #parent_node.update_u_for_child(child_idx, self.cpuct)

    def revert_virtual_loss_and_update(self, child_idx, virtual_loss, value):
        # revert the virtual loss effect and apply the backpropagated value of its child node
        with self.lock:
            self.n_sum -= virtual_loss - 1
            self.n[child_idx] -= virtual_loss - 1
            self.w[child_idx] += virtual_loss + value

            self.q[child_idx] = self.w[child_idx] / self.n[child_idx]
            #parent_node.update_u_for_child(child_idx, self.cpuct)
            self.nb_total_expanded_child_nodes += 1
            self.nb_expandable_child_nodes += self.nb_direct_child_nodes

