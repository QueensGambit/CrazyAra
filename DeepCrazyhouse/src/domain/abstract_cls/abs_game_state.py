"""
@file: _GameState.py
Created on 14.10.18
@project: crazy_ara_refactor
@author: queensgambit

Please describe what the content of this file is about
"""

from abc import ABC, abstractmethod
import chess.variant


class AbsGameState(ABC):
    """Abstract class for the GameState child class"""

    def __init__(self, board):
        self.uci_variant = board.uci_variant
        self.board = board
        self._fen_dic = {}
        self._board_occ = 0

    @abstractmethod
    def apply_move(self, move: chess.Move):  # , remember_state=False):
        """Force the child to implement apply_move method"""

    @abstractmethod
    def get_state_planes(self):
        """Force the child to implement get_state_planes method"""
        # return board_to_planes(self.board, 0, normalize=True)

    @abstractmethod
    def get_pythonchess_board(self):
        """ Force the child to implement get_pythonchess_board method"""

    def is_draw(self):
        """ Check if you can claim a draw - it's assumed that the draw is always claimed """
        return self.board.can_claim_draw()

    @abstractmethod
    def is_win(self):
        """Force the child to implement is_win method"""

    @abstractmethod
    def is_loss(self):
        """Force the child to implement is_loss method"""

    def get_legal_moves(self):
        """ Find legal moves based on the board state"""
        return self.board.legal_moves

    @abstractmethod
    def is_white_to_move(self):
        """Force the child to implement is_white_to_move method"""

    @abstractmethod
    def mirror_policy(self) -> bool:
        """Force the child to implement mirror_policy method"""

    def __str__(self):
        return self.board.fen()

    def get_board_fen(self):
        """ Create an identifier string for the board state"""
        return self.board.fen()

    def get_transposition_key(self):
        """
        Returns an identifier key for the current board state excluding move counters.
        Calling ._transposition_key() is faster than .fen()
        :return:
        """
        return self.board._transposition_key()  # protected member access(pylint error)

    @abstractmethod
    def new_game(self):
        """Force the child to implement new_game method"""

    def get_halfmove_counter(self):
        """ Return the number of steps towards the 40 move rule without progress """
        return self.board.halfmove_clock

    def get_fullmove_number(self):
        """ Returns the current full move number"""
        return self.board.fullmove_number

    @abstractmethod
    def is_variant_end(self):
        """ Checks if the current game state is a terminal state"""
        raise NotImplementedError("is_variant_end() hasn't been implemented yet")
