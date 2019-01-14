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
    def __init__(self, board):
        self.board = board
        self._fen_dic = {}

    def apply_move(self, move: chess.Move):  # , remember_state=False):
        self.board.push(move)

    @abstractmethod
    def get_state_planes(self):
        pass
        # return board_to_planes(self.board, 0, normalize=True)

    def get_pythonchess_board(self):
        return self.board

    def is_draw(self):
        # check if you can claim a draw - its assumed that the draw is always claimed
        return self.board.can_claim_draw()

    @abstractmethod
    def is_won(self):
        pass
        # only a is_won() and no is_lost() function is needed because the game is over
        #  after the player found checkmate successfully
        # return self.board.is_checkmate()

    def get_legal_moves(self):
        return self.board.legal_moves

    def is_white_to_move(self):
        return self.board.turn

    def __str__(self):
        return self.board.fen()

    def get_board_fen(self):
        # create an identifier string for the board state
        return self.board.fen()

    def get_transposition_key(self):
        """
        Returns an identifier key for the current board state excluding move counters.
        Calling ._transposition_key() is faster than .fen()
        :return:
        """
        return self.board._transposition_key()

    @abstractmethod
    def new_game(self):
        pass

    def get_halfmove_counter(self):
        return self.board.halfmove_clock

    def get_fullmove_number(self):
        return self.board.fullmove_number
