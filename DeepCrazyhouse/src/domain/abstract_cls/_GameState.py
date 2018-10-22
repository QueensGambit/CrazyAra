"""
@file: _GameState.py
Created on 14.10.18
@project: crazy_ara_refactor
@author: queensgambit

Please describe what the content of this file is about
"""

import chess
import chess.variant


class _GameState:

    def __init__(self, board):
        self.board = board
        self._fen_dic = {}

    def apply_move(self, move: chess.Move, remember_state=False):
        self.board.push(move)

    def get_state_planes(self):

        raise NotImplementedError('get_state_planes() should return board_to_planes(self.board, 0, normalize=True)')
        return board_to_planes(self.board, 0, normalize=True)

    def get_pythonchess_board(self):
        return self.board

    def is_draw(self):
        # check if you can claim a draw - its assumed that the draw is always claimed
        return self.board.can_claim_draw()

    def is_won(self):
        raise NotImplementedError()
        # only a is_won() and no is_lost() function is needed because the game is over
        #  after the player found checkmate successfully
        return self.board.is_checkmate()

    def get_legal_moves(self):
        return self.board.legal_moves

    def is_white_to_move(self):
        return self.board.turn

    def __str__(self):
        return self.board.fen()

    def get_board_fen(self):
        # create an identifier string for the board state by dropping the move clock
        return self.board.fen()
        #return self.board.fen().rsplit(' ', 1)[0]

    def new_game(self):
        raise NotImplementedError
