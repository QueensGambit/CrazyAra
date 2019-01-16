"""
@file: game_state.py
Last changed on 16.01.19
@project: crazy_ara_cleaning
@author: queensgambit and matuiss

File to group everything to recognize the game state
"""
import chess
from chess.variant import CrazyhouseBoard
from DeepCrazyhouse.src.domain.crazyhouse.input_representation import board_to_planes
from DeepCrazyhouse.src.domain.abstract_cls.abs_game_state import AbsGameState


class GameState(AbsGameState):
    """File to group everything to recognize the game state"""

    def __init__(self, board=CrazyhouseBoard()):
        AbsGameState.__init__(self, board)
        self.board = board
        self._fen_dic = {}
        self._board_occ = 0

    def apply_move(self, move: chess.Move):  # , remember_state=False):
        """ Apply the move on the board"""
        self.board.push(move)

        # if remember_state is True:
        #    self._remember_board_state()

    def get_state_planes(self):
        """Transform the current board state to a plane"""
        return board_to_planes(self.board, board_occ=self._board_occ, normalize=True)
        # return np.random.random((34, 8, 8))

    def get_pythonchess_board(self):
        """ Get the board by calling a method"""
        return self.board

    def is_draw(self):
        """ Check if you can claim a draw - its assumed that the draw is always claimed """
        return self.can_claim_threefold_repetition() or self.board.can_claim_fifty_moves()
        # return self.board.can_claim_draw()

    def can_claim_threefold_repetition(self):
        """
        Custom implementation for threefold-repetition check which uses the board_occ variable.
        :return: True if claim is legal else False
        """
        return self._board_occ >= 2

    def is_won(self):
        """ Check if you can claim the win by checkmate"""
        # only a is_won() and no is_lost() function is needed because the game is over
        return self.board.is_checkmate()  # after the player found checkmate successfully

    def get_legal_moves(self):
        """ Returns the legal moves based on current board state"""
        # return list(self.board.legal_moves)
        legal_moves = []
        for move in self.board.generate_legal_moves():
            legal_moves.append(move)
        return legal_moves

    def is_white_to_move(self):
        """ Returns true if its whites turn"""
        return self.board.turn

    def __str__(self):
        return self.board.fen()

    def new_game(self):
        """ Create a new board on the starting position"""
        self.board = CrazyhouseBoard()
        self._fen_dic = {}

    def set_fen(self, fen):  # , remember_state=True
        """ Returns the fen of the current state"""
        self.board.set_fen(fen)

        # if remember_state is True:
        #    self._remember_board_state()

    # def _remember_board_state(self):
    # calculate the transposition key
    #    transposition_key = self.get_transposition_key()
    # update the number of board occurrences
    # self._board_occ = self._transposition_table[transposition_key]
    # increase the counter for this transposition key
    #    self._transposition_table.update((transposition_key,))

    def is_check(self):
        """ Check if the king of the player of the turn is in check"""
        return self.board.is_check()

    def are_pocket_empty(self):
        """ Checks if at least one player has a piece available in their pocket """
        return not self.board.pockets[chess.WHITE] and not self.board.pockets[chess.BLACK]
