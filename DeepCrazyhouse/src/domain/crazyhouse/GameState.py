import chess
from chess.variant import CrazyhouseBoard
from DeepCrazyhouse.src.domain.crazyhouse.input_representation import board_to_planes
from DeepCrazyhouse.src.domain.abstract_cls._GameState import _GameState
import numpy as np

class GameState(_GameState):

    def __init__(self, board=CrazyhouseBoard()):
        _GameState.__init__(self, board)
        self.board = board
        self._fen_dic = {}
        self._board_occ = 0

    def apply_move(self, move: chess.Move, remember_state=False):
        # apply the move on the board
        self.board.push(move)

        if remember_state is True:
            self._remember_board_state()

    def get_state_planes(self):
        return board_to_planes(self.board, board_occ=self._board_occ, normalize=True)
        #return np.random.random((34, 8, 8))

    def get_pythonchess_board(self):
        return self.board

    def is_draw(self):
        # check if you can claim a draw - its assumed that the draw is always claimed
        return self.board.can_claim_draw()

    def is_won(self):
        # only a is_won() and no is_lost() function is needed because the game is over
        #  after the player found checkmate successfully
        return self.board.is_checkmate()

    def get_legal_moves(self):
        return self.board.legal_moves

    def is_white_to_move(self):
        return self.board.turn

    def __str__(self):
        return self.board.fen()

    def new_game(self):
        self.board = CrazyhouseBoard()
        self._fen_dic = {}

    def set_fen(self, fen, remember_state=True):
        self.board.set_fen(fen)

        if remember_state is True:
            self._remember_board_state()

    def _remember_board_state(self):
        fen = self.board.board_fen()
        if fen in self._fen_dic:
            # create a new entry in the dictionary if there exists one
            self._fen_dic[fen] += 1
        else:
            # create a new entry in the dictionary if there exists one
            self._fen_dic[fen] = 1
        # receive the number of occurrence given the fen list
        self._board_occ = self._fen_dic[fen] - 1

    def are_pocket_empty(self):
        """
        Checks wether at least one player has a piece available in their pocket
        :return:
        """

        return len(self.board.pockets[chess.WHITE]) == 0 and len(self.board.pockets[chess.BLACK]) == 0
