"""
@file: game_state.py
Last changed on 16.01.19
@project: crazy_ara_cleaning
@author: queensgambit and matuiss

File to group everything to recognize the game state
"""
import chess

## Variants Includes:

# 0 - "is960", board.chess960 = True (lichess: "Chess960")
# from chess import Board(chess960=True), CrazyhouseBoard(chess960=True), ...

# variants: In the following order. Names in "" are given as uci_variant name:
# 1 - "chess" (lichess: "Standard")
# from chess import Board
# 2 - "crazyhouse" (lichess: "Crazyhouse")
# from chess.variant import CrazyhouseBoard
# 3 - "kingofthehill" (lichess: "King of the Hill")
# from chess.variant import KingOfTheHillBoard
# 4- "3check" (lichess: "Three-check")
# from chess.variant import ThreeCheckBoard
# 5- "giveaway" (lichess: "Antichess")
# from chess.variant import GiveawayBoard
# 6- "atomic" (lichess: "Atomic")
# from chess.variant import AtomicBoard
# 7- "horde" (lichess: "Horde")
# from chess.variant import HordeBoard
# 8- "racingkings" (lichess: "Racing Kings")
# from chess.variant import RacingKingsBoard

from DeepCrazyhouse.src.domain.variants.input_representation import board_to_planes
from DeepCrazyhouse.src.domain.abstract_cls.abs_game_state import AbsGameState
from DeepCrazyhouse.configs.main_config import main_config


class GameState(AbsGameState):
    """File to group everything to recognize the game state"""

    def __init__(self, board: chess.Board):
        AbsGameState.__init__(self, board)

    def apply_move(self, move: chess.Move):
        """ Apply the move on the board"""
        self.board.push(move)

    def get_state_planes(self):
        """Transform the current board state to a plane"""
        return board_to_planes(self.board, board_occ=self._board_occ, normalize=True, mode=main_config['mode'])

    def get_pythonchess_board(self):
        """ Get the board by calling a method"""
        return self.board

    def is_draw(self):
        """ Check if you can claim a draw - its assumed that the draw is always claimed """
        return self.board.is_variant_draw() or self.can_claim_threefold_repetition() or self.board.can_claim_fifty_moves()
        # return self.board.can_claim_draw()

    def can_claim_threefold_repetition(self):
        """
        Custom implementation for threefold-repetition check which uses the board_occ variable.
        :return: True if claim is legal else False
        """
        return self._board_occ >= 2

    def is_win(self):
        """ Check if you can claim the win by checkmate"""
        if self.board.uci_variant in ["giveaway"]:
            return self.board.is_variant_win()
        else:
            raise NotImplementedError

    def is_loss(self):
        """ Check if the current player to move has lost due checkmate or the variant_loss definition"""
        # only a is_won() and no is_lost() function is needed because the game is over
        if self.board.uci_variant in ["crazyhouse", "chess"]:
            return self.board.is_checkmate()  # after the player found checkmate successfully
        elif self.board.uci_variant in ["kingofthehill", "3check", "horde", "atomic"]:
            return self.board.is_checkmate() or self.board.is_variant_loss()
        elif self.board.uci_variant in ["giveaway", "racingkings"]:
            return self.board.is_variant_loss()
        else:
            raise Exception("Unhandled variant: %s" % self.board.uci_variant)

    def is_variant_loss(self):
        return self.board.is_variant_loss()

    def get_legal_moves(self):
        """ Returns the legal moves based on current board state"""
        return [*self.board.legal_moves]  # is same as list(self.board.legal_moves)

    def is_white_to_move(self):
        """ Returns true if its whites turn"""
        return self.board.turn

    def __str__(self):
        return self.board.fen()

    def new_game(self):
        """ Create a new board on the starting position"""
        self.board.reset()
        self._fen_dic = {}

    def set_fen(self, fen):  # , remember_state=True
        """ Returns the fen of the current state"""
        self.board.set_fen(fen)

    def is_check(self):
        """ Check if the king of the player of the turn is in check"""
        return self.board.is_check()

    def are_pocket_empty(self):
        """ Checks if at least one player has a piece available in their pocket """
        return self.uci_variant == "crazyhouse" and not self.board.pockets[chess.WHITE] and not self.board.pockets[chess.BLACK]

    def is_variant_end(self):
        """ Checks if the current game state is a terminal state"""
        return self.is_loss() or self.is_draw()
