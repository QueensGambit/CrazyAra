import unittest
import chess
from chess.variant import CrazyhouseBoard
from src.domain.util import get_board_position_index, get_row_col
from DeepCrazyhouse.src.model.crazy_house.output_representation import get_move_planes
from DeepCrazyhouse.src.model.crazy_house.move_presentation import get_move_representation


def construct_move_from_positions(from_pos, to_pos, promotion=None, drop=None):
    from_idx = get_board_position_index(from_pos[0], from_pos[1])
    to_idx = get_board_position_index(to_pos[0], to_pos[1])
    return chess.Move(from_idx, to_idx, promotion, drop)


def assert_round_trip(test_case: unittest.TestCase, board: CrazyhouseBoard, move, comment=""):
    print(f"Asserting round-trip ({comment})")
    rt_move = get_move_representation(board, get_move_planes(move), is_white_to_move=True)
    print(f"Given {move} -> {rt_move}")
    test_case.assertTrue(move == rt_move, "round trip was not successful")


class MovePlaneRoundTripTests(unittest.TestCase):
    """
    Tests the conversion from moves to planes and back
    Queen moves | 56     ->  0..55
    Knight moves | 8     -> 56..63
    Under promotions | 9  -> 64..72
    Drop | 5             -> 73..77
    """

    def test_row_col_plane_index_round_trip(self):
        positions = [[1, 4], [0, 1]]
        for position in positions:
            row = position[0]
            col = position[1]

            print(f"Testing coordinate round trip ({row},{col})")
            rt_row, rt_col = get_row_col(get_board_position_index(row, col))
            self.assertEqual(rt_row, row)
            self.assertEqual(rt_col, col)

    def test_move_planes_round_trip_given_pawn_move_expect_round_trip(self):
        board = CrazyhouseBoard()

        move = construct_move_from_positions([1, 4], [2, 4])
        assert_round_trip(self, board, move, comment="pawn move")

    def test_move_planes_round_trip_given_knight_move_expect_round_trip(self):
        board = CrazyhouseBoard()

        def test_move_planes_round_trip_given_knight_move_expect_round_trip(self):
            board = CrazyhouseBoard()

            move = construct_move_from_positions([4, 4], [2, 3])
            assert_round_trip(self, board, move, comment="knight move 0")  # A

            move = construct_move_from_positions([4, 4], [3, 2])
            assert_round_trip(self, board, move, comment="knight move 1")  # B

            move = construct_move_from_positions([4, 4], [2, 5])
            assert_round_trip(self, board, move, comment="knight move 2")  # C

            move = construct_move_from_positions([4, 4], [3, 6])
            assert_round_trip(self, board, move, comment="knight move 3")  # D

            move = construct_move_from_positions([4, 4], [6, 3])
            assert_round_trip(self, board, move, comment="knight move 4")  # E

            move = construct_move_from_positions([4, 4], [5, 2])
            assert_round_trip(self, board, move, comment="knight move 5")  # F

            move = construct_move_from_positions([4, 4], [6, 5])
            assert_round_trip(self, board, move, comment="knight move 2")  # G

            move = construct_move_from_positions([4, 4], [5, 6])
            assert_round_trip(self, board, move, comment="knight move 2")  # H

    def test_move_planes_round_trip_given_promotion_expect_round_trip(self):
        board = CrazyhouseBoard()

        move = construct_move_from_positions([6, 4], [7, 4], promotion=3)
        assert_round_trip(self, board, move, comment="under promotion (straight)")

        move = construct_move_from_positions([6, 4], [7, 3], promotion=3)
        assert_round_trip(self, board, move, comment="under promotion (left)")

        move = construct_move_from_positions([6, 4], [7, 5], promotion=3)
        assert_round_trip(self, board, move, comment="under promotion (right)")

    def test_move_planes_round_trip_given_drop_expect_round_trip(self):
        board = CrazyhouseBoard()

        move = construct_move_from_positions([5, 4], [5, 4], drop=2)
        assert_round_trip(self, board, move, comment="dropping (bishop)")
