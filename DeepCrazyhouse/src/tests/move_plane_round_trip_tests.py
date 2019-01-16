"""
@file: move_plane_round_trip_tests.py
Changed last on 16.01.19
@project: crazy_ara_cleaning
@author: queensgambit and matuiss2

Testing move and board plane

This file contains the test cases for testing the moving plane
"""
import unittest
import chess
from chess.variant import CrazyhouseBoard
from DeepCrazyhouse.src.domain.util import get_board_position_index, get_row_col
from DeepCrazyhouse.src.model.crazy_house.output_representation import get_move_planes  # outdated
from DeepCrazyhouse.src.model.crazy_house.move_presentation import get_move_representation  # outdated


def construct_move_from_positions(from_pos, to_pos, promotion=None, drop=None):
    """Creates the move representation(start and end placement)"""
    from_idx = get_board_position_index(from_pos[0], from_pos[1])
    to_idx = get_board_position_index(to_pos[0], to_pos[1])
    return chess.Move(from_idx, to_idx, promotion, drop)


def assert_round_trip(test_case: unittest.TestCase, board: CrazyhouseBoard, move, comment=""):
    """Assert if the move is working on the board(outdated)"""
    print(f"Asserting round-trip ({comment})")
    rt_move = get_move_representation(board, get_move_planes(move), is_white_to_move=True)  # outdated
    print(f"Given {move} -> {rt_move}")
    test_case.assertTrue(move == rt_move, "round trip was not successful")


def construct_move_and_assert(test_case, pos1, pos2, comment, promotion=None, drop=None):
    """Merge construct_move_from_positions and assert_round_trip"""
    move = construct_move_from_positions(pos1, pos2, promotion=promotion, drop=drop)
    assert_round_trip(test_case, CrazyhouseBoard(), move, comment=comment)


class MovePlaneRoundTripTests(unittest.TestCase):
    """
    Tests the conversion from moves to planes and back
    Queen moves | 56     ->  0..55
    Knight moves | 8     -> 56..63
    Under promotions | 9  -> 64..72
    Drop | 5             -> 73..77
    """

    def test_row_col_plane_index_round_trip(self):
        """Test if the row and column value are giving the right positions"""
        row = [1, 4]
        col = [0, 1]
        print(f"Testing coordinate round trip ({row},{col})")
        rt_row, rt_col = get_row_col(get_board_position_index(row, col))
        self.assertEqual(rt_row, row)
        self.assertEqual(rt_col, col)

    def test_move_planes_round_trip_given_pawn_move_expect_round_trip(self):
        """Test if the pawns are moving correctly"""
        construct_move_and_assert(self, [1, 4], [2, 4], "pawn move")

    def test_move_planes_round_trip_given_knight_move_expect_round_trip(self):
        """Test if the knights are moving correctly"""
        construct_move_and_assert(self, [4, 4], [2, 3], "knight move 0")  # A
        construct_move_and_assert(self, [4, 4], [3, 2], "knight move 1")  # B
        construct_move_and_assert(self, [4, 4], [2, 5], "knight move 2")  # C
        construct_move_and_assert(self, [4, 4], [5, 2], "knight move 3")  # D
        construct_move_and_assert(self, [4, 4], [3, 6], "knight move 4")  # E
        construct_move_and_assert(self, [4, 4], [6, 3], "knight move 5")  # F
        construct_move_and_assert(self, [4, 4], [6, 5], "knight move 6")  # G
        construct_move_and_assert(self, [4, 4], [5, 6], "knight move 7")  # h

    def test_move_planes_round_trip_given_promotion_expect_round_trip(self):
        """Test if the under promotions are working correctly"""
        construct_move_and_assert(self, [6, 4], [7, 4], "under promotion (straight)", promotion=3)
        construct_move_and_assert(self, [6, 4], [7, 3], "under promotion (left)", promotion=3)
        construct_move_and_assert(self, [6, 4], [7, 5], "under promotion (right)", promotion=3)

    def test_move_planes_round_trip_given_drop_expect_round_trip(self):
        """Test if the pieces drops are working correctly"""
        construct_move_and_assert(self, [5, 4], [5, 4], "dropping (bishop)", drop=2)
