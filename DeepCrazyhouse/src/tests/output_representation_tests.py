import unittest
import chess
import numpy as np
from DeepCrazyhouse.src.model.crazy_house.output_representation import (
    get_plane_offset_id_from_queen_movement_vector,
    get_move_planes,
    get_board_position_index,
)


class OutputRepresentationTests(unittest.TestCase):
    """
    Tests the correct behaviour for converting Move objects into their corresponding plane representation
    Queen moves | 56     ->  0..55
    Knight moves | 8     -> 56..63
    Underpromotions | 9  -> 64..72
    Drop | 5             -> 73..77
    """

    def test_get_plane_offset_given_id_from_queen_movement_vector_expect_correct_indices(self):
        plane_ids = np.zeros(56)
        for direction in [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]:
            for length in range(1, 8):
                movement_vector = np.array(direction) * length
                plane_id = get_plane_offset_id_from_queen_movement_vector(movement_vector)

                # check that id lies within [0..55]
                self.assertGreaterEqual(plane_id, 0, "id has to be in range [0..55]")
                self.assertLess(plane_id, 56, "id has to be in range [0..55]")

                plane_ids[plane_id] += 1

        # assert that all ids where selected once (and only once)
        self.assertTrue(np.all(plane_ids == 1), "some ids where selected never or more than once")

    def test_get_move_planes_given_all_positions_expect_correct_position_on_plane(self):
        for col in range(8):
            for row in range(8):
                from_idx = get_board_position_index(row, col)
                to_idx = get_board_position_index(row, col + 1)
                planes = get_move_planes(chess.Move(from_idx, to_idx))
                flat_plane = np.sum(planes, axis=0)  # flatten all planes onto a single board

                # test that only a single position is selected
                self.assertTrue(np.sum(flat_plane) == 1, "more than one position was selected")

                # test that the correct position is selected
                self.assertTrue(flat_plane[row, col] == 1, "an incorrect position was selected")

    def test_get_move_planes_given_knight_moves_expect_correct_plane_selection(self):
        # place the knight in the center of the board
        # and test all combinations
        col = 4
        row = 4

        aggregated_selected_boards = np.zeros(8)

        for x in [-1, 1]:
            for y in [-1, 1]:
                for x_dominates in [True, False]:
                    movement_vector = [y * (1 if x_dominates else 2), x * (2 if x_dominates else 1)]

                    from_idx = get_board_position_index(row, col)
                    to_idx = get_board_position_index(row + movement_vector[0], col + movement_vector[1])
                    selected_boards = np.sum(get_move_planes(chess.Move(from_idx, to_idx))[56:64, :, :], axis=(1, 2))
                    # test that only a single position is selected
                    self.assertTrue(np.sum(selected_boards) == 1, "more than one board was selected")

                    aggregated_selected_boards += selected_boards

        # test that all ids where selected once
        self.assertTrue(np.all(aggregated_selected_boards == 1), "some boards where selected never or more than once")

    def test_get_move_planes_given_underpromotions_moves_expect_correct_plane_selection(self):
        col = 4
        row = 6
        aggregated_selected_boards = np.zeros(9)

        for piece in ["n", "b", "r"]:
            for x_direction in [-1, 0, 1]:
                movement_vector = [1, x_direction]

                from_idx = get_board_position_index(row, col)
                to_idx = get_board_position_index(row + movement_vector[0], col + movement_vector[1])
                selected_boards = np.sum(get_move_planes(chess.Move(from_idx, to_idx, piece))[64:73, :, :], axis=(1, 2))
                # test that only a single position is selected
                self.assertTrue(np.sum(selected_boards) == 1, "more than one board was selected")

                aggregated_selected_boards += selected_boards

        # test that all ids where selected once
        self.assertTrue(np.all(aggregated_selected_boards == 1), "some boards where selected never or more than once")

    def test_get_move_planes_given_queen_promotion_moves_expect_correct_plane_selection(self):
        col = 4
        row = 6
        aggregated_selected_boards = np.zeros(56)

        for x_direction in [-1, 0, 1]:
            movement_vector = [1, x_direction]

            from_idx = get_board_position_index(row, col)
            to_idx = get_board_position_index(row + movement_vector[0], col + movement_vector[1])

            selected_boards = np.sum(get_move_planes(chess.Move(from_idx, to_idx, "Q"))[0:56, :, :], axis=(1, 2))
            # test that only a single position is selected
            self.assertTrue(np.sum(selected_boards) == 1, "more than one board was selected")

            aggregated_selected_boards += selected_boards

        # test that 3 ids where selected once
        selected = aggregated_selected_boards[aggregated_selected_boards > 0]
        self.assertTrue(len(selected) == 3, "more or less than 3 boards where selected")
        self.assertTrue(np.all(selected == 1), "some boards where selected never or more than once")

    def test_get_move_planes_given_drops_expect_correct_plane_selection(self):
        aggregated_selected_boards = np.zeros(5)

        for piece in ["p", "n", "b", "r", "q"]:
            from_idx = get_board_position_index(4, 4)
            to_idx = get_board_position_index(4, 4)
            selected_boards = np.sum(
                get_move_planes(chess.Move(from_idx, to_idx, None, piece))[73:78, :, :], axis=(1, 2)
            )

            # test that only a single position is selected
            self.assertTrue(np.sum(selected_boards) == 1, "more than one board was selected")

            aggregated_selected_boards += selected_boards

        # test that all ids where selected once
        self.assertTrue(np.all(aggregated_selected_boards == 1), "some boards where selected never or more than once")
