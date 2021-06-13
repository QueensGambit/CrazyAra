"""
@file: label_test.py
Created on 25.09.18
@project: crazy_ara_refactor
@author: queensgambit

Tests the functionality of the LABEL and LABEL_MIRRORED list based on the conversion to board and move planes
"""

import unittest
import chess.variant
from DeepCrazyhouse.src.domain.variants.output_representation import (
    policy_to_move,
    move_to_policy,
    policy_to_moves,
    policy_to_best_move,
)
from DeepCrazyhouse.src.domain.variants.constants import LABELS, LABELS_MIRRORED
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset


class MoveRoundTripTest(unittest.TestCase):
    """ TODO: docstring"""

    def __init__(self, *args, **kwargs):
        super(MoveRoundTripTest, self).__init__(*args, **kwargs)

    def test_move_roundtrip_white(self):
        """ TODO: docstring"""
        move = chess.Move.from_uci("e2e4")

        policy_vec = move_to_policy(move, mirror_policy=False)
        mv_converted = policy_to_move(policy_vec, mirror_policy=False)

        self.assertTrue(LABELS[policy_vec.argmax()] == move.uci())
        self.assertTrue(move == mv_converted)

    def test_move_roundtrip_black(self):
        """ TODO: docstring"""
        move = chess.Move.from_uci("e7e5")

        # TODO: This does not work for racing kings
        policy_vec = move_to_policy(move, mirror_policy=True)
        mv_conv = policy_to_move(policy_vec, mirror_policy=True)

        self.assertTrue(LABELS_MIRRORED[policy_vec.argmax()] == move.uci())
        self.assertTrue(move == mv_conv)

    def test_loaded_dataset_black_move(self):
        """
        Loads the dataset file and checks the first move policy vector for black for correctness
        :return:
        """
        _, _, _, yp_val, _ = load_pgn_dataset(
            dataset_type="test", part_id=0, verbose=True, normalize=True
        )

        board = chess.variant.CrazyhouseBoard()
        # push a dummy move
        board.push_uci("e2e4")

        mv_conv0 = policy_to_move(yp_val[1], is_white_to_move=False)
        mv_conv1, prob = policy_to_best_move(board, yp_val[1])

        self.assertEqual(prob, 1, msg="The policy vector has to be one hot encoded.")

        selected_moves, move_probabilities = policy_to_moves(board, yp_val[1])
        mv_conv2 = selected_moves[0]

        self.assertGreater(move_probabilities[0], 0, msg="The move probability must be greater 0")
        self.assertEqual(move_probabilities[0], 1, msg="The policy vector has to be one hot encoded.")

        converted_moves = [mv_conv0, mv_conv1, mv_conv2]

        for mv_converted in converted_moves:
            mv_converted_is_legal = False

            # check if the move is legal in the starting position
            for move in board.legal_moves:
                if move == mv_converted:
                    mv_converted_is_legal = True

            self.assertTrue(
                mv_converted_is_legal,
                msg="Convert move %s is not a legal move in the starting position for BLACK" % mv_converted.uci(),
            )

    def test_loaded_dataset_white_move(self):
        """
        Loads the dataset file and checks the first move policy vector for white for correctness
        :return:
        """
        _, _, _, yp_val, _ = load_pgn_dataset(
            dataset_type="test", part_id=0, verbose=True, normalize=True
        )

        board = chess.variant.CrazyhouseBoard()

        mv_conv0 = policy_to_move(yp_val[1], is_white_to_move=True)
        mv_conv1, prob = policy_to_best_move(board, yp_val[1])

        self.assertEqual(prob, 1, msg="The policy vector has to be one hot encoded.")

        selected_moves, move_probabilities = policy_to_moves(board, yp_val[1])
        mv_conv2 = selected_moves[0]

        self.assertGreater(move_probabilities[0], 0, msg="The move probability must be greater 0")
        self.assertEqual(move_probabilities[0], 1, msg="The policy vector has to be one hot encoded.")

        converted_moves = [mv_conv0, mv_conv1, mv_conv2]

        for mv_converted in converted_moves:
            mv_converted_is_legal = False

            # check if the move is legal in the starting position
            for move in board.legal_moves:
                if move == mv_converted:
                    mv_converted_is_legal = True

            self.assertTrue(
                mv_converted_is_legal,
                msg="Convert move %s is not a legal move in the starting position for BLACK" % mv_converted.uci(),
            )
