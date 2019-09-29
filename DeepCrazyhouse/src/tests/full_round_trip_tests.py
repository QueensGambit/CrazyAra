"""
@file: FullRoundTripTests.py
Created on 22.08.18
@project: DeepCrazyhouse
@author: queensgambit

Loads the plane representation for the test dataset and iterates through all board positions and moves.
"""

import unittest
from multiprocessing import Pool
from copy import deepcopy
import logging
import chess
import chess.pgn
import numpy as np
from DeepCrazyhouse.src.domain.variants.input_representation import planes_to_board
from DeepCrazyhouse.src.domain.variants.output_representation import policy_to_move
from DeepCrazyhouse.src.preprocessing.pgn_to_planes_converter import PGN2PlanesConverter
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset

# import the Colorer to have a nicer logging printout
from DeepCrazyhouse.src.runtime.color_logger import enable_color_logging

enable_color_logging()


def board_single_game(params_inp):
    """
    Iterates through a game based on the pgn file and compares the board of the pgn file
     with reconvert plane representation.

    :param params_inp: Tuple containing the pgn, plane_representation and the start index of the game
    :return: Boolean indicating if the all board states have been equal and the start_idx of the pgn
    """
    pgn, x_test, start_idx = params_inp

    pgn = deepcopy(pgn)
    cur_game = chess.pgn.read_game(pgn)

    all_ok = True

    # Iterate through all moves and play them on a board.
    board = cur_game.board()

    for i, move in enumerate(cur_game.main_line()):

        x_test_single_img = np.expand_dims(x_test[i], axis=0)
        mat_board = planes_to_board(x_test_single_img[0])

        cur_ok = board == mat_board
        all_ok = all_ok and cur_ok

        if not cur_ok:
            logging.error("mat_board != board: - idx: %d", i)
            logging.error("%s -> mat_board.fen", mat_board.fen)
            logging.error("%s -> board.fen", board.fen)

        board.push(move)

    return all_ok, start_idx


def moves_single_game(params_inp):
    """
    Iterates over all moves of a given pgn file and compares the reconverted policy representation with
    the move stored in the pgn file

    :param params_inp: pgn file, policy vector, starting index of the game
    :return: Boolean indicating if the all moves have been equal in uci notation and the start_idx of the pgn
    """
    pgn, yp_test, start_idx = params_inp

    all_ok = True

    pgn = deepcopy(pgn)
    cur_game = chess.pgn.read_game(pgn)

    # Iterate through all moves and play them on a board.
    board = cur_game.board()

    for i, move in enumerate(cur_game.main_line()):

        # get the move in python chess format based on the policy representation
        converted_move = policy_to_move(yp_test[i], is_white_to_move=board.turn)

        cur_ok = converted_move == move
        all_ok = all_ok and cur_ok

        if not cur_ok:
            logging.error("mat_move != move: %s - %s", converted_move, move)

        board.push(move)

    return all_ok, start_idx


class FullRoundTripTests(unittest.TestCase):  # Too many instance attributes (10/7)
    """ Load all games from the pgn and test all moves"""

    def __init__(self, part_id=0, *args, **kwargs):
        """
        Constructor
        :param part_id: Part id to choose for file selection. This way you can test different variants at a time.
        :param args:
        :param kwargs:
        """
        super(FullRoundTripTests, self).__init__(*args, **kwargs)
        logging.info("loading test dataset...")
        self._s_idcs_test, self._x_test, self._yv_test, self._yp_test, _, self._pgn_datasets_test = load_pgn_dataset(
            dataset_type="test", part_id=part_id, print_statistics=True, normalize=False, print_parameters=True
        )
        logging.info("loading test pgn file...")
        self._pgn_filename = self._pgn_datasets_test["parameters/pgn_name"][0].decode("UTF8")
        self._batch_size = self._pgn_datasets_test["parameters/batch_size"][0]
        # self._min_elo_both = self._pgn_datasets_test["parameters/min_elo_both"][0]
        # Rating cap at 90% cumulative rating for all varaints
        self._min_elo_both = {
            "Crazyhouse": 2000,
            "Chess960": 1950,
            "King of the Hill": 1925,
            "Three-check": 1900,
            "Antichess": 1925,
            "Atomic": 1900,
            "Horde": 1900,
            "Racing Kings": 1900
        }
        self._start_indices = self._pgn_datasets_test["start_indices"]

        converter = PGN2PlanesConverter(
            limit_nb_games_to_analyze=0,
            nb_games_per_file=self._batch_size,
            max_nb_files=1,
            min_elo_both=self._min_elo_both,
            termination_conditions=["Normal"],
            log_lvl=logging.DEBUG,
            compression="lz4",
            clevel=5,
            dataset_type="test",
            first_pgn_to_analyze=self._pgn_filename
        )
        self._all_pgn_sel, _, _, _, _ = converter.filter_pgn()
        print(len(self._all_pgn_sel))

    def test_board_states(self):
        """ Loads all games from the pgn file and calls the board_single_game() routine """
        logging.info("start board test...")
        logging.info("preparing input parameter...")
        params_inp = []  # create a param input list which will concatenate the pgn with it's corresponding game index
        for i in range(self._batch_size):
            if i < self._batch_size - 1:
                # select all board positions given by the start index to the start index of the next game
                x_test = self._x_test[self._start_indices[i] : self._start_indices[i + 1], :, :, :]
            else:
                # for the last batch only take the remaining items in the vector
                x_test = self._x_test[self._start_indices[i] :, :, :, :]

            params_inp.append((self._all_pgn_sel[i], x_test, self._start_indices[i]))

        pool = Pool()

        logging.info("start board test...")
        for game_ok, start_idx in pool.map(board_single_game, params_inp):
            self.assertTrue(game_ok)
            if game_ok:
                logging.debug("Board States - Game StartIdx %d [OK]", start_idx)
            else:
                logging.error("Board States - Game StartIdx %d [NOK]", start_idx)

        pool.close()
        pool.join()
        logging.info("board test done...")

    def test_moves(self):
        """ Loads all moves from all games in the pgn file and calls the moves_single_game() routine"""
        logging.info("start move comparision test...")
        logging.info("preparing input parameter...")

        params_inp = []  # create a param input list which will concatenate the pgn with it's corresponding game index
        for i in range(self._batch_size):
            if i < self._batch_size - 1:
                yp_test = self._yp_test[self._start_indices[i] : self._start_indices[i + 1], :]
            else:
                yp_test = self._yp_test[self._start_indices[i] :, :]

            params_inp.append((self._all_pgn_sel[i], yp_test, self._start_indices[i]))

        pool = Pool()
        logging.info("start board test...")
        for game_ok, start_idx in pool.map(moves_single_game, params_inp):
            self.assertTrue(game_ok)
            if game_ok is True:
                logging.debug("Moves - Game StartIdx %d [OK]", start_idx)
            else:
                logging.error("Moves - Game StartIdx %d [NOK]", start_idx)

        pool.close()
        pool.join()
        logging.info("move comparision test done...")


if __name__ == "__main__":
    # test out all supported variants
    nb_variants = 8
    for part_id in range(nb_variants):
        t = FullRoundTripTests(part_id=0)
        t.test_board_states()
        t.test_moves()
