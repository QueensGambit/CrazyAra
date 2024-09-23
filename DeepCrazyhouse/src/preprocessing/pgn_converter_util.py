"""
@file: pgn_converter.py
Created on 09.06.18
@project: DeepCrazyhouse
@author: queensgambit

Converts a given board state defined by a python-chess object to the plane representation which can be learned by a CNN
"""

import numpy as np
import logging
import chess.pgn
from DeepCrazyhouse.src.domain.variants.constants import NB_LAST_MOVES
from DeepCrazyhouse.src.domain.variants.output_representation import move_to_policy
from DeepCrazyhouse.src.domain.variants.input_representation import board_to_planes
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.src.domain.variants.game_state import mirror_policy
from DeepCrazyhouse.src.preprocessing.game_phase_detector import get_game_phase


NB_ITEMS_METADATA = 18  # constant which defines how many meta data items will be stored in a matrix
# 2019-09-28: Increased NB_ITEMS_METADATA from 17 to 18 for chess 960


def get_planes_from_pgn(params):
    """
    Wrapper of the method get_planes_from_game() which loads a pgn first and then calls get_planes_from_game().
    This method is intended to be used for multiprocessing
    :param params: (pgn, game_idx, mv_hist_len, mate_in_one)
    :return: metadata: nd.array - Numpy array which contains string type meta information about the games
             game_idx: int - id which describes the order of the games (first game starts with id=0)
             x: nd.array - All boards of a game which corresponds to the given pgn-file
             y_value: nd.array - Vector which describes the game outcome either [-1, 0, 1]
             y_policy: nd.array - Numpy matrix defining the policy distribution for each board state
             plys_to_end - array of how many plys to the end of the game for each position.
             This can be used to apply discounting
             phase_vector - array of the game phase of each position
    """
    (pgn, game_idx, mate_in_one) = params

    game = chess.pgn.read_game(pgn)

    if game is None:
        print("game is None!")

    metadata = np.zeros((1, NB_ITEMS_METADATA), dtype="S128")  # store the meta-data of the game in a buffer
    row = 0

    # add the header to the metadata dictionary for the first game
    if game_idx == 0:
        metadata = np.zeros((2, NB_ITEMS_METADATA), dtype="S128")

        for i, key in enumerate(game.headers):
            if i == NB_ITEMS_METADATA:
                logging.warning("The number of meta data items exceeded the metadata items of the current game.")
                break
            metadata[row, i] = key.encode("ascii", "ignore")
        row = 1

    # export the meta-data content
    for i, key in enumerate(game.headers):
        metadata[row, i] = game.headers[key].encode("ascii", "ignore")
        # only save the first 17 metadata attributes
        if i == NB_ITEMS_METADATA - 1:
            break

    results = get_planes_from_game(game, mate_in_one)

    return metadata, game_idx, results[0], results[1], results[2], results[3], results[4]


def get_planes_from_move_sequence(board: chess.Board, y_init, all_moves, mate_in_one=False):
    """
    Returns all plane descriptions of a given game and their corresponding target values:
    - the game outcome (-1, 0, 1)
    - the next move which will be played in each position

    :param board: Board object which is a python-chess object
    :param y_init: Evaluation of the initial board position
    :param all_moves: List of all moves to be applied to the position
    :param mate_in_one: Decide weather only to export the position before the last mate-in-one move
                        (this option is for evaluation and DEBUG purposes)
    :return: x - the position description of all moves in the game
             y_value - the target values of the scene description. Here the game outcome.
                  returns -1 if the current player lost, +1 if the current player won, 0 for draw
             y_policy - the policy vector one-hot encoded indicating the next move the player current player chose
              in this position
             plys_to_end - array of how many plys to the end of the game for each position.
              This can be used to apply discounting
             phase_vector - array of the game phase of each position
    """

    fen_dic = {}  # A dictionary which maps the fen description to its number of occurrences
    x = []
    y_value = []
    y_policy = []
    plys_to_end = []  # save the number of plys until the end of the game for each position that was considered
    phase_vector = []  # save all phases that occurred during the game

    # Iterate through all moves (except the last one) and play them on a board.
    # you don't want to push the last move on the board because you had no movement policy to learn from in this case
    # The moves get pushed at the end of the for-loop and is only used in the next loop.
    # Therefore, we can iterate over 'all' moves
    for plys, move in enumerate(all_moves):
        board_occ = 0  # by default the positions hasn't occurred before
        fen = board.fen()
        # remove the halfmove counter & move counter from this fen to make repetitions possible
        fen = fen[: fen.find(" ") + 2]
        # save the board state to the fen dictionary
        if fen in list(fen_dic.keys()):
            board_occ = fen_dic[fen]
            fen_dic[fen] += 1
        else:
            fen_dic[fen] = 1  # create a new entry
        # we insert the move i (and not i+1), because the start is the empty board position
        next_move = all_moves[plys]

        # check if you need to export a mate_in_one_scenario
        if not mate_in_one or plys == len(all_moves) - 1:

            # if specified phase is not None
            # check if the current game phase is the phase the dataset is created for

            curr_phase = get_game_phase(board, phase_definition=main_config["phase_definition"])[4]

            if main_config["phase"] is None or curr_phase == main_config["phase"]:
                # build the last move vector by putting the most recent move on top followed by the remaining past moves
                last_moves = [None] * NB_LAST_MOVES
                if plys != 0:
                    last_moves[0:min(plys, NB_LAST_MOVES)] = all_moves[max(plys-NB_LAST_MOVES, 0):plys][::-1]

                # receive the board and the evaluation of the current position in plane representation
                # We don't want to store float values because the integer datatype is cheaper,
                #  that's why normalize is set to false
                x_cur = board_to_planes(board, board_occ, normalize=False, mode=main_config["mode"], last_moves=last_moves)

                # add the evaluation of 1 position to the list
                x.append(x_cur)
                y_value.append(y_init)
                # add the next move defined in policy vector notation to the policy list
                # the network always sees the board as if he's the white player, that's the move is mirrored fro black
                y_policy.append(move_to_policy(next_move, mirror_policy=mirror_policy(board)))
                plys_to_end.append(len(all_moves) - 1 - plys)

                phase_vector.append(curr_phase)

        y_init *= -1  # flip the y_init value after each move
        board.push(move)  # push the next move on the board

    # check if there has been any moves and stack the lists
    if x and y_value and y_policy:
        x = np.stack(x, axis=0)
        y_value = np.stack(y_value, axis=0)
        y_policy = np.stack(y_policy, axis=0)

    return x, y_value, y_policy, plys_to_end, phase_vector


def get_planes_from_game(game, mate_in_one=False):
    """
    Returns all plane descriptions of a given game and their corresponding target values:
    - the game outcome (-1, 0, 1)
    - the next move which will be played in each position

    :param game: Game handle which is a python-chess object
    (e.g. mv_hist_len = 8 means that the current position and the 7 previous positions are exported)
    :param mate_in_one: Decide weather only to export the position before the last mate-in-one move
                        (this option is for evaluation and DEBUG purposes)
    :return: x - the position description of all moves in the game
             y_value - the target values of the scene description. Here the game outcome.
                  returns -1 if the current player lost, +1 if the current player won, 0 for draw
             y_policy - the policy vector one-hot encoded indicating the next move the player current player chose
              in this position
             plys_to_end - array of how many plys to the end of the game for each position.
              This can be used to apply discounting
             phase_vector - array of the game phase of each position
    """

    board = game.board()  # get the initial board state
    # update the y value accordingly
    if board.turn == chess.WHITE:
        y_init = 1
    else:
        y_init = -1
    if game.headers["Result"] == "0-1":
        y_init *= -1
    elif game.headers["Result"] == "1/2-1/2":
        y_init = 0

    all_moves = []  # Extract all moves first and save them into a list
    for move in game.main_line():
        all_moves.append(move)

    try:
        return get_planes_from_move_sequence(board, y_init, all_moves, mate_in_one)
    except Exception:
        print("game.headers:")
        print(game.headers)
        print("game", game)

