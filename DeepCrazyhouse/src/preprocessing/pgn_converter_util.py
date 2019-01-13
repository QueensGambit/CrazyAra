"""
@file: pgn_converter.py
Created on 09.06.18
@project: DeepCrazyhouse
@author: queensgambit

Converts a given board state defined by a python-chess object to the plane representation which can be learned by a CNN
"""

import numpy as np
import chess.pgn
from DeepCrazyhouse.src.domain.crazyhouse.output_representation import move_to_policy
from DeepCrazyhouse.src.domain.crazyhouse.input_representation import board_to_planes

# constant which defines how many meta data items will be stored in a matrix
NB_ITEMS_METADATA = 17


def get_planes_from_pgn(params):
    """
    Wrapper of the method get_planes_from_game() which loads a pgn first and then calls get_planes_from_game().
    This method is intended to be used for multiprocessing
    :param params: (pgn, game_idx, mv_hist_len, mate_in_one)
    :return: game - game which corresponds to the given pgn-file
             game_idx - id which describes the order of the games (first game starts with id=0)

    """
    (pgn, game_idx, mate_in_one) = params

    game = chess.pgn.read_game(pgn)

    if game is None:
        print("game is None!")

    # store the meta-data of the game in a buffer
    metadata = np.zeros((1, NB_ITEMS_METADATA), dtype="S128")
    row = 0

    # add the header to the metadata dictionary for the first game
    if game_idx == 0:
        metadata = np.zeros((2, NB_ITEMS_METADATA), dtype="S128")

        for i, key in enumerate(game.headers):
            metadata[row, i] = key.encode("ascii", "ignore")
        row = 1

    # export the meta-data content
    for i, key in enumerate(game.headers):
        metadata[row, i] = game.headers[key].encode("ascii", "ignore")
        # only save the first 17 metadata attributes
        if i == NB_ITEMS_METADATA - 1:
            break

    # get the image planes and targets
    x, y_value, y_policy = get_planes_from_game(game, mate_in_one)

    return metadata, game_idx, x, y_value, y_policy


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
    """

    # fen dic is a dictionary which maps the fen description to its number of occurrences
    fen_dic = {}

    x = []
    y_value = []
    y_policy = []

    # get the initial board state
    board = game.board()

    # create the target value
    # default is a draw
    y_init = 0

    # update the y value accordingly
    if game.headers["Result"] == "1-0":
        if board.turn == chess.WHITE:
            y_init = 1
        else:
            y_init = -1
    elif game.headers["Result"] == "0-1":
        if board.turn == chess.WHITE:
            y_init = -1
        else:
            y_init = 1

    # Extract all moves first and save them into a list
    all_moves = []
    for move in game.main_line():
        all_moves.append(move)
    # Iterate through all moves (except the last one) and play them on a board.
    # you don't want to push the last move on the board because you had no movement policy to learn from in this case
    # The moves get pushed at the end of the for-loop and is only used in the next loop.
    # Therefore we can iterate over 'all' moves
    for i, move in enumerate(all_moves):
        # by default the positions hasn't occurred before
        board_occ = 0
        fen = board.fen()
        # remove the halfmove counter & move counter from this fen to make repetitions possible
        fen = fen[: fen.find(" ") + 2]
        # save the board state to the fen dictionary
        if fen in list(fen_dic.keys()):
            fen_dic[fen] += 1
            board_occ = fen_dic[fen]
        else:
            # create a new entry
            fen_dic[fen] = 1
        # we insert the move i (and not i+1), because the start is the empty board position
        y_policy = all_moves[i]

        # check if you need to export a mate_in_one_scenario
        if mate_in_one is False or i == len(all_moves) - 1:
            # receive the board and the evaluation of the current position in plane representation
            # We don't want to store float values because the integer datatype is cheaper,
            #  that's why normalize is set to false
            x_cur = board_to_planes(board, board_occ, normalize=False)
            # add the evaluation of 1 position to the list
            x.append(x_cur)
            y_value.append(y_init)
            # add the next move defined in policy vector notation to the policy list
            # the network always sees the board as if he's the white player, that's the move is mirrored fro black
            y_policy.append(move_to_policy(y_policy, is_white_to_move=board.turn))

        # flip the y_init value after each move
        y_init *= -1

        # push the next move on the board
        board.push(move)

    # check if there has been any moves
    if x and y_value and y_policy:
        x = np.stack(x, axis=0)
        y_value = np.stack(y_value, axis=0)
        y_policy = np.stack(y_policy, axis=0)
    else:
        print("game.headers:")
        print(game.headers)
        raise Exception("The given pgn file's mainline is empty!")

    return x, y_value, y_policy
