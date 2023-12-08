"""
@file: validate_train_results_util.py
Created on 03.12.23
@project: CrazyAra
@author: queensgambit

Utility functions for validate_train_results.ipynb
"""
import sys
sys.path.insert(0, '../../../')
import chess
import numpy as np
from copy import deepcopy
import chess.svg
from IPython.display import SVG

import torch
from DeepCrazyhouse.src.domain.variants.input_representation import planes_to_board
from DeepCrazyhouse.src.domain.variants.output_representation import policy_to_moves, \
    policy_to_move
from DeepCrazyhouse.configs.train_config import TrainConfig

from DeepCrazyhouse.src.training.lr_schedules.lr_schedules import *
from DeepCrazyhouse.src.domain.variants.plane_policy_representation import FLAT_PLANE_IDX


def predict_single(net, x, ctx, select_policy_from_plane=False):
    """Evaluate a given position in numpy representation and return its prediction output."""
    out = [None, None]

    with torch.no_grad():
        pred = net(torch.Tensor(np.expand_dims(x, axis=0)).to(ctx))
        out[0] = pred[0].to(torch.device("cpu")).numpy()
        out[1] = pred[1].to(torch.device("cpu")).softmax(dim=1).numpy()
    if select_policy_from_plane:
        out[1] = out[1][:, FLAT_PLANE_IDX]

    return out


def eval_pos(net, x_mate, yp_mate, ctx, tc: TrainConfig, mode: int, verbose: bool = False):
    """Evaluates a given mating position in numpy representation and returns its prediction."""
    select_policy_from_plane = tc.select_policy_from_plane
    board = planes_to_board(x_mate, normalized_input=tc.normalize, mode=mode)
    if verbose is True:
        print("{0}'s turn".format(chess.COLOR_NAMES[board.turn]))
        if board.uci_variant == "crazyhouse":
            print("black/white {0}".format(board.pockets))
    pred = predict_single(net, x_mate, ctx, select_policy_from_plane=select_policy_from_plane)

    true_move = policy_to_move(yp_mate, mirror_policy=board.turn == chess.BLACK)

    opts = 5
    pred_moves, probs = policy_to_moves(board, pred[1][0])
    pred_moves = pred_moves[:opts]

    legal_move_cnt = board.legal_moves.count()
    mate_move_cnt = str(board.legal_moves).count('#')

    is_mate_5_top = False

    for pred_move in pred_moves:
        board_5_top = deepcopy(board)
        board_5_top.push(pred_move)
        if board_5_top.is_checkmate() is True:
            is_mate_5_top = True
            break

    board.push(pred_moves[0])

    is_checkmate = False
    if board.is_checkmate() is True:
        is_checkmate = True

    filtered_pred = sorted(pred[1][0], reverse=True)

    if verbose is True:
        plt.barh(range(opts)[::-1], filtered_pred[:opts])
        ax = plt.gca()
        ax.set_yticks(range(opts)[::-1])
        ax.set_yticklabels(pred_moves)
        plt.title('True Move:' + str(true_move) +
                  '\nEval:' + str(pred[0][0]))
        plt.show()

    return pred, pred_moves, true_move, board, is_checkmate, is_mate_5_top, legal_move_cnt, mate_move_cnt


def clean_string(np_string):
    string = str(np_string).replace("b'", "")
    string = string.replace("'", "")
    string = string.replace('"', '')

    return string


def show_first_x_examples(nb_examples, net, site_mate, ctx, tc: TrainConfig, mode: int, x_mate, yp_mate):
    for i in range(nb_examples):
        print(clean_string(site_mate[i]))
        result = eval_pos(net, x_mate[i], yp_mate[i], ctx, tc, mode, verbose=True)
        pred, pred_moves, true_move, board, is_checkmate, is_mate_5_top, legal_move_cnt, mate_move_cnt = result
        pred_move = pred_moves[0]
        pred_arrow = chess.svg.Arrow(pred_move.from_square, pred_move.to_square)
        SVG(data=chess.svg.board(board=board, arrows=[pred_arrow], size=400))


def show_mating_fail_examples(max_nb_examples, net, site_mate, ctx, tc: TrainConfig, mode: int, x_mate, yp_mate):
    mate_missed = 0
    for i in range(1000):
        result = eval_pos(net, x_mate[i], yp_mate[i], ctx, tc, mode, verbose=True)
        pred, pred_moves, true_move, board, is_checkmate, is_mate_5_top, legal_move_cnt, mate_move_cnt = result

        if is_mate_5_top is False:
            mate_missed += 1
            print(clean_string(site_mate[i]))
            result = eval_pos(net, x_mate[i], yp_mate[i], ctx, tc, mode, verbose=True)
            pred, pred_moves, true_move, board, is_checkmate, is_mate_5_top, legal_move_cnt, mate_move_cnt = result

            pred_move = pred_moves[0]
            pred_arrow = chess.svg.Arrow(pred_move.from_square, pred_move.to_square)
            SVG(data=chess.svg.board(board=board, arrows=[pred_arrow], size=400))
        if mate_missed == max_nb_examples:
            break
