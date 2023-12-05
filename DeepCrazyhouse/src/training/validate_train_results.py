"""
@file: validate_train_results.py
Created on 03.12.23
@project: CrazyAra
@author: queensgambit

Runs an evaluation of the trained model on the test set
and some additional evaluations mostly for demonstration purposes.
"""

import os
import sys

sys.path.insert(0, '../../../')
import glob
import chess
import shutil
import logging
import numpy as np
from pathlib import Path
from copy import deepcopy

import torch

from DeepCrazyhouse.src.training.train_util import get_metrics, value_to_wdl_label, prepare_plys_label
from DeepCrazyhouse.src.domain.variants.input_representation import board_to_planes, planes_to_board
from DeepCrazyhouse.src.domain.variants.output_representation import policy_to_moves, policy_to_best_move, \
    policy_to_move
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.runtime.color_logger import enable_color_logging
from DeepCrazyhouse.configs.main_config import main_config
from DeepCrazyhouse.configs.train_config import TrainConfig, TrainObjects

from DeepCrazyhouse.src.training.lr_schedules.lr_schedules import *
from DeepCrazyhouse.src.domain.variants.plane_policy_representation import FLAT_PLANE_IDX
from DeepCrazyhouse.src.domain.variants.constants import NB_POLICY_MAP_CHANNELS, NB_LABELS, MODE_CHESS, MODE_CRAZYHOUSE


def predict_single(net, x, select_policy_from_plane=False):
    out = [None, None]

    with torch.no_grad():
        pred = net(torch.Tensor(np.expand_dims(x, axis=0)).to(ctx))
        out[0] = pred[0].to(torch.device("cpu")).numpy()
        out[1] = pred[1].to(torch.device("cpu")).softmax(dim=1).numpy()
    if select_policy_from_plane:
        out[1] = out[1][:, FLAT_PLANE_IDX]

    return out


def eval_pos(net, x_mate, yp_mate, verbose=False, select_policy_from_plane=False):
    board = planes_to_board(x_mate, normalized_input=tc.normalize, mode=mode)
    if verbose is True:
        print("{0}'s turn".format(chess.COLOR_NAMES[board.turn]))
        if board.uci_variant == "crazyhouse":
            print("black/white {0}".format(board.pockets))
    pred = predict_single(net, x_mate, select_policy_from_plane=select_policy_from_plane)

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


def main():
    print('load current best model:', model_params_path)


    load_torch_state(model, torch.optim.SGD(model.parameters(), lr=tc.max_lr), model_tar_path, tc.device_id)

    # In[ ]:

    print('best val_loss: %.5f with v_policy_acc: %.5f at k_steps_best %d' % (
    val_loss_best, val_p_acc_best, k_steps_best))

    # ## Convert to onnx

    # In[ ]:

    if tc.use_wdl and tc.use_plys_to_end:
        outputs = [main_config['value_output'] + '_output', main_config['policy_output'] + '_output',
                   main_config['auxiliary_output'] + '_output',
                   main_config['wdl_output'] + '_output', main_config['plys_to_end_output'] + '_output']
    else:
        outputs = [main_config['value_output'] + '_output', main_config['policy_output'] + '_output', ]

    if tc.framework == 'mxnet':
        convert_mxnet_model_to_onnx(best_model_arch_path, best_model_params_path,
                                    outputs,
                                    tuple(input_shape), tuple([1, 8, 16, 64]), True)
    elif tc.framework == 'pytorch':
        model_prefix = "%s-%04d" % (model_name, k_steps_best)
        with torch.no_grad():
            ctx = get_context(tc.context, tc.device_id)
            dummy_input = torch.zeros(1, input_shape[0], input_shape[1], input_shape[2]).to(ctx)
            export_to_onnx(model, 1,
                           dummy_input,
                           Path(tc.export_dir) / Path("best-model"), model_prefix, tc.use_wdl and tc.use_plys_to_end,
                           True)

    # In[ ]:

    print("Saved json, weight & onnx files of the best model to %s" % (tc.export_dir + "best-model"))

    # ## Show move predictions

    # In[ ]:

    idx = 0

    # In[ ]:

    if mode == MODE_CHESS:
        start_board = chess.Board()
    elif mode == MODE_CRAZYHOUSE:
        start_board = chess.variant.CrazyhouseBoard()
    else:
        start_board = planes_to_board(x_val[idx], normalized_input=tc.normalize, mode=mode)
    board = start_board
    print(chess.COLOR_NAMES[board.turn])
    if board.uci_variant == "crazyhouse":
        print(board.pockets)
    board

    # In[ ]:

    # In[ ]:

    if tc.framework == 'pytorch':
        net = model
        net.eval()

    # In[ ]:

    x_start_pos = board_to_planes(board, normalize=tc.normalize, mode=mode)
    pred = predict_single(net, x_start_pos, tc.select_policy_from_plane)
    pred

    # In[ ]:

    policy_to_best_move(board, yp_val[idx])

    # In[ ]:

    opts = 5
    selected_moves, probs = policy_to_moves(board, pred[1][0])
    selected_moves[:opts]

    # In[ ]:

    plt.barh(range(opts)[::-1], probs[:opts])
    ax = plt.gca()
    ax.set_yticks(range(opts)[::-1])
    ax.set_yticklabels(selected_moves[:opts])

    # In[ ]:

    board = start_board
    board.push_uci('e2e4')
    board.push_uci('e7e5')
    board.push_uci('f1c4')
    board.push_uci('b8c6')
    board.push_uci('d1h5')
    x_scholar_atck = board_to_planes(board, normalize=tc.normalize, mode=mode)
    board

    # In[ ]:

    pred = predict_single(net, x_scholar_atck, tc.select_policy_from_plane)

    selected_moves, probs = policy_to_moves(board, pred[1][0])
    plt.barh(range(opts)[::-1], probs[:opts])
    ax = plt.gca()
    ax.set_yticks(range(opts)[::-1])
    ax.set_yticklabels(selected_moves[:opts])

    # In[ ]:

    board.push(selected_moves[0])
    board

    # ### Performance on test dataset
    #

    # In[ ]:

    s_idcs_test, x_test, yv_test, yp_test, yplys_test, pgn_datasets_test = load_pgn_dataset(dataset_type='test',
                                                                                            part_id=0,
                                                                                            verbose=True,
                                                                                            normalize=True)
    test_data = get_data_loader(x_test, yv_test, yp_test, yplys_test, tc, shuffle=False)

    # In[ ]:

    if tc.framework == 'mxnet':
        metrics = metrics_gluon

    evaluate_metrics(to.metrics, test_data, net, nb_batches=None, sparse_policy_label=tc.sparse_policy_label, ctx=ctx,
                     apply_select_policy_from_plane=tc.select_policy_from_plane, use_wdl=tc.use_wdl,
                     use_plys_to_end=tc.use_plys_to_end)

    # ### Show result on mate-in-one problems

    # In[ ]:

    s_idcs_mate, x_mate, yv_mate, yp_mate, yplys_mate, pgn_dataset_mate = load_pgn_dataset(dataset_type='mate_in_one',
                                                                                           part_id=0,
                                                                                           verbose=True,
                                                                                           normalize=tc.normalize)
    yplys_mate = np.ones(len(yv_mate))
    mate_data = get_data_loader(x_mate, yv_mate, yp_mate, yplys_mate, tc, shuffle=False)

    # ### Mate In One Performance

    # In[ ]:

    evaluate_metrics(to.metrics, mate_data, net, nb_batches=None, sparse_policy_label=tc.sparse_policy_label, ctx=ctx,
                     apply_select_policy_from_plane=tc.select_policy_from_plane, use_wdl=tc.use_wdl,
                     use_plys_to_end=tc.use_plys_to_end)

    # ### Show some example mate problems

    # In[ ]:

    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.ast_node_interactivity = "all"

    # ### Evaluate Performance

    # In[ ]:


    # In[ ]:

    nb_pos = len(x_mate)
    mates_found = []
    mates_5_top_found = []
    legal_mv_cnts = []
    mate_mv_cnts = []

    for i in range(nb_pos):
        pred, pred_moves, true_move, board, is_mate, is_mate_5_top, legal_mv_cnt, mate_mv_cnt = eval_pos(net, x_mate[i],
                                                                                                         yp_mate[i],
                                                                                                         select_policy_from_plane=tc.select_policy_from_plane)
        mates_found.append(is_mate)
        legal_mv_cnts.append(legal_mv_cnt)
        mate_mv_cnts.append(mate_mv_cnt)
        mates_5_top_found.append(is_mate_5_top)

    # In[ ]:

    np.array(mate_mv_cnts).mean()

    # In[ ]:

    np.array(legal_mv_cnts).mean()

    # ### Random Guessing Baseline

    # In[ ]:

    np.array(mate_mv_cnts).mean() / np.array(legal_mv_cnts).mean()

    # ### Prediciton Performance

    # In[ ]:

    print('mate_in_one_acc:', sum(mates_found) / nb_pos)

    # In[ ]:

    sum(mates_5_top_found) / nb_pos

    # In[ ]:

    pgn_dataset_mate.tree()

    # In[ ]:

    metadata = np.array(pgn_dataset_mate['metadata'])
    metadata[0, :]
    metadata[1, :]

    # In[ ]:

    site_mate = metadata[1:, 1]

    # In[ ]:

    def clean_string(np_string):
        string = str(site_mate[i]).replace("b'", "")
        string = string.replace("'", "")
        string = string.replace('"', '')

        return string

    # In[ ]:

    import chess.svg
    from IPython.display import SVG, HTML

    # ## Show the result of the first 17 examples

    # In[ ]:

    for i in range(17):
        print(clean_string(site_mate[i]))
        pred, pred_moves, true_move, board, is_checkmate, is_mate_5_top, legal_move_cnt, mate_move_cnt = eval_pos(net,
                                                                                                                  x_mate[
                                                                                                                      i],
                                                                                                                  yp_mate[
                                                                                                                      i],
                                                                                                                  verbose=True,
                                                                                                                  select_policy_from_plane=tc.select_policy_from_plane)
        pred_move = pred_moves[0]
        pred_arrow = chess.svg.Arrow(pred_move.from_square, pred_move.to_square)
        SVG(data=chess.svg.board(board=board, arrows=[pred_arrow], size=400))

    # ## Show examples where it failed

    # In[ ]:

    mate_missed = 0
    for i in range(1000):
        pred, pred_moves, true_move, board, is_checkmate, is_mate_5_top, legal_move_cnt, mate_move_cnt = eval_pos(net,
                                                                                                                  x_mate[
                                                                                                                      i],
                                                                                                                  yp_mate[
                                                                                                                      i],
                                                                                                                  verbose=False,
                                                                                                                  select_policy_from_plane=tc.select_policy_from_plane)
        if is_mate_5_top is False:
            mate_missed += 1
            print(clean_string(site_mate[i]))
            pred, pred_moves, true_move, board, is_checkmate, is_mate_5_top, legal_move_cnt, mate_move_cnt = eval_pos(
                net,
                x_mate[
                    i],
                yp_mate[
                    i],
                verbose=True,
                select_policy_from_plane=tc.select_policy_from_plane)
            pred_move = pred_moves[0]
            pred_arrow = chess.svg.Arrow(pred_move.from_square, pred_move.to_square)
            SVG(data=chess.svg.board(board=board, arrows=[pred_arrow], size=400))
        if mate_missed == 15:
            break


if __name__ == "__main__":
    main()
