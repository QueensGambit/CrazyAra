"""
@file: game_phase_detector.py
Created on 08.06.2023
@project: CrazyAra
@author: HelpstoneX

Analyses a given board state defined by a python-chess object and outputs the game phase according to a given definition
"""


import chess
import chess.pgn
import numpy as np
import matplotlib.pyplot as plt
import io


def get_majors_and_minors_count(board):
    pieces_left = bin(board.queens | board.rooks | board.knights | board.bishops).count("1")
    return pieces_left


def is_backrank_sparse(board, max_pieces_allowed=3):
    white_backrank_sparse = bin(board.occupied_co[chess.WHITE] & chess.BB_RANK_1).count("1") <= max_pieces_allowed
    black_backrank_sparse = bin(board.occupied_co[chess.BLACK] & chess.BB_RANK_8).count("1") <= max_pieces_allowed
    return white_backrank_sparse or black_backrank_sparse


def score(num_white_pieces_in_region, num_black_pieces_in_region, rank):
    score_map = {
        (0, 0): 0,
        (1, 0): 1 + (8 - rank),
        (2, 0): 2 + (rank - 2) if rank > 2 else 0,
        (3, 0): 3 + (rank - 1) if rank > 1 else 0,
        (4, 0): 3 + (rank - 1) if rank > 1 else 0,
        (0, 1): 1 + rank,
        (1, 1): 5 + abs(3 - rank),
        (2, 1): 4 + rank,
        (3, 1): 5 + rank,
        (0, 2): 2 + (6 - rank) if rank < 6 else 0,
        (1, 2): 4 + (6 - rank),
        (2, 2): 7,
        (0, 3): 3 + (7 - rank) if rank < 7 else 0,
        (1, 3): 5 + (6 - rank),
        (0, 4): 3 + (7 - rank) if rank < 7 else 0
    }
    return score_map.get((num_white_pieces_in_region, num_black_pieces_in_region), 0)


def get_mixedness(board):

    mix = 0

    for rank_idx in range(7):  # use ranks 1 to 7 (indices 0 to 6)
        for file_idx in range(7):  # use files A to G (indices 0 to 6)
            num_white_pieces_in_region = 0
            num_black_pieces_in_region = 0
            for dx in [0, 1]:
                for dy in [0, 1]:
                    square = chess.square(file_idx+dx, rank_idx+dy)
                    if board.piece_at(square):
                        if board.piece_at(square).color == chess.WHITE:
                            num_white_pieces_in_region += 1
                        else:
                            num_black_pieces_in_region += 1
            mix += score(num_white_pieces_in_region, num_black_pieces_in_region, rank_idx + 1)

    return mix


def get_game_phase(board, definition="lichess"):
    """
    TODO fill docstring
    """
    if definition == "lichess":
        # returns the game phase based on the lichess definition implemented in:
        # https://github.com/lichess-org/scalachess/blob/master/src/main/scala/Divider.scala

        num_majors_and_minors = get_majors_and_minors_count(board)
        backrank_sparse = is_backrank_sparse(board)
        mixedness_score = get_mixedness(board)

        if num_majors_and_minors <= 6:
            return "endgame", num_majors_and_minors, backrank_sparse, mixedness_score, 2
        elif num_majors_and_minors <= 10 or backrank_sparse or (mixedness_score > 150):
            return "midgame", num_majors_and_minors, backrank_sparse, mixedness_score, 1
        else:
            return "opening", num_majors_and_minors, backrank_sparse, mixedness_score, 0

    else:
        return "not implemented yet"


if __name__ == "__main__":
    print(get_game_phase(chess.Board()))

    #pgn = open("download_pgns/lichess_db_standard_rated_2013-01.pgn", encoding="utf-8-sig")
    pgn = io.StringIO("1. e4 Nc6 2. Nf3 e5 3. Bb5 a6 4. Ba4 b5 { C70 Ruy Lopez: Morphy Defense, Caro Variation } 5. Bb3 d6 6. O-O Nh6 7. d4 g6 8. d5 Nb4 9. a3 Nxc2 10. Qxc2 Ng4 11. h3 Nf6 12. Nc3 h5 13. Bg5 Bh6 14. Bxh6 Rxh6 15. Qd2 c5 16. Qxh6 c4 17. Bc2 Ke7 18. b3 cxb3 19. Bxb3 Rb8 20. Qg5 b4 21. Ne2 a5 22. Ba4 Ba6 23. Rfe1 Rc8 24. Bc6 Bd3 25. axb4 Bxe4 26. bxa5 Bd3 27. Ned4 Rxc6 28. dxc6 Kf8 29. Qh6+ Kg8 30. a6 exd4 31. Nxd4 Ne4 32. a7 Qa8 33. c7 Nxf2 34. Kxf2 f5 35. Qxg6+ Kh8 36. c8=Q+ Qxc8 37. a8=Q Qxa8 38. Rxa8# { White wins. } 1-0")

    stats_by_move_idx = dict()
    games_parsed = 0
    moves_parsed = 0
    midgame_start_moves = list()
    endgame_start_moves = list()

    phase_to_phase_id = {"opening": 0, "midgame": 1, "endgame": 2}
    mid_to_open_per_game = list()
    end_to_mid_per_game = list()
    end_to_open_per_game = list()
    all_prev_switch_per_game = list()

    while True:
        curr_game = chess.pgn.read_game(pgn)
        games_parsed += 1
        if games_parsed % 100 == 0:
            print(games_parsed)
        if curr_game is None or games_parsed > 100000:
            break

        curr_board = curr_game.board()
        midgame_started = False
        endgame_started = False

        curr_phase = "opening"
        mid_to_open_counter = 0
        end_to_mid_counter = 0
        end_to_open_counter = 0
        all_prev_switch_counter = 0

        for idx, move in enumerate(curr_game.main_line()):

            phase, num_maj_and_min, backrank_sparse, mix_score, _ = get_game_phase(curr_board)
            print(get_game_phase(curr_board), move)

            if curr_phase == "midgame" and phase == "opening":
                mid_to_open_counter += 1
                all_prev_switch_counter += 1

            if curr_phase == "endgame" and phase == "midgame":
                end_to_mid_counter += 1
                all_prev_switch_counter += 1

            if curr_phase == "endgame" and phase == "opening":
                end_to_open_counter += 1
                all_prev_switch_counter += 1

            if not midgame_started and phase == "midgame":
                midgame_started = True
                midgame_start_moves.append(idx)

            if not endgame_started and phase == "endgame":
                endgame_started = True
                endgame_start_moves.append(idx)

            if idx not in stats_by_move_idx:
                stats_by_move_idx[idx] = [[phase_to_phase_id[phase], num_maj_and_min, int(backrank_sparse),
                                           mix_score, int(num_maj_and_min <= 10), int(mix_score > 150)]]
            else:
                stats_by_move_idx[idx].append([phase_to_phase_id[phase], num_maj_and_min, int(backrank_sparse),
                                               mix_score, int(num_maj_and_min <= 10), int(mix_score > 150)])
            curr_board.push(move)
            moves_parsed += 1
            curr_phase = phase

        mid_to_open_per_game.append((mid_to_open_counter, curr_game.__str__()))
        end_to_mid_per_game.append((end_to_mid_counter, curr_game.__str__()))
        end_to_open_per_game.append((end_to_open_counter, curr_game.__str__()))
        all_prev_switch_per_game.append((all_prev_switch_counter, curr_game.__str__()))

    mid_trans_list = [x for x, _ in mid_to_open_per_game]
    print(np.mean(mid_trans_list))
    print(max(mid_trans_list))
    print(mid_to_open_per_game[np.argmax(mid_trans_list)][1])

    mid_trans_list_bool = [1 if x > 0 else 0 for x, _ in mid_to_open_per_game]
    print(np.mean(mid_trans_list_bool))

    num_maj_and_min_data = list()
    mix_score_data = list()
    num_samples_data = list()
    opening_moves = list()
    midgame_moves = list()
    endgame_moves = list()

    num_opening_moves = 0
    num_midgame_moves = 0
    num_endgame_moves = 0

    for move_idx, stats in stats_by_move_idx.items():
        stats_array = np.array(stats)
        game_phases = list(stats_array[:, 0])

        num_opening_moves += game_phases.count(0)
        num_midgame_moves += game_phases.count(1)
        num_endgame_moves += game_phases.count(2)

        if 0 < move_idx < 100:
            opening_moves += game_phases.count(0)*[move_idx/2]
            midgame_moves += game_phases.count(1)*[move_idx/2]
            endgame_moves += game_phases.count(2)*[move_idx/2]

        if 0 < move_idx < 120 and move_idx % 10 == 0:

            num_maj_and_mins = stats_array[:, 1]
            backrank_sparses = stats_array[:, 2]
            mix_scores = stats_array[:, 3]
            middle_piece_conds = stats_array[:, 4]
            mix_conds = stats_array[:, 5]
            samples = len(num_maj_and_mins)

            num_maj_and_min_data.append(num_maj_and_mins)
            mix_score_data.append(mix_scores)
            num_samples_data.append(samples)

    plt.boxplot(num_maj_and_min_data)
    plt.xticks(list(range(1, len(num_maj_and_min_data)+1)), [5*x for x in range(1, len(num_maj_and_min_data)+1)])
    plt.title("major and minor pieces left")
    plt.xlabel("move")
    plt.show()

    plt.boxplot(mix_score_data)
    plt.xticks(list(range(1, len(mix_score_data)+1)), [5*x for x in range(1, len(mix_score_data)+1)])
    plt.title("board mixedness")
    plt.xlabel("move")
    plt.show()

    plt.bar(list(range(len(num_samples_data))), np.array(num_samples_data))
    plt.xticks(list(range(len(num_samples_data))), [5*x for x in range(len(num_samples_data))])
    plt.title("num_samples")
    plt.xlabel("move")
    plt.show()

    bins = np.linspace(0, 50, 100)
    plt.hist(opening_moves, bins, alpha=0.5, label="opening")
    plt.hist(midgame_moves, bins, alpha=0.5, label="midgame")
    plt.hist(endgame_moves, bins, alpha=0.5, label="endgame")
    plt.xlabel("move")
    plt.title("game phase distribution")
    plt.legend()
    plt.axvline(np.mean(midgame_start_moves)/2, 0, 1, color="black")
    plt.axvline(np.mean(endgame_start_moves)/2, 0, 1, color="black")
    plt.show()

    bins = np.linspace(0, 50, 50)
    plt.hist(np.array(midgame_start_moves)/2, bins, alpha=0.5, label="midgame_start")
    plt.hist(np.array(endgame_start_moves)/2, bins, alpha=0.5, label="endgame_start")
    plt.xlabel("move")
    plt.title("game phase start distribution")
    plt.legend()
    plt.show()

    plt.bar(["opening_positions", "midgame_positions", "endgame_positions"], [num_opening_moves, num_midgame_moves, num_endgame_moves])
    plt.title("game phase position counts")
    plt.show()

    print("end")

