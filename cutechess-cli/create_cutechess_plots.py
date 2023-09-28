"""
@file: create_cutechess_plots
Created on 28.09.2023
@project: CrazyAra
@author: Felix

creates plots based on the results of different cutechess match configurations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    all_match_info_df = pd.read_csv("all_matches_outcomes.csv", index_col=0)
    batch_sizes = [1, 8, 16, 32, 64]
    matchups = [("correct_phases", "no_phases"),]
                #("specific_opening", "no_phases"),
                #("specific_midgame", "no_phases"),
                #("specific_endgame", "no_phases")]
    y_lim = (-10, 175)
    plys_ylim = (70, 140)
    all_match_info_df = all_match_info_df.sort_values(by=["playerA", "bsize", "nodes", "movetime"])

    for playerA, playerB in matchups:
        # batch sizes by nodes
        matchup_df = all_match_info_df[(all_match_info_df["playerA"] == f"ClassicAra_{playerA}") &
                                       (all_match_info_df["playerB"] == f"ClassicAra_{playerB}")]
        nodes_experiments_df = matchup_df[matchup_df["nodes"] != 0]
        fig, ax = plt.subplots()

        for batch_size in batch_sizes:
            curr_bs_df = nodes_experiments_df[nodes_experiments_df["bsize"] == batch_size]
            plt.plot(curr_bs_df["nodes"], curr_bs_df["A_elo_diff"], label=f"Batch Size: {batch_size}", marker=".")
        plt.axhline(y=0, color="black", linestyle="-")
        plt.xlabel("Number of Nodes")
        plt.ylabel("Relative Elo")
        plt.ylim(*y_lim)
        plt.legend()
        ax.grid(axis='y')
        plt.title(f"{playerA} vs {playerB}")
        plt.show()

        # batch sizes by movetime
        movetime_experiments_df = matchup_df[matchup_df["movetime"] != 0]
        fig2, ax2 = plt.subplots()

        for batch_size in batch_sizes:
            curr_movetime_df = movetime_experiments_df[movetime_experiments_df["bsize"] == batch_size]
            plt.plot(curr_movetime_df["movetime"], curr_movetime_df["A_elo_diff"], label=f"Batch Size: {batch_size}", marker=".")
        plt.axhline(y=0, color="black", linestyle="-")
        plt.xlabel("Movetime [ms]")
        plt.ylabel("Relative Elo")
        plt.ylim(*y_lim)
        plt.legend()
        ax2.grid(axis='y')
        plt.title(f"{playerA} vs {playerB}")
        plt.show()

        # p_a_win_plys_mean by movetime
        movetime_experiments_df = matchup_df[matchup_df["movetime"] != 0]
        fig2, ax2 = plt.subplots()

        for batch_size in batch_sizes:
            curr_movetime_df = movetime_experiments_df[movetime_experiments_df["bsize"] == batch_size]
            plt.plot(curr_movetime_df["movetime"], curr_movetime_df["player_a_w_plys_mean"], label=f"Batch Size: {batch_size}", marker=".")
        plt.xlabel("Movetime [ms]")
        plt.ylabel("Avg Plys")
        plt.legend()
        plt.ylim(*plys_ylim)
        ax2.grid(axis='y')
        plt.title(f"A Wins {playerA} vs {playerB}")
        plt.show()

        # p_b_win_plys_mean by movetime
        movetime_experiments_df = matchup_df[matchup_df["movetime"] != 0]
        fig2, ax2 = plt.subplots()

        for batch_size in batch_sizes:
            curr_movetime_df = movetime_experiments_df[movetime_experiments_df["bsize"] == batch_size]
            plt.plot(curr_movetime_df["movetime"], curr_movetime_df["player_b_w_plys_mean"], label=f"Batch Size: {batch_size}", marker=".")
        plt.xlabel("Movetime [ms]")
        plt.ylabel("Avg Plys")
        plt.legend()
        plt.ylim(*plys_ylim)
        ax2.grid(axis='y')
        plt.title(f"B wins {playerA} vs {playerB}")
        plt.show()

        # draw_plys_mean by movetime
        movetime_experiments_df = matchup_df[matchup_df["movetime"] != 0]
        fig2, ax2 = plt.subplots()

        for batch_size in batch_sizes:
            curr_movetime_df = movetime_experiments_df[movetime_experiments_df["bsize"] == batch_size]
            plt.plot(curr_movetime_df["movetime"], curr_movetime_df["draw_plys_mean"], label=f"Batch Size: {batch_size}", marker=".")
        plt.xlabel("Movetime [ms]")
        plt.ylabel("Avg Plys")
        plt.legend()
        plt.ylim(*plys_ylim)
        ax2.grid(axis='y')
        plt.title(f"Draw {playerA} vs {playerB}")
        plt.show()

        # draw rate by movetime
        movetime_experiments_df = matchup_df[matchup_df["movetime"] != 0]
        fig2, ax2 = plt.subplots()

        for batch_size in batch_sizes:
            curr_movetime_df = movetime_experiments_df[movetime_experiments_df["bsize"] == batch_size]
            plt.plot(curr_movetime_df["movetime"], curr_movetime_df["draws_pct"], label=f"Batch Size: {batch_size}", marker=".")

        plt.xlabel("Movetime [ms]")
        plt.ylabel("Draw Percentage")
        plt.legend()
        ax2.grid(axis='y')
        plt.title(f"Draw Percentage {playerA} vs {playerB}")
        plt.show()


    # p_a win plys mean, p_b win plys mean
    # white win plys mean, black win plys mean, draw plys mean
    # all plys mean

    # boxplots batch size, ?
    # a wins, b wins, draw boxplots plys
    # nodes oder movetime festsetzen

    # all nets vs each single net in one plot


    # draw rate


    # avg plys (white wins, black wins, draw

    print(all_match_info_df.head(5))
