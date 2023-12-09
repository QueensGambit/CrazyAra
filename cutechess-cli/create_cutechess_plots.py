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


def phase_importance():

    use960 = True
    prefix_960 = "960_" if use960 else ""
    all_match_info_df = pd.read_csv("all_matches_outcomes.csv", index_col=0)
    batch_sizes = [64]
    metric = "nodes"
    matchups = [(f"{prefix_960}correct_opening", f"{prefix_960}no_phases"),
                (f"{prefix_960}correct_midgame", f"{prefix_960}no_phases"),
                (f"{prefix_960}correct_endgame", f"{prefix_960}no_phases"),]
                #("specific_endgame", "no_phases")]

    translation_dict = {f"{prefix_960}correct_opening": "opening expert",
                        f"{prefix_960}correct_midgame": "midgame expert",
                        f"{prefix_960}correct_endgame": "endgame expert"}
    y_lim = (-175, 75)
    plys_ylim = (70, 140)
    all_match_info_df = all_match_info_df.sort_values(by=["playerA", "bsize", "nodes", "movetime"])

    plt.rc('axes', titlesize=15)  # fontsize of the axes title
    plt.rc('axes', labelsize=15)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    plt.rc('legend', fontsize=12)  # legend fontsize
    plt.rc('figure', titlesize=40)  # fontsize of the figure title
    #N_colors = 7
    #plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.hot(np.linspace(0, 1, N_colors)))

    fig, ax = plt.subplots()

    for playerA, playerB in matchups:
        # batch sizes by nodes
        matchup_df = all_match_info_df[(all_match_info_df["playerA"] == f"{prefix_960}ClassicAra_{playerA}") &
                                       (all_match_info_df["playerB"] == f"{prefix_960}ClassicAra_{playerB}")]
        nodes_experiments_df = matchup_df[matchup_df[metric] != 0]


        for batch_size in batch_sizes:
            curr_bs_df = nodes_experiments_df[nodes_experiments_df["bsize"] == batch_size]
            plt.plot(curr_bs_df[metric], curr_bs_df["A_elo_diff"], label=translation_dict[playerA], marker=".")
            plt.fill_between(x=curr_bs_df[metric], y1=curr_bs_df["A_elo_diff"] + curr_bs_df["A_elo_err"],
                             y2=curr_bs_df["A_elo_diff"] - curr_bs_df["A_elo_err"], alpha=0.2)

    plt.axhline(y=0, color="black", linestyle="-")
    if metric == "nodes":
        plt.xlabel("Number of Nodes")
    elif metric == "movetime":
        plt.xlabel("Movetime [ms]")
    plt.ylabel("Relative Elo")
    plt.ylim(*y_lim)
    #ax.set_xticks(curr_bs_df["nodes"])
    plt.legend(loc="upper left")
    ax.grid(axis='y')
    #plt.title(f"{playerA} vs {playerB}")

    if metric == "nodes":
        plt.savefig(f'{prefix_960}specific_phases_nodes.pdf', bbox_inches='tight')
    elif metric == "movetime":
        plt.savefig(f'{prefix_960}specific_phases_movetime.pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":

    phase_importance()

    all_match_info_df = pd.read_csv("all_matches_outcomes.csv", index_col=0)
    batch_sizes = [1, 16,  64]
    use960 = True
    prefix_960 = "960_" if use960 else ""
    matchups = [("960_cont_learning", "960_no_phases"),]
                #("specific_opening", "no_phases"),
                #("no_phases", "specific_endgame"),]
                #("specific_endgame", "no_phases")]
    y_lim = (-260, -80)
    plys_ylim = (70, 140)

    all_match_info_df = all_match_info_df.sort_values(by=["playerA", "bsize", "nodes", "movetime"])

    plt.rc('axes', titlesize=15)  # fontsize of the axes title
    plt.rc('axes', labelsize=15)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    plt.rc('legend', fontsize=12)  # legend fontsize
    plt.rc('figure', titlesize=40)  # fontsize of the figure title
    N_colors = 4
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.hot(np.linspace(0, 1, N_colors)))

    for playerA, playerB in matchups:
        # batch sizes by nodes
        matchup_df = all_match_info_df[(all_match_info_df["playerA"] == f"{prefix_960}ClassicAra_{playerA}") &
                                       (all_match_info_df["playerB"] == f"{prefix_960}ClassicAra_{playerB}")]
        nodes_experiments_df = matchup_df[matchup_df["nodes"] != 0]
        fig, ax = plt.subplots()

        for batch_size in batch_sizes:
            curr_bs_df = nodes_experiments_df[nodes_experiments_df["bsize"] == batch_size]
            plt.plot(curr_bs_df["nodes"], curr_bs_df["A_elo_diff"], label=f"Batch Size: {batch_size}", marker=".")
            plt.fill_between(x=curr_bs_df["nodes"], y1=curr_bs_df["A_elo_diff"] + curr_bs_df["A_elo_err"],
                             y2=curr_bs_df["A_elo_diff"] - curr_bs_df["A_elo_err"], alpha=0.2)
        plt.axhline(y=0, color="black", linestyle="-")
        plt.xlabel("Number of Nodes")
        plt.ylabel("Relative Elo")
        plt.ylim(*y_lim)
        #ax.set_xticks(curr_bs_df["nodes"])
        plt.legend(loc="lower right")
        ax.grid(axis='y')
        #plt.title(f"{playerA} vs {playerB}")
        plt.savefig(f'{prefix_960}{playerA}_vs_{playerB}.pdf', bbox_inches='tight')
        plt.show()

        # batch sizes by movetime
        movetime_experiments_df = matchup_df[matchup_df["movetime"] != 0]
        fig2, ax2 = plt.subplots()

        for batch_size in batch_sizes:
            curr_movetime_df = movetime_experiments_df[movetime_experiments_df["bsize"] == batch_size]
            plt.plot(curr_movetime_df["movetime"], curr_movetime_df["A_elo_diff"], label=f"Batch Size: {batch_size}", marker=".")
            plt.fill_between(x=curr_movetime_df["movetime"], y1=curr_movetime_df["A_elo_diff"] + curr_movetime_df["A_elo_err"],
                             y2=curr_movetime_df["A_elo_diff"] - curr_movetime_df["A_elo_err"], alpha=0.2)
        plt.axhline(y=0, color="black", linestyle="-")
        plt.xlabel("Movetime [ms]")
        plt.ylabel("Relative Elo")
        plt.ylim(*y_lim)
        plt.legend(loc="lower right")
        ax2.grid(axis='y')
        #plt.title(f"{playerA} vs {playerB}")
        plt.savefig(f'{prefix_960}{playerA}_vs_{playerB}_movetime.pdf', bbox_inches='tight')
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
