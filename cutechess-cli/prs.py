# -*- coding: utf-8 -*-
"""
prs.py

Cutechess pgn result summarizer

Read cutechess pgn output and generate result info including time forfeit
stalled connections, win, draws and loses and others.

Authors: noelben
         QueensGambit

# updates:
 - 2019.06.01, corrected ply-count calculation (QueensGambit)
 - 2019.06.16, added elo calculation based on cutechess (QueensGambit)

Requirements:
    python 3

"""


import argparse
import os
import re
import pandas as pd
try:
    import statistics as stats
    import numpy as np
except ImportError as err:
    print('{}, please use python 3.'.format(err))


VERSION = '0.3.0'
WHITE = 0
BLACK = 1


class Elo:
    """
    Class to calculate the elo difference including error margin based on nubmer wins, losses and draw
    Elo calculation is based from cutechess code:
    https://github.com/cutechess/cutechess/blob/1da8c2fc0ff28b9b25b53d7f52504a4b887f35ce/projects/lib/src/elo.cpp
    """
    def __init__(self, wins : int, losses : int, draws :int):
        n = wins + losses + draws
        w = wins / n
        l = losses / n
        d = draws / n
        self.m_mu = w + d / 2.0

        devW = w * (1 - self.m_mu) ** 2
        devL = l * (0 - self.m_mu) ** 2
        devD = d * (0.5 - self.m_mu) ** 2

        self.m_stdev = np.sqrt(devW + devL + devD) / np.sqrt(n)

    def _diff(self):
        return self.m_mu

    def _diff(self, p):
        return -400 * np.log10(1 / p - 1)

    def _phiInv(self, p):
        return np.sqrt(2) * self._erfInv(2 * p - 1)

    def _erfInv(self, x):
        a = 8 * (np.pi - 3) / (3 * np.pi * (4 - np.pi))
        y = np.log(1 - x * x)
        z = 2 / (np.pi * a) + y / 2
        ret = np.sqrt(np.sqrt(z * z - y / a) - z)
        if x < 0:
            return -ret
        return ret

    def elo_diff(self):
        """
        Return the elo difference
        :return: Elo difference
        """
        return  -400 * np.log10(1 / self.m_mu - 1)

    def error_margin(self):
        """
        Returns the error margin for the elo difference
        :return: Error Margin
        """
        muMin = self.m_mu + self._phiInv(0.025) * self.m_stdev
        muMax = self.m_mu + self._phiInv(0.975) * self.m_stdev
        return (self._diff(muMax) - self._diff(muMin)) / 2.0


def get_players(data):
    """ Read data and return unique player names """
    players = []
    for n in data:
        players.append(n[0])

    return sorted(list(set(players)))


def get_game_headers(pgn):
    """
    Read pgn and return a list of header dict
    """
    h = []
    ter = '?'
    
    with open(pgn, 'r') as f:
        for lines in f:
            line = lines.rstrip()
            if '[White ' in line:
                wp = line.split('"')[1]
            elif '[Black ' in line:
                bp = line.split('"')[1]
            elif '[Result ' in line:
                res = line.split('"')[1]
            elif '[PlyCount ' in line:
                ply = int(line.split('"')[1])
            elif '[Termination ' in line:
                ter = line.split('"')[1]
            elif '[TimeControl ' in line or '[WhiteTimeControl ' in line:
                # WhiteTimeControl would appear when TC are different
                d = {
                    'White': wp,
                    'Black': bp,
                    'Result': res,
                    'Termination': ter,
                    'PlyCount': ply
                }
                h.append(d)
                ter = '?'

    return h


def save_win(data, name, color, ply):
    """
    """
    data.append([name, color, 1, 0, 0, ply, 0, 0])


def save_draw(data, name, color):
    """
    """
    data.append([name, color, 0, 0, 1, 0, 0, 0])


def save_loss(data, ter, name, color):
    """
    """
    if ter != '?':
        if ter == 'time forfeit':
            data.append([name, color, 0, 1, 0, 0, 1, 0])
        elif ter == 'stalled connection':
            data.append([name, color, 0, 1, 0, 0, 0, 1])
        else:
            # Other reason of lossing
            data.append([name, color, 0, 1, 0, 0, 0, 0])
    else:
        data.append([name, color, 0, 1, 0, 0, 0, 0])


def extract_info(data, ter, wp, bp, res, ply):
    """       
    """
    # color 0 = white, 1 = black
    # [name, color, win, loss, draw, win_ply_count, tf, sc]
    # tf = time forfeit, sc = stalled connection
    # Only record the ply_count for the player that won

    if ter != '?':
        if res == '1-0':
            save_loss(data, ter, bp, BLACK)
            save_win(data, wp, WHITE, ply)
        elif res == '0-1':
            save_loss(data, ter, wp, WHITE)
            save_win(data, bp, BLACK, ply)
        elif res == '1/2-1/2':
            save_draw(data, wp, WHITE)
            save_draw(data, bp, BLACK)
    else:
        if res == '1-0':
            save_loss(data, ter, bp, BLACK)
            save_win(data, wp, WHITE, ply)
        elif res == '0-1':
            save_loss(data, ter, wp, WHITE)
            save_win(data, bp, BLACK, ply)
        elif res == '1/2-1/2':
            save_draw(data, wp, WHITE)
            save_draw(data, bp, BLACK)

    return data


def print_summary(players, data):
    """
    """
    # (1) Summary table    
    print('{:28.27}: {:>6s} {:>19s} {:>6s} {:>6s} {:>5s} {:>5s} {:>5s} {:>3s} {:>3s}'.
          format('Name','Games', 'Elo', 'Pts', 'Pts%','Win','Loss','Draw','Tf','Sc'))

    table_content = list()

    for p in players:
        g = 0
        pts, wwins, bwins, wloss, bloss, draws, tf, sc, wdraws, bdraws = 0,0,0,0,0,0,0,0,0,0
        
        # [name, color, win, loss, draw, win_ply_count, tf, sc]
        for n in data:
            if p == n[0]:
                pts += n[2] + n[4]/2.
                tf += n[6]
                sc += n[7]
                draws += n[4]
                if n[1] == WHITE:
                    wwins += n[2]
                    wloss += n[3]
                    wdraws += n[4]
                else:
                    bwins += n[2]
                    bloss += n[3]
                    bdraws += n[4]
                g += 1
        pct = 0.0
        if g:
            pct = 100*(wwins+bwins + draws/2) / g

        wins = wwins+bwins
        losses = wloss+bloss
        elo = Elo(wins, losses, draws)

        assert pts == wwins + bwins + 0.5*draws
        assert pct == 100*pts/g
        assert g == wins + losses + draws

        w_pts = wwins + 0.5*wdraws
        b_pts = bwins + 0.5*bdraws
        w_pct = 100*w_pts / (wwins + wloss + wdraws)
        b_pct = 100*b_pts / (bwins + bloss + bdraws)

        print('{:28.27}: {:>6d} {:>8.2f} +/-{:>7.2f} {:>6.1f} {:>6.1f} {:>5d} {:>5d} '
              '{:>5d} {:>3d} {:>3d}'.
              format(p, g, elo.elo_diff(), elo.error_margin(), pts, pct, wins, losses, draws, tf, sc))
        table_content.append([p, g, elo.elo_diff(), elo.error_margin(), pts, pct, wins, losses, draws, tf, sc, wwins, bwins, wloss, bloss, wdraws, bdraws, w_pct, b_pct])
    return table_content


def print_wins(players, data):
    """
    """
    # (1) Wins / Draws table
    t_wwins, t_bwins, t_draws = 0, 0, 0
    print('{:28.27}: {:>6s} {:>6s} {:>6s}'.
          format('Name', 'W_win', 'B_win', 'Draw'))
    for p in players:
        pts, wwins, bwins, wloss, bloss, draws = 0, 0, 0, 0, 0, 0
        
        # [name, color, win, loss, draw, win_ply_count, tf, sc]
        for n in data:
            if p == n[0]:
                pts += n[2] + n[4]/2
                if n[1] == WHITE:
                    wwins += n[2]
                    wloss += n[3]
                    t_wwins += n[2]
                else:
                    bwins += n[2]
                    bloss += n[3]
                    t_bwins += n[2]
                draws += n[4]
                t_draws += n[4]

        print('{:28.27}: {:6d} {:6d} {:6d}'.format(p, wwins, bwins, draws))
    return t_wwins, t_bwins, t_draws

        
def print_win_ply(players, data):
    """
    """
    # (3) Win ply count table
    table_content = list()
    print()
    print('{:28.27}: {:>6s} {:>6s}'.format('Name', 'Wapc', 'Sd'))
    for p in players:
        winplycount = []
        
        # [name, color, win, loss, draw, win_ply_count, tf, sc]
        for n in data:
            if p == n[0] and n[5] != 0:
                winplycount.append(n[5])
                
        if len(winplycount) > 1:
            print('{:28.27}: {:>6.0f} {:>6.0f}'.format(p, stats.mean(winplycount),
                                                    stats.stdev(winplycount)))
            table_content.append([p, stats.mean(winplycount), stats.stdev(winplycount), winplycount])
        elif len(winplycount) == 1:
            print('{:28.27}: {:>6.0f} {:>6.0f}'.format(p, winplycount[0], 0))
            table_content.append([p, winplycount[0], 0, winplycount])
        else:
            print('{:28.27}: {:>6.0f} {:>6.0f}'.format(p, 0, 0))
            table_content.append([p, 0, 0, winplycount])
    return table_content


def process_pgn(pgnfn):
    """
    Read pgnfn and print result stats
    """
    data = []
    f_games, u_games = 0, 0  # f_games = finished games, u = unfinished
    
    game_headers = get_game_headers(pgnfn)
    plys_info = list()  # plys, winning_player, winning_color
    for h in game_headers:
        wp = h['White']
        bp = h['Black']
        res = h['Result']
        ter = h['Termination']
        ply = h['PlyCount']

        if res == '1-0' or res == '0-1' or res == '1/2-1/2':
            f_games += 1
            # Extract info from this game header and save it to data
            extract_info(data, ter, wp, bp, res, ply)

            if res == '1-0':
                plys_info.append([ply, wp, WHITE])
            elif res == '0-1':
                plys_info.append([ply, bp, BLACK])
            elif res == '1/2-1/2':
                plys_info.append([ply, "draw", -1])

        else:
            u_games += 1

    players = get_players(data)
    print('File: {}\n'.format(pgnfn))    
    summary_table_content = print_summary(players, data)
    print()    
    t_wwins, t_bwins, t_draws = print_wins(players, data)
    assert (summary_table_content[0][8] + summary_table_content[1][8] == t_draws)
    assert (summary_table_content[0][-8] + summary_table_content[1][-8] == t_wwins)
    assert (summary_table_content[0][-7] + summary_table_content[1][-7] == t_bwins)
    assert (summary_table_content[0][-6] + summary_table_content[1][-6] == t_bwins)
    assert (summary_table_content[0][-5] + summary_table_content[1][-5] == t_wwins)
    t_draws = t_draws / 2  # draws are counted twice without this
    print()
    ply_table_content = print_win_ply(players, data)
    print()

    # Overall game summary
    print('Finished games   : {}'.format(f_games))
    wwin_pct = 100*t_wwins/f_games
    bwin_pct = 100*t_bwins/f_games
    draws_pct = 100*t_draws/f_games
    w_pct = 100*(t_wwins+0.5*t_draws)/f_games
    b_pct = 100*(t_bwins+0.5*t_draws)/f_games
    print('White wins       : {} ({:0.1f}%)'.format(t_wwins,
          wwin_pct))
    print('Black wins       : {} ({:0.1f}%)'.format(t_bwins,
          bwin_pct))
    print('Draws            : {} ({:0.1f}%)'.format(t_draws,
          draws_pct))
    print('Unfinished games : {} ({:0.1f}%)'.format(u_games,
          100*u_games/(f_games+u_games)))

    player_a_win_plys = [ply for ply, player, _ in plys_info if player == summary_table_content[0][0]]
    player_b_win_plys = [ply for ply, player, _ in plys_info if player == summary_table_content[1][0]]
    white_win_plys = [ply for ply, _, color in plys_info if color == WHITE]
    black_win_plys = [ply for ply, _, color in plys_info if color == BLACK]
    draw_plys = [ply for ply, _, color in plys_info if color == -1]
    all_plys = [ply for ply, _, _ in plys_info]

    assert len(player_a_win_plys) == summary_table_content[0][-13]
    assert len(player_b_win_plys) == summary_table_content[1][-13]
    assert len(white_win_plys) == t_wwins
    assert len(black_win_plys) == t_bwins
    assert len(draw_plys) == t_draws
    assert player_a_win_plys == ply_table_content[0][3]
    assert player_b_win_plys == ply_table_content[1][3]

    player_a_w_plys_mean, player_a_win_plys_std = stats.mean(player_a_win_plys), stats.stdev(player_a_win_plys)
    player_b_w_plys_mean, player_b_win_plys_std = stats.mean(player_b_win_plys), stats.stdev(player_b_win_plys)
    white_w_plys_mean, white_w_plys_std = stats.mean(white_win_plys), stats.stdev(white_win_plys)
    black_w_plys_mean, black_w_plys_std = stats.mean(black_win_plys), stats.stdev(black_win_plys)
    draw_plys_mean, draw_plys_std = stats.mean(draw_plys), stats.stdev(draw_plys)
    all_plys_mean, all_plys_std = stats.mean(all_plys), stats.stdev(all_plys)

    plys_agg_info = [player_a_w_plys_mean, player_a_win_plys_std, player_b_w_plys_mean, player_b_win_plys_std,
                     white_w_plys_mean, white_w_plys_std, black_w_plys_mean, black_w_plys_std,
                     draw_plys_mean, draw_plys_std, all_plys_mean, all_plys_std]

    color_agg_info = [t_wwins, t_bwins, t_draws, wwin_pct, bwin_pct, draws_pct, w_pct, b_pct, f_games, u_games]


    # Legend
    print()
    print('Tf = time forfeit')
    print('Sc = stalled connection')
    print('Wapc = win average ply count, lower is better')
    print('Sd = standard deviation\n')
    return data, summary_table_content, plys_agg_info, color_agg_info


def main():
    parser = argparse.ArgumentParser(add_help=False,
             description='About: Read cutechess pgn file and output \
             results summary.')
    parser.add_argument('-p', '--pgn', help='input cutechess pgn filename, \
                        default is mygames.pgn',
                        default='mygames.pgn', required=False)
    parser.add_argument('-h', '--help', action='help',
                default=argparse.SUPPRESS,
                help='will show this help, use python 3 to run this script')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(VERSION))

    args = parser.parse_args()
    
    process_pgn(args.pgn)


if __name__ == "__main__":

    all_match_info = list()
    cutechess_results_folder = "cutechess_results"

    for idx, filename in enumerate(os.listdir(cutechess_results_folder)):
        f = os.path.join(cutechess_results_folder, filename)
        if os.path.isfile(f) and filename[-4:] == ".pgn":
            print(filename, f)
            numbers_in_filename = re.findall(r'\d+', filename)
            bsize = int(numbers_in_filename[-1])
            movetime = int(numbers_in_filename[-3])
            nodes = int(numbers_in_filename[-2])
            simuls = nodes*2

            data, summary_table_content, plys_agg_info, color_agg_info = process_pgn(f)
            match_info = [bsize, movetime, nodes, simuls, *summary_table_content[0], *summary_table_content[1],
                          *plys_agg_info, *color_agg_info]
            all_match_info.append(match_info)

    all_match_info_df = pd.DataFrame(all_match_info, columns=["bsize", "movetime", "nodes", "simuls", "playerA",
                                                              "A_games", "A_elo_diff", "A_elo_err", "A_pts", "A_pct",
                                                              "A_wins", "A_losses", "A_draws", "A_tf", "A_sc",
                                                              "A_wwins", "A_bwins", "A_wlosses", "A_blosses",
                                                              "A_wdraws", "A_bdraws", "A_w_pct", "A_b_pct", "playerB",
                                                              "B_games", "B_elo_diff", "B_elo_err", "B_pts", "B_pct",
                                                              "B_wins", "B_losses", "B_draws", "B_tf", "B_sc",
                                                              "B_wwins", "B_bwins", "B_wlosses", "B_blosses",
                                                              "B_wdraws", "B_bdraws", "B_w_pct", "B_b_pct",
                                                              "player_a_w_plys_mean", "player_a_win_plys_std",
                                                              "player_b_w_plys_mean", "player_b_win_plys_std",
                                                              "white_w_plys_mean", "white_w_plys_std",
                                                              "black_w_plys_mean", "black_w_plys_std",
                                                              "draw_plys_mean", "draw_plys_std", "all_plys_mean",
                                                              "all_plys_std", "t_wwins", "t_bwins", "t_draws",
                                                              "wwin_pct", "bwin_pct", "draws_pct", "white_pct",
                                                              "black_pct", "f_games", "u_games"])

    all_match_info_df.to_csv("all_matches_outcomes_new.csv")

    print("done")
    #main()


