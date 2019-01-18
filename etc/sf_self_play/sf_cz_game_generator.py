"""
@file: sf_cz_game_generator.py
Created on 31.12.18
@project: CrazyAra
@author: queensgambit

Script to generate self played games by stockfish.
The script is basic wrapper for the cutechess-cli executable.

For transferability a fixed node count is used for every game for BLACK and WHITE.
The fixed node count is defined at NB_NODES.
For each game the number of nodes is permutated by +/- RANDOM_FACTOR_NODES to have more diverse games.
Moreover an opening suite is used, defined by OPENING_BOOK_PATH.
"""

import random
import multiprocessing
import argparse
import os
import getpass
import time
import subprocess

VERSION = 1.2

# Fixed Settings
# --------------
# define a max search depth to avoid wasting time in mate in X situations
MAX_DEPTH = int(22)
# average number of nodes per game
NB_NODES = int(1e6)
# use some random node counts per game for 20% +/- this would be
# 1e6  +/- 1e5
RANDOM_FACTOR_NODES = 0.2

# changeable settings
DEF_HASH = int(256)
CUTECHESS_PATH = THREADS = SF_PATH = OPENING_BOOK_PATH = PGN_PATH = VARIANT = None


def main():  # Too many local variables (20/15)
    """Execute the SF self-play game generation"""
    parser = argparse.ArgumentParser(description="Sf self play game generator v%.1f" % VERSION)
    username = getpass.getuser()

    # check for os
    if os.name == "nt":
        # add .exe for windows
        def_cutechess_cli = "cutechess-cli.exe"
        def_sf_path = "C:/Programs/crazyhouse_engines/stockfish/stockfish-x86_64-modern.exe"
    else:
        # linux or mac
        def_cutechess_cli = "cutechess-cli"
        def_sf_path = "/home/%s/Programs/crazyhouse_engines/stockfish/stockfish-x86_64-modern" % username

    parser.add_argument(
        "--cutechess_cli_path",
        default=def_cutechess_cli,
        type=str,
        help="cutechess executable path (default: %s)" % def_cutechess_cli,
    )
    parser.add_argument(
        "--sf_path", default=def_sf_path, type=str, help="cutechess executable path (default: %s)" % def_sf_path
    )
    parser.add_argument(
        "--opening_book_path",
        default="1k_cz_lichess_startpos.pgn",
        type=str,
        help="opening book path (default: 1k_cz_lichess_startpos.pgn",
    )
    def_pgnout_path = "sf_vs_sf_" + username + ".pgn"  # create the output file based on the username
    parser.add_argument(
        "--pgnout_path",
        default=def_pgnout_path,
        type=str,
        help="filepath where the games will be stored (default: %s)" % def_pgnout_path,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(multiprocessing.cpu_count() - 1, 1),
        help="number of threads for generating games (default: %d number of cores-1 detected by python)"
        % max(multiprocessing.cpu_count() - 1, 1),
    )
    parser.add_argument("--hash", type=int, default=DEF_HASH, help="hash size in mb (default: %d)" % DEF_HASH)
    available_variants = "'3check': Three-check Chess\
                        '5check': Five-check Chess\
                        'atomic': Atomic Chess\
                        'berolina': Berolina Chess\
                        'capablanca': Capablanca Chess\
                        'caparandom': Capablanca Random Chess\
                        'checkless': Checkless Chess\
                        'chessgi': Chessgi (Drop Chess)\
                        'crazyhouse': Crazyhouse (Drop Chess)\
                        'extinction': Extinction Chess\
                        'fischerandom': Fischer Random Chess/Chess 960\
                        'gothic': Gothic Chess\
                        'horde': Horde Chess (v2)\
                        'janus': Janus Chess\
                        'kinglet': Kinglet Chess\
                        'kingofthehill': King of the Hill Chess\
                        'loop': Loop Chess (Drop Chess)\
                        'losers': Loser's Chess\
                        'racingkings': Racing Kings Chess\
                        'standard': Standard Chess."

    parser.add_argument(
        "--variant",
        default="crazyhouse",
        type=str,
        help="define the chess variant (default: crazyhouse)" + available_variants,
    )
    args = parser.parse_args()
    nodes = []

    i = 0
    # start the self play learning loop
    while True:
        # use a random fixed number of nodes for white and black on every move
        random_nodes_white = NB_NODES * random.random() * RANDOM_FACTOR_NODES * random.choice([-1, 1])
        random_nodes_black = NB_NODES * random.random() * RANDOM_FACTOR_NODES * random.choice([-1, 1])
        current_nodes_white = int(NB_NODES + random_nodes_white)
        current_nodes_black = int(NB_NODES + random_nodes_black)
        nodes.append(current_nodes_white)
        nodes.append(current_nodes_white)
        sf_engine_cmd = []
        local_time = time.localtime()
        # description of the current game
        event = "Game %d - Nodes: %d-%d - Hash: %d - Threads: %d" % (
            i + 1,
            current_nodes_white,
            current_nodes_black,
            args.hash,
            args.threads,
        )

        for current_nodes in [current_nodes_white, current_nodes_black]:
            sf_engine_cmd.append(
                " -engine cmd="
                + args.sf_path
                + " tc=inf nodes="
                + str(current_nodes)
                + " option.Hash="
                + str(args.hash)
                + " option.Threads="
                + str(args.threads)
                + " depth="
                + str(MAX_DEPTH)
                + " proto=uci"
            )

        game_cmd = (
            " -variant "
            + str(args.variant)
            + " -event "
            + '"'
            + event
            + '"'
            + " -openings file="
            + args.opening_book_path
            + " order=random -pgnout "
            + args.pgnout_path
        )

        cmd_str = args.cutechess_cli_path + sf_engine_cmd[0] + sf_engine_cmd[1] + game_cmd
        print(
            "%s - %s - Threads: %d" % (time.asctime(local_time), event, args.threads)
        )  # print the current game description
        process_start = subprocess.Popen(cmd_str, shell=True)  # start the game with the cutechess-cli
        process_start.wait()
        i += 1


if __name__ == "__main__":
    main()
