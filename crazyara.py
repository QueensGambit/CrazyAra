"""
@file: crazyara.py
Created on 18.10.18
@project: crazy_ara_refactor
@author: queensgambit

Universal chess interface (CUI) communication protocol interface for the engine.
The protocol was published by Stefan-Meyer Kahlen (ShredderChess) and can be accessed at:
http://wbec-ridderkerk.nl/html/UCIProtocol.html
"""

from __future__ import print_function
import sys
import chess.pgn

import traceback

client = {
    'name': 'CrazyAra',
    'version': '0.2.0',
    'authors': 'Johannes Czech, Moritz Willig, Alena Beyer et al.'
}


INTRO_PT1 = "" \
        "                                  _                                           \n" \
        "                   _..           /   ._   _.  _        /\   ._   _.           \n" \
        "                 .' _ `\         \_  |   (_|  /_  \/  /--\  |   (_|           \n" \
        "                /  /e)-,\                         /                           \n" \
        "               /  |  ,_ |                    __    __    __    __             \n" \
        "              /   '-(-.)/          bw     8 /__////__////__////__////         \n" \
        "            .'--.   \  `                 7 ////__////__////__////__/          \n" \
        "           /    `\   |                  6 /__////__////__////__////           \n" \
        "         /`       |  / /`\.-.          5 ////__////__////__////__/            \n" \
        "       .'        ;  /  \_/__/         4 /__////__////__////__////             \n" \
        "     .'`-'_     /_.'))).-` \         3 ////__////__////__////__/              \n" \
        "    / -'_.'---;`'-))).-'`\_/        2 /__////__////__////__////               \n" \

INTRO_PT2 = "" \
            "   (__.'/   /` .'`                 1 ////__////__////__////__/                \n" \
            "    (_.'/ /` /`                       a  b  c  d  e  f  g  h                  \n" \
            "      _|.' /`                                                                 \n" \
            "jgs.-` __.'|  Developers: Johannes Czech, Moritz Willig, Alena Beyer et al.   \n" \
            "    .-'||  |  Source-Code: QueensGambit/CrazyAra (GPLv3-License)              \n" \
            "       \_`/   Inspiration: A0-paper by Silver, Hubert, Schrittwieser et al.  \n" \
            "              ASCII-Art: Joan G. Stark, Chappell, Burton                      \n"

log_file_path = "CrazyAra-log.txt"

try:
    log_file = open(log_file_path, 'w')
except:
    log_file = None
    # print out the error message
    traceback_text = traceback.format_exc()
    print("! An error occured while trying to open the log_file %s !" % log_file_path)
    print(traceback_text)


def log_print(text: str):
    print(text)
    if log_file:
        log_file.write("< %s\n" % text)
        log_file.flush()


def log(text: str):
    if log_file:
        log_file.write("> %s\n" % text)
        log_file.flush()


print(INTRO_PT1, end="")
print(INTRO_PT2, end="")

# GLOBAL VARIABLES
MCTSAGENT = None
RAWNETAGENT = None
GAMESTATE = None

SWITCHED_TO_RAW_NET = False

SETUP_DONE = False

# SETTINGS
S = {
    'UCI_Variant': 'crazyhouse',
    # set the context in which the neural networks calculation will be done
    # choose 'gpu' using the settings if there is one available
    "context": 'cpu',
    "use_raw_network": False,
    "threads": 8,
    "playouts_empty_pockets": 256,
    "playouts_filled_pockets": 512,
    "playouts_update_stats": 256,
    "centi_cpuct": 100,
    "centi_dirichlet_epsilon": 10,
    "centi_dirichlet_alpha": 20,
    "max_search_time_s": 300,
    "max_search_depth": 40,
    "centi_temperature": 0,
    "centi_clip_quantil": 0,
    "virtual_loss": 3,
    "threshold_time_for_raw_net_ms": 30000
}


def setup_network():
    """
    Load the libraries and the weights of the neural network
    :return:
    """

    global GAMESTATE
    global SETUP_DONE
    global RAWNETAGENT
    global MCTSAGENT
    global S

    if SETUP_DONE is False:
        from DeepCrazyhouse.src.domain.crazyhouse.GameState import GameState
        from DeepCrazyhouse.src.domain.agent.NeuralNetAPI import NeuralNetAPI
        from DeepCrazyhouse.src.domain.agent.player.RawNetAgent import RawNetAgent
        from DeepCrazyhouse.src.domain.agent.player.MCTSAgent import MCTSAgent

        net = NeuralNetAPI(ctx=S['context'])

        RAWNETAGENT = RawNetAgent(net, temperature=S['centi_temperature'], clip_quantil=S['centi_clip_quantil'])

        MCTSAGENT = MCTSAgent(net, cpuct=S['centi_cpuct']/100, nb_playouts_empty_pockets=S['playouts_empty_pockets'],
                              nb_playouts_filled_pockets=S['playouts_filled_pockets'], max_search_depth=S['max_search_depth'],
                              nb_playouts_update=S['playouts_update_stats'], max_search_time_s=S['max_search_time_s'],
                              dirichlet_alpha=S['centi_dirichlet_alpha']/100, dirichlet_epsilon=S['centi_dirichlet_epsilon']/100,
                              virtual_loss=S['virtual_loss'], threads=S['threads'], temperature=S['centi_temperature']/100,
                              clip_quantil=S['centi_clip_quantil']/100)

        GAMESTATE = GameState()

        SETUP_DONE = True


def perform_action(cmd_list):
    """
    
    :return: 
    """

    global SWITCHED_TO_RAW_NET
    global AGENT
    global GAMESTATE
    global MCTSAGENT
    global RAWNETAGENT

    if len(cmd_list) >= 5:
        if cmd_list[1] == 'wtime' and cmd_list[3] == 'btime':

            wtime = int(cmd_list[2])
            btime = int(cmd_list[4])

            if GAMESTATE.is_white_to_move() is True:
                my_time = wtime
            else:
                my_time = btime

            if SWITCHED_TO_RAW_NET is False and int(my_time) < S['threshold_time_for_raw_net_ms']:
                log_print('Switching to raw network for fast mode...')
                # switch to RawNetwork-Agent
                SWITCHED_TO_RAW_NET = True

            elif SWITCHED_TO_RAW_NET is True and my_time >= S['threshold_time_for_raw_net_ms']:
                log_print('Switching back to MCTS network for slow mode...')
                # switch to RawNetwork-Agent
                SWITCHED_TO_RAW_NET = False

    if SWITCHED_TO_RAW_NET is True or S['use_raw_network'] is True:
        value, selected_move, confidence, _ = RAWNETAGENT.perform_action(GAMESTATE)
    else:
        value, selected_move, confidence, _ = MCTSAGENT.perform_action(GAMESTATE)

    log_print('bestmove %s' % selected_move.uci())


def setup_gamestate(cmd_list):

    position_type = cmd_list[1]
    if position_type == "startpos":
        GAMESTATE.new_game()

    elif position_type == "fen":
        sub_command_offset = cmd_list.index("moves") if "moves" in cmd_list else len(cmd_list)
        fen = " ".join(cmd_list[2:sub_command_offset])

        GAMESTATE.set_fen(fen)

    mv_list = cmd_list[3:]
    for move in mv_list:
        GAMESTATE.apply_move(chess.Move.from_uci(move))


def set_options(cmd_list):
    """
    Updates the internal options as requested by the use via the uci-protocoll
    An example call could be: "setoption name nb_threads value 1"
    :param cmd_list: List of received of commands
    :return:
    """
    # SETTINGS
    global S

    if cmd_list[1] == 'name' and cmd_list[3] == 'value':
        option_name = cmd_list[2]

        if option_name not in S:
            raise Exception("The given option %s wasn't found in the settings list" % option_name)

        if option_name in ['UCI_Variant', 'context', 'use_raw_network']:
            value = cmd_list[4]
        else:
            value = int(cmd_list[4])

        if option_name == 'use_raw_network':
            if value == 'true':
                S['use_raw_network'] = True
            else:
                S['use_raw_network'] = False
        else:
            S[option_name] = value

        log_print('Updated option %s to %s' % (option_name, value))


# main waiting loop for processing command line inputs
while True:
    line = input()

    # wait for an std-in input command
    if line:
        try:
            # split the line to a list which makes parsing easier
            cmd_list = line.rstrip().split(' ')
            # extract the first command from the list for evaluation
            main_cmd = cmd_list[0]

            # write the given command to the log-file
            log(line)

            if main_cmd == 'uci':
                log_print('id name %s %s' % (client['name'], client['version']))
                log_print('id author %s' % client['authors'])
                # tell the GUI all possible options
                log_print('option name UCI_Variant type combo default crazyhouse var crazyhouse')
                log_print('option name context type combo default cpu var cpu var gpu')
                log_print('option name use_raw_network type check default false')
                log_print('option name threads type spin default 8 min 1 max 4096')
                log_print('option name playouts_empty_pockets type spin default 256 min 56 max 8192')
                log_print('option name playouts_filled_pockets type spin default 512 min 56 max 8192')
                log_print('option name playouts_update_stats type spin default 256 min 56 max 8192')
                log_print('option name centi_cpuct type spin default 100 min 1 max 500')
                log_print('option name centi_dirichlet_epsilon type spin default 10 min 0 max 100')
                log_print('option name centi_dirichlet_alpha type spin default 20 min 0 max 100')
                log_print('option name max_search_time_s type spin default 300 min 1 max 9000')
                log_print('option name max_search_depth type spin default 40 min 1 max 100')
                log_print('option name centi_temperature type spin default 0 min 0 max 100')
                log_print('option name centi_clip_quantil type spin default 0 min 0 max 100')
                log_print('option name virtual_loss type spin default 3 min 0 max 10')
                log_print('option name threshold_time_for_raw_net_ms type spin default 30000 min 1 max 999999999')

                # verify that all options have been sent
                log_print('uciok')

            elif main_cmd == 'isready':
                setup_network()
                log_print('readyok')
            elif main_cmd == 'ucinewgame':
                pass
            elif main_cmd == "position":
                setup_gamestate(cmd_list)
            elif main_cmd == "setoption":
                set_options(cmd_list)
            elif main_cmd == 'go':
                perform_action(cmd_list)
            elif main_cmd == 'quit' or 'exit':
                sys.exit(0)
        except:
            # log the error message to the log-file and exit the script
            traceback_text = traceback.format_exc()
            log_print(traceback_text)
            sys.exit(-1)
