/*
 * CrazyAra, a deep learning chess variant engine
 * Copyright (C) 2018 Johannes Czech, Moritz Willig, Alena Beyer
 * Copyright (C) 2019 Johannes Czech
 *
 * CrazyAra is free software: You can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * @file: crazyara.cpp
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#include "crazyara.h"

#include "bitboard.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"
#include "syzygy/tbprobe.h"
#include "movegen.h"
#include "search.h"

#include "evalinfo.h"
#include "domain/crazyhouse/constants.h"
#include "sfutil.h"
#include "uci.h"
#include "constants.h"
#include "board.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "domain/variants.h"
#include "optionsuci.h"

using namespace std;

// allocate memory
std::string LABELS_MIRRORED[NB_LABELS];
std::unordered_map<Move, size_t> MV_LOOKUP = {};
std::unordered_map<Move, size_t> MV_LOOKUP_MIRRORED = {};

CrazyAra::CrazyAra()
{
    states = new StatesManager();
}

void CrazyAra::welcome()
{
    start_logger("CrazyAra.log");
    std::cout << intro << std::endl;
}

void CrazyAra::uci_loop(int argc, char *argv[])
{
    Board pos;
    string token, cmd;
    //    StateListPtr states(new std::deque<StateInfo>(1));
    auto uiThread = std::make_shared<Thread>(0);

    StateInfo* newState = new StateInfo;
    pos.set(StartFENs[CRAZYHOUSE_VARIANT], false, CRAZYHOUSE_VARIANT, newState, uiThread.get());
    states->activeStates.push_back(newState);

    for (int i = 1; i < argc; ++i)
        cmd += std::string(argv[i]) + " ";

    size_t it = 0;

    std::vector<std::string> commands = {
        //"uci",
        //"isready",
        //"position startpos moves e2e4 b8c6 b1c3 e7e6 d2d4 f8b4 g1f3 g8f6 f1d3 d7d5 e4e5 f6e4 e1g1 b4c3 b2c3 e4c3 c1g5 c3d1 g5d8 d1f2",
        //"go"
        //"position startpos moves e2e4 b8c6 b1c3 e7e6 d2d4 f8b4 g1f3 g8f6 f1d3 d7d5 e4e5 f6e4 e1g1 b4c3 b2c3 e4c3 c1g5 c3d1 g5d8 d1f2 f1f2 P@e3",
        //"go",
        //"position startpos moves e2e4 b8c6 b1c3 e7e6 d2d4 f8b4 g1f3 g8f6 f1d3 d7d5 e4e5 f6e4 e1g1 b4c3 b2c3 e4c3 c1g5 c3d1 g5d8 d1f2 f1f2 P@e3 f2f1 N@f2",
        //"go",
        //"position startpos moves e2e4 b8c6 b1c3 e7e6 d2d4 f8b4 g1f3 g8f6 f1d3 d7d5 e4e5 f6e4 e1g1 b4c3 b2c3 e4c3 c1g5 c3d1 g5d8 d1f2 f1f2 P@e3 f2f1 N@f2 f1f2 e3f2",
        //"go",
        //"position startpos moves e2e4 b8c6 b1c3 e7e6 d2d4 f8b4 g1f3 g8f6 f1d3 d7d5 e4e5 f6e4 e1g1 b4c3 b2c3 e4c3 c1g5 c3d1 g5d8 d1f2 f1f2 P@e3 f2f1 N@f2 f1f2 e3f2 g1f2 P@b2",
        //"go",
        //"quit"
    };

    do {

        if (it < commands.size()) {
            cmd = commands[it];
            cout << ">>" << cmd << endl;
        }
        else if (argc == 1 && !getline(cin, cmd)) // Block here waiting for input or EOF
            cmd = "quit";

        istringstream is(cmd);

        token.clear(); // Avoid a stale if getline() returns empty or blank line
        is >> skipws >> token;

        if (    token == "quit"
                ||  token == "stop")
            Threads.stop = true;

        // The GUI sends 'ponderhit' to tell us the user has played the expected move.
        // So 'ponderhit' will be sent if we were told to ponder on the same move the
        // user has played. We should continue searching but switch from pondering to
        // normal search.
        else if (token == "ponderhit")
            Threads.main()->ponder = false; // Switch to normal search

        else if (token == "uci")
            sync_cout << engine_info()
                      << Options << sync_endl
                      << "uciok"  << sync_endl;

        else if (token == "setoption")  OptionsUCI::setoption(is);
        else if (token == "go")         go(&pos, is);
        else if (token == "position")   position(&pos, is);
        else if (token == "ucinewgame") sync_cout << "info string newgame" << sync_endl;
        else if (token == "isready") {
            if (is_ready()) {
                sync_cout << "readyok" << sync_endl;
            }
        }

        // Additional custom non-UCI commands, mainly for debugging
        else if (token == "root")  mctsAgent->print_root_node();
        else if (token == "flip")  pos.flip();
        else if (token == "bench") cout << "dummy"; //bench(pos, is, states);
        else if (token == "d")     sync_cout << pos << sync_endl;
        else if (token == "eval")  cout << "dummy"; //sync_cout << Eval::trace(pos) << sync_endl;
        else
            sync_cout << "Unknown command: " << cmd << sync_endl;

        ++it;
    } while (token != "quit" && argc == 1); // Command line args are one-shot
}

// go() is called when engine receives the "go" UCI command. The function sets
// the thinking time and other parameters from the input string, then starts
// the search.

void CrazyAra::go(Board *pos, istringstream &is) {

    //      Search::LimitsType limits;
    SearchLimits limits;

    SearchLimits searchLimits;
    string token;
    bool ponderMode = false;

    searchLimits.startTime = now(); // As early as possible!

    while (is >> token) {
        if (token == "searchmoves")
            while (is >> token);
        //                      searchLimits.searchmoves.push_back(UCI::to_move(pos, token));

        else if (token == "wtime")     is >> limits.time[WHITE];
        else if (token == "btime")     is >> limits.time[BLACK];
        else if (token == "winc")      is >> limits.inc[WHITE];
        else if (token == "binc")      is >> limits.inc[BLACK];
        else if (token == "movestogo") is >> limits.movestogo;
        else if (token == "depth")     is >> limits.depth;
        else if (token == "nodes")     is >> limits.nodes;
        else if (token == "movetime")    is >> searchLimits.movetime;
        //      else if (token == "mate")      is >> limits.mate;
        //      else if (token == "perft")     is >> limits.perft;
        else if (token == "infinite")  limits.infinite = true;
        else if (token == "ponder")    ponderMode = true;
    }
    //  EvalInfo res = rawAgent->evalute_board_state(pos);
    //  rawAgent->perform_action(pos);

    Move selectedMove = mctsAgent->perform_action(pos, &searchLimits);

    // inform the mcts agent of the move, so the tree can potentially be reused later
    mctsAgent->apply_move_to_tree(selectedMove, true);
}

// position() is called when engine receives the "position" UCI command.
// The function sets up the position described in the given FEN string ("fen")
// or the starting position ("startpos") and then makes the moves given in the
// following move list ("moves").

void CrazyAra::position(Board *pos, istringstream& is) {

    Move m;
    string token, fen;

    Variant variant = UCI::variant_from_name(Options["UCI_Variant"]);

    is >> token;
    if (token == "startpos")
    {
        fen = StartFENs[CRAZYHOUSE_VARIANT]; //variant];
        is >> token; // Consume "moves" token if any
    }
    else if (token == "fen")
        while (is >> token && token != "moves")
            fen += token + " ";
    else
        return;

    auto uiThread = std::make_shared<Thread>(0);
    pos->set(fen, Options["UCI_Chess960"], CRAZYHOUSE_VARIANT, new StateInfo, uiThread.get());

    states->clear_states();
    states->swap_states();

    Move lastMove = MOVE_NULL;

    // Parse move list (if any)
    while (is >> token && (m = UCI::to_move(*pos, token)) != MOVE_NONE)
    {
        // TODO: Careful this causes a memory leak
        // TODO: position startpos moves e2e4 c7c5 g1f3 b8c6 b1c3 e7e5 f1c4 f8e7 e1g1 d7d6 d2d3 g8f6 f3g5 e8g8 g5f7 f8f7 P@g5 c6d4 g5f6 e7f6 c4f7 g8f7 N@d5 P@h3 R@g3
        // position fen r1bq4/pp3kpp/3p1b2/2pNp3/3nP3/2NP2Rp/PPP2PPP/R1BQ1RK1/bn b - - 0 13
        // check why c8fg4 is proposed -> was due to states list
        StateInfo *newState = new StateInfo;
        pos->do_move(m, *newState); //states->back());
        states->activeStates.push_back(newState);
        sync_cout << "info string consume move" << sync_endl;
        lastMove = m;
    }

    // inform the mcts agent of the move, so the tree can potentially be reused later
    if (lastMove != MOVE_NULL) {
        mctsAgent->apply_move_to_tree(lastMove, false);
    }

    sync_cout << "info string position " << pos->fen() << sync_endl;

}

void CrazyAra::init()
{
    OptionsUCI::init(Options);
    Bitboards::init();
    Position::init();
    Bitbases::init();
    Search::init();
    Constants::init();
}

bool CrazyAra::is_ready()
{
    if (!networkLoaded) {
        SearchSettings searchSettings;
        searchSettings.batchSize = 8; // 8//128; //1; //28;
        netSingle = new NeuralNetAPI("cpu", 1,
                                     "/home/queensgambit/Programming/Deep_Learning/models/risev2/json/",
                                     "/home/queensgambit/Programming/Deep_Learning/models/risev2/params/");
        rawAgent = new RawNetAgent(netSingle, PlaySettings(), 0, 0, true);
        NeuralNetAPI** netBatches = new NeuralNetAPI*[searchSettings.threads];
        for (size_t i = 0; i < searchSettings.threads; ++i) {
            netBatches[i] = new NeuralNetAPI("cpu", searchSettings.batchSize,
                                             "/home/queensgambit/Programming/Deep_Learning/models/risev2/json/",
                                             "/home/queensgambit/Programming/Deep_Learning/models/risev2/params/");
        }
        mctsAgent = new MCTSAgent(netSingle, netBatches, searchSettings, PlaySettings(), states);
        networkLoaded = true;
    }

    return networkLoaded;
}

string CrazyAra::engine_info()
{
    stringstream ss;
    ss << "id name " << name << " " << version << "\n";
    ss << "id author " << authors;
    return ss.str();
}
