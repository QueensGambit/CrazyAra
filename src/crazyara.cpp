/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018  Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019  Johannes Czech

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: crazyara.cpp
 * Created on 12.06.2019
 * @author: queensgambit
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
#include "util/sfutil.h"
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
    sync_cout << intro << sync_endl;
}

void CrazyAra::uci_loop(int argc, char *argv[])
{
    Board pos;
    string token, cmd;
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
//"position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 e1g1 g8f6 d2d3 e8g8 b1c3 d7d6 c1g5 h7h6 g5f6 d8f6 c4b5 g8h8 c3d5 f6d8 b5c6 b7c6 N@e7 c6d5 N@g6 h8h7 g6f8 d8f8 e7d5 B@d8 R@b5 c7c6 b5c5 c6d5 c5c8 a8c8 B@f5 B@g6 f5c8 N@f4 e4d5 f4g2 R@d2 g2f4 P@g3 N@h3 c8h3 f4h3 g1g2 N@f4 g3f4 h3f4 g2g1 P@g2 N@g5 h6g5 f3g5",
//"go wtime 26864 btime 39522 winc 2000 binc 2000",
//"position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 e1g1 g8f6 d2d3 e8g8 b1c3 d7d6 c1g5 h7h6 g5f6 d8f6 c4b5 g8h8 c3d5 f6d8 b5c6 b7c6 N@e7 c6d5 N@g6 h8h7 g6f8 d8f8 e7d5 B@d8 R@b5 c7c6 b5c5 c6d5 c5c8 a8c8 B@f5 B@g6 f5c8 N@f4 e4d5 f4g2 R@d2 g2f4 P@g3 N@h3 c8h3 f4h3 g1g2 N@f4 g3f4 h3f4 g2g1 P@g2 N@g5 h6g5 f3g5 d8g5 B@g8",
//"go wtime 25991 btime 38380 winc 2000 binc 2000",
//"position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 e1g1 g8f6 d2d3 e8g8 b1c3 d7d6 c1g5 h7h6 g5f6 d8f6 c4b5 g8h8 c3d5 f6d8 b5c6 b7c6 N@e7 c6d5 N@g6 h8h7 g6f8 d8f8 e7d5 B@d8 R@b5 c7c6 b5c5 c6d5 c5c8 a8c8 B@f5 B@g6 f5c8 N@f4 e4d5 f4g2 R@d2 g2f4 P@g3 N@h3 c8h3 f4h3 g1g2 N@f4 g3f4 h3f4 g2g1 P@g2 N@g5 h6g5 f3g5 d8g5 B@g8 f8g8 d1h5",
//"go wtime 25153 btime 37253 winc 2000 binc 2000",
//"position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 e1g1 g8f6 d2d3 e8g8 b1c3 d7d6 c1g5 h7h6 g5f6 d8f6 c4b5 g8h8 c3d5 f6d8 b5c6 b7c6 N@e7 c6d5 N@g6 h8h7 g6f8 d8f8 e7d5 B@d8 R@b5 c7c6 b5c5 c6d5 c5c8 a8c8 B@f5 B@g6 f5c8 N@f4 e4d5 f4g2 R@d2 g2f4 P@g3 N@h3 c8h3 f4h3 g1g2 N@f4 g3f4 h3f4 g2g1 P@g2 N@g5 h6g5 f3g5 d8g5 B@g8 f8g8 d1h5 g6h5 N@f6",
//"go wtime 24348 btime 36106 winc 2000 binc 2000",
//"quit"
    };

    do {

        if (it < commands.size()) {
            cmd = commands[it];
            sync_cout << ">>" << cmd << sync_endl;
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
        else if (token == "ucinewgame") new_game();
        else if (token == "isready") {
            if (is_ready()) {
                sync_cout << "readyok" << sync_endl;
            }
        }

        // Additional custom non-UCI commands, mainly for debugging
        else if (token == "root")  mctsAgent->print_root_node();
        else if (token == "flip")  pos.flip();
        else if (token == "d")     sync_cout << pos << sync_endl;
        else
            sync_cout << "Unknown command: " << cmd << sync_endl;

        ++it;
    } while (token != "quit" && argc == 1); // Command line args are one-shot
}

void CrazyAra::go(Board *pos, istringstream &is) {
    SearchLimits searchLimits;
    searchLimits.moveOverhead = TimePoint(Options["Move_Overhead"]);
    searchLimits.nodes = Options["Nodes"];

    string token;
    bool ponderMode = false;

    searchLimits.startTime = now(); // As early as possible!

    while (is >> token) {
        if (token == "searchmoves")
            while (is >> token);
        else if (token == "wtime")     is >> searchLimits.time[WHITE];
        else if (token == "btime")     is >> searchLimits.time[BLACK];
        else if (token == "winc")      is >> searchLimits.inc[WHITE];
        else if (token == "binc")      is >> searchLimits.inc[BLACK];
        else if (token == "movestogo") is >> searchLimits.movestogo;
        else if (token == "depth")     is >> searchLimits.depth;
        else if (token == "nodes")     is >> searchLimits.nodes;
        else if (token == "movetime")    is >> searchLimits.movetime;
        //      else if (token == "mate")      is >> searchLimits.mate;
        //      else if (token == "perft")     is >> searchLimits.perft;
        else if (token == "infinite")  searchLimits.infinite = true;
        else if (token == "ponder")    ponderMode = true;
    }
    //  EvalInfo res = rawAgent->evalute_board_state(pos);
    //  rawAgent->perform_action(pos);
    Move selectedMove = mctsAgent->perform_action(pos, &searchLimits);

    // inform the mcts agent of the move, so the tree can potentially be reused later
    mctsAgent->apply_move_to_tree(selectedMove, true);
}

void CrazyAra::position(Board *pos, istringstream& is) {

    Move m;
    string token, fen;
    Variant variant = UCI::variant_from_name(Options["UCI_Variant"]);

    is >> token;
    if (token == "startpos")
    {
        fen = StartFENs[variant];
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
        StateInfo *newState = new StateInfo;
        pos->do_move(m, *newState);
        states->activeStates.push_back(newState);
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
        searchSettings = new SearchSettings(Options);
        netSingle = new NeuralNetAPI(Options["Context"], 1);
        rawAgent = new RawNetAgent(netSingle, PlaySettings(), 0, 0, true);
        NeuralNetAPI** netBatches = new NeuralNetAPI*[searchSettings->threads];
        for (size_t i = 0; i < searchSettings->threads; ++i) {
            netBatches[i] = new NeuralNetAPI(Options["Context"], searchSettings->batchSize);
        }
        mctsAgent = new MCTSAgent(netSingle, netBatches, searchSettings, PlaySettings(), states);
        networkLoaded = true;
    }
    return networkLoaded;
}

void CrazyAra::new_game()
{
    mctsAgent->reset_time_buffer_counter();
    sync_cout << "info string newgame" << sync_endl;
}

string CrazyAra::engine_info()
{
    stringstream ss;
    ss << "id name " << name << " " << version << "\n";
    ss << "id author " << authors;
    return ss.str();
}
