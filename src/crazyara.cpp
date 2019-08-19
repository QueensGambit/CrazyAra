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
#include "constants.h"
#include "board.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "domain/variants.h"
#include "optionsuci.h"
#include "tests/benchmarkpositions.h"

using namespace std;

// allocate memory
string LABELS_MIRRORED[NB_LABELS];
unordered_map<Move, size_t> MV_LOOKUP = {};
unordered_map<Move, size_t> MV_LOOKUP_MIRRORED = {};

CrazyAra::CrazyAra()
{
    states = new StatesManager();
}

void CrazyAra::welcome()
{
    start_logger("CrazyAra.log");
    cout << intro << endl;
}

void CrazyAra::uci_loop(int argc, char *argv[])
{
    Board pos;
    string token, cmd;
    EvalInfo evalInfo;
    auto uiThread = make_shared<Thread>(0);

    StateInfo* newState = new StateInfo;
    pos.set(StartFENs[CRAZYHOUSE_VARIANT], false, CRAZYHOUSE_VARIANT, newState, uiThread.get());
    states->activeStates.push_back(newState);

    for (int i = 1; i < argc; ++i)
        cmd += string(argv[i]) + " ";

    size_t it = 0;

	// this is debug vector which can contain uci commands which will be automaticly processed when the executable is launched
    vector<string> commands = {
//        "isready",
//        "ucinewgame",
//        "position startpos",
//        "position startpos moves e2e3",
//        "position startpos moves e2e3 g8f6",
//        "isready",
//        "go wtime 60000 btime 60000",
//        "position startpos moves e2e3 g8f6 d2d4 g7g6",
//        "isready",
//        "go wtime 58839 btime 58835"
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

		if (token == "quit") {
			break;
		}
		else if (token == "uci") {
			cout << engine_info()
				<< Options << endl
				<< "uciok" << endl;
		}
        else if (token == "setoption")  OptionsUCI::setoption(is);
        else if (token == "go")         go(&pos, is, evalInfo);
        else if (token == "position")   position(&pos, is);
        else if (token == "ucinewgame") new_game();
        else if (token == "isready") {
            if (is_ready()) {
                cout << "readyok" << endl;
            }
        }

        // Additional custom non-UCI commands, mainly for debugging
        else if (token == "benchmark")  benchmark(is);
        else if (token == "root")       mctsAgent->print_root_node();
        else if (token == "flip")       pos.flip();
        else if (token == "d")          cout << pos << endl;
        else
            cout << "Unknown command: " << cmd << endl;

        ++it;
    } while (token != "quit" && argc == 1); // Command line args are one-shot
}

void CrazyAra::go(Board *pos, istringstream &is,  EvalInfo& evalInfo, bool applyMoveToTree) {
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
        else if (token == "movetime")  is >> searchLimits.movetime;
        else if (token == "infinite")  searchLimits.infinite = true;
        else if (token == "ponder")    ponderMode = true;
    }
    //  EvalInfo res = rawAgent->evalute_board_state(pos);
    //  rawAgent->perform_action(pos);
    mctsAgent->perform_action(pos, &searchLimits, evalInfo);

    if (applyMoveToTree) {
        // inform the mcts agent of the move, so the tree can potentially be reused later
        mctsAgent->apply_move_to_tree(evalInfo.bestMove, true);
    }
}

void CrazyAra::go(string fen, string goCommand, EvalInfo& evalInfo)
{
    Board pos;
    string token, cmd;
    auto uiThread = make_shared<Thread>(0);

    StateInfo* newState = new StateInfo;
    pos.set(StartFENs[CRAZYHOUSE_VARIANT], false, CRAZYHOUSE_VARIANT, newState, uiThread.get());

    istringstream is("fen " + fen);
    position(&pos, is);
    istringstream isGoCommand(goCommand);
    go(&pos, isGoCommand, evalInfo, false);
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

    auto uiThread = make_shared<Thread>(0);
    pos->set(fen, false, CRAZYHOUSE_VARIANT, new StateInfo, uiThread.get());
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
    cout << "info string position " << pos->fen() << endl;
}

void CrazyAra::benchmark(istringstream &is)
{
    int passedCounter = 0;
    EvalInfo evalInfo;
    BenchmarkPositions benchmark;
    string moveTime;
    is >> moveTime;
    string goCommand = "go movetime " + moveTime;
    int totalNPS = 0;
    int totalDepth = 0;

    for (TestPosition pos : benchmark.positions) {
        go(pos.fen, goCommand, evalInfo);
        string uciMove = UCI::move(evalInfo.bestMove, false);
        if (uciMove != pos.blunderMove) {
            cout << "passed      -- " << uciMove << " != " << pos.blunderMove << endl;
            passedCounter++;
        }
        else {
            cout << "failed      -- " << uciMove << " == " << pos.blunderMove << endl;
        }
        cout << "alternative -- ";
        if (uciMove == pos.alternativeMove) {
            cout << uciMove << " == " << pos.alternativeMove << endl;
        }
        else {
            cout << uciMove << " != " << pos.alternativeMove << endl;
        }
        totalNPS += evalInfo.nps;
        totalDepth += evalInfo.depth;
    }

    cout << endl << "Summary" << endl;
    cout << "----------------------" << endl;
    cout << "Passed:\t\t" << passedCounter << "/" << benchmark.positions.size() << endl;
    cout << "NPS:\t\t" << setw(2) << totalNPS /  benchmark.positions.size() << endl;
    cout << "PV-Depth:\t" << setw(2) << totalDepth /  benchmark.positions.size() << endl;
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
        searchSettings = new SearchSettings();
        searchSettings->threads = Options["Threads"];
        searchSettings->batchSize = Options["Batch_Size"];
        searchSettings->useTranspositionTable = Options["Use_Transposition_Table"];
        searchSettings->uInit = float(Options["Centi_U_Init_Divisor"]) / 100.0f;
        searchSettings->uMin = Options["Centi_U_Min"] / 100.0f;
        searchSettings->uBase = Options["U_Base"];
        searchSettings->qValueWeight = Options["Centi_Q_Value_Weight"] / 100.0f;
        searchSettings->enhanceChecks = Options["Enhance_Checks"];
        searchSettings->enhanceCaptures = Options["Enhance_Captures"];
        searchSettings->cpuctInit = Options["Centi_CPuct_Init"] / 100.0f;
        searchSettings->cpuctBase = Options["CPuct_Base"];
        searchSettings->dirichletEpsilon = Options["Centi_Dirichlet_Epsilon"] / 100.0f;
        searchSettings->dirichletAlpha = Options["Centi_Dirichlet_Alpha"] / 100.0f;
        searchSettings->virtualLoss = Options["Virtual_Loss"];
        searchSettings->qThreshInit = Options["Centi_Q_Thresh_Init"] / 100.0f;
        searchSettings->qThreshMax = Options["Centi_Q_Thresh_Max"] / 100.0f;
        searchSettings->qThreshBase = Options["Q_Thresh_Base"];
        searchSettings->randomMoveFactor = Options["Centi_Random_Move_Factor"]  / 100.0f;

        playSettings = new PlaySettings();
        playSettings->temperature = Options["Centi_Temperature"] / 100.0f;
        playSettings->temperatureMoves = Options["Temperature_Moves"];

        string modelDirectory = Options["Model_Directory"];
        netSingle = new NeuralNetAPI(Options["Context"], 1, modelDirectory);
        rawAgent = new RawNetAgent(netSingle, PlaySettings(), 0, 0, true);
        NeuralNetAPI** netBatches = new NeuralNetAPI*[searchSettings->threads];
        for (size_t i = 0; i < searchSettings->threads; ++i) {
            netBatches[i] = new NeuralNetAPI(Options["Context"], searchSettings->batchSize, modelDirectory);
        }
        mctsAgent = new MCTSAgent(netSingle, netBatches, searchSettings, *playSettings, states);
        networkLoaded = true;
    }
    return networkLoaded;
}

void CrazyAra::new_game()
{
    mctsAgent->clear_game_history();
    cout << "info string newgame" << endl;
}

string CrazyAra::engine_info()
{
    stringstream ss;
    ss << "id name " << name << " " << version << " (" << __DATE__ << ")" << "\n";
    ss << "id author " << authors;
    return ss.str();
}
