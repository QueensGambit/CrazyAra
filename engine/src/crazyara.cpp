/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018       Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019-2020  Johannes Czech

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
#include "constants.h"
#include "board.h"
#include "domain/variants.h"
#include "optionsuci.h"
#include "tests/benchmarkpositions.h"
#include "util/communication.h"
#ifdef MXNET
#include "nn/mxnetapi.h"
#elif defined TENSORRT
#include "nn/tensorrtapi.h"
#endif

using namespace std;

// allocate memory
string LABELS_MIRRORED[NB_LABELS];
unordered_map<Move, size_t> MV_LOOKUP = {};
unordered_map<Move, size_t> MV_LOOKUP_MIRRORED = {};
unordered_map<Move, size_t> MV_LOOKUP_CLASSIC = {};
unordered_map<Move, size_t> MV_LOOKUP_MIRRORED_CLASSIC = {};

CrazyAra::CrazyAra()
{
    searchSettings = nullptr;  // will be initialized in init_search_settings()
    playSettings = nullptr;    // will be initialized in init_play_settings()
#ifdef USE_RL
    rlSettings = nullptr;      // will be initialized in init_rl_settings()
#endif
    netSingle = nullptr;       // will be initialized in is_ready()
    states = new StatesManager();
#ifdef MODE_CRAZYHOUSE
    variant = CRAZYHOUSE_VARIANT;
#else
    variant = CHESS_VARIANT;
#endif
    useRawNetwork = false;     // will be initialized in init_search_settings()
    is960 = false;
}

CrazyAra::~CrazyAra()
{
    delete searchSettings;
    delete playSettings;
#ifdef USE_RL
    delete rlSettings;
#endif
    delete netSingle;
    delete states;
}

void CrazyAra::welcome()
{
    cout << intro << endl;
}

void CrazyAra::uci_loop(int argc, char *argv[])
{
    Board pos;
    string token, cmd;
    EvalInfo evalInfo;
    auto uiThread = make_shared<Thread>(0);

    StateInfo* newState = new StateInfo;
    variant = UCI::variant_from_name(Options["UCI_Variant"]);
    pos.set(StartFENs[variant], is960, variant, newState, uiThread.get());
    states->activeStates.push_back(newState);

    for (int i = 1; i < argc; ++i)
        cmd += string(argv[i]) + " ";

    size_t it = 0;

	// this is debug vector which can contain uci commands which will be automaticly processed when the executable is launched
    vector<string> commands = {
//        "isready",
//        "go",
//        "position startpos moves e2e4 e7e5",
//        "go",
//        "setoption name UCI_Variant value crazyhouse",
//        "ucinewgame",
//        "position startpos",
//        "position startpos moves a2a4",
//        "position startpos moves a2a4 b7b6",
//        "position startpos moves a2a4 b7b6 a1a3",
//        "position startpos moves a2a4 b7b6 a1a3 e7e5"
//        "isready",
//        "go movetime 3000",
//        "ucinewgame",
//        "benchmark 1000",
//        "ucinewgame",
//        "position startpos",
//        "position startpos moves e2e3",
//        "position startpos moves e2e3 g8f6",
//        "ucinewgame",
//        "isready",
//        "go wtime 60000 btime 60000",
//        "position startpos moves e2e3 g8f6 d2d4 g7g6",
//        "isready",
//        "go wtime 58839 btime 58835",
//        "ucinewgame"
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

        if (token == "stop" || token == "quit") {
            mctsAgent->stop_search();
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
#ifdef USE_RL
        else if (token == "selfplay")   selfplay(is);
        else if (token == "arena")      arena(is);
#endif
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
    if (useRawNetwork) {
        rawAgent->perform_action(pos, &searchLimits, evalInfo);
    }
    else {
        mctsAgent->perform_action(pos, &searchLimits, evalInfo);
        if (applyMoveToTree) {
            // inform the mcts agent of the move, so the tree can potentially be reused later
            mctsAgent->apply_move_to_tree(evalInfo.bestMove, true, pos);
        }
    }
}

void CrazyAra::go(const string& fen, string goCommand, EvalInfo& evalInfo)
{
    Board pos;
    string token, cmd;
    auto uiThread = make_shared<Thread>(0);
    variant = UCI::variant_from_name(Options["UCI_Variant"]);

    StateInfo* newState = new StateInfo;
    pos.set(StartFENs[variant], is960, variant, newState, uiThread.get());

    istringstream is("fen " + fen);
    position(&pos, is);
    istringstream isGoCommand(goCommand);
    go(&pos, isGoCommand, evalInfo, false);
}

void CrazyAra::position(Board *pos, istringstream& is) {

    Move m;
    string token, fen;
    variant = UCI::variant_from_name(Options["UCI_Variant"]);

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
    pos->set(fen, is960, variant, new StateInfo, uiThread.get());
    states->clear_states();
    states->swap_states();
    Move lastMove = MOVE_NULL;

    // Parse move list (if any)
    while (is >> token && (m = UCI::to_move(*pos, token)) != MOVE_NONE)
    {
        StateInfo* newState = new StateInfo;
        pos->do_move(m, *newState);
        states->activeStates.push_back(newState);
        lastMove = m;
    }
    // inform the mcts agent of the move, so the tree can potentially be reused later
    if (lastMove != MOVE_NULL && !useRawNetwork) {
        mctsAgent->apply_move_to_tree(lastMove, false, pos);
    }
    info_string("position", pos->fen());
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
    vector<int> nps;

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
        const int cur_nps = evalInfo.calculate_nps();
        totalNPS += cur_nps;
        totalDepth += evalInfo.depth;
        nps.push_back(cur_nps);
    }

    sort(nps.begin(), nps.end());

    cout << endl << "Summary" << endl;
    cout << "----------------------" << endl;
    cout << "Passed:\t\t" << passedCounter << "/" << benchmark.positions.size() << endl;
    cout << "NPS (avg):\t" << setw(2) << totalNPS /  benchmark.positions.size() << endl;
    cout << "NPS (median):\t" << setw(2) << nps[nps.size()/2] << endl;
    cout << "PV-Depth:\t" << setw(2) << totalDepth /  benchmark.positions.size() << endl;
}

#ifdef USE_RL
void CrazyAra::selfplay(istringstream &is)
{
    SearchLimits searchLimits;
    searchLimits.nodes = size_t(Options["Nodes"]);
    SelfPlay selfPlay(rawAgent, mctsAgent, &searchLimits, playSettings, rlSettings);
    size_t numberOfGames;
    is >> numberOfGames;
    selfPlay.go(numberOfGames, states, variant);
    cout << "readyok" << endl;
}

void CrazyAra::arena(istringstream &is)
{
    SearchLimits searchLimits;
    searchLimits.nodes = size_t(Options["Nodes"]);
    SelfPlay selfPlay(rawAgent, mctsAgent, &searchLimits, playSettings, rlSettings);
    NeuralNetAPI* netSingle = create_new_net_single(Options["Model_Directory_Contender"]);
    NeuralNetAPI** netBatches = create_new_net_batches(Options["Model_Directory_Contender"]);
    MCTSAgent* mctsAgentContender = create_new_mcts_agent(netSingle, netBatches, states);
    size_t numberOfGames;
    is >> numberOfGames;
    TournamentResult tournamentResult = selfPlay.go_arena(mctsAgentContender, numberOfGames, states, variant);

    cout << "Arena summary" << endl;
    cout << "Score of Contender vs Producer: " << tournamentResult << endl;
    if (tournamentResult.score() > 0.5f) {
        cout << "replace" << endl;
    }
    else {
        cout << "keep" << endl;
    }
    write_tournament_result_to_csv(tournamentResult, "arena_results.csv");
    delete mctsAgentContender;
    delete netSingle;
}

void CrazyAra::init_rl_settings()
{
    rlSettings = new RLSettings();
    rlSettings->numberChunks = Options["Selfplay_Number_Chunks"];
    rlSettings->chunkSize = Options["Selfplay_Chunk_Size"];
    rlSettings->quickSearchNodes = Options["Quick_Nodes"];
    rlSettings->quickSearchProbability = Options["Centi_Quick_Probability"] / 100.0f;
    rlSettings->quickSearchQValueWeight = Options["Centi_Quick_Q_Value_Weight"] / 100.0f;
    rlSettings->lowPolicyClipThreshold = Options["Milli_Policy_Clip_Thresh"] / 1000.0f;
    rlSettings->quickDirichletEpsilon = Options["Centi_Quick_Dirichlet_Epsilon"] / 100.0f;
    rlSettings->nodeRandomFactor = Options["Centi_Node_Random_Factor"] / 100.0f;
    rlSettings->rawPolicyProbabilityTemperature = Options["Centi_Raw_Prob_Temperature"] / 100.0f;
    rlSettings->resignProbability = Options["Centi_Resign_Probability"] / 100.0f;
    rlSettings->resignThreshold = Options["Centi_Resign_Threshold"] / 100.0f;
    rlSettings->reuseTreeForSelpay = Options["Reuse_Tree"];
}
#endif

void CrazyAra::init()
{
    OptionsUCI::init(Options);
    Bitboards::init();
    Position::init();
    Bitbases::init();
    Search::init();
    Tablebases::init(UCI::variant_from_name(Options["UCI_Variant"]), Options["SyzygyPath"]);
}

bool CrazyAra::is_ready()
{
    if (!networkLoaded) {
        init_search_settings();
        init_play_settings();
#ifdef USE_RL
        init_rl_settings();
#endif
        netSingle = create_new_net_single(Options["Model_Directory"]);
        NeuralNetAPI** netBatches = create_new_net_batches(Options["Model_Directory"]);
        mctsAgent = create_new_mcts_agent(netSingle, netBatches, states);
        rawAgent = new RawNetAgent(netSingle, playSettings, false);
        Constants::init(mctsAgent->is_policy_map());
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
    ss << "id name " << engineName << " " << engineVersion << " (" << __DATE__ << ")" << "\n";
    ss << "id author " << engineAuthors;
    return ss.str();
}

NeuralNetAPI* CrazyAra::create_new_net_single(const string& modelDirectory)
{
#ifdef MXNET
    return new MXNetAPI(Options["Context"], int(Options["First_Device_ID"]), 1, modelDirectory, false);
#elif defined TENSORRT
    return new TensorrtAPI(int(Options["First_Device_ID"]), 1, modelDirectory, Options["Precision"]);
#endif
    return nullptr;
}

NeuralNetAPI** CrazyAra::create_new_net_batches(const string& modelDirectory)
{
#ifdef TENSORRT
    const bool useTensorRT = bool(Options["Use_TensorRT"]);
#else
    const bool useTensorRT = false;
#endif
    NeuralNetAPI** netBatches = new NeuralNetAPI*[size_t(searchSettings->threads)];
    size_t netBatchIdx = 0;
    for (int deviceId = int(Options["First_Device_ID"]); deviceId <= int(Options["Last_Device_ID"]); ++deviceId) {
        for (size_t i = 0; i < size_t(Options["Threads_per_GPU"]); ++i) {
    #ifdef MXNET
            netBatches[netBatchIdx] = new MXNetAPI(Options["Context"], deviceId, searchSettings->batchSize, modelDirectory, useTensorRT);
    #elif defined TENSORRT
            netBatches[netBatchIdx] = new TensorrtAPI(deviceId, searchSettings->batchSize, modelDirectory, Options["Precision"]);
    #endif
            ++netBatchIdx;
        }
    }
    return netBatches;
}

MCTSAgent *CrazyAra::create_new_mcts_agent(NeuralNetAPI* netSingle, NeuralNetAPI** netBatches, StatesManager* states)
{
    return new MCTSAgent(netSingle, netBatches, searchSettings, playSettings, states);
}

void CrazyAra::init_search_settings()
{
    searchSettings = new SearchSettings();
    validate_device_indices(Options);
    searchSettings->threads = Options["Threads_per_GPU"] * get_num_gpus(Options);
    searchSettings->batchSize = Options["Batch_Size"];
    searchSettings->useTranspositionTable = Options["Use_Transposition_Table"];
//    searchSettings->uInit = float(Options["Centi_U_Init_Divisor"]) / 100.0f;     currently disabled
//    searchSettings->uMin = Options["Centi_U_Min"] / 100.0f;                      currently disabled
//    searchSettings->uBase = Options["U_Base"];                                   currently disabled
    searchSettings->qValueWeight = Options["Centi_Q_Value_Weight"] / 100.0f;
    searchSettings->enhanceChecks = Options["Enhance_Checks"];                   //currently disabled
    searchSettings->enhanceCaptures = Options["Enhance_Captures"];               //currently disabled
    searchSettings->cpuctInit = Options["Centi_CPuct_Init"] / 100.0f;
    searchSettings->cpuctBase = Options["CPuct_Base"];
    searchSettings->dirichletEpsilon = Options["Centi_Dirichlet_Epsilon"] / 100.0f;
    searchSettings->dirichletAlpha = Options["Centi_Dirichlet_Alpha"] / 100.0f;
    searchSettings->nodePolicyTemperature = Options["Centi_Node_Temperature"] / 100.0f;
    searchSettings->virtualLoss = Options["Centi_Virtual_Loss"] / 100.0f;
    searchSettings->qThreshInit = Options["Centi_Q_Thresh_Init"] / 100.0f;
    searchSettings->qThreshMax = Options["Centi_Q_Thresh_Max"] / 100.0f;
    searchSettings->qThreshBase = Options["Q_Thresh_Base"];
    searchSettings->randomMoveFactor = Options["Centi_Random_Move_Factor"]  / 100.0f;
    searchSettings->allowEarlyStopping = Options["Allow_Early_Stopping"];
    useRawNetwork = Options["Use_Raw_Network"];
    searchSettings->useSolver = Options["Use_Solver"];
#ifdef SUPPORT960
    is960 = Options["UCI_Chess960"];
#endif
}

void CrazyAra::init_play_settings()
{
    playSettings = new PlaySettings();
    playSettings->initTemperature = Options["Centi_Temperature"] / 100.0f;
    playSettings->temperatureMoves = Options["Temperature_Moves"];
    playSettings->temperatureDecayFactor = Options["Centi_Temperature_Decay"] / 100.0f;
#ifdef USE_RL
    playSettings->meanInitPly = Options["MeanInitPly"];
    playSettings->maxInitPly = Options["MaxInitPly"];
#endif
}

size_t get_num_gpus(OptionsMap& option)
{
    return size_t(option["Last_Device_ID"] - option["First_Device_ID"] + 1);
}

void validate_device_indices(OptionsMap& option)
{
    if (option["Last_Device_ID"] < option["First_Device_ID"]) {
        info_string("Last_Device_ID:", option["Last_Device_ID"]);
        info_string("First_Device_ID:", option["First_Device_ID"]);
        info_string("Last_Device_ID is smaller than First_Device_ID.");
        info_string("Last_Device_ID will be set to ", option["First_Device_ID"]);
        option["Last_Device_ID"] = option["First_Device_ID"];
    }
}
