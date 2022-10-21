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

#include <thread>
#include <fstream>
#include "mctsagent.h"
#include "search.h"
#include "evalinfo.h"
#include "constants.h"
#include "state.h"
#include "optionsuci.h"
#include "../tests/benchmarkpositions.h"
#include "util/communication.h"
#if defined(MODE_XIANGQI) || defined(MODE_BOARDGAMES)
#include "piece.h"
#endif
#ifdef MXNET
#include "nn/mxnetapi.h"
#elif defined TENSORRT
#include "nn/tensorrtapi.h"
#elif defined OPENVINO
#include "nn/openvinoapi.h"
#endif


CrazyAra::CrazyAra():
    rawAgent(nullptr),
    mctsAgent(nullptr),
    netSingle(nullptr),         // will be initialized in is_ready()
#ifdef USE_RL
    netSingleContender(nullptr),
    mctsAgentContender(nullptr),
#endif
    searchSettings(SearchSettings()),
    searchLimits(SearchLimits()),
    playSettings(PlaySettings()),
    variant(StateConstants::DEFAULT_VARIANT()),
    useRawNetwork(false),      // will be initialized in init_search_settings()
    networkLoaded(false),
    ongoingSearch(false),
#ifdef SUPPORT960
    is960(true),
#else
    is960(false),
#endif
    changedUCIoption(false)
{
}

CrazyAra::~CrazyAra()
{
}

void CrazyAra::welcome()
{
    cout << intro << endl;
}

void CrazyAra::uci_loop(int argc, char *argv[])
{
    unique_ptr<StateObj> state = make_unique<StateObj>();
    string token, cmd;
    EvalInfo evalInfo;
    variant = StateConstants::variant_to_int(Options["UCI_Variant"]);
    state->set(StateConstants::start_fen(variant), is960, variant);

    for (int i = 1; i < argc; ++i)
        cmd += string(argv[i]) + " ";

    size_t it = 0;

    // this is debug vector which can contain uci commands which will be automatically processed when the executable is launched
    vector<string> commands = {
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
            stop_search();
        }
        else if (token == "uci") {
            cout << engine_info()
                 << Options << endl
                 << "uciok" << endl;
        }
        else if (token == "setoption")  set_uci_option(is, *state.get());
        else if (token == "go")         go(state.get(), is, evalInfo);
        else if (token == "position")   position(state.get(), is);
        else if (token == "ucinewgame") ucinewgame();
        else if (token == "isready")    is_ready<true>();

        // Additional custom non-UCI commands, mainly for debugging
        else if (token == "benchmark")  benchmark(is);
        else if (token == "root")       mctsAgent->print_root_node();
        else if (token == "tree")      export_search_tree(is);
        else if (token == "flip")       state->flip();
        else if (token == "d")          cout << *(state.get()) << endl;
        else if (token == "activeuci") activeuci();
#ifdef USE_RL
        else if (token == "selfplay")   selfplay(is);
        else if (token == "arena")      arena(is);
        // Test if the new modes are also usable for chess and others

        else if (token == "match")   multimodel_arena(is, "", "", true);
        else if (token == "tournament")   roundrobin(is);
#endif   
        else
            cout << "Unknown command: " << cmd << endl;

        ++it;
    } while (token != "quit" && argc == 1); // Command line args are one-shot

    wait_to_finish_last_search();
}

void CrazyAra::prepare_search_config_structs()
{
    OptionsUCI::init_new_search(searchLimits, Options);

    if (changedUCIoption) {
        init_search_settings();
        init_play_settings();
        changedUCIoption = false;
    }
}

void CrazyAra::go(StateObj* state, istringstream &is,  EvalInfo& evalInfo)
{
    wait_to_finish_last_search();
    ongoingSearch = true;
    prepare_search_config_structs();

    string token;
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
    }

    if (useRawNetwork) {
        rawAgent->set_search_settings(state, &searchLimits, &evalInfo);
        rawAgent->lock();  // lock() rawAgent to avoid calling stop() immediatly
        mainSearchThread = thread(run_agent_thread, rawAgent.get());
    }
    else {
        mctsAgent->set_search_settings(state, &searchLimits, &evalInfo);
        mctsAgent->lock(); // lock() mctsAgent to avoid calling stop() immediatly
        mainSearchThread = thread(run_agent_thread, mctsAgent.get());
    }
}

void CrazyAra::go(const string& fen, string goCommand, EvalInfo& evalInfo)
{
    unique_ptr<StateObj> state = make_unique<StateObj>();
    string token, cmd;

    variant = StateConstants::variant_to_int(Options["UCI_Variant"]);
    state->set(StateConstants::start_fen(variant), is960, variant);

    istringstream is("fen " + fen);

    position(state.get(), is);
    istringstream isGoCommand(goCommand);
    go(state.get(), isGoCommand, evalInfo);
}

void CrazyAra::wait_to_finish_last_search()
{
    if (ongoingSearch) {
        mainSearchThread.join();
        ongoingSearch = false;
    }
}

void CrazyAra::stop_search()
{
    if (mctsAgent != nullptr) {
        mctsAgent->stop();
        wait_to_finish_last_search();
    }
}

void CrazyAra::position(StateObj* state, istringstream& is)
{
    wait_to_finish_last_search();

    Action action;
    string token, fen;
    variant = StateConstants::variant_to_int(Options["UCI_Variant"]);

    is >> token;
    if (token == "startpos")
    {
        fen = StateConstants::start_fen(variant);
        is >> token; // Consume "moves" token if any
    }
    else if (token == "fen") {
        while (is >> token && token != "moves")
            fen += token + " ";
        fen = fen.substr(0, fen.length()-1);  // remove last ' ' to avoid parsing problems
    }
    else
        return;

    state->set(fen, is960, variant);
    Action lastMove = ACTION_NONE;

    // Parse move list (if any)
    while (is >> token && (action = state->uci_to_action(token)) != ACTION_NONE)
    {
        state->do_action(action);
        lastMove = action;
    }
    // inform the mcts agent of the move, so the tree can potentially be reused later
    if (lastMove != MOVE_NULL && !useRawNetwork) {
        mctsAgent->apply_move_to_tree(lastMove, false);
    }
    info_string("position", state->fen());
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
        string uciMove = StateConstants::action_to_uci(evalInfo.bestMove, false);
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

void CrazyAra::export_search_tree(istringstream &is)
{
    string depth, filename;
    is >> depth;
    is >> filename;
    if (depth == "") {
        mctsAgent->export_search_tree(2, "tree.gv");
        return;
    }
    if (filename == "") {
        mctsAgent->export_search_tree(std::stoi(depth), "tree.gv");
        return;
    }
    mctsAgent->export_search_tree(std::stoi(depth), filename);
}

void CrazyAra::activeuci()
{
    for (const auto& it : Options)
        cout << "option name " << it.first << " value " << string(Options[it.first]) << endl;
    cout << "readyok" << endl;
}

#ifdef USE_RL
void CrazyAra::selfplay(istringstream &is)
{
    prepare_search_config_structs();
    SelfPlay selfPlay(rawAgent.get(), mctsAgent.get(), &searchLimits, &playSettings, &rlSettings, Options);
    size_t numberOfGames;
    is >> numberOfGames;
    selfPlay.go(numberOfGames, variant);
    cout << "readyok" << endl;
}

void CrazyAra::arena(istringstream &is)
{
    prepare_search_config_structs();
    SelfPlay selfPlay(rawAgent.get(), mctsAgent.get(), &searchLimits, &playSettings, &rlSettings, Options);
    netSingleContender = create_new_net_single(Options["Model_Directory_Contender"]);
    netBatchesContender = create_new_net_batches(Options["Model_Directory_Contender"]);
    mctsAgentContender = create_new_mcts_agent(netSingleContender.get(), netBatchesContender, &searchSettings);
    size_t numberOfGames;
    is >> numberOfGames;
    TournamentResult tournamentResult = selfPlay.go_arena(mctsAgentContender.get(), numberOfGames, variant);

    cout << "Arena summary" << endl;
    cout << "Score of Contender vs Producer: " << tournamentResult << endl;
    if (tournamentResult.score() > 0.5f) {
        cout << "replace" << endl;
    }
    else {
        cout << "keep" << endl;
    }
    write_tournament_result_to_csv(tournamentResult, "arena_results.csv");
}

void CrazyAra::multimodel_arena(istringstream &is, const string &modelDirectory1, const string &modelDirectory2, bool isModelInInputStream)
{
    SearchLimits searchLimits;
    searchLimits.nodes = size_t(Options["Nodes"]);

    // create two MCTS agents
    int type;
    int folder;
    is >> type;
    string modelDir1 = modelDirectory1;
    if (isModelInInputStream)
    {
        is >> folder;
        modelDir1 = "m" + std::to_string(folder) + "/";
    }
    auto mcts1 = create_new_mcts_agent(netSingle.get(), netBatches, &searchSettings, static_cast<MCTSAgentType>(type));
    if (modelDir1 != "")
    {
        netSingle = create_new_net_single(modelDir1);
        netBatches = create_new_net_batches(modelDir1);
        mcts1 = create_new_mcts_agent(netSingle.get(), netBatches, &searchSettings, static_cast<MCTSAgentType>(type));
    }

    is >> type;
    string modelDir2 = modelDirectory2;
    if (isModelInInputStream)
    {
        is >> folder;
        modelDir2 = "m" + std::to_string(folder) + "/";
    }
    auto mcts2 = create_new_mcts_agent(netSingle.get(), netBatches, &searchSettings, static_cast<MCTSAgentType>(type));
    if (modelDir2 != "")
    {
        netSingleContender = create_new_net_single(modelDir2);
        netBatchesContender = create_new_net_batches(modelDir2);
        mcts2 = create_new_mcts_agent(netSingleContender.get(), netBatchesContender, &searchSettings, static_cast<MCTSAgentType>(type));
    }

    SelfPlay selfPlay(rawAgent.get(), mcts1.get(), &searchLimits, &playSettings, &rlSettings, Options);
    size_t numberOfGames;
    is >> numberOfGames;
    TournamentResult tournamentResult = selfPlay.go_arena(mcts2.get(), numberOfGames, variant);

    cout << "Arena summary" << endl;
    cout << "Score of Agent1 vs Agent2: " << tournamentResult << endl;
    write_tournament_result_to_csv(tournamentResult, "mcts_arena_results.csv");
}

void CrazyAra::roundrobin(istringstream &is)
{
    int type;
    int numberofgames;
    is >> numberofgames;
    struct modelstring
    {
        int number_of_mcts_agent;
        int number_of_model_folder;
    };

    int i = 0;
    std::vector<modelstring> agents;
    std::vector<int> numbers;
    while (!is.eof())
    {
        is >> type;
        int tmp1 = type;
        is >> type;

        modelstring tmp;
        tmp.number_of_mcts_agent = tmp1;
        tmp.number_of_model_folder = type;
        std::cout << "ini " << tmp1 << " " << type << std::endl;
        agents.push_back(tmp);
        numbers.push_back(i);
        i++;
    }

    std::vector<std::string> combinations = comb(numbers, 2);
    std::string delimiter = " ";
    for (int i = 0; i < combinations.size(); i++)
    {
        std::string s = combinations[i];

        int token1 = std::stoi(s.substr(0, s.find(delimiter)));
        int token2 = std::stoi(s.substr(2, s.find(delimiter)));
        std::string comb = std::to_string(agents[token1].number_of_mcts_agent) + " " + std::to_string(agents[token2].number_of_mcts_agent);
        std::string m1 = "m" + std::to_string(agents[token1].number_of_model_folder) + "/";
        std::string m2 = "m" + std::to_string(agents[token2].number_of_model_folder) + "/";
        std::istringstream iss(comb + " " + std::to_string(numberofgames));
        multimodel_arena(iss, m1, m2, false);
    }

    exit(0);
}

void CrazyAra::init_rl_settings()
{
    rlSettings.numberChunks = Options["Selfplay_Number_Chunks"];
    rlSettings.chunkSize = Options["Selfplay_Chunk_Size"];
    rlSettings.quickSearchNodes = Options["Quick_Nodes"];
    rlSettings.quickSearchProbability = Options["Centi_Quick_Probability"] / 100.0f;
    rlSettings.quickSearchQValueWeight = Options["Centi_Quick_Q_Value_Weight"] / 100.0f;
    rlSettings.lowPolicyClipThreshold = Options["Milli_Policy_Clip_Thresh"] / 1000.0f;
    rlSettings.quickDirichletEpsilon = Options["Centi_Quick_Dirichlet_Epsilon"] / 100.0f;
    rlSettings.nodeRandomFactor = Options["Centi_Node_Random_Factor"] / 100.0f;
    rlSettings.rawPolicyProbabilityTemperature = Options["Centi_Raw_Prob_Temperature"] / 100.0f;
    rlSettings.resignProbability = Options["Centi_Resign_Probability"] / 100.0f;
    rlSettings.resignThreshold = Options["Centi_Resign_Threshold"] / 100.0f;
    rlSettings.reuseTreeForSelpay = Options["Reuse_Tree"];
    rlSettings.epdFilePath = string(Options["EPD_File_Path"]);
    if (rlSettings.epdFilePath != "<empty>" and rlSettings.epdFilePath != "") {
        std::ifstream epdFile (rlSettings.epdFilePath);
        if (!epdFile.is_open()) {
            throw invalid_argument("Given epd file: " + rlSettings.epdFilePath + " could not be opened.");
        }
    }
}
#endif

std::string read_string_from_file(const std::string &file_path){
    const std::ifstream input_stream(file_path, std::ios_base::binary);

    if (input_stream.fail()) {
        throw std::runtime_error("Failed to open file");
    }

    std::stringstream buffer;
    buffer << input_stream.rdbuf();

    return buffer.str();
}

void CrazyAra::init()
{
    OptionsUCI::init(Options);
#ifdef MODE_XIANGQI
    UCI::init(Options);
    pieceMap.init();
#endif
#ifdef MODE_BOARDGAMES
    UCI::init(Options);
    pieceMap.init();
    fstream variantIni;
    const string file_path = "variants.ini";
    string variantInitContent = read_string_from_file(file_path);
    std::stringstream ss(variantInitContent);
    variants.parse_istream<false>(ss);
    Options["UCI_Variant"].set_combo(variants.get_keys());
#endif
#ifdef SF_DEPENDENCY
    Bitboards::init();
    Position::init();
    Bitbases::init();
    Search::init();
#endif
#if defined(MODE_XIANGQI) || defined(MODE_BOARDGAMES)
    // This is a workaround for compatibility with Fairy-Stockfish
    // Option with key "Threads" is also removed. (See /3rdparty/Fairy-Stockfish/src/ucioption.cpp)
    Options.erase("Hash");
    Options.erase("Use NNUE");
#endif
}

template<bool verbose>
bool CrazyAra::is_ready()
{
    bool hasReplied = false;
    if (!networkLoaded) {
        const size_t timeoutMS = Options["Timeout_MS"];
        TimeOutReadyThread timeoutThread(timeoutMS);
        thread tTimeoutThread;
        if (timeoutMS != 0) {
            tTimeoutThread = thread(run_timeout_thread, &timeoutThread);
        }
        init_search_settings();
        init_play_settings();
#ifdef USE_RL
        init_rl_settings();
#endif
        netSingle = create_new_net_single(string(Options["Model_Directory"]));
        netSingle->validate_neural_network();
        netBatches = create_new_net_batches(string(Options["Model_Directory"]));
        netBatches.front()->validate_neural_network();
        mctsAgent = create_new_mcts_agent(netSingle.get(), netBatches, &searchSettings);
        rawAgent = make_unique<RawNetAgent>(netSingle.get(), &playSettings, false);
        StateConstants::init(mctsAgent->is_policy_map());
        timeoutThread.kill();
        if (timeoutMS != 0) {
            tTimeoutThread.join();
        }
        hasReplied = timeoutThread.has_replied();
        networkLoaded = true;
    }
    wait_to_finish_last_search();
    if (verbose && !hasReplied) {
        cout << "readyok" << endl;
    }
    return networkLoaded;
}

void CrazyAra::ucinewgame()
{
    if (networkLoaded) {
        wait_to_finish_last_search();
        mctsAgent->clear_game_history();
        cout << "info string newgame" << endl;
    }
}

string CrazyAra::engine_info()
{
    stringstream ss;
    ss << "id name " << engineName << " " << engineVersion << " (" << __DATE__ << ")" << "\n";
    ss << "id author " << engineAuthors;
    return ss.str();
}

unique_ptr<NeuralNetAPI> CrazyAra::create_new_net_single(const string& modelDirectory)
{
#ifdef MXNET
    return make_unique<MXNetAPI>(Options["Context"], int(Options["First_Device_ID"]), 1, modelDirectory, Options["Precision"], false);
#elif defined TENSORRT
    return make_unique<TensorrtAPI>(int(Options["First_Device_ID"]), 1, modelDirectory, Options["Precision"]);
#elif defined OPENVINO
    return make_unique<OpenVinoAPI>(int(Options["First_Device_ID"]), 1, modelDirectory, Options["Threads_NN_Inference"]);
#endif
    return nullptr;
}

vector<unique_ptr<NeuralNetAPI>> CrazyAra::create_new_net_batches(const string& modelDirectory)
{
    vector<unique_ptr<NeuralNetAPI>> netBatches;
#ifdef MXNET
    #ifdef TENSORRT
        const bool useTensorRT = bool(Options["Use_TensorRT"]);
    #else
        const bool useTensorRT = false;
    #endif
#endif
    for (int deviceId = int(Options["First_Device_ID"]); deviceId <= int(Options["Last_Device_ID"]); ++deviceId) {
        for (size_t i = 0; i < size_t(Options["Threads"]); ++i) {
    #ifdef MXNET
            netBatches.push_back(make_unique<MXNetAPI>(Options["Context"], deviceId, searchSettings.batchSize, modelDirectory, Options["Precision"], useTensorRT));
    #elif defined TENSORRT
            netBatches.push_back(make_unique<TensorrtAPI>(deviceId, searchSettings.batchSize, modelDirectory, Options["Precision"]));
    #elif defined OPENVINO
            netBatches.push_back(make_unique<OpenVinoAPI>(deviceId, searchSettings.batchSize, modelDirectory, Options["Threads_NN_Inference"]));
    #endif
        }
    }
    return netBatches;
}

void CrazyAra::set_uci_option(istringstream &is, StateObj& state)
{
    // these three UCI-Options may trigger a network reload, keep an eye on them
    const string prevModelDir = Options["Model_Directory"];
    const int prevThreads = Options["Threads"];
    const string prevUciVariant = Options["UCI_Variant"];
    const int prevFirstDeviceID = Options["First_Device_ID"];
    const int prevLastDeviceID = Options["Last_Device_ID"];

    OptionsUCI::setoption(is, variant, state);
    changedUCIoption = true;
    if (networkLoaded) {
        if (string(Options["Model_Directory"]) != prevModelDir || int(Options["Threads"]) != prevThreads || string(Options["UCI_Variant"]) != prevUciVariant ||
            int(Options["First_Device_ID"]) != prevFirstDeviceID || int(Options["Last_Device_ID"] != prevLastDeviceID)) {
            networkLoaded = false;
            is_ready<false>();
        }
    }
}

unique_ptr<MCTSAgent> CrazyAra::create_new_mcts_agent(NeuralNetAPI* netSingle, vector<unique_ptr<NeuralNetAPI>>& netBatches, SearchSettings* searchSettings, MCTSAgentType type)
{   
    switch (type) {
    case MCTSAgentType::kDefault:
        return make_unique<MCTSAgent>(netSingle, netBatches, searchSettings, &playSettings);
    case MCTSAgentType::kBatch1:
        info_string("TYP 1 -> Batch 1");
        return make_unique<MCTSAgentBatch>(netSingle, netBatches, searchSettings, &playSettings , 1, false);
    case MCTSAgentType::kBatch3:
        info_string("TYP 2 -> Batch 3");
        return make_unique<MCTSAgentBatch>(netSingle, netBatches, searchSettings, &playSettings , 3, false);
    case MCTSAgentType::kBatch5:
        info_string("TYP 3 -> Batch 5");
        return make_unique<MCTSAgentBatch>(netSingle, netBatches, searchSettings, &playSettings , 5, false);
    case MCTSAgentType::kBatch3_reducedNodes:
        info_string("TYP 4 -> Batch 3 Split");
        return make_unique<MCTSAgentBatch>(netSingle, netBatches, searchSettings, &playSettings , 3, true);
    case MCTSAgentType::kBatch5_reducedNodes:
        info_string("TYP 5 -> Batch 5 Split");
        return make_unique<MCTSAgentBatch>(netSingle, netBatches, searchSettings, &playSettings , 5, true);
    case MCTSAgentType::kTrueSight:
        info_string("TYP 6 -> TrueSight");
        return make_unique<MCTSAgentTrueSight>(netSingle, netBatches, searchSettings, &playSettings);
    case MCTSAgentType::kRandom:
        info_string("TYP 7 -> Random");
        return make_unique<MCTSAgentRandom>(netSingle, netBatches, searchSettings, &playSettings);
    default:
      info_string("Unknown MCTSAgentType");
      return nullptr;
  }
}

void CrazyAra::init_search_settings()
{
    validate_device_indices(Options);
    searchSettings.multiPV = Options["MultiPV"];
    searchSettings.threads = Options["Threads"] * get_num_gpus(Options);
    searchSettings.batchSize = Options["Batch_Size"];
    searchSettings.useMCGS = Options["Search_Type"] == "mcgs";
//    searchSettings.uInit = float(Options["Centi_U_Init_Divisor"]) / 100.0f;     currently disabled
//    searchSettings.uMin = Options["Centi_U_Min"] / 100.0f;                      currently disabled
//    searchSettings.uBase = Options["U_Base"];                                   currently disabled
    searchSettings.qValueWeight = Options["Centi_Q_Value_Weight"] / 100.0f;
    searchSettings.qVetoDelta = Options["Centi_Q_Veto_Delta"] / 100.0f;
    searchSettings.epsilonChecksCounter = round((1.0f / Options["Centi_Epsilon_Checks"]) * 100.0f);
    searchSettings.epsilonGreedyCounter = round((1.0f / Options["Centi_Epsilon_Greedy"]) * 100.0f);
//    searchSettings.enhanceCaptures = Options["Enhance_Captures"];               //currently disabled
    searchSettings.cpuctInit = Options["Centi_CPuct_Init"] / 100.0f;
    searchSettings.cpuctBase = Options["CPuct_Base"];
    searchSettings.dirichletEpsilon = Options["Centi_Dirichlet_Epsilon"] / 100.0f;
    searchSettings.dirichletAlpha = Options["Centi_Dirichlet_Alpha"] / 100.0f;
    searchSettings.nodePolicyTemperature = Options["Centi_Node_Temperature"] / 100.0f;
    searchSettings.virtualLoss = Options["Centi_Virtual_Loss"] / 100.0f;
    searchSettings.randomMoveFactor = Options["Centi_Random_Move_Factor"]  / 100.0f;
    searchSettings.allowEarlyStopping = Options["Allow_Early_Stopping"];
    useRawNetwork = Options["Use_Raw_Network"];
#ifdef SUPPORT960
    is960 = Options["UCI_Chess960"];
#endif
    searchSettings.useNPSTimemanager = Options["Use_NPS_Time_Manager"];
    if (string(Options["SyzygyPath"]).empty() || string(Options["SyzygyPath"]) == "<empty>") {
        searchSettings.useTablebase = false;
    }
    else {
        searchSettings.useTablebase = true;
    }
    searchSettings.reuseTree = Options["Reuse_Tree"];
    searchSettings.mctsSolver = Options["MCTS_Solver"];
}

void CrazyAra::init_play_settings()
{
    playSettings.initTemperature = Options["Centi_Temperature"] / 100.0f;
    playSettings.temperatureMoves = Options["Temperature_Moves"];
    playSettings.temperatureDecayFactor = Options["Centi_Temperature_Decay"] / 100.0f;
    playSettings.quantileClipping = Options["Centi_Quantile_Clipping"] / 100.0f;
#ifdef USE_RL
    playSettings.meanInitPly = Options["MeanInitPly"];
    playSettings.maxInitPly = Options["MaxInitPly"];
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

std::vector<std::string> comb(std::vector<int> N, int K)
{
    std::string bitmask(K, 1); // K leading 1's
    bitmask.resize(N.size(), 0); // N-K trailing 0's
    std::vector<std::string> p ;
    // print integers and permute bitmask

    do {
        std::string c = "";
        for (int i = 0; i < N.size(); ++i) // [0..N-1] integers
        {
            if (bitmask[i]){
                c.append(std::to_string(N[i])+ " ");
            } 
        }
        p.push_back(c);
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

    return p;
}
