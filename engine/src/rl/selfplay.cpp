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
 * @file: selfplay.cpp
 * Created on 16.09.2019
 * @author: queensgambit
 *
 */

#ifdef USE_RL
#include "selfplay.h"

#include "thread.h"
#include <iostream>
#include <fstream>
#include "state.h"
#include "util/blazeutil.h"
#include "util/randomgen.h"


void play_move_and_update(const EvalInfo& evalInfo, StateObj* state, GamePGN& gamePGN, Result& gameResult)
{
    string sanMove = state->action_to_san(evalInfo.bestMove, evalInfo.legalMoves, false, false);
    state->do_action(evalInfo.bestMove);
    gameResult = state->check_result();

    if (is_win(gameResult)) {
        // replace '+' by '#'
        if (sanMove[sanMove.size()-1] == '+') {
            sanMove[sanMove.size()-1] = '#';
        }
        // add '#'
        else {
            sanMove += "#";
        }
    }
    gamePGN.gameMoves.emplace_back(sanMove);
}


string load_random_fen(string filepath)
{
    if (filepath == "<empty>" or filepath == "") {
        return "";
    }
    std::ifstream myfile (filepath);
    // https://stackoverflow.com/questions/33532108/pick-a-random-line-from-text-file-in-c
    std::random_device seed;
    std::mt19937 prng(seed());
    std::string line;
    std::string result;
    for(std::size_t n = 0; std::getline(myfile, line); n++) {
        std::uniform_int_distribution<> dist(0, n);
        if (dist(prng) < 1)
            result = line;
    }
    // remove last ";"
    if (result.back() == ';') {
        result.pop_back();
    }
    return result;
}


SelfPlay::SelfPlay(RawNetAgent* rawAgent, MCTSAgent* mctsAgent, SearchLimits* searchLimits, PlaySettings* playSettings,
                   RLSettings* rlSettings, OptionsMap& options):
    rawAgent(rawAgent), mctsAgent(mctsAgent), searchLimits(searchLimits), playSettings(playSettings),
    rlSettings(rlSettings), gameIdx(0), gamesPerMin(0), samplesPerMin(0), options(options)
{
    is960 = options["UCI_Chess960"];
    string suffix960 = "";
    if (is960) {
        suffix960 = "960";
    }
      
    #ifndef MODE_STRATEGO
    if (not is960 && string(options["UCI_Variant"]) == "chess") {
        // TODO: do we want standard instead of chess ?
        gamePGN.variant = "standard";
    } else {
        gamePGN.variant = string(options["UCI_Variant"]) + suffix960;
    }
    #else
        gamePGN.variant = "stratego";
    #endif

    time_t     now = time(0);
    struct tm  tstruct;
    char       date[80];
    tstruct = *localtime(&now);
    strftime(date, sizeof(date), "%Y.%m.%d %X", &tstruct);
    gamePGN.date = date;

    gamePGN.event = "SelfPlay";
    gamePGN.site = "Darmstadt, GER";
    gamePGN.round = "?";
    gamePGN.is960 = is960;
    this->exporter = new TrainDataExporter(string("data_") + mctsAgent->get_device_name() + string(".zarr"),
                                           rlSettings->numberChunks, rlSettings->chunkSize);
    filenamePGNSelfplay = string("games_") + mctsAgent->get_device_name() + string(".pgn");
    filenamePGNArena = string("arena_games_")+ mctsAgent->get_device_name() + string(".pgn");
    fileNameGameIdx = string("gameIdx_") + mctsAgent->get_device_name() + string(".txt");

    // delete content of files
    ofstream pgnFile;
    pgnFile.open(filenamePGNSelfplay, std::ios_base::trunc);
    pgnFile.close();
    ofstream idxFile;
    idxFile.open(fileNameGameIdx, std::ios_base::trunc);
    idxFile.close();

    backupNodes = searchLimits->nodes;
    backupQValueWeight = mctsAgent->get_q_value_weight();
    backupDirichletEpsilon = mctsAgent->get_dirichlet_noise();
}

SelfPlay::~SelfPlay()
{
    delete exporter;
}

void SelfPlay::adjust_node_count(SearchLimits* searchLimits, int randInt)
{
    size_t maxRandomNodes = size_t(searchLimits->nodes * rlSettings->nodeRandomFactor);
    if (maxRandomNodes != 0) {
        searchLimits->nodes += (size_t(randInt) % maxRandomNodes) - maxRandomNodes / 2;
    }
}

bool SelfPlay::is_quick_search() {
    if (rlSettings->quickSearchProbability < 0.01f) {
        return false;
    }
    return float(rand()) / RAND_MAX < rlSettings->quickSearchProbability;
}

bool SelfPlay::is_resignation_allowed() {
    if (rlSettings->resignProbability < 0.01f) {
        return false;
    }
    return float(rand()) / RAND_MAX < rlSettings->resignProbability;
}

void SelfPlay::check_for_resignation(const bool allowResingation, const EvalInfo &evalInfo, const StateObj* state, Result &gameResult)
{
    if (!allowResingation) {
        return;
    }
    if (evalInfo.bestMoveQ[0] < rlSettings->resignThreshold) {
        if (state->side_to_move() == WHITE) {
            gameResult = WHITE_WIN;
        }
        else {
            gameResult = BLACK_WIN;
        }
    }
}

void SelfPlay::reset_search_params(bool isQuickSearch)
{
    searchLimits->nodes = backupNodes;
    if (isQuickSearch) {
        mctsAgent->update_q_value_weight(backupQValueWeight);
        mctsAgent->update_dirichlet_epsilon(backupDirichletEpsilon);
    }
}

void SelfPlay::generate_game(int variant, bool verbose)
{
    chrono::steady_clock::time_point gameStartTime = chrono::steady_clock::now();

    size_t ply = size_t(random_exponential<float>(1.0f/playSettings->meanInitPly) + 0.5f);
    ply = clip_ply(ply, playSettings->maxInitPly);

    srand(unsigned(int(time(nullptr))));
    // load position from file if epd filepath was set
    string startingFen = load_random_fen(rlSettings->epdFilePath);
    unique_ptr<StateObj> state = init_starting_state_from_raw_policy(*rawAgent, ply, gamePGN, variant, is960, rlSettings->rawPolicyProbabilityTemperature, startingFen);
    EvalInfo evalInfo;
    Result gameResult;
    exporter->new_game();

    size_t generatedSamples = 0;
    const bool allowResignation = is_resignation_allowed();
    do {
        searchLimits->startTime = now();
        const int randInt = rand();
        const bool isQuickSearch = is_quick_search();

        if (isQuickSearch) {
            searchLimits->nodes = rlSettings->quickSearchNodes;
            mctsAgent->update_q_value_weight(rlSettings->quickSearchQValueWeight);
            mctsAgent->update_dirichlet_epsilon(rlSettings->quickDirichletEpsilon);
        }
        adjust_node_count(searchLimits, randInt);
        mctsAgent->set_search_settings(state.get(), searchLimits, &evalInfo);
        mctsAgent->perform_action();
        if (rlSettings->reuseTreeForSelpay) {
            mctsAgent->apply_move_to_tree(evalInfo.bestMove, true);
        }

        if (!isQuickSearch && !exporter->is_file_full()) {
            if (rlSettings->lowPolicyClipThreshold > 0) {
                sharpen_distribution(evalInfo.policyProbSmall, rlSettings->lowPolicyClipThreshold);
            }
            exporter->save_sample(state.get(), evalInfo);
            ++generatedSamples;
        }
        play_move_and_update(evalInfo, state.get(), gamePGN, gameResult);
        reset_search_params(isQuickSearch);
        check_for_resignation(allowResignation, evalInfo, state.get(), gameResult);
    }
    while(gameResult == NO_RESULT);

    // export all training samples of the generated game
    exporter->export_game_samples(gameResult);

    set_game_result_to_pgn(gameResult);
    write_game_to_pgn(filenamePGNSelfplay, verbose);
    clean_up(gamePGN, mctsAgent);

    // measure time statistics
    if (verbose) {
        const float elapsedTimeMin = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - gameStartTime).count() / 60000.f;
        speed_statistic_report(elapsedTimeMin, generatedSamples);
    }
    ++gameIdx;
}

Result SelfPlay::generate_arena_game(MCTSAgent* whitePlayer, MCTSAgent* blackPlayer, int variant, bool verbose, const string& fen)
{
    gamePGN.white = whitePlayer->get_name();
    gamePGN.black = blackPlayer->get_name();
    unique_ptr<StateObj> state = make_unique<StateObj>();
    //unique_ptr<StateObj> state = init_starting_state_from_raw_policy(*rawAgent, 0, gamePGN, variant, is960, rlSettings->rawPolicyProbabilityTemperature);
    
    if (fen != "") {
        // set starting fen
        state->set(fen, is960, variant);
    } else {
        // create new starting fen and return it
        state->init(variant, is960);
    }
    gamePGN.fen = state->fen();
    EvalInfo evalInfo;

    MCTSAgent* activePlayer;
    MCTSAgent* passivePlayer;
    // preserve the current active states
    Result gameResult;
    do {
        searchLimits->startTime = now();
        if (state->side_to_move() == WHITE) {
            activePlayer = whitePlayer;
            passivePlayer = blackPlayer;
        }
        else {
            activePlayer = blackPlayer;
            passivePlayer = whitePlayer;
        }
        activePlayer->set_search_settings(state.get(), searchLimits, &evalInfo);
        activePlayer->perform_action();
        activePlayer->apply_move_to_tree(evalInfo.bestMove, true);
        if (state->steps_from_null() != 0) {
            passivePlayer->apply_move_to_tree(evalInfo.bestMove, false);
        }
        play_move_and_update(evalInfo, state.get(), gamePGN, gameResult);
    }
    while(gameResult == NO_RESULT);
    set_game_result_to_pgn(gameResult);
    write_game_to_pgn(filenamePGNArena, verbose);
    clean_up(gamePGN, whitePlayer);
    blackPlayer->clear_game_history();
    return gameResult;
}

void clean_up(GamePGN& gamePGN, MCTSAgent* mctsAgent)
{
    gamePGN.new_game();
    mctsAgent->clear_game_history();
}

void SelfPlay::write_game_to_pgn(const std::string& pgnFileName, bool verbose)
{
    ofstream pgnFile;
    pgnFile.open(pgnFileName, std::ios_base::app);
    if (verbose) {
        cout << endl << gamePGN << endl;
    }
    pgnFile << gamePGN << endl;
    pgnFile.close();
}

void SelfPlay::set_game_result_to_pgn(Result res)
{
    gamePGN.result = result[res];
}

void SelfPlay::reset_speed_statistics()
{
    gameIdx = 0;
    gamesPerMin = 0;
    samplesPerMin = 0;
}

void SelfPlay::speed_statistic_report(float elapsedTimeMin, size_t generatedSamples)
{
    // compute running cumulative average
    gamesPerMin = (gameIdx * gamesPerMin + (1 / elapsedTimeMin)) / (gameIdx + 1);
    samplesPerMin = (gameIdx * samplesPerMin + (generatedSamples / elapsedTimeMin)) / (gameIdx + 1);

    cout << "    games    |  games/min  | samples/min " << endl
         << "-------------+-------------+-------------" << endl
         << std::setprecision(5)
         << setw(13) << gameIdx << '|'
         << setw(13) << gamesPerMin << '|'
         << setw(13) << samplesPerMin << endl << endl;
}

void SelfPlay::export_number_generated_games() const
{
    ofstream gameIdxFile;
    gameIdxFile.open(fileNameGameIdx);
    gameIdxFile << gameIdx;
    gameIdxFile.close();
}


void SelfPlay::go(size_t numberOfGames, int variant)
{
    reset_speed_statistics();
    gamePGN.white = mctsAgent->get_name();
    gamePGN.black = mctsAgent->get_name();

    if (numberOfGames == 0) {
        while(!exporter->is_file_full()) {
            generate_game(variant, true);
        }
    }
    else {
        for (size_t idx = 0; idx < numberOfGames; ++idx) {
            generate_game(variant, true);
        }
    }
    export_number_generated_games();
}

TournamentResult SelfPlay::go_arena(MCTSAgent *mctsContender, size_t numberOfGames, int variant)
{
    TournamentResult tournamentResult;
    tournamentResult.playerA = mctsContender->get_name();
    tournamentResult.playerB = mctsAgent->get_name();
    Result gameResult;

    for (size_t idx = 0; idx < numberOfGames; ++idx) {
        if (idx % 2 == 0) {
            string startFen = load_random_fen(rlSettings->epdFilePath);
            // use default or in case of chess960 a random starting position
            gameResult = generate_arena_game(mctsContender, mctsAgent, variant, true, startFen);
            if (gameResult == WHITE_WIN) {
                ++tournamentResult.numberWins;
            }
            else if (gameResult == BLACK_WIN){
                ++tournamentResult.numberLosses;
            }
        }
        else {
            // use same starting position as before stored via gamePGN.fen
            gameResult = generate_arena_game(mctsAgent, mctsContender, variant, true, gamePGN.fen);
            if (gameResult == BLACK_WIN) {
                ++tournamentResult.numberWins;
            }
            else if (gameResult == WHITE_WIN){
                ++tournamentResult.numberLosses;
            }
        }
        if (gameResult == DRAWN) {
            ++tournamentResult.numberDraws;
        }
    }
    return tournamentResult;
}

unique_ptr<StateObj> init_starting_state_from_raw_policy(RawNetAgent &rawAgent, size_t plys, GamePGN &gamePGN, int variant, bool is960, float rawPolicyProbTemp, const string& fen)
{
    unique_ptr<StateObj> state= make_unique<StateObj>();
    if (fen != "") {
        // set starting fen
        state->set(fen, is960, variant);
    } else {
        // create new starting fen and return it
        state->init(variant, is960);
    }
    gamePGN.fen = state->fen();
    
    for (size_t ply = 0; ply < plys; ++ply) {
        EvalInfo eval;
        rawAgent.set_search_settings(state.get(), nullptr, &eval);
        rawAgent.evaluate_board_state();
        apply_raw_policy_temp(eval, rawPolicyProbTemp);
        const size_t moveIdx = random_choice(eval.policyProbSmall);
        eval.bestMove = eval.legalMoves[moveIdx];

        if (state->leads_to_terminal(eval.bestMove)) {
            break;
        }
        else {
            gamePGN.gameMoves.push_back(state->action_to_san(eval.legalMoves[moveIdx], eval.legalMoves, false, true));
            state->do_action(eval.bestMove);
        }
    }
    return state;
}

unique_ptr<StateObj> init_starting_state_from_fixed_move(GamePGN &gamePGN, int variant, bool is960, const vector<Action>& actions)
{
    unique_ptr<StateObj> state= make_unique<StateObj>();
    state->init(variant, is960);
    gamePGN.fen = state->fen();
    for (Action action : actions) {
        gamePGN.gameMoves.push_back(state->action_to_san(action, {}, false, true));
        state->do_action(action);
    }
    return state;
}

size_t clip_ply(size_t ply, size_t maxPly)
{
    if (ply > maxPly) {
        return size_t(rand()) % maxPly;
    }
    return ply;
}

void apply_raw_policy_temp(EvalInfo &eval, float rawPolicyProbTemp)
{
    if (float(rand()) / RAND_MAX < rawPolicyProbTemp) {
        float temp = 2.0f;
        const float prob = float(rand()) / INT_MAX;
        if (prob < 0.05f) {
            temp = 10.0f;
        }
        else if (prob < 0.25f) {
            temp = 5.0f;
        }
        apply_temperature(eval.policyProbSmall, temp);
    }
}
#endif
