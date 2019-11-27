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
#include "../domain/variants.h"
#include "../util/blazeutil.h"
#include "../util/randomgen.h"

SelfPlay::SelfPlay(RawNetAgent* rawAgent, MCTSAgent* mctsAgent, SearchLimits* searchLimits, PlaySettings* playSettings, RLSettings* rlSettings):
    rawAgent(rawAgent), mctsAgent(mctsAgent), searchLimits(searchLimits), playSettings(playSettings), rlSettings(rlSettings),
    gameIdx(0), gamesPerMin(0), samplesPerMin(0)
{
    gamePGN.variant = "crazyhouse";
    gamePGN.event = "CrazyAra-SelfPlay";
    gamePGN.site = "Darmstadt, GER";
    gamePGN.date = "?";  // TODO: Change this later
    gamePGN.round = "?";
    gamePGN.is960 = false;
    this->exporter = new TrainDataExporter(string("data_") + mctsAgent->get_device_name() + string(".zarr"),
                                           rlSettings->numberChunks, rlSettings->chunkSize);
    filenamePGNSelfplay = string("games_") + mctsAgent->get_device_name() + string(".pgn");
    filenamePGNArena = string("arena_games_")+ mctsAgent->get_device_name() + string(".pgn");
    fileNameGameIdx = string("gameIdx_") + mctsAgent->get_device_name() + string(".txt");

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

void SelfPlay::reset_search_params(bool isQuickSearch)
{
    searchLimits->nodes = backupNodes;
    if (isQuickSearch) {
        mctsAgent->update_q_value_weight(backupQValueWeight);
        mctsAgent->update_dirichlet_epsilon(backupDirichletEpsilon);
    }
}

void SelfPlay::generate_game(Variant variant, StatesManager* states, bool verbose)
{
    chrono::steady_clock::time_point gameStartTime = chrono::steady_clock::now();

    size_t ply = size_t(random_exponential<float>(1.0f/playSettings->meanInitPly) + 0.5f);
    ply = clip_ply(ply, playSettings->maxInitPly);

    Board* position = init_starting_pos_from_raw_policy(*rawAgent, ply, gamePGN, variant, states);
    EvalInfo evalInfo;
    states->swap_states();
    bool leadsToTerminal = false;
    Node* nextRoot;
    exporter->new_game();

    srand(unsigned(int(time(nullptr))));
    size_t generatedSamples = 0;
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
        mctsAgent->perform_action(position, searchLimits, evalInfo);
        mctsAgent->apply_move_to_tree(evalInfo.bestMove, true);
        nextRoot = mctsAgent->get_opponents_next_root();

        if (nextRoot != nullptr) {
            leadsToTerminal = nextRoot->is_terminal();
        }
        if (!isQuickSearch && !exporter->is_file_full()) {
            sharpen_distribution(evalInfo.policyProbSmall, rlSettings->lowPolicyClipThreshold);
            exporter->save_sample(position, evalInfo);
            ++generatedSamples;
        }
        StateInfo* newState = new StateInfo;
        states->activeStates.push_back(newState);
        position->do_move(evalInfo.bestMove, *(newState));
        gamePGN.gameMoves.push_back(pgn_move(evalInfo.bestMove,
                                            false,
                                            *mctsAgent->get_root_node()->get_pos(),
                                            evalInfo.legalMoves,
                                            leadsToTerminal && int(nextRoot->get_value()) == -1));
        reset_search_params(isQuickSearch);
    }
    while(!leadsToTerminal);

    // export all training samples of the generated game
    exporter->export_game_samples(get_terminal_node_result(nextRoot));

    set_game_result_to_pgn(nextRoot);
    write_game_to_pgn(filenamePGNSelfplay, verbose);
    clean_up(gamePGN, mctsAgent, states, position);

    // measure time statistics
    if (verbose) {
        const float elapsedTimeMin = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - gameStartTime).count() / 60000.f;
        speed_statistic_report(elapsedTimeMin, generatedSamples);
    }
    ++gameIdx;
}

Result SelfPlay::generate_arena_game(MCTSAgent* whitePlayer, MCTSAgent* blackPlayer, Variant variant, StatesManager* states, bool verbose)
{
    gamePGN.white = whitePlayer->get_name();
    gamePGN.black = blackPlayer->get_name();
    Board* position = init_board(variant, states);
    EvalInfo evalInfo;
    bool isTerminal = false;

    MCTSAgent* activePlayer;
    MCTSAgent* passivePlayer;
    // preserve the current active states
    states->swap_states();

    const Node* nextRoot;
    do {
        searchLimits->startTime = now();
        if (position->side_to_move() == WHITE) {
            activePlayer = whitePlayer;
            passivePlayer = blackPlayer;
        }
        else {
            activePlayer = blackPlayer;
            passivePlayer = whitePlayer;
        }
        activePlayer->perform_action(position, searchLimits, evalInfo);
        activePlayer->apply_move_to_tree(evalInfo.bestMove, true);
        if (position->plies_from_null() != 0) {
            passivePlayer->apply_move_to_tree(evalInfo.bestMove, false);
        }
        nextRoot = activePlayer->get_opponents_next_root();
        if (nextRoot != nullptr) {
            isTerminal = nextRoot->is_terminal();
        }
        StateInfo* newState = new StateInfo;
        states->activeStates.push_back(newState);
        position->do_move(evalInfo.bestMove, *(newState));
        gamePGN.gameMoves.push_back(pgn_move(evalInfo.bestMove,
                                            false,
                                            *activePlayer->get_root_node()->get_pos(),
                                            evalInfo.legalMoves,
                                            isTerminal && int(nextRoot->get_value()) == -1));
    }
    while(!isTerminal);
    set_game_result_to_pgn(nextRoot);
    write_game_to_pgn(filenamePGNArena, verbose);
    Result gameResult = get_terminal_node_result(nextRoot);
    clean_up(gamePGN, whitePlayer, states, position);
    blackPlayer->clear_game_history();
    return gameResult;
}

void clean_up(GamePGN& gamePGN, MCTSAgent* mctsAgent, StatesManager* states, Board* position) {
    gamePGN.new_game();
    mctsAgent->clear_game_history();
    states->swap_states();
    states->clear_states();
    position->set_state_info(new StateInfo);
    delete position;
}

void SelfPlay::write_game_to_pgn(const std::string& pngFileName, bool verbose)
{
    ofstream pgnFile;
    pgnFile.open(pngFileName, std::ios_base::app);
    if (verbose) {
        cout << endl << gamePGN << endl;
    }
    pgnFile << gamePGN << endl;
    pgnFile.close();
}

void SelfPlay::set_game_result_to_pgn(const Node* terminalNode)
{
    gamePGN.result = result[get_terminal_node_result(terminalNode)];
}

void SelfPlay::reset_speed_statistics()
{
    gameIdx = 0;
    gamesPerMin = 0;
    samplesPerMin = 0;
}

void SelfPlay::speed_statistic_report(float elapsedTimeMin, size_t generatedSamples)
{
    // compute running cummulative average
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


void SelfPlay::go(size_t numberOfGames, StatesManager* states, float policySharpening)
{
    reset_speed_statistics();
    gamePGN.white = mctsAgent->get_name();
    gamePGN.black = mctsAgent->get_name();

    if (numberOfGames == 0) {
        while(!exporter->is_file_full()) {
            generate_game(CRAZYHOUSE_VARIANT, states, true);
        }
    }
    else {
        for (size_t idx = 0; idx < numberOfGames; ++idx) {
            generate_game(CRAZYHOUSE_VARIANT, states, true);
        }
    }
    export_number_generated_games();
}

TournamentResult SelfPlay::go_arena(MCTSAgent *mctsContender, size_t numberOfGames, StatesManager* states)
{
    TournamentResult tournamentResult;
    tournamentResult.playerA = mctsContender->get_name();
    tournamentResult.playerB = mctsAgent->get_name();
    Result gameResult;
    for (size_t idx = 0; idx < numberOfGames; ++idx) {
        if (idx % 2 == 0) {
            gameResult = generate_arena_game(mctsContender, mctsAgent, CRAZYHOUSE_VARIANT, states, true);
            if (gameResult == WHITE_WIN) {
                ++tournamentResult.numberWins;
            }
            else {
                ++tournamentResult.numberLosses;
            }
        }
        else {
            gameResult = generate_arena_game(mctsAgent, mctsContender, CRAZYHOUSE_VARIANT, states, true);
            if (gameResult == BLACK_WIN) {
                ++tournamentResult.numberWins;
            }
            else {
                ++tournamentResult.numberLosses;
            }
        }
        if (gameResult == DRAWN) {
            ++tournamentResult.numberDraws;
        }
    }
    return tournamentResult;
}

Board* init_board(Variant variant, StatesManager* states)
{
    Board* position = new Board();
    auto uiThread = make_shared<Thread>(0);

    StateInfo* newState = new StateInfo;
    position->set(StartFENs[variant], false, variant, newState, uiThread.get());
    states->activeStates.push_back(newState);
    return position;
}

Board* init_starting_pos_from_raw_policy(RawNetAgent &rawAgent, size_t plys, GamePGN &gamePGN, Variant variant, StatesManager *states)
{
    Board* position = init_board(variant, states);

    for (size_t ply = 0; ply < plys; ++ply) {
        EvalInfo eval;
        rawAgent.evaluate_board_state(position, eval);
        const size_t moveIdx = random_choice(eval.policyProbSmall);
        eval.bestMove = eval.legalMoves[moveIdx];

        if (leads_to_terminal(*position, eval.bestMove)) {
            break;
        }
        else {
            gamePGN.gameMoves.push_back(pgn_move(eval.legalMoves[moveIdx],
                                                 false,
                                                 *position,
                                                 eval.legalMoves,
                                                 false,
                                                 true));
            StateInfo* newState = new StateInfo;
            states->activeStates.push_back(newState);
            position->do_move(eval.bestMove, *(newState));
        }
    }

    return position;
}

size_t clip_ply(size_t ply, size_t maxPly)
{
    if (ply > maxPly) {
        return size_t(rand()) % maxPly;
    }
    return ply;
}
#endif
