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

SelfPlay::SelfPlay(MCTSAgent* mctsAgent, size_t numberChunks, size_t chunkSize):
    mctsAgent(mctsAgent), gameIdx(0), gamesPerMin(0), samplesPerMin(0)
{
    gamePGN.variant = "crazyhouse";
    gamePGN.event = "CrazyAra-SelfPlay";
    gamePGN.site = "Darmstadt, GER";
    gamePGN.date = "?";  // TODO: Change this later
    gamePGN.round = "?";
    gamePGN.is960 = false;
    this->exporter = new TrainDataExporter(string("data_") + mctsAgent->get_device_name() + string(".zarr"),
                                           numberChunks, chunkSize);
    filenamePGNSelfplay = string("games_") + mctsAgent->get_device_name() + string(".pgn");
    filenamePGNArena = string("arena_games_")+ mctsAgent->get_device_name() + string(".pgn");
    fileNameGameIdx = string("gameIdx_") + mctsAgent->get_device_name() + string(".txt");
}

SelfPlay::~SelfPlay()
{
    delete exporter;
}

void SelfPlay::generate_game(Variant variant, SearchLimits& searchLimits, StatesManager* states)
{
    chrono::steady_clock::time_point gameStartTime = chrono::steady_clock::now();
    Board* position = init_board(variant, states);
    EvalInfo evalInfo;
    states->swap_states();
    bool leadsToTerminal = false;
    exporter->new_game();
    do {
        searchLimits.startTime = now();
        mctsAgent->perform_action(position, &searchLimits, evalInfo);
        mctsAgent->apply_move_to_tree(evalInfo.bestMove, true);
        const Node* nextRoot = mctsAgent->get_opponents_next_root();
        if (nextRoot != nullptr) {
            leadsToTerminal = nextRoot->is_terminal();
        }
        if (!exporter->is_file_full()) {
            exporter->save_sample(position, evalInfo, size_t(position->game_ply()));
        }
        StateInfo* newState = new StateInfo;
        states->activeStates.push_back(newState);
        position->do_move(evalInfo.bestMove, *(newState));
        gamePGN.gameMoves.push_back(pgn_move(evalInfo.bestMove,
                                            false,
                                            *mctsAgent->get_root_node()->get_pos(),
                                            evalInfo.legalMoves,
                                            leadsToTerminal && int(nextRoot->get_value()) == -1));
    }
    while(!leadsToTerminal);

    int16_t result = position->side_to_move() == WHITE ? LOSS : WIN;
    // game contains how many moves have been made at the end of the game
    exporter->export_game_samples(result, size_t(position->game_ply()));

    set_game_result_to_pgn(mctsAgent->get_opponents_next_root());
    write_game_to_pgn(filenamePGNSelfplay);
    clean_up(gamePGN, mctsAgent, states, position);

    // measure time statistics
    const float elapsedTimeMin = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - gameStartTime).count() / 60000.f;
    speed_statistic_report(elapsedTimeMin, position->game_ply());
}

Result SelfPlay::generate_arena_game(MCTSAgent* whitePlayer, MCTSAgent* blackPlayer, Variant variant, SearchLimits &searchLimits, StatesManager* states)
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
        searchLimits.startTime = now();
        if (position->side_to_move() == WHITE) {
            activePlayer = whitePlayer;
            passivePlayer = blackPlayer;
        }
        else {
            activePlayer = blackPlayer;
            passivePlayer = whitePlayer;
        }
        activePlayer->perform_action(position, &searchLimits, evalInfo);
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
    write_game_to_pgn(filenamePGNArena);
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

void SelfPlay::write_game_to_pgn(const std::string& pngFileName)
{
    ofstream pgnFile;
    pgnFile.open(pngFileName, std::ios_base::app);
    cout << endl << gamePGN << endl;
    pgnFile << gamePGN << endl;
    pgnFile.close();
}

void SelfPlay::set_game_result_to_pgn(const Node* terminalNode)
{
    gamePGN.result = result[get_terminal_node_result(terminalNode)];
}

Board* SelfPlay::init_board(Variant variant, StatesManager* states)
{
    Board* position = new Board();
    auto uiThread = make_shared<Thread>(0);

    StateInfo* newState = new StateInfo;
    position->set(StartFENs[variant], false, variant, newState, uiThread.get());
    states->activeStates.push_back(newState);
    return position;
}

void SelfPlay::reset_speed_statistics()
{
    gameIdx = 0;
    gamesPerMin = 0;
    samplesPerMin = 0;
}

void SelfPlay::speed_statistic_report(float elapsedTimeMin, int generatedSamples)
{
    // compute running cummulative average
    gamesPerMin = (gameIdx * gamesPerMin + (1 / elapsedTimeMin)) / (gameIdx + 1);
    samplesPerMin = (gameIdx * samplesPerMin + (generatedSamples / elapsedTimeMin)) / (gameIdx + 1);

    cout << "    games    |  games/min  | samples/min " << endl
         << "-------------+-------------+-------------" << endl
         << std::setprecision(5)
         << setw(13) << ++gameIdx << '|'
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

void SelfPlay::go(size_t numberOfGames, SearchLimits& searchLimits, StatesManager* states)
{
    reset_speed_statistics();
    gamePGN.white = mctsAgent->get_name();
    gamePGN.black = mctsAgent->get_name();

    if (numberOfGames == 0) {
        while(!exporter->is_file_full()) {
            generate_game(CRAZYHOUSE_VARIANT, searchLimits, states);
        }
    }
    else {
        for (size_t idx = 0; idx < numberOfGames; ++idx) {
            generate_game(CRAZYHOUSE_VARIANT, searchLimits, states);
        }
    }
    export_number_generated_games();
}

TournamentResult SelfPlay::go_arena(MCTSAgent *mctsContender, size_t numberOfGames, SearchLimits &searchLimits, StatesManager* states)
{
    TournamentResult tournamentResult;
    Result gameResult;
    for (size_t idx = 0; idx < numberOfGames; ++idx) {
        if (idx % 2 == 0) {
            gameResult = generate_arena_game(mctsContender, mctsAgent, CRAZYHOUSE_VARIANT, searchLimits, states);
            if (gameResult == WHITE_WIN) {
                ++tournamentResult.numberWins;
            }
            else {
                ++tournamentResult.numberLosses;
            }
        }
        else {
            gameResult = generate_arena_game(mctsAgent, mctsContender, CRAZYHOUSE_VARIANT, searchLimits, states);
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
#endif
