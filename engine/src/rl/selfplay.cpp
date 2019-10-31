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

SelfPlay::SelfPlay(MCTSAgent* mctsAgent):
    mctsAgent(mctsAgent)
{
    gamePGN.variant = "crazyhouse";
    gamePGN.event = "CrazyAra-SelfPlay";
    gamePGN.site = "Darmstadt, GER";
    gamePGN.date = "?";  // TODO: Change this later
    gamePGN.round = "?";
    gamePGN.is960 = false;
    this->exporter = new TrainDataExporter("data.zarr", 200, 128);
}

void SelfPlay::generate_game(Variant variant, SearchLimits& searchLimits, StatesManager* states)
{
    Board* position = init_board(variant);
    EvalInfo evalInfo;
    states->swap_states();
    bool leadsToTerminal = false;
    do {
        searchLimits.startTime = now();
        mctsAgent->perform_action(position, &searchLimits, evalInfo);
        mctsAgent->apply_move_to_tree(evalInfo.bestMove, true);
        const Node* nextRoot = mctsAgent->get_opponents_next_root();
        if (nextRoot != nullptr) {
            leadsToTerminal = nextRoot->is_terminal();
        }
        StateInfo* newState = new StateInfo;
        states->activeStates.push_back(newState);
        position->do_move(evalInfo.bestMove, *(newState));
        gamePGN.gameMoves.push_back(pgn_move(evalInfo.bestMove,
                                            false,
                                            *mctsAgent->get_root_node()->get_pos(),
                                            evalInfo.legalMoves,
                                            leadsToTerminal && int(nextRoot->get_value()) == -1));
        if (!exporter->is_file_full()) {
            exporter->export_pos(position, evalInfo, size_t(position->game_ply()));
            exporter->export_best_move_q(evalInfo, size_t(position->game_ply()));
        }
    }
    while(!leadsToTerminal);

    int16_t result = position->side_to_move() == WHITE ? LOSS : WIN;
    // we set one less than actual plys because the last terminal node isn't part of the training data
    exporter->export_game_result(result, 0, size_t(position->game_ply())-1);

    cout << "info string terminal fen " << mctsAgent->get_opponents_next_root()->get_pos()->fen() << " move " << UCI::move(evalInfo.bestMove, evalInfo.isChess960)<< endl;
    set_game_result_to_pgn(mctsAgent->get_opponents_next_root());
    write_game_to_pgn("games.pgn");
    gamePGN.new_game();
    mctsAgent->clear_game_history();
    states->swap_states();
    states->clear_states();
}

Result SelfPlay::generate_arena_game(MCTSAgent* whitePlayer, MCTSAgent* blackPlayer, Variant variant, SearchLimits &searchLimits, StatesManager* states)
{
    gamePGN.white = whitePlayer->get_name();
    gamePGN.black = blackPlayer->get_name();
    Board* position = init_board(variant);
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
    write_game_to_pgn("arena_games.pgn");
    gamePGN.new_game();
    Result gameResult = get_terminal_node_result(nextRoot);
    whitePlayer->clear_game_history();
    blackPlayer->clear_game_history();
    states->swap_states();
    states->clear_states();
    return gameResult;
}

void SelfPlay::write_game_to_pgn(const std::string& pngFileName)
{
    ofstream pgnFile;
    pgnFile.open(pngFileName, std::ios_base::app);  // TODO: Change to more meaningful filename
    cout << endl << gamePGN << endl;
    pgnFile << gamePGN << endl;
    pgnFile.close();
}

void SelfPlay::set_game_result_to_pgn(const Node* terminalNode)
{
    gamePGN.result = result[get_terminal_node_result(terminalNode)];
}

Board* SelfPlay::init_board(Variant variant)
{
    Board* position = new Board();
    auto uiThread = make_shared<Thread>(0);

    StateInfo* newState = new StateInfo;
    position->set(StartFENs[variant], false, variant, newState, uiThread.get());
    return position;
}

void SelfPlay::go(size_t numberOfGames, SearchLimits& searchLimits, StatesManager* states)
{
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
