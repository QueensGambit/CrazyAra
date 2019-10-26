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
}

void SelfPlay::generate_game(Variant variant, SearchLimits& searchLimits)
{
    Board* position = init_board(variant);
    EvalInfo evalInfo;

    bool isTerminal = false;
    do {
        searchLimits.startTime = now();
        mctsAgent->perform_action(position, &searchLimits, evalInfo);
        mctsAgent->apply_move_to_tree(evalInfo.bestMove, true);
        const Node* nextRoot = mctsAgent->get_opponents_next_root();
        if (nextRoot != nullptr) {
            isTerminal = nextRoot->is_terminal();
        }
        position->do_move(evalInfo.bestMove, *(new StateInfo));
        gamePGN.gameMoves.push_back(pgnMove(evalInfo.bestMove,
                                            false,
                                            *mctsAgent->get_root_node()->get_pos(),
                                            evalInfo.legalMoves,
                                            isTerminal));
    }
    while(!isTerminal);

    cout << "info string terminal fen " << mctsAgent->get_opponents_next_root()->get_pos()->fen() << " move " << UCI::move(evalInfo.bestMove, evalInfo.isChess960)<< endl;
    mctsAgent->export_game_results();
    set_game_result_to_pgn(mctsAgent->get_opponents_next_root());
    write_game_to_pgn("games.pgn");
    gamePGN.new_game();
    mctsAgent->clear_game_history();
    delete position;
}

Result SelfPlay::generate_arena_game(MCTSAgent* whitePlayer, MCTSAgent* blackPlayer, Variant variant, SearchLimits &searchLimits)
{
    gamePGN.white = whitePlayer->get_name();
    gamePGN.black = blackPlayer->get_name();
    Board* position = init_board(variant);
    EvalInfo evalInfo;
    bool isTerminal = false;

    MCTSAgent* activePlayer;
    MCTSAgent* passivePlayer;
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
        passivePlayer->apply_move_to_tree(evalInfo.bestMove, true);
        nextRoot = activePlayer->get_opponents_next_root();
        if (nextRoot != nullptr) {
            isTerminal = nextRoot->is_terminal();
        }
        position->do_move(evalInfo.bestMove, *(new StateInfo));
        gamePGN.gameMoves.push_back(pgnMove(evalInfo.bestMove,
                                            false,
                                            *activePlayer->get_root_node()->get_pos(),
                                            evalInfo.legalMoves,
                                            isTerminal));
    }
    while(!isTerminal);
    set_game_result_to_pgn(nextRoot);
    write_game_to_pgn("arena_games.pgn");
    gamePGN.new_game();
    activePlayer->clear_game_history();
    passivePlayer->clear_game_history();
    delete position;
    return get_terminal_node_result(nextRoot);
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

void SelfPlay::go(size_t numberOfGames, SearchLimits& searchLimits)
{
    gamePGN.white = mctsAgent->get_name();
    gamePGN.black = mctsAgent->get_name();
    for (size_t idx = 0; idx < numberOfGames; ++idx) {
        generate_game(CRAZYHOUSE_VARIANT, searchLimits);
    }
}

TournamentResult SelfPlay::go_arena(MCTSAgent *mctsContender, size_t numberOfGames, SearchLimits &searchLimits)
{
    TournamentResult tournamentResult;
    Result gameResult;
    for (size_t idx = 0; idx < numberOfGames; ++idx) {
        if (idx % 2 == 0) {
            gameResult = generate_arena_game(mctsContender, mctsAgent, CRAZYHOUSE_VARIANT, searchLimits);
            if (gameResult == WHITE_WIN) {
                ++tournamentResult.numberWins;
            }
            else {
                ++tournamentResult.numberLosses;
            }
        }
        else {
            gameResult = generate_arena_game(mctsAgent, mctsContender, CRAZYHOUSE_VARIANT, searchLimits);
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
