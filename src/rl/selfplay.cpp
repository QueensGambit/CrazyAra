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
    gamePGN.white = "CrazyAra-0.6.0";  // TODO: Make version dependent
    gamePGN.black = "CrazyAra-0.6.0";
    gamePGN.is960 = false;
}

void SelfPlay::generate_game(Variant variant, SearchLimits& searchLimits)
{
    Board* position = new Board();
    string token, cmd;
    auto uiThread = make_shared<Thread>(0);

    StateInfo* newState = new StateInfo;
    position->set(StartFENs[variant], false, variant, newState, uiThread.get());
    EvalInfo evalInfo;

    bool isTerminal = false;
    do {
        searchLimits.startTime = now();
        mctsAgent->perform_action(position, &searchLimits, evalInfo);
        mctsAgent->apply_move_to_tree(evalInfo.bestMove, true);
        const Node* nextRoot = mctsAgent->get_opponents_next_root();
        if (nextRoot != nullptr) {
//            position = nextRoot->get_pos();
            isTerminal = nextRoot->is_terminal();
        }
        position->do_move(evalInfo.bestMove, *(new StateInfo));
        gamePGN.gameMoves.push_back(pgnMove(evalInfo.bestMove,
                                            false,
                                            *mctsAgent->get_root_node()->get_pos(),
                                            isTerminal));
    }
    while(!isTerminal);

    cout << "info string terminal fen " << mctsAgent->get_opponents_next_root()->get_pos()->fen() << " move " << UCI::move(evalInfo.bestMove, evalInfo.isChess960)<< endl;
    mctsAgent->export_game_results();
    set_game_result_to_pgn();
    write_game_to_pgn();
    gamePGN.new_game();
    mctsAgent->clear_game_history();
}

void SelfPlay::write_game_to_pgn()
{
    ofstream pgnFile;
    pgnFile.open("games.pgn", std::ios_base::app);  // TODO: Change to more meaningful filename
    cout << endl << gamePGN << endl;
    pgnFile << gamePGN << endl;
    pgnFile.close();
}

void SelfPlay::set_game_result_to_pgn()
{
    if (int(mctsAgent->get_opponents_next_root()->get_value()) == 0) {
        gamePGN.result = "1/2-1/2";
    }
    else if ( mctsAgent->get_opponents_next_root()->get_pos()->side_to_move() == BLACK) {
        gamePGN.result = "1-0";
    }
    else {
        gamePGN.result = "0-1";
    }
}

void SelfPlay::go(size_t numberOfGames, SearchLimits& searchLimits)
{
    for (size_t idx = 0; idx < numberOfGames; ++idx) {
        generate_game(CRAZYHOUSE_VARIANT, searchLimits);
    }
}
