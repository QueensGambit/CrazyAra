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
 * @file: selfplay.h
 * Created on 16.09.2019
 * @author: queensgambit
 *
 * Functionality for running CrazyAra in self play mode
 */

#ifndef SELFPLAY_H
#define SELFPLAY_H

#include "../agents/mctsagent.h"
#include "gamepgn.h"
#include "../manager/statesmanager.h"

#ifdef USE_RL
class SelfPlay
{
private:
    MCTSAgent* mctsAgent;
    GamePGN gamePGN;
    EvalInfo evalInfo;

    /**
     * @brief generate_game Generates a new game in self play mode
     * @param variant Current chess variant
     * @param searchLimits Search limits struct
     */
    void generate_game(Variant variant, SearchLimits& searchLimits);

    /**
     * @brief write_game_to_pgn Writes the game log to a pgn file
     */
    void write_game_to_pgn();

    /**
     * @brief set_game_result Sets the game result to the gamePGN object
     */
    void set_game_result_to_pgn();

public:
    SelfPlay(MCTSAgent* mctsAgent);

    /**
     * @brief go Starts the self play game generation for a given number of games
     * @param numberOfGames Number of games to generate
     * @param searchLimits Search limit struct
     */
    void go(size_t numberOfGames, SearchLimits& searchLimits);
};
#endif

#endif // SELFPLAY_H
