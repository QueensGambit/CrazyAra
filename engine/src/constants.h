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
 * @file: constants.h
 * Created on 13.05.2019
 * @author: queensgambit
 *
 * Definition of all constants for the game of Crazyhouse
 *
 * ! DO NOT CHANGE THE LABEL LIST OTHERWISE YOU WILL BREAK THE MOVE MAPPING OF THE NETWORK !
 *
 * This file contains the description of the policy vector which is predicted by the neural network and
 * constants for designing the input planes representation.
 * Each index of the array LABEL array corresponds to a unique move in uci-representation
 */

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>
#include <unordered_map>
#include <iostream>
#include "state.h"

using namespace std;

// meta data
#ifdef MODE_CRAZYHOUSE
const string engineName = "CrazyAra";
#elif defined MODE_LICHESS
const string engineName = "MultiAra";
#elif defined MODE_XIANGQI
const string engineName = "XiangqiAra";
#elif defined MODE_BOARDGAMES
const string engineName = "BoardAra";
#elif defined MODE_STRATEGO
const string engineName = "StrategoAra";
#elif defined MODE_OPEN_SPIEL
const string engineName = "OpenSpielAra";
#else  // MODE_CHESS
const string engineName = "ClassicAra";
#endif

const string engineVersion = "1.0.5";
#ifdef MODE_CRAZYHOUSE
const string engineAuthors = "Johannes Czech, Moritz Willig, Alena Beyer and CrazyAra developers (see AUTHORS file)";
#elif defined MODE_LICHESS
const string engineAuthors = "Johannes Czech, Maximilian Alexander Gehrke and CrazyAra developers (see AUTHORS file)";
#elif defined MODE_XIANGQI
const string engineAuthors = "Johannes Czech, Maximilian Langer and CrazyAra developers (see AUTHORS file)";
#elif defined MODE_BOARDGAMES
const string engineAuthors = "Johannes Czech, Rumei Ma and CrazyAra developers (see AUTHORS file)";
#elif defined MODE_STRATEGO
const string engineAuthors = "Johannes Czech, Jannis Blüml and CrazyAra developers (see AUTHORS file)";
#elif defined MODE_OPEN_SPIEL
const string engineAuthors = "Johannes Czech and CrazyAra developers (see AUTHORS file)";
#else  // MODE_CHESS
const string engineAuthors = "Johannes Czech and CrazyAra developers (see AUTHORS file)";
#endif

#define LOSS_VALUE -1
#define DRAW_VALUE 0
#define WIN_VALUE 1
#define PRESERVED_ITEMS 8
// Pre-initialized index when no forced win was found: 2^16 - 1
#define NO_CHECKMATE 65535
#define Q_VALUE_DIFF 0.1f
#define Q_INIT -1.0f
#define DEPTH_INIT 64
#define Q_TRANSPOS_DIFF 0.01
#define MAX_HASH_SIZE 100000000
#ifdef MODE_CHESS
#define VALUE_TO_CENTI_PARAM 1.4f
#else
#define VALUE_TO_CENTI_PARAM 1.2f
#endif
#define TIME_EXPECT_GAME_LENGTH 38
#define TIME_THRESH_MOVE_PROP_SYSTEM 35
#define TIME_PROP_MOVES_TO_GO 14
#define TIME_INCREMENT_FACTOR 0.7f
#define TIME_BUFFER_FACTOR 30
#define NONE_IDX uint16_t(-1)

#ifndef MODE_POMMERMAN
#define TERMINAL_NODE_CACHE 8192
#else
#define TERMINAL_NODE_CACHE 1
#endif

const std::string result[] = {"1/2-1/2", "1-0", "0-1"};

#endif // CONSTANTS_H
