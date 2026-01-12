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
 * @file: searchsettings.h
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * Struct which stores all relevant search settings for the Search-Agents
 */

#ifndef SEARCHSETTINGS_H
#define SEARCHSETTINGS_H

#include <cstdlib>
#include <cstdint>

enum SearchPlayerMode {
    MODE_SINGLE_PLAYER,
    MODE_TWO_PLAYER
};

enum VirtualStyle {
    VIRTUAL_LOSS,
    VIRTUAL_VISIT,
    VIRTUAL_OFFSET,
    VIRTUAL_MIX
};

enum GamePhaseDefinition {
    LICHESS,
    MOVECOUNT
};

struct SearchSettings
{
    uint_fast16_t multiPV;
    size_t threads;
    uint_fast32_t batchSize;
    float dirichletEpsilon;
    float dirichletAlpha;
    // policy temperature which can be applied on the every nodes' policy
    float nodePolicyTemperature;
    float qValueWeight;
    // describes how much better the highest Q-Value has to be to replace the candidate move with the highest visit count
    float qVetoDelta;
    bool verbose;
    uint_fast8_t epsilonChecksCounter;
//    bool enhanceCaptures;   currently not support
//    bool useFutureQValues;  currently not supported
    bool useMCGS;
    float cpuctInit;
    float cpuctBase;
    float uInit;
    float uMin;
    float uBase;
    float randomMoveFactor;

    // If true, the exact given node count doesn't need to reached, but search can be stopped earlier
    bool allowEarlyStopping;
    // early break out based on max node visits in tree; increases time for falling eval
    bool useNPSTimemanager;
    // boolean indicator if tablebases were loaded correctly
    bool useTablebase;
    // If true random exploration is used
    uint_fast8_t epsilonGreedyCounter;
    // If the tree or parts of the treee can be reused for the next search
    bool reuseTree;
    // If true, then the MCTS solver for terminals and tablebases will be active
    bool mctsSolver;
    // Defines the nubmer of players within the MCTS search. Available are MODE_SINGLE_PLAYER and MODE_TWO_PLAYER
    SearchPlayerMode searchPlayerMode;
    // Define the virtual style to avoid conflict between different threads in within the same mini-batch
    VirtualStyle virtualStyle;
    // Defines the number of visits to switch from virtual-visit to virtual-loss
    uint_fast32_t virtualMixThreshold;
    // Defines the strength of the virtual offset
    double virtualOffsetStrenght;
    // Defines the type of game phase definition to be used
    GamePhaseDefinition gamePhaseDefinition;
    // Number of games to generate in parallel
    uint_fast32_t numberParallelGames;
    SearchSettings();

    uint_fast32_t get_main_batch_size() const {
        return batchSize * numberParallelGames;
    }
};

#endif // SEARCHSETTINGS_H
