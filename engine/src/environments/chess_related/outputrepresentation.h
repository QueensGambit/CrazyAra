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
 * @file: outputrepresentation.h
 * Created on 13.05.2019
 * @author: queensgambit
 *
 * Provides all methods to convert a move to policy representation and back.
 */

#ifndef OUTPUTREPRESENTATION_H
#define OUTPUTREPRESENTATION_H

#include <climits>
#include <array>
#include <blaze/Math.h>
#include "constants.h"
#include "state.h"

using blaze::HybridVector;
using blaze::DynamicVector;
using action_idx_map = Action[USHRT_MAX];

using namespace std;

void apply_softmax(DynamicVector<float> &policyProbSmall);

namespace uci_labels {
/**
 * @brief generate_uci_labels Generates all possible uci moves for every possible board state.
 * This is function was imported from the initial python version and is based on:
 * https://github.com/Zeta36/chess-alpha-zero/blob/master/src/chess_zero/config.py#L88
 * @param promoted_to
 * @return vector<string>
 */
vector<string> generate_uci_labels(const vector<string>& promotionPieces);

/**
 * @brief get_square_destinations Returns all possible square destination for a given square
 * @param rankIdx Rank index
 * @param fileIdx File index
 * @return vector of squares
 */
vector<tuple<int,int>> get_square_destinations(int rankIdx, int fileIdx);

/**
 * @brief generate_dropping_moves Returns all legal dropping moves in UCI notation
 * @param labels Vector which will be appended
 */
void generate_dropping_moves(vector<string>& labels);

array<string, 8> files();
array<string, 8> ranks();
array<string, 6> pieces();
vector<string> promotion_pieces();
}

struct OutputRepresentation{
    static vector<std::string> LABELS;
    static vector<std::string> LABELS_MIRRORED;
    static action_idx_map MV_LOOKUP;
    static action_idx_map MV_LOOKUP_MIRRORED;
    static action_idx_map MV_LOOKUP_CLASSIC;
    static action_idx_map MV_LOOKUP_MIRRORED_CLASSIC;
    /**
     * @brief init_labels Generates all labels in uci move notation. First creating all possible chess moves and
     *  later adding all possible dropping moves for MODE_CRAZYHOUSE or MODE_LICHESS.
     */
    static void init_labels();
    /**
     * @brief init_policy_constants Fills the hash maps for a action to nn index binding.
     * @param isPolicyMap describes if a policy map head is used for the NN.
     * @param is960 defines if 960 variant should be supported
     */
    static void init_policy_constants(bool isPolicyMap, bool is960);


};

#endif // OUTPUTREPRESENTATION_H
