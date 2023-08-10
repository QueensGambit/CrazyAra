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
 * @file: outputrepresentation.cpp
 * Created on 13.05.2019
 * @author: queensgambit
 */

#include <tuple>
#include "outputrepresentation.h"
#include "policymaprepresentation.h"
#include "sfutil.h"
#include "constants.h"
#include "stateobj.h"
using namespace std;

void apply_softmax(DynamicVector<float> &policyProbSmall)
{
    policyProbSmall = softmax(policyProbSmall);
}

void OutputRepresentation::init_policy_constants(bool isPolicyMap, bool is960)
{
    // fill mirrored label list and look-up table
    for (size_t mvIdx = 0; mvIdx < StateConstants::NB_LABELS(); mvIdx++) {
        LABELS_MIRRORED[mvIdx] = mirror_move(LABELS[mvIdx]);
        std::vector<Move> moves = make_move(LABELS[mvIdx], is960);
        for (Move move : moves) {
            isPolicyMap ? MV_LOOKUP[move] = FLAT_PLANE_IDX[mvIdx] : MV_LOOKUP[move] = mvIdx;

            MV_LOOKUP_CLASSIC[move] = mvIdx;
        }
        std::vector<Move> moves_mirrored = make_move(LABELS_MIRRORED[mvIdx], is960);
        for (Move move : moves_mirrored) {
            isPolicyMap ? MV_LOOKUP_MIRRORED[move] = FLAT_PLANE_IDX[mvIdx] : MV_LOOKUP_MIRRORED[move] = mvIdx;
            MV_LOOKUP_MIRRORED_CLASSIC[move] = mvIdx;
        }
    }
}

array<string, 8> uci_labels::files() {
    return {"a", "b", "c", "d", "e", "f", "g", "h"};
}

array<string, 8> uci_labels::ranks() {
    return {"1", "2", "3", "4", "5", "6", "7", "8"};
}

array<string, 6> uci_labels::pieces() {
    return {"P", "N", "B", "R", "Q", "K"};
}

vector<string> uci_labels::promotion_pieces()
{
#ifdef MODE_LICHESS
    return {"q", "r", "b", "n", "k"};  // king promotion was added to support the antichess variant
#endif
    return {"q", "r", "b", "n"};
}

vector<tuple<int,int>> uci_labels::get_square_destinations(int rank1, int fileIdx) {
    vector<tuple<int,int>> destinations;
    for (int i = 0; i < 8; ++i) {
        tuple<int,int> t{i, rank1};
        destinations.emplace_back(t);
    }
    for (int i = 0; i < 8; ++i) {
        tuple<int,int> t{fileIdx, i};
        destinations.emplace_back(t);
    }
    for (int i = -7; i < 8; ++i) {
        tuple<int,int> t{fileIdx + i, rank1 + i};
        destinations.emplace_back(t);
    }
    for (int i = -7; i < 8; ++i) {
        tuple<int,int> t{fileIdx + i, rank1 - i};
        destinations.emplace_back(t);
    }
    array<int,8> knightJumpsfileOffsets = {-2, -1, -2, 1, 2, -1, 2, 1};
    array<int,8> knightJumpsRankOffsets = {-1, -2, 1, -2, -1, 2, 1, 2};
    for (int i = 0; i < 8; ++i) {
        int fileOffset = knightJumpsfileOffsets[i];
        int rankOffset = knightJumpsRankOffsets[i];
        tuple<int,int> t{fileIdx + fileOffset, rank1 + rankOffset};
        destinations.emplace_back(t);
    }
    return destinations;
}


vector<string> uci_labels::generate_uci_labels(const vector<string>& promotionPieces) {
    vector<string> labels;
    const auto ranks = uci_labels::ranks();
    const auto files = uci_labels::files();
    // generate classical moves
    for (int fileIdx = 0; fileIdx < 8; ++fileIdx) {
        for (int rankIdx = 0; rankIdx < 8; ++rankIdx) {
            vector<tuple<int,int>> destinations = uci_labels::get_square_destinations(rankIdx, fileIdx);
            for (tuple<int,int> curTuple : destinations) {
                int fileIdx2 = std::get<0>(curTuple);
                int rankIdx2 = std::get<1>(curTuple);
                if ((fileIdx != fileIdx2 || rankIdx != rankIdx2) && fileIdx2 >= 0 && fileIdx2 < 8 && rankIdx2 >= 0 && rankIdx2 < 8) {
                    string move = files[fileIdx] + ranks[rankIdx] + files[fileIdx2] + uci_labels::ranks()[rankIdx2];
                    labels.emplace_back(move);
                }
            }
        }
    }
    // generate promotion moves
    for (int fileIdx = 0; fileIdx < 8; ++fileIdx) {
        string file = files[fileIdx];
        for (string p : promotionPieces) {
            labels.emplace_back(file + "2" + file + "1" + p);
            labels.emplace_back(file + "7" + file + "8" + p);
            if (fileIdx > 0) {
                string l_l = files[fileIdx - 1];
                labels.emplace_back(file + "2" + l_l + "1" + p);
                labels.emplace_back(file + "7" + l_l + "8" + p);
            }
            if (fileIdx < 7) {
                string l_r = files[fileIdx + 1];
                labels.emplace_back(file + "2" + l_r + "1" + p);
                labels.emplace_back(file + "7" + l_r + "8" + p);
            }
        }
    }
    return labels;
}

void uci_labels::generate_dropping_moves(vector<string>& labels) {
    // iterate over all chess squares
    const auto pieces = uci_labels::pieces();
    for (int l1 = 0; l1 < 8; ++l1) {
        for (int n1 = 0; n1 < 8; ++n1) {
            // add all possible dropping moves
            // we don't want the first symbol which is empty and not the last which is the king
            for (auto pieceIt = pieces.begin(); pieceIt != pieces.end()-1; ++pieceIt) {
                // make sure not to add pawn dropping in the first or last row
                if (*pieceIt != "P" || !(uci_labels::ranks()[n1] == "1" || uci_labels::ranks()[n1] == "8")) {
                    const string move = *pieceIt + "@" + uci_labels::files()[l1] + uci_labels::ranks()[n1];
                    labels.emplace_back(move);
                }
            }
        }
    }
}

void OutputRepresentation::init_labels()
{
#ifdef MODE_CRAZYHOUSE
    // start with the normal chess labels
    LABELS = uci_labels::generate_uci_labels(uci_labels::promotion_pieces());
    uci_labels::generate_dropping_moves(LABELS);
#elif MODE_LICHESS
    // start with the normal chess labels
    LABELS = uci_labels::generate_uci_labels(uci_labels::promotion_pieces());
    uci_labels::generate_dropping_moves(LABELS);
#elif MODE_CHESS
    // start with the normal chess labels
    LABELS = uci_labels::generate_uci_labels(uci_labels::promotion_pieces());
#endif
    if (LABELS.size() != StateConstants::NB_LABELS()) {
        cerr << "LABELS.size() != StateConstants::NB_LABELS():" <<  LABELS.size() << " " << StateConstants::NB_LABELS() << endl;
        assert(false);
    }
    LABELS_MIRRORED.resize(LABELS.size());
}
