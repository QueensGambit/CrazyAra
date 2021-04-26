#include <iostream>
#include "fairyoutputrepresentation.h"
#include "fairypolicymaprepresentation.h"
#include "fairystate.h"
#include "fairyutil.h"

using namespace std;
using uci_labels::nbRanks;
using uci_labels::nbFiles;

vector<tuple<int, int>> uci_labels::get_destinations(int rankIdx, int fileIdx) {
    vector<tuple<int, int>> destinations;
    for (int i = 0; i < nbFiles; ++i) {
        tuple<int, int> tmp{rankIdx, i};
        destinations.emplace_back(tmp);
    }

    for (int i = 0; i < nbRanks; ++i) {
        tuple<int, int> tmp{i, fileIdx};
        destinations.emplace_back(tmp);
    }

    // horse moves
    array<int, 8> horseRankOffsets = {-2, -1, 1, 2, 2, 1, -1, -2};
    array<int, 8> horseFileOffsets = {-1, -2, -2, -1, 1, 2, 2, 1};
    for (int i = 0; i < horseFileOffsets.size(); ++i) {
        int rankOffset = horseRankOffsets[i];
        int fileOffset = horseFileOffsets[i];
        tuple<int, int> tmp{rankIdx + rankOffset, fileIdx + fileOffset};
        destinations.emplace_back(tmp);
    }

    // elephant moves
    vector<int> elephantRankOffsets;
    vector<int> elephantFileOffsets;
    if ((rankIdx == 0 && fileIdx == 2) || (rankIdx == 0 && fileIdx == 6)
        || (rankIdx == 2 && fileIdx == 0) || (rankIdx == 2 && fileIdx == 4) || (rankIdx == 2 && fileIdx == 8)
        || (rankIdx == 7 && fileIdx == 0) || (rankIdx == 7 && fileIdx == 4) || (rankIdx == 7 && fileIdx == 8)
        || (rankIdx == 9 && fileIdx == 2) || (rankIdx == 9 && fileIdx == 6)) {
        elephantRankOffsets = {2, 2, -2, -2};
        elephantFileOffsets = {-2, 2, -2, 2};
    } else if (rankIdx == 4 && (fileIdx == 2 || fileIdx == 6)) {
        elephantRankOffsets = {-2, -2};
        elephantFileOffsets = {-2, 2};
    } else if (rankIdx == 5 && (fileIdx == 2 || fileIdx == 6)) {
        elephantRankOffsets = {2, 2};
        elephantFileOffsets = {-2, 2};
    }
    for (int i = 0; i < elephantFileOffsets.size(); ++i) {
        int rankOffset = elephantRankOffsets[i];
        int fileOffset = elephantFileOffsets[i];
        tuple<int, int> tmp{rankIdx + rankOffset, fileIdx + fileOffset};
        destinations.emplace_back(tmp);
    }

    // advisor diagonal moves from mid palace
    if (fileIdx == 4 && (rankIdx == 1 || rankIdx == 8)) {
        array<int, 4> advisorRankOffsets = {-1, 1, 1, -1};
        array<int, 4> advisorFileOffsets = {-1, -1, 1, 1};
        for (int i = 0; i < advisorFileOffsets.size(); ++i) {
            int rankOffset = advisorRankOffsets[i];
            int fileOffset = advisorFileOffsets[i];
            tuple<int, int> tmp{rankIdx + rankOffset, fileIdx + fileOffset};
            destinations.emplace_back(tmp);
        }
    }
    return destinations;
}

vector<string> uci_labels::generate_uci_labels() {
    vector<string> labels;
    const array<string, nbRanks> ranks = uci_labels::ranks();
    const array<string, nbFiles> files = uci_labels::files();
    for (int rankIdx = 0; rankIdx < nbRanks; ++rankIdx) {
         for (int fileIdx = 0; fileIdx < nbFiles; ++fileIdx) {
            vector<tuple<int, int>> destinations = uci_labels::get_destinations(rankIdx, fileIdx);
            for (tuple<int, int> destination : destinations) {
                int rankIdx2 = get<0>(destination);
                int fileIdx2 = get<1>(destination);
                if ((fileIdx != fileIdx2 || rankIdx != rankIdx2)
                    && fileIdx2 >= 0 && fileIdx2 < nbFiles && rankIdx2 >= 0 && rankIdx2 < nbRanks) {
                    string move = files[fileIdx] + ranks[rankIdx] + files[fileIdx2] + ranks[rankIdx2];
                    labels.emplace_back(move);
                }
            }
        }
    }

    // advidor moves to mid palace
    labels.emplace_back("d1e2");
    labels.emplace_back("f1e2");
    labels.emplace_back("d3e2");
    labels.emplace_back("f3e2");
    labels.emplace_back("d10e9");
    labels.emplace_back("f10e9");
    labels.emplace_back("d8e9");
    labels.emplace_back("f8e9");
    return labels;
}

string uci_labels::mirror_move(const string &ucciMove) {
    // a10b10
        if (ucciMove.size() == 6) {
            return string(1, ucciMove[0]) + string(1, '1') + string(1, ucciMove[3]) + string(1, '1');
        }
        else if (ucciMove.size() == 5) {
            // a10a9
            if (isdigit(ucciMove[2])) {
                int rankTo = ucciMove[4] - '0';
                int rankToMirrored = 10 - rankTo + 1;
                return string(1, ucciMove[0]) + string(1, '1') + string(1, ucciMove[3]) + to_string(rankToMirrored);
            }
            // a9a10
            else {
                int rankFrom = ucciMove[1] - '0';
                int rankFromMirrored = 10 - rankFrom + 1;
                return string(1, ucciMove[0]) + to_string(rankFromMirrored) + string(1, ucciMove[2]) + string(1, '1');
            }
        }
        // a1b1
        else {
            string moveMirrored;
            for (size_t i = 0; i < ucciMove.length(); ++i) {
                if (isdigit(ucciMove[i])) {
                    int rank = ucciMove[i] - '0';
                    int rankMirrored = 10 - rank + 1;
                    moveMirrored += to_string(rankMirrored);
                }
                else {
                    moveMirrored += ucciMove[i];
                }
            }
            return moveMirrored;
        }
}

array<string, 9> uci_labels::files() {
    return {"a", "b", "c", "d", "e", "f", "g", "h", "i"};
}

array<string, 10> uci_labels::ranks() {
    return {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};
}

void FairyOutputRepresentation::init_labels() {
    LABELS = uci_labels::generate_uci_labels();
    if (LABELS.size() != StateConstantsFairy::NB_LABELS()) {
        cerr << "LABELS.size() != StateConstantsFairy::NB_LABELS():" << LABELS.size() << " "
             << StateConstantsFairy::NB_LABELS() << endl;
        assert(false);
    }
    LABELS_MIRRORED.resize(LABELS.size());
}

void FairyOutputRepresentation::init_policy_constants(bool isPolicyMap) {
    for (size_t i = 0; i < StateConstantsFairy::NB_LABELS(); ++i) {
        LABELS_MIRRORED[i] = uci_labels::mirror_move(LABELS[i]);

        Square fromSquare = get_origin_square(LABELS[i]);
        Square toSquare = get_destination_square(LABELS[i]);
        Move move = make_move(fromSquare, toSquare);
        isPolicyMap ? MV_LOOKUP[move] = FLAT_PLANE_IDX[i] : MV_LOOKUP[move] = i;
        MV_LOOKUP_CLASSIC[move] = i;

        Square fromSquareMirrored = get_origin_square(LABELS_MIRRORED[i]);
        Square toSquareMirrored = get_destination_square(LABELS_MIRRORED[i]);
        Move moveMirrored = make_move(fromSquareMirrored, toSquareMirrored);
        isPolicyMap ? MV_LOOKUP_MIRRORED[moveMirrored] = FLAT_PLANE_IDX[i] : MV_LOOKUP_MIRRORED[moveMirrored] = i;
        MV_LOOKUP_MIRRORED_CLASSIC[moveMirrored] = i;
    }
}
