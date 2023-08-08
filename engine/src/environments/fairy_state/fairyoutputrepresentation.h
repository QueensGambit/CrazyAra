#ifndef FAIRYOUTPUTREPRESENTATION_H
#define FAIRYOUTPUTREPRESENTATION_H

#include <climits>
#include <array>
#include <unordered_map>
#include <blaze/Math.h>
#include "state.h"

using blaze::HybridVector;
using blaze::DynamicVector;
using action_idx_map = unordered_map<Action, int_fast32_t>;

using namespace std;

namespace uci_labels {
    const int nbRanks = 10;
    const int nbFiles = 9;

    /**
     * @brief get_destinations Returns all possible destinations on a Xiangqi board for a given square index.
     */
    vector<tuple<int, int>> get_destinations(int rankIdx, int fileIdx);

    vector<string> generate_uci_labels();

    /**
     * @brief generate_uci_labels_cfour Returns a vector of all possible uci moves for connect four.
     * {"a10a1", "a10b1", ..., "a10g1"}
     * @return Vector of UCI-Strings
     */
    vector<string> generate_uci_labels_cfour_and_flipello();

    /**
     * @brief generate_uci_labels_breakthrough_and_clobber Returns a vector of all possible uci moves for breakthrough
     * @return Vector of UCI-Strings
     */
    void generate_uci_labels_breakthrough_and_clobber(vector<string>& labels);

    string mirror_move(const string &ucciMove);

    // For the ucci labels we begin with index 0 for the ranks
    array<string, nbRanks> ranks();

    array<string, nbFiles> files();
}

struct FairyOutputRepresentation {
    static vector<string> LABELS;
    static vector<string> LABELS_MIRRORED;
    static action_idx_map MV_LOOKUP;
    static action_idx_map MV_LOOKUP_MIRRORED;
    static action_idx_map MV_LOOKUP_CLASSIC;
    static action_idx_map MV_LOOKUP_MIRRORED_CLASSIC;

    /**
     * @brief init_labels Generates all labels in ucci move notation.
     */
    static void init_labels();

    /**
     * @brief init_policy_constants Fills the hash maps for a action to Neural Network index binding.
     * @param isPolicyMap describes if a policy map head is used for the Neural Network.
     */
    static void init_policy_constants(bool isPolicyMap);
};

#endif //FAIRYOUTPUTREPRESENTATION_H
