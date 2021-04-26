#ifndef FAIRYUTIL_H
#define FAIRYUTIL_H

#include <types.h>

using namespace std;

const unordered_map<char, File> FILE_LOOKUP = {
        {'a', FILE_A},
        {'b', FILE_B},
        {'c', FILE_C},
        {'d', FILE_D},
        {'e', FILE_E},
        {'f', FILE_F},
        {'g', FILE_G},
        {'h', FILE_H},
        {'i', FILE_I}};

// Note that we have 10 ranks but use a char to Rank lookup...
const unordered_map<char, Rank> RANK_LOOKUP = {
        {'1', RANK_1},
        {'2', RANK_2},
        {'3', RANK_3},
        {'4', RANK_4},
        {'5', RANK_5},
        {'6', RANK_6},
        {'7', RANK_7},
        {'8', RANK_8},
        {'9', RANK_9}};

/**
 * @brief get_origin_square Returns the origin square for a valid ucciMove
 * @param uciMove uci-Move in string notation
 * @return origin square
 */
Square get_origin_square(const string &uciMove);

/**
 * @brief get_origin_square Returns the destination square for a valid ucciMove
 * @param uciMove uci-Move in string notation
 * @return destination square
 */
Square get_destination_square(const string &uciMove);

#endif //FAIRYUTIL_H
