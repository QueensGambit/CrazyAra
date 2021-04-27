#include "fairyutil.h"
#include <unordered_map>


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


Square get_origin_square(const string& uciMove)
{
    File fromFile = FILE_LOOKUP.at(uciMove[0]);
    Rank fromRank = isdigit(uciMove[2]) ? RANK_10 : RANK_LOOKUP.at(uciMove[1]);
    return make_square(fromFile, fromRank);
}

Square get_destination_square(const string& uciMove)
{
    File toFile;
    Rank toRank;
    if (uciMove.size() == 6) {
        toFile = FILE_LOOKUP.at(uciMove[3]);
        toRank = RANK_10;
    }
    else if (uciMove.size() == 5) {
        if (isdigit(uciMove[2])) {
            toFile = FILE_LOOKUP.at(uciMove[3]);
            toRank = RANK_LOOKUP.at(uciMove[4]);
        }
        else {
            toFile = FILE_LOOKUP.at(uciMove[2]);
            toRank = RANK_10;
        }
    }
    else {
        toFile = FILE_LOOKUP.at(uciMove[2]);
        toRank = RANK_LOOKUP.at(uciMove[3]);
    }
    return make_square(toFile, toRank);
}

char file_to_uci(File file) {
    for (auto it = FILE_LOOKUP.begin(); it != FILE_LOOKUP.end(); ++it) {
        if (it->second == file) {
            return it->first;
        }
    }
    return char();
}

std::string rank_to_uci(Rank rank) {
    for (auto it = RANK_LOOKUP.begin(); it != RANK_LOOKUP.end(); ++it) {
        if (it->second == rank) {
            return std::string(1, it->first);
        }
    }
    return "10";
}
