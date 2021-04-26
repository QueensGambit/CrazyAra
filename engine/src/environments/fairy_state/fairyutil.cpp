#include "fairyutil.h"
#include <unordered_map>


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
