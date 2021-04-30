#ifndef FAIRYBOARD_H
#define FAIRYBOARD_H

#include <position.h>
#include <blaze/Math.h>
#include "state.h"

using blaze::StaticVector;
using blaze::DynamicVector;

class FairyBoard : public Position
{
public:
    FairyBoard();
    FairyBoard(const FairyBoard& b);
    ~FairyBoard();
    FairyBoard& operator=(const FairyBoard &b);

    int get_pocket_count(Color c, PieceType pt) const;
    Key hash_key() const;
    bool is_terminal() const;
    size_t number_repetitions() const;
};

Result get_result(const FairyBoard &pos, bool inCheck);
std::string wxf_move(Move m, const FairyBoard& pos);
std::string uci_move(Move m);

#endif //FAIRYBOARD_H
