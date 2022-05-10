#include "fairyinputrepresentation.h"
#include "fairystate.h"

using namespace std;

void set_bits_from_bitmap(Bitboard bitboard, size_t channel, float *inputPlanes, Color color) {
    size_t p = 0;
    while (bitboard != Bitboard(0)) {
        if (bitboard & Bitboard(0x1)) {
            if (color == WHITE) {
                int col = std::abs(9-std::floor(p/9));
                int row = p % 9;
                inputPlanes[channel * StateConstantsFairy::NB_SQUARES() + col * 9 + row] = 1;
            }
            else {
                inputPlanes[channel * StateConstantsFairy::NB_SQUARES() + p] = 1;
            }
        }
        // Largeboards use 12 files per rank, xiangqi boards only use 9 files per rank
        (p+1) % 9 == 0 ? bitboard = bitboard >> 4 : bitboard = bitboard >> 1;
        p++;
    }
}

void board_to_planes(const FairyBoard* pos, bool normalize, float *inputPlanes) {
    fill(inputPlanes, inputPlanes + StateConstantsFairy::NB_VALUES_TOTAL(), 0.0f);
    size_t currentChannel = 0;
    Color me = pos->side_to_move();
    Color you = ~me;
#ifdef MODE_BOARDGAMES
    const vector<PieceType> pieces = {PAWN};
#else
    // pieces (ORDER: King, Advisor, Elephant, Horse, Rook, Cannon, Soldier)
    const vector<PieceType> pieces = {KING, FERS, ELEPHANT, HORSE, ROOK, CANNON, SOLDIER};
#endif

    // pieces
    for (Color color : {me, you}) {
        for (PieceType piece : pieces) {
            const Bitboard pieces = pos->pieces(color, piece);
#ifdef MODE_BOARDGAMES
            set_bits_from_bitmap(pieces, currentChannel, inputPlanes, BLACK);  // disable mirroring
#else
            set_bits_from_bitmap(pieces, currentChannel, inputPlanes, me);
#endif
            currentChannel++;
        }
    }

#ifndef MODE_BOARDGAMES
    // pocket count
    for (Color color : {me, you}) {
        for (PieceType piece : {FERS, ELEPHANT, HORSE, ROOK, CANNON, SOLDIER}) {
            int pocket_cnt = pos->get_pocket_count(color, piece);
            if (pocket_cnt > 0) {
                std::fill(inputPlanes + currentChannel * StateConstantsFairy::NB_SQUARES(),
                          inputPlanes + (currentChannel + 1) * StateConstantsFairy::NB_SQUARES(),
                          normalize ? pocket_cnt / StateConstantsFairy::MAX_NB_PRISONERS() : pocket_cnt);
            }
            currentChannel++;
        }
    }
#endif

    // color
    if (me == WHITE) {
        std::fill(inputPlanes + currentChannel * StateConstantsFairy::NB_SQUARES(),
                  inputPlanes + (currentChannel + 1) * StateConstantsFairy::NB_SQUARES(), 1.0f);
    }
    currentChannel++;

    // total move count
    std::fill(inputPlanes + currentChannel * StateConstantsFairy::NB_SQUARES(),
              inputPlanes + (currentChannel + 1) * StateConstantsFairy::NB_SQUARES(),
              normalize ? (std::floor(pos->game_ply() / 2 )) / StateConstantsFairy::MAX_FULL_MOVE_COUNTER() : std::floor(pos->game_ply() / 2 ));
}
