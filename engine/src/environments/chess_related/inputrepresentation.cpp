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
 * @file: inputrepresentation.cpp
 * Created on 27.05.2019
 * @author: queensgambit
 */

#include "inputrepresentation.h"
#include <iostream>
#include <deque>
#include "stateobj.h"
#include "sfutil.h"
using namespace std;

void set_bits_from_bitmap(Bitboard bitboard, size_t channel, float *inputPlanes, bool flipBoard) {
    size_t p = 0;
    // set the individual bits for the pieces
    // https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
    while (bitboard != 0) {
        if (bitboard & 0x1) {
            if (flipBoard) {
                //                                                         row            col
                inputPlanes[channel * StateConstants::NB_SQUARES() + (7 - (p / 8)) * 8 + (p % 8)] = 1;
            }
            else {
                inputPlanes[channel * StateConstants::NB_SQUARES() + p] = 1;
            }
        }
        bitboard >>= 1;
        p++;
    }
}


inline bool flip_board(const Board *pos) {
#ifdef MODE_LICHESS
    if (pos->is_race()) {
        return false;
    }
#endif
    return pos->side_to_move() == BLACK;
}

struct PlaneData {
    const Board* pos;
    bool flipBoard;
    float* inputPlanes;
    size_t currentChannel;
    bool normalize;
    PlaneData(const Board* pos, float* inputPlanes, bool normalize):
        pos(pos), flipBoard(flip_board(pos)), inputPlanes(inputPlanes), currentChannel(0), normalize(normalize)
    {
        // intialize the input_planes with 0
        std::fill(inputPlanes, inputPlanes+StateConstants::NB_VALUES_TOTAL(), 0.0f);
    }
    inline Color me() {
        return pos->side_to_move();
    }
    inline Color you() {
        return ~pos->side_to_move();
    }
    template<bool increment>
    inline void set_plane_to_one() {
        set_plane_to_value<increment>(1.0f);
    }
    template<bool increment>
    inline void set_plane_to_value(float value) {
        std::fill(inputPlanes + currentChannel * StateConstants::NB_SQUARES(), inputPlanes + (currentChannel+1) * StateConstants::NB_SQUARES(), value);
        if (increment) {
            ++currentChannel;
        }
    }
    inline void set_plane_to_bitboard(Bitboard bitboard) {
        set_bits_from_bitmap(bitboard, currentChannel, inputPlanes, flipBoard);
        ++currentChannel;
    }
    template<bool increment>
    inline void set_single_square_to_one(Square sq, size_t channelOffset=0) {
        set_single_square_to_value<increment>(sq, 1.0f, channelOffset);
    }
    template<bool increment>
    inline void set_single_square_to_value(Square sq, float value, size_t channelOffset=0) {
        Square squareToSet = flipBoard ? vertical_flip(sq) : sq;
        inputPlanes[(currentChannel+channelOffset) * StateConstants::NB_SQUARES() + squareToSet] = value;
        if (increment) {
            ++currentChannel;
        }
    }
};


inline void set_plane_pieces(PlaneData& p)
{
    for (Color color : {p.me(), p.you()}) {
        for (PieceType piece: {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
            const Bitboard pieces = p.pos->pieces(color, piece);
            // set the individual bits for the pieces
            // https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/value
            p.set_plane_to_bitboard(pieces);
        }
    }
}

inline void set_plane_repetition(PlaneData& p, size_t boardRepetition)
{
    if (boardRepetition >= 1) {
        p.set_plane_to_one<true>();
        if (boardRepetition >= 2) {
            p.set_plane_to_one<true>();
            return;
        }
        ++p.currentChannel;
        return;
    }
    p.currentChannel+= 2;
}

#ifdef CRAZYHOUSE
inline void set_plane_pockets(PlaneData& p)
{
    for (Color color : {p.me(), p.you()}) {
        for (PieceType piece: {PAWN, KNIGHT, BISHOP, ROOK, QUEEN}) {
            // unfortunately you can't use a loop over count_in_hand() PieceType because of template arguments
            int pocket_cnt = p.pos->get_pocket_count(color, piece);
            if (pocket_cnt > 0) {
                p.set_plane_to_value<false>(p.normalize ? pocket_cnt / StateConstants::MAX_NB_PRISONERS() : pocket_cnt);
            }
            p.currentChannel++;
        }
    }
}

inline void set_plane_promoted_pieces(PlaneData& p)
{
    p.set_plane_to_bitboard(p.pos->promoted_pieces() & p.pos->pieces(p.me()));
    p.set_plane_to_bitboard(p.pos->promoted_pieces() & p.pos->pieces(p.you()));
}
#endif

inline void set_plane_ep_square(PlaneData& p)
{
    if (p.pos->ep_square() != SQ_NONE) {
        p.set_single_square_to_one<false>(p.pos->ep_square());
    }
    p.currentChannel++;
}

inline void set_plane_color_info(PlaneData(& p))
{
    if (p.me() == WHITE) {
        p.set_plane_to_one<true>();
        return;
    }
    p.currentChannel++;
}

inline void set_plane_total_move_count(PlaneData& p)
{
    // stockfish starts counting from 0, the full move counter starts at 1 in FEN
    p.set_plane_to_value<true>(p.normalize ? ((p.pos->game_ply()/2)+1) / StateConstants::MAX_FULL_MOVE_COUNTER() : ((p.pos->game_ply()/2)+1));
}

inline void set_plane_castling_rights(PlaneData& p)
{
    if (p.me() == WHITE) {
        if (p.pos->can_castle(WHITE_OO)) {
            p.set_plane_to_one<false>();
        }
        p.currentChannel++;
        if (p.pos->can_castle(WHITE_OOO)) {
            p.set_plane_to_one<false>();
        }
        p.currentChannel++;
        if (p.pos->can_castle(BLACK_OO)) {
            p.set_plane_to_one<false>();
        }
        p.currentChannel++;
        if (p.pos->can_castle(BLACK_OOO)) {
            p.set_plane_to_one<false>();
        }
        p.currentChannel++;
        return;
    }
    // second player to move
    if (p.pos->can_castle(BLACK_OO)) {
        p.set_plane_to_one<false>();
    }
    p.currentChannel++;
    if (p.pos->can_castle(BLACK_OOO)) {
        p.set_plane_to_one<false>();
    }
    p.currentChannel++;
    if (p.pos->can_castle(WHITE_OO)) {
        p.set_plane_to_one<false>();
    }
    p.currentChannel++;
    if (p.pos->can_castle(WHITE_OOO)) {
        p.set_plane_to_one<false>();
    }
    p.currentChannel++;
}

void set_no_progress_counter(PlaneData& p)
{
    p.set_plane_to_value<true>(p.normalize ? p.pos->rule50_count() / StateConstants::MAX_NB_NO_PROGRESS(): p.pos->rule50_count());
}

#ifdef THREECHECK
void set_remaining_checks(PlaneData& p)
{
    if (p.pos->is_three_check()) {
        for (Color color : {p.me(), p.you()}) {
            if (p.pos->checks_given(color) != 0) {
                p.set_plane_to_one<true>();
                if (p.pos->checks_given(color) >= 2) {
                    p.set_plane_to_one<false>();
                }
                p.currentChannel++;
            }
            else {
                p.currentChannel += 2;
            }
        }
        return;
    }
    p.currentChannel += 4;
}
#endif

#ifdef MODE_LICHESS
inline void set_variant_and_960(PlaneData& p)
{
    // set the is960 boolean flag when active
    if (p.pos->is_chess960()) {
        p.set_plane_to_one<false>();
    }

    const size_t preCurrentChannel = p.currentChannel;
    // set the current active variant as a one-hot encoded entry
    p.currentChannel += StateConstants::CHANNEL_MAPPING_VARIANTS().at(p.pos->variant());
    p.set_plane_to_one<false>();
    p.currentChannel = preCurrentChannel + StateConstants::NB_CHANNELS_VARIANTS();
}
#endif

#if defined(MODE_CHESS) || defined(MODE_LICHESS)
inline void set_last_moves(PlaneData& p)
{
    const size_t preCurrentChannel = p.currentChannel;
    // (VI) Fill the bits of the last move planes
    for (const Move move : p.pos->get_last_moves()) {
        p.set_single_square_to_one<true>(from_sq(move));
        p.set_single_square_to_one<true>(to_sq(move));
    }
    p.currentChannel = preCurrentChannel + StateConstants::NB_CHANNELS_HISTORY();
}
#endif

inline void set_960(PlaneData& p)
{
    if (p.pos->is_chess960()) {
        p.set_plane_to_one<false>();
    }
    ++p.currentChannel;
}

inline void set_piece_masks(PlaneData& p)
{
    for (Color color : {p.me(), p.you()}) {
        const Bitboard pieces = p.pos->pieces(color, ALL_PIECES);
        // set the individual bits for the pieces
        // https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/value
        p.set_plane_to_bitboard(pieces);
    }
}

inline void set_checkerboard(PlaneData& p)
{
    bool targetModulo = 1;
    for (uint row = 0; row < StateConstants::BOARD_HEIGHT(); ++row) {
        for (uint col = 0; col < StateConstants::BOARD_WIDTH(); ++col) {
            if (col % 2 == targetModulo) {
                p.inputPlanes[p.currentChannel * StateConstants::NB_SQUARES() + row * StateConstants::BOARD_WIDTH() + col] = 1.0f;
            }
        }
        targetModulo = !targetModulo;
    }
    ++p.currentChannel;
}

inline void set_material_diff(PlaneData& p)
{
    float relativeCount = p.pos->count<PAWN>(p.me()) - p.pos->count<PAWN>(p.you());
    p.set_plane_to_value<true>(p.normalize ? relativeCount / StateConstants::NORMALIZE_PIECE_NUMBER() : relativeCount);
    relativeCount = p.pos->count<KNIGHT>(p.me()) - p.pos->count<KNIGHT>(p.you());
    p.set_plane_to_value<true>(p.normalize ? relativeCount / StateConstants::NORMALIZE_PIECE_NUMBER() : relativeCount);
    relativeCount = p.pos->count<BISHOP>(p.me()) - p.pos->count<BISHOP>(p.you());
    p.set_plane_to_value<true>(p.normalize ? relativeCount / StateConstants::NORMALIZE_PIECE_NUMBER() : relativeCount);
    relativeCount = p.pos->count<ROOK>(p.me()) - p.pos->count<ROOK>(p.you());
    p.set_plane_to_value<true>(p.normalize ? relativeCount / StateConstants::NORMALIZE_PIECE_NUMBER() : relativeCount);
    relativeCount = p.pos->count<QUEEN>(p.me()) - p.pos->count<QUEEN>(p.you());
    p.set_plane_to_value<true>(p.normalize ? relativeCount / StateConstants::NORMALIZE_PIECE_NUMBER() : relativeCount);
}

inline int count_bits_in_bitboard(int bitboard) {
    // https://www.tutorialspoint.com/c-cplusplus-program-to-count-set-bits-in-an-integer
   int count = 0;
   while(bitboard != 0) {
      if((bitboard & 1) == 1) {
         count++;
      }
      bitboard = bitboard >> 1; //right shift 1 bit
   }
   return count;
}

inline void set_attack_planes(PlaneData& p)
{
    // https://www.chessprogramming.org/Square_Attacked_By
    for (Square sq = SQ_A1; sq <= SQ_H8; ++sq) {
        const Bitboard attackers = p.pos->attackers_to(sq);
        for (Color color : {p.me(), p.you()}) {
            const Bitboard myAttackers = attackers & p.pos->pieces(color);
            const float nbAttackers = p.normalize ? float(count_bits_in_bitboard(myAttackers)) / StateConstants::NORMALIZE_ATTACKERS() : count_bits_in_bitboard(myAttackers);
            p.set_single_square_to_value<false>(sq, nbAttackers);
        }
        ++p.currentChannel;
    }
}

inline void set_checkers(PlaneData& p)
{
    p.set_plane_to_bitboard(p.pos->checkers());
}

inline void set_pinners(PlaneData& p)
{
    for (Color color : {p.me(), p.you()}) {
        p.set_plane_to_bitboard(p.pos->pinners(color));
    }
}

inline void set_check_moves(PlaneData& p, const vector<Action>& legalMoves)
{
    for (auto it = legalMoves.begin(); it != legalMoves.end(); ++it) {
        const Move move = Move(*it);
        if (p.pos->gives_check(move)) {
            p.set_single_square_to_one<false>(from_sq(move));
            p.set_single_square_to_one<false>(to_sq(move), 1);
        }
    }
    p.currentChannel += 2;
}

inline void set_mobility(PlaneData& p, const vector<Action>& legalMoves)
{
    p.set_plane_to_value<true>(p.normalize ? legalMoves.size() / StateConstants::NORMALIZE_MOBILITY() : legalMoves.size());
}

inline void set_opposite_bishops(PlaneData& p)
{
    if (p.pos->opposite_bishops()) {
        p.set_plane_to_one<false>();
    }
    ++p.currentChannel;
}

inline void set_material_count(PlaneData& p)
{
    float relativeCount = p.pos->count<PAWN>(p.me());
    p.set_plane_to_value<true>(p.normalize ? relativeCount / StateConstants::NORMALIZE_PIECE_NUMBER() : relativeCount);
    relativeCount = p.pos->count<KNIGHT>(p.me());
    p.set_plane_to_value<true>(p.normalize ? relativeCount / StateConstants::NORMALIZE_PIECE_NUMBER() : relativeCount);
    relativeCount = p.pos->count<BISHOP>(p.me());
    p.set_plane_to_value<true>(p.normalize ? relativeCount / StateConstants::NORMALIZE_PIECE_NUMBER() : relativeCount);
    relativeCount = p.pos->count<ROOK>(p.me());
    p.set_plane_to_value<true>(p.normalize ? relativeCount / StateConstants::NORMALIZE_PIECE_NUMBER() : relativeCount);
    relativeCount = p.pos->count<QUEEN>(p.me());
    p.set_plane_to_value<true>(p.normalize ? relativeCount / StateConstants::NORMALIZE_PIECE_NUMBER() : relativeCount);
}

#ifdef MODE_CHESS
inline void board_to_planes_v_2_7(PlaneData& planeData, const vector<Action>& legalMoves)
{
    set_plane_pieces(planeData);
    set_plane_ep_square(planeData);
    assert(planeData.currentChannel == StateConstants::NB_CHANNELS_POS());
    set_plane_castling_rights(planeData);
    assert(planeData.currentChannel == StateConstants::NB_CHANNELS_POS() + StateConstants::NB_CHANNELS_CONST());
    set_last_moves(planeData);
    assert(planeData.currentChannel == StateConstants::NB_CHANNELS_POS() + StateConstants::NB_CHANNELS_CONST() + StateConstants::NB_LAST_MOVES() * StateConstants::NB_CHANNELS_PER_HISTORY());
    set_960(planeData);
    assert(planeData.currentChannel == StateConstants::NB_CHANNELS_POS() + StateConstants::NB_CHANNELS_CONST() + StateConstants::NB_LAST_MOVES() * StateConstants::NB_CHANNELS_PER_HISTORY() + StateConstants::NB_CHANNELS_VARIANTS());
    set_piece_masks(planeData);
    set_checkerboard(planeData);
    set_material_diff(planeData);
    set_opposite_bishops(planeData);
    set_checkers(planeData);
    set_check_moves(planeData, legalMoves);
    set_mobility(planeData, legalMoves);
}

inline void board_to_planes_v_2_8(PlaneData& planeData, const vector<Action>& legalMoves)
{
    board_to_planes_v_2_7(planeData, legalMoves);
    set_material_count(planeData);
}

inline void board_to_planes_v3(PlaneData& planeData, size_t boardRepetition)
{
    set_plane_pieces(planeData);
    set_plane_repetition(planeData, boardRepetition);
    set_plane_ep_square(planeData);
    assert(planeData.currentChannel == StateConstants::NB_CHANNELS_POS());
    set_plane_castling_rights(planeData);
    set_no_progress_counter(planeData);
    assert(planeData.currentChannel == StateConstants::NB_CHANNELS_POS() + StateConstants::NB_CHANNELS_CONST());
    set_last_moves(planeData);
    assert(planeData.currentChannel == StateConstants::NB_CHANNELS_POS() + StateConstants::NB_CHANNELS_CONST() + StateConstants::NB_LAST_MOVES() * StateConstants::NB_CHANNELS_PER_HISTORY());
    set_960(planeData);
    assert(planeData.currentChannel == StateConstants::NB_CHANNELS_POS() + StateConstants::NB_CHANNELS_CONST() + StateConstants::NB_LAST_MOVES() * StateConstants::NB_CHANNELS_PER_HISTORY() + StateConstants::NB_CHANNELS_VARIANTS());
    set_piece_masks(planeData);
    set_checkerboard(planeData);
    set_material_diff(planeData);
    set_opposite_bishops(planeData);
    set_checkers(planeData);
    set_material_count(planeData);
    assert(planeData.currentChannel == StateConstants::NB_CHANNELS_TOTAL());
}
#endif

void board_to_planes(const Board *pos, size_t boardRepetition, bool normalize, float *inputPlanes)
{
    // Fill in the piece positions
    // Iterate over both color starting with WHITE
    PlaneData planeData(pos, inputPlanes, normalize);

#if VERSION == 2
#if SUB_VERSION == 7
    board_to_planes_v_2_7(planeData, pos->legal_actions());
#elif SUB_VERSION == 8
    board_to_planes_v_2_8(planeData, pos->legal_actions());
#endif
    assert(planeData.currentChannel == StateConstants::NB_CHANNELS_TOTAL());
    return;
#endif

#if VERSION == 3
    board_to_planes_v3(planeData, boardRepetition);
    return;
#endif

    // (I) Set the pieces for both players
    set_plane_pieces(planeData);

    // (II) Fill in the Repetition Data
    // set how often the position has already occurred in the game (default 0 times)
    // this is used to check for claiming the 3 fold repetition rule
    // A game to test out if everything is working correctly is: https://lichess.org/jkItXBWy#73
    set_plane_repetition(planeData, boardRepetition);

#ifndef MODE_CHESS
    // (III) Fill in the Prisoners / Pocket Pieces
    // iterate over all pieces except the king
    set_plane_pockets(planeData);

    // (IV) Fill in the promoted pieces
    // iterate over all promoted pieces according to the mask and set the according bit
    set_plane_promoted_pieces(planeData);
#endif

    // (V) En Passant Square
    // mark the square where an en-passant capture is possible
    set_plane_ep_square(planeData);
    assert(planeData.currentChannel == StateConstants::NB_CHANNELS_POS());

    // (VI) Constant Value Inputs
    // (VI.1) Color
    set_plane_color_info(planeData);

    // (VI.2) Total Move Count
    set_plane_total_move_count(planeData);

    // (IV.3) Castling Rights
    // check for King Side Castling
    set_plane_castling_rights(planeData);

    // (VI.4) No Progress Count
    // define a no 'progress' counter
    // it gets incremented by 1 each move
    // however, whenever a piece gets dropped, a piece is captured or a pawn is moved, it is reset to 0
    // halfmove_clock is an official metric in fen notation
    //  -> see: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
    set_no_progress_counter(planeData);

#ifdef MODE_LICHESS
    // set the remaining checks (only needed for "3check")
    set_remaining_checks(planeData);
#endif
    assert(planeData.currentChannel == StateConstants::NB_CHANNELS_POS() + StateConstants::NB_CHANNELS_CONST());

#ifdef MODE_LICHESS
    // (V) Variants specification
    set_variant_and_960(planeData);
#endif
#ifdef MODE_CHESS
    // (V) Variants specification
    // set the is960 boolean flag when active
    set_960(planeData);
#endif
    assert(planeData.currentChannel == StateConstants::NB_CHANNELS_POS() + StateConstants::NB_CHANNELS_CONST() + StateConstants::NB_CHANNELS_VARIANTS());

#if defined(MODE_CHESS) || defined(MODE_LICHESS)
    set_last_moves(planeData);
#endif
    assert(planeData.currentChannel == StateConstants::NB_CHANNELS_TOTAL());
}

