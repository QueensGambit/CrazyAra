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
 * @file: board.cpp
 * Created on 23.05.2019
 * @author: queensgambit
 */

#ifndef MODE_POMMERMAN
#include "board.h"
#include "stateobj.h"
#include "uci.h"
#include <iostream>
#include "movegen.h"

using namespace std;

Board::Board()
{
}

Board::Board(const Board &b)
{
    // TODO: Change to usage of swap
    operator=(b);
}

Board::~Board()
{
}

#ifdef CRAZYHOUSE
Bitboard Board::promoted_pieces() const
{
    return promotedPieces;
}

int Board::get_pocket_count(Color c, PieceType pt) const
{
    return pieceCountInHand[c][pt];
}
#endif

Key Board::hash_key() const
{
    return st->key;
}

void Board::set_state_info(StateInfo *st)
{
    this->st = st;
}

StateInfo *Board::get_state_info() const
{
    return st;
}

Board& Board::operator=(const Board &b)
{
    std::copy(b.board, b.board+SQUARE_NB, this->board );
    std::copy(b.byTypeBB, b.byTypeBB+PIECE_TYPE_NB, this->byTypeBB);
    std::copy(b.byColorBB, b.byColorBB+COLOR_NB, this->byColorBB);
    std::copy(b.pieceCount, b.pieceCount+PIECE_NB, this->pieceCount);
#ifdef HORDE
    std::copy(&b.pieceList[0][0], &b.pieceList[0][0]+PIECE_NB*SQUARE_NB, &this->pieceList[0][0]);
#else
    std::copy(&b.pieceList[0][0], &b.pieceList[0][0]+PIECE_NB*16, &this->pieceList[0][0]);
#endif
#ifdef CRAZYHOUSE
    std::copy(&b.pieceCountInHand[0][0], &b.pieceCountInHand[0][0]+COLOR_NB*PIECE_TYPE_NB, &this->pieceCountInHand[0][0]);
    promotedPieces = b.promotedPieces;
#endif
    std::copy(b.index, b.index+SQUARE_NB, this->index);
    std::copy(b.castlingRightsMask, b.castlingRightsMask+SQUARE_NB, this->castlingRightsMask);
#if defined(GIVEAWAY) || defined(EXTINCTION) || defined(TWOKINGS)
    std::copy(b.castlingKingSquare, b.castlingKingSquare+COLOR_NB, this->castlingKingSquare);
#endif
    std::copy(b.castlingRookSquare, b.castlingRookSquare+CASTLING_RIGHT_NB, castlingRookSquare);
    std::copy(b.castlingPath, b.castlingPath+CASTLING_RIGHT_NB, this->castlingPath);
    this->gamePly = b.gamePly;
    sideToMove = b.sideToMove;
    psq = b.psq;
    thisThread = b.thisThread;
    st = b.st;
    chess960 = b.chess960;
    var = b.var;
    subvar = b.subvar;
#if defined(MODE_CHESS) || defined(MODE_LICHESS)
    lastMoves = b.lastMoves;  // vectors and deques are deeply copied by default
#endif
    return *this;
}

int Board::plies_from_null() const
{
    return st->pliesFromNull;
}

vector<Action> Board::legal_actions() const
{
    vector<Action> legalMoves;
    // generate the legal moves and save them in the list
    for (const ExtMove& move : MoveList<LEGAL>(*this)) {
        legalMoves.push_back(move.move);
    }
    return legalMoves;
}

size_t Board::total_move_cout() const
{
    return size_t(gamePly / 2);
}

size_t Board::number_repetitions() const
{
    if (st->repetition == 0) {
        return 0;
    }
    else if (st->repetition) {
        return 1;
    }
    return 2;
}

bool Board::can_claim_3fold_repetition() const
{
    // The repetition info stores the ply distance to the next previous
    // occurrence of the same position.
    // It is negative in the 3-fold case, or zero if the position was not repeated.
    return st->repetition < 0;
}

bool Board::is_50_move_rule_draw() const
{
#ifdef CRAZYHOUSE
    if (is_house()) {} else
#endif
        if (st->rule50 > 99 && (!checkers() || MoveList<LEGAL>(*this).size())) {
            return true;
        }
    return false;
}

bool Board::is_terminal() const
{
    // 3-fold-repetition and 50 move rul draw is handled outside move generation
    if (can_claim_3fold_repetition() || is_50_move_rule_draw() || draw_by_insufficient_material()) {
        return true;
    }

    for (const ExtMove move : MoveList<LEGAL>(*this)) {
        return false;
    }
    return true;
}

bool Board::draw_by_insufficient_material() const
{
    // fast return options (insufficient material can never by reached in some variants)
#ifdef CRAZYHOUSE
    if (is_house()) {
        return false;
    }
#endif
#ifdef KOTH
    if (is_koth()) {
        return false;
    }
#endif
#ifdef THREECHECK
    if (is_three_check()) {
        return false;
    }
#endif
#ifdef ANTI
    if (is_anti()) {
        return false;
    }
#endif
#ifdef RACE
    if (is_race()) {
        return false;
    }
#endif
#ifdef HORDE
    if (is_horde()) {
        // it seems not be worth to handle all cases here
        return false;
    }
#endif

    // default early stopping
    if (this->count<ALL_PIECES>() > 4) {
        return false;
    }

    // check for chess and atomic
    return (this->count<ALL_PIECES>() == 2) ||                                      // 1) KK
           (this->count<ALL_PIECES>() == 3 && this->count<BISHOP>() == 1) ||        // 2) KB vs K
           (this->count<ALL_PIECES>() == 3 && this->count<KNIGHT>() == 1) ||        // 3) KN vs K
           (this->count<ALL_PIECES>() == 4 &&
           (this->count<KNIGHT>(WHITE) == 2 || this->count<KNIGHT>(BLACK) == 2));   // 4) KNN vs K
}

void Board::add_move_to_list(Move m)
{
    if (StateConstants::NB_LAST_MOVES() == 0) {
        return;
    }
    lastMoves.push_front(m);
    if (lastMoves.size() > StateConstants::NB_LAST_MOVES()) {
        lastMoves.pop_back();
    }
}

void Board::do_move(Move m, StateInfo &newSt)
{
    add_move_to_list(m);
    Position::do_move(m, newSt);
}

void Board::do_move(Move m, StateInfo &newSt, bool givesCheck)
{
    add_move_to_list(m);
    Position::do_move(m, newSt, givesCheck);
}

void Board::undo_move(Move m)
{
    if (!lastMoves.empty()) {
        // make sure the lastMoves deque is not empty, otherwise crash will occur
        lastMoves.pop_front();
    }
    Position::undo_move(m);
}

deque<Move> Board::get_last_moves() const
{
    return lastMoves;
}

int Board::get_board_piece_count(Color color, PieceType pieceType) const
{
    return pieceCount[make_piece(color, pieceType)];
}

void Board::set(const string &fenStr, bool isChess960, Variant v, StateInfo *si, Thread *th)
{
    lastMoves.clear();
    Position::set(fenStr, isChess960, v, si, th);
}

void Board::set(const string &code, Color c, Variant v, StateInfo *si)
{
    lastMoves.clear();
    Position::set(code, c, v, si);
}

std::string pgn_move(Move m, bool chess960, const Board& pos, const std::vector<Action>& legalMoves, bool leadsToWin, bool bookMove)
{
    std::string move;

    const Square from = from_sq(m);
    const Square to = to_sq(m);

    if (m == MOVE_NONE) {
        return "(none)";
    }

    if (m == MOVE_NULL) {
        return "0000";
    }

    bool rank_ambiguous;
    bool file_ambiguous;

    string ambiguous = "";
    if (is_pgn_move_ambiguous(m, pos, legalMoves, file_ambiguous, rank_ambiguous)) {
        if (file_ambiguous && rank_ambiguous) {
            ambiguous = UCI::square(from);
        }
        else if (file_ambiguous) {
            ambiguous = char('1' + rank_of(from));
        }
        else {
            ambiguous = char('a' + file_of(from));
        }
    }
    else {
        ambiguous = "";
    }

    if (type_of(m) == CASTLING) {
        if (file_of(from) < file_of(to)) {
            move = "O-O";
        } else {
            move = "O-O-O";
        }
    }
    else if (pos.capture(m)) {
        if (pos.piece_on(from) == W_PAWN || pos.piece_on(from) == B_PAWN) {
            move = std::string{"abcdefgh "[file_of(from)]} + "x";
        }
        else {
            move = std::string{" PNBRQK  PNBRQK "[pos.piece_on(from)]} + ambiguous + "x";
        }
        move += UCI::square(to);
    }

#ifdef CRAZYHOUSE
    else if (type_of(m) == DROP) {
        move = std::string{" PNBRQK  PNBRQK "[dropped_piece(m)], '@'} + UCI::square(to);
    }
#endif
    else {
        if (pos.piece_on(from) == W_PAWN || pos.piece_on(from) == B_PAWN) {
            move = UCI::square(to);
        }
        else {
            move = std::string{" PNBRQK  PNBRQK "[pos.piece_on(from)]} + ambiguous + UCI::square(to);
        }
    }

    if (type_of(m) == PROMOTION) {
        move += " PNBRQK"[promotion_type(m)];
    }

    if (pos.gives_check(m)) {
        if (leadsToWin) {
            move += "#";
        }
        else {
            move += "+";
        }
    }

    if (bookMove) {
        move += " {book}";
    }
    return move;
}


bool is_pgn_move_ambiguous(Move m, const Board& pos, const std::vector<Action> &legalMoves, bool &isFileAmbiguous, bool &isRankAmbiguous)
{
    bool ambiguous = false;
    isFileAmbiguous = false;
    isRankAmbiguous = false;
    const Square from = from_sq(m);
    const Square to = to_sq(m);

    for (Action move: legalMoves) {
        const Square cur_from = from_sq(Move(move));
        const Square cur_to = to_sq(Move(move));
        if (to == cur_to && from != cur_from && pos.piece_on(from) == pos.piece_on(cur_from)) {
            ambiguous = true;
            if (file_of(from) == file_of(cur_from)) {
                isFileAmbiguous = true;
            }
            if (rank_of(from) == rank_of(cur_from)) {
                isRankAmbiguous = true;
            }
        }
    }
    return ambiguous;
}

bool leads_to_terminal(const Board &pos, Move m, StateListPtr& states)
{
    Board posCheckTerminal = Board(pos);
    states->emplace_back();
    posCheckTerminal.do_move(m, states->back());
    return posCheckTerminal.is_terminal();
}

Tablebases::WDLScore probe_wdl(Board& pos, Tablebases::ProbeState* result)
{
    return Tablebases::probe_wdl(pos, result);
}

int probe_dtz(Board &pos, Tablebases::ProbeState *result)
{
    return Tablebases::probe_dtz(pos, result);
}

void generate_dtz_values(const vector<Move>& legalMoves, Board& pos, DynamicVector<int>& dtzValues) {
    StateListPtr states = StateListPtr(new std::deque<StateInfo>(0));
    // fill dtz value vector
    for (size_t idx = 0; idx < legalMoves.size(); ++idx) {
        states->emplace_back();
        pos.do_move(legalMoves[idx], states->back());
        Tablebases::ProbeState result;
        int dtzValue = -probe_dtz(pos, &result);
        if (result != Tablebases::FAIL) {
            dtzValues[idx] = dtzValue;
        }
        else {
            cerr << "DTZ tablebase look-up failed!";
        }
        pos.undo_move(legalMoves[idx]);
    }
}

bool enhance_move_type(float increment, float thresh, const vector<Move>& legalMoves, const DynamicVector<bool>& moveType, DynamicVector<float>& policyProbSmall)
{
    bool update = false;
    for (size_t i = 0; i < legalMoves.size(); ++i) {
        if (moveType[i] && policyProbSmall[i] < thresh) {
            policyProbSmall[i] += increment;
            update = true;
        }
    }
    return update;
}

bool is_check(const Board* pos, Move move)
{
    return pos->gives_check(move);
}

bool is_capture(const Board* pos, Move move)
{
    return pos->capture(move);
}

#endif

unsigned int get_majors_and_minors_count(const Board& pos)
{
    return pos.count<QUEEN>() + pos.count<ROOK>() + pos.count<KNIGHT>() + pos.count<BISHOP>();
}

bool is_backrank_sparse(const Board& pos)
{
    Bitboard backrankPiecesWhiteBb = pos.pieces(WHITE, ALL_PIECES) & rank_bb(Rank(0));
    Bitboard backrankPiecesBlackBb = pos.pieces(BLACK, ALL_PIECES) & rank_bb(Rank(7));

    // True if either white or black backrank is sparse (three or less pieces)
    return (popcount(backrankPiecesWhiteBb) <= 3) || (popcount(backrankPiecesBlackBb) <= 3);
}

int score_region(int numWhitePiecesInRegion, int numBlackPiecesInRegion, int rank)
{
    if (numWhitePiecesInRegion == 1 && numBlackPiecesInRegion == 0) {
        return 1 + (8 - rank);
    }
    else if (numWhitePiecesInRegion == 2 && numBlackPiecesInRegion == 0) {
        return 2 + ((rank > 2) ? (rank - 2) : 0);
    }
    else if (numWhitePiecesInRegion == 3 && numBlackPiecesInRegion == 0) {
        return 3 + ((rank > 1) ? (rank - 1) : 0);
    }
    else if (numWhitePiecesInRegion == 4 && numBlackPiecesInRegion == 0) {
        return 3 + ((rank > 1) ? (rank - 1) : 0);
    }
    else if (numWhitePiecesInRegion == 0 && numBlackPiecesInRegion == 1) {
        return 1 + rank;
    }
    else if (numWhitePiecesInRegion == 1 && numBlackPiecesInRegion == 1) {
        return 5 + abs(3 - rank);
    }
    else if (numWhitePiecesInRegion == 2 && numBlackPiecesInRegion == 1) {
        return 4 + rank;
    }
    else if (numWhitePiecesInRegion == 3 && numBlackPiecesInRegion == 1) {
        return 5 + rank;
    }
    else if (numWhitePiecesInRegion == 0 && numBlackPiecesInRegion == 2) {
        return 2 + ((rank < 6) ? (6 - rank) : 0);
    }
    else if (numWhitePiecesInRegion == 1 && numBlackPiecesInRegion == 2) {
        return 4 + (6 - rank);
    }
    else if (numWhitePiecesInRegion == 2 && numBlackPiecesInRegion == 2) {
        return 7;
    }
    else if (numWhitePiecesInRegion == 0 && numBlackPiecesInRegion == 3) {
        return 3 + ((rank < 7) ? (7 - rank) : 0);
    }
    else if (numWhitePiecesInRegion == 1 && numBlackPiecesInRegion == 3) {
        return 5 + (6 - rank);
    }
    else if (numWhitePiecesInRegion == 0 && numBlackPiecesInRegion == 4) {
        return 3 + ((rank < 7) ? (7 - rank) : 0);
    }

    return 0;  // for 0 white and 0 black and all other (incorrect) options with a sum that is bigger than 4

}

int get_mixedness(const Board& pos)
{
    int mix = 0;

    for (int rankIdx = 0; rankIdx < 7; ++rankIdx) { // use ranks 1 to 7 (indices 0 to 6)
        for (int fileIdx = 0; fileIdx < 7; ++fileIdx) { // use files A to G (indices 0 to 6)
            int numWhitePiecesInRegion = 0;
            int numBlackPiecesInRegion = 0;
            for (int dx = 0; dx < 2; ++dx) {
                for (int dy = 0; dy < 2; ++dy) {
                    Square currSquare = make_square(File(fileIdx + dx), Rank(rankIdx + dy));
                    Piece currPiece = pos.piece_on(currSquare);

                    if (currPiece != NO_PIECE) {
                        if (color_of(currPiece) == WHITE)
                        {
                            numWhitePiecesInRegion++;
                        }
                        else {
                            numBlackPiecesInRegion++;
                        }
                    }
                }
            }
            mix += score_region(numWhitePiecesInRegion, numBlackPiecesInRegion, rankIdx + 1);
        }
    }

    return mix;
}

GamePhase Board::get_phase(unsigned int numPhases, GamePhaseDefinition gamePhaseDefinition) const
{
    if (gamePhaseDefinition == LICHESS) {

        assert(numPhases == 3);  // lichess definition requires three models to be loaded

        // returns the game phase based on the lichess definition implemented in:
        // https://github.com/lichess-org/scalachess/blob/master/src/main/scala/Divider.scala
        unsigned int numMajorsAndMinors = get_majors_and_minors_count(*this);

        if (numMajorsAndMinors <= 6)
        {
            return GamePhase(2);
        }
        else
        {
            bool backrankSparse = is_backrank_sparse(*this);
            int mixednessScore = get_mixedness(*this);

            if (numMajorsAndMinors <= 10 || backrankSparse || mixednessScore > 150)
            {
                return GamePhase(1);
            }
            else
            {
                return GamePhase(0);
            }
        }
    }
    else if (gamePhaseDefinition == MOVECOUNT) {
        if (numPhases == 1) { // directly return phase 0 if there is only a single network loaded
            return GamePhase(0);
        }
        else {  // use naive phases by move count
            double averageMovecountPerGame = 42.85;
            double phaseLength = std::round(averageMovecountPerGame / numPhases);
            size_t movesCompleted = this->total_move_cout();
            double gamePhaseDouble = movesCompleted / phaseLength;
            if (gamePhaseDouble > numPhases - 1) { // ensure that all higher results are attributed to the last phase
                return GamePhase(numPhases - 1);
            }
            else {
                return GamePhase(gamePhaseDouble); // truncated to Integer value
            }
        }
    }
  return GamePhase(0);
}
