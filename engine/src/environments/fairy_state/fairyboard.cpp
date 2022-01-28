#include "fairyboard.h"
#include "apiutil.h"
#include "fairyutil.h"


FairyBoard::FairyBoard() {}

FairyBoard::FairyBoard(const FairyBoard &b) {
    operator=(b);
}

FairyBoard::~FairyBoard() {}

FairyBoard& FairyBoard::operator=(const FairyBoard &b) {
    std::copy(b.board, b.board+SQUARE_NB, this->board);
    std::copy(b.byTypeBB, b.byTypeBB+PIECE_TYPE_NB, this->byTypeBB);
    std::copy(b.byColorBB, b.byColorBB+COLOR_NB, this->byColorBB);
    std::copy(b.pieceCount, b.pieceCount+PIECE_NB, this->pieceCount);
    std::copy(b.castlingRightsMask, b.castlingRightsMask+SQUARE_NB, this->castlingRightsMask);
    std::copy(b.castlingRookSquare, b.castlingRookSquare+CASTLING_RIGHT_NB, castlingRookSquare);
    std::copy(b.castlingPath, b.castlingPath+CASTLING_RIGHT_NB, this->castlingPath);
    this->gamePly = b.gamePly;
    sideToMove = b.sideToMove;
    psq = b.psq;
    thisThread = b.thisThread;
    st = b.st;
    tsumeMode = b.tsumeMode;
    chess960 = b.chess960;
    std::copy(&b.pieceCountInHand[0][0], &b.pieceCountInHand[0][0]+COLOR_NB*PIECE_TYPE_NB, &this->pieceCountInHand[0][0]);
    promotedPieces = b.promotedPieces;
    var = b.var;
    return *this;
}

int FairyBoard::get_pocket_count(Color c, PieceType pt) const {
    return Position::count_in_hand(c, pt);
}

Key FairyBoard::hash_key() const {
    return state()->key;
}

bool FairyBoard::is_terminal() const {
    // "Unlike in chess, in which stalemate is a draw, in xiangqi, it is a loss for the stalemated player."
    // -- https://en.wikipedia.org/wiki/Xiangqi
    if (this->number_repetitions() != 0) {
        return true;
    }
    for (const ExtMove move : MoveList<LEGAL>(*this)) {
        return false;
    }
    return true;
}

size_t FairyBoard::number_repetitions() const {
    StateInfo *st = state();
    // st->repetition:
    // "It is the ply distance from the previous
    // occurrence of the same position, negative in the 3-fold case, or zero" -- fairy/position.cpp
    // if the position was not repeated.
    if (st->repetition > 0) {
        return 1;
    }
    if (st->repetition < 0) {
        return 2;
    }
    return 0;
}

Result get_result(const FairyBoard &pos, bool inCheck) {
    if (pos.is_terminal()) {
        if (!inCheck) {
            return DRAWN;
        }
        if (pos.side_to_move() == BLACK) {
            return WHITE_WIN;
        }
        else {
            return BLACK_WIN;
        }
    }
    return NO_RESULT;
}

std::string wxf_move(Move m, const FairyBoard& pos) {
    Notation notation = NOTATION_XIANGQI_WXF;

    std::string wxf = "";

    Color us = pos.side_to_move();
    Square from = from_sq(m);
    Square to = to_sq(m);

    wxf += SAN::piece(pos, m, notation);
    SAN::Disambiguation d = SAN::disambiguation_level(pos, m, notation);
    wxf += disambiguation(pos, from, notation, d);

    if (rank_of(from) == rank_of(to)) {
        wxf += "=";
    }
    else if (relative_rank(us, to, pos.max_rank()) > relative_rank(us, from, pos.max_rank())) {
        wxf += "+";
    }
    else {
        wxf += "-";
    }

    if (type_of(m) != DROP) {
        wxf += file_of(to) == file_of(from) ? std::to_string(std::abs(rank_of(to) - rank_of(from))) : SAN::file(pos, to, notation);
    }
    else {
        wxf += SAN::square(pos, to, notation);
    }
    return wxf;
}

std::string uci_move(Move m) {
    std::string uciMove;

    Square from = from_sq(m);
    Square to = to_sq(m);

    char fromFile = file_to_uci(file_of(from));
    std::string fromRank = rank_to_uci(rank_of(from));
    char toFile = file_to_uci(file_of(to));
    std::string toRank = rank_to_uci(rank_of(to));

    return std::string(1, fromFile) + fromRank + std::string(1, toFile) + toRank;
}
