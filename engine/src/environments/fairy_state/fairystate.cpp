#include "fairystate.h"
#include "fairyinputrepresentation.h"
#include "position.h"
#include "movegen.h"
#include "variant.h"


action_idx_map FairyOutputRepresentation::MV_LOOKUP = {};
action_idx_map FairyOutputRepresentation::MV_LOOKUP_MIRRORED = {};
action_idx_map FairyOutputRepresentation::MV_LOOKUP_CLASSIC = {};
action_idx_map FairyOutputRepresentation::MV_LOOKUP_MIRRORED_CLASSIC = {};
vector<std::string> FairyOutputRepresentation::LABELS;
vector<std::string> FairyOutputRepresentation::LABELS_MIRRORED;

FairyState::FairyState() :
        State(),
        states(StateListPtr(new std::deque<StateInfo>(0))),
        variantNumber(0) {}

FairyState::FairyState(const FairyState &f) :
        State(),
        board(f.board),
        states(StateListPtr(new std::deque<StateInfo>(0))),
        variantNumber(f.variantNumber){
    states->emplace_back(f.states->back());
}

std::vector<Action> FairyState::legal_actions() const {
    std::vector<Action> legalMoves;
    for (const ExtMove &move : MoveList<LEGAL>(board)) {
        legalMoves.push_back(Action(move.move));
    }
    return legalMoves;
}

void FairyState::set(const string &fenStr, bool isChess960, int variant) {
    states = StateListPtr(new std::deque<StateInfo>(1));
    Thread *thread;
#ifdef MODE_BOARDGAMES
    board.set(variants.find(StateConstantsFairy::available_variants()[variant])->second, fenStr, isChess960, &states->back(), thread, false);
    variantNumber = variant;
#else
    board.set(variants.find("xiangqi")->second, fenStr, isChess960, &states->back(), thread, false);
#endif
}

void FairyState::get_state_planes(bool normalize, float *inputPlanes, Version version) const {
    board_to_planes(&board, normalize, inputPlanes);
}

unsigned int FairyState::steps_from_null() const {
    return board.game_ply();
}

bool FairyState::is_chess960() const {
    return false;
}

string FairyState::fen() const {
    return board.fen();
}

void FairyState::do_action(Action action) {
    states->emplace_back();
    board.do_move(Move(action), states->back());
}

void FairyState::undo_action(Action action) {
    board.undo_move(Move(action));
}

void FairyState::prepare_action() {
    // pass
}

int FairyState::side_to_move() const {
    return board.side_to_move();
}

Key FairyState::hash_key() const {
    return board.hash_key();
}

void FairyState::flip() {
    board.flip();
}

Action FairyState::uci_to_action(string &uciStr) const {
    return Action(UCI::to_move(board, uciStr));
}

TerminalType FairyState::is_terminal(size_t numberLegalMoves, float &customTerminalValue) const {
    Value value;
    bool gameEnd = board.is_game_end(value, board.game_ply());

    if (gameEnd) {
        if (value == VALUE_DRAW) {
            return TERMINAL_DRAW;
        }
        if (value < VALUE_DRAW) {
            return TERMINAL_LOSS;
        }
        return TERMINAL_WIN;
    }

    if (numberLegalMoves == 0) {
#ifdef MODE_BOARDGAMES
        if(variantNumber == 3){ //variant clobber
            return TERMINAL_LOSS;
        }

        return TERMINAL_DRAW;
#else   // Xinagqi
        // "Unlike in chess, in which stalemate is a draw, in xiangqi, it is a loss for the stalemated player."
        // -- https://en.wikipedia.org/wiki/Xiangqi
        return TERMINAL_LOSS;
#endif
    }
    if (this->number_repetitions() != 0) {
        // "If one side perpetually checks and the other side perpetually chases, the checking side has to stop or be ruled to have lost."
        // -- https://en.wikipedia.org/wiki/Xiangqi
        return TERMINAL_WIN;
    }
    return TERMINAL_NONE;
}

bool FairyState::gives_check(Action action) const {
    return board.gives_check(Move(action));
}

void FairyState::print(ostream &os) const
{
    os << board;
}

FairyState* FairyState::clone() const {
    return new FairyState(*this);
}

unsigned int FairyState::number_repetitions() const {
    return board.number_repetitions();
}

string FairyState::action_to_san(Action action, const std::vector<Action> &legalActions, bool leadsToWin, bool bookMove) const {
    return UCI::move(board, Move(action));
}

Tablebase::WDLScore FairyState::check_for_tablebase_wdl(Tablebase::ProbeState &result) {
	return Tablebase::WDLScoreNone;  // TODO
}

void FairyState::set_auxiliary_outputs(const float* auxiliaryOutputs) {

}

void FairyState::init(int variant, bool isChess960)
{
    states = StateListPtr(new std::deque<StateInfo>(1));
    board.set(variants.find(StateConstantsFairy::available_variants()[variant])->second, variants.find(StateConstantsFairy::available_variants()[variant])->second->startFen, isChess960, &states->back(), nullptr, false);
    variantNumber = variant;
}

GamePhase FairyState::get_phase(unsigned int numPhases, GamePhaseDefinition gamePhaseDefinition) const
{
    // TODO: Implement phase definition here
    return GamePhase(0);
}
