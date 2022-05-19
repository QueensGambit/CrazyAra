#ifndef FAIRYSTATE_H
#define FAIRYSTATE_H

#include "fairyboard.h"
#include "fairyoutputrepresentation.h"
#include "state.h"
#include "uci.h"
#include "variant.h"

class StateConstantsFairy : public StateConstantsInterface<StateConstantsFairy>
{
public:
    static uint BOARD_WIDTH() {
        return NB_SQUARES_HORIZONTAL();
    }
    static uint BOARD_HEIGHT() {
        return NB_SQUARES_VERTICAL();
    }
    static uint NB_SQUARES() {
        return BOARD_WIDTH() * BOARD_HEIGHT();
    }
    static uint NB_CHANNELS_TOTAL() {
        return NB_CHANNELS_POS() + NB_CHANNELS_CONST();
    }
    static uint NB_LABELS() {
        return 49;
    }
    static uint NB_LABELS_POLICY_MAP() {
        return 4500;
    }
    static uint NB_AUXILIARY_OUTPUTS() {
            return 0U;
    }
    static uint NB_PLAYERS() {
        return 2;
    }
    template<PolicyType p, MirrorType m>
    static MoveIdx action_to_index(Action action) {
#ifdef MODE_BOARDGAMES
        const int nRowsCfour = 6;
        const int a10a1 = 443088896;
        vector<int> a10aX = {a10a1};
        for (int idx = 0; idx < nRowsCfour-1; ++idx) {
            a10aX.emplace_back(a10aX.back()+12);  // increment by 12 for each row
        }
        vector<int> a10gX;
        for (int idx = 0; idx < nRowsCfour; ++idx) {
            a10gX.emplace_back(a10aX[idx]+7);  // increment by 7 for end of row
        }
        vector<int> prefix;
        for (int idx = 0; idx < nRowsCfour; ++idx) {
            prefix.emplace_back(idx*7);
        }
        for (int idx = 0; idx < nRowsCfour; ++idx) {
            // check if action is in between a given row
            // e.g. action >= a10a1 && action <= a10g1
            if (action >= a10aX[idx] && action <= a10gX[idx]) {
                const MoveIdx moveIdx = action - a10aX[idx] + prefix[idx];
                return moveIdx;
            }
        }
#endif
        switch (p) {
            case normal:
                switch (m) {
                    case notMirrored:
                        return FairyOutputRepresentation::MV_LOOKUP[action];
                    case mirrored:
                        return FairyOutputRepresentation::MV_LOOKUP_MIRRORED[action];
                    default:
                        return FairyOutputRepresentation::MV_LOOKUP[action];
                }
            case classic:
                switch (m) {
                    case notMirrored:
                        return FairyOutputRepresentation::MV_LOOKUP_CLASSIC[action];
                    case mirrored:
                        return FairyOutputRepresentation::MV_LOOKUP_MIRRORED_CLASSIC[action];
                    default:
                        return FairyOutputRepresentation::MV_LOOKUP_CLASSIC[action];
                }
            default:
                return FairyOutputRepresentation::MV_LOOKUP[action];
        }
    }
    // Currently only ucci notation is supported
    static string action_to_uci(Action action, bool is960) {
        Move m = Move(action);
        Square from = from_sq(m);
        Square to = to_sq(m);

        if (m == MOVE_NONE) {
            return Options["Protocol"] == "usi" ? "resign" : "(none)";
        }
        if (m == MOVE_NULL) {
            return "0000";
        }

        if (is_pass(m) && Options["Protocol"] == "xboard") {
            return "@@@@";
        }
        string fromSquare = rank_of(from) < RANK_10 ? string{char('a' + file_of(from)), char('1' + rank_of(from))}
                                                    : string{char('a' + file_of(from)), '1', '0'};
        string toSquare = rank_of(to) < RANK_10 ? string{char('a' + file_of(to)), char('1' + rank_of(to))}
                                                    : string{char('a' + file_of(to)), '1', '0'};
        return fromSquare + toSquare;
    }
    static void init(bool isPolicyMap) {
        FairyOutputRepresentation::init_labels();
        FairyOutputRepresentation::init_policy_constants(isPolicyMap);
    }

#ifdef MODE_BOARDGAMES
    static uint NB_SQUARES_HORIZONTAL() {
        return 7;
    }
    static uint NB_SQUARES_VERTICAL() {
        return 7;
    }
    static uint NB_CHANNELS_POS() {
        return 2;
    }
    static uint NB_CHANNELS_CONST() {
        return 2;
    }
    static float MAX_NB_PRISONERS() {
        return 0;
    }
    static float MAX_FULL_MOVE_COUNTER() {
        return 500;
    }
#endif

    static std::vector<std::string> available_variants() {
        return {"cfour",
                "tictactoe",
                "breakthrough",
                "clobber",
//                "xiangqi"
                };
    }

    static std::string start_fen(int variant) {
        switch (variant) {
        case 0: //cfour
            return "7/7/7/7/7/7[PPPPPPPPPPPPPPPPPPPPPppppppppppppppppppppp] w - - 0 1";
        case 1: //tictactoe
            return "3/3/3 w - - 0 1";
        case 2: //breakthrough
            return "pppppppp/pppppppp/8/8/8/8/PPPPPPPP/PPPPPPPP w - - 0 1";
        case 3: // clobber
            return "PpPpP/pPpPp/PpPpP/pPpPp/PpPpP/pPpPp w - - 0 1";
//        case 3:
//            return "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";
        default:
            return "";
        }
    }

};

class FairyState : public State
{
private:
    FairyBoard board;
    StateListPtr states;

public:
    FairyState();
    FairyState(const FairyState& f);

    std::vector<Action> legal_actions() const override;
    void set(const std::string &fenStr, bool isChess960, int variant) override;
    void get_state_planes(bool normalize, float *inputPlanes, Version version) const override;
    unsigned int steps_from_null() const override;
    bool is_chess960() const override;
    std::string fen() const override;
    void do_action(Action action) override;
    void undo_action(Action action) override;
    void prepare_action() override;
    int side_to_move() const override;
    Key hash_key() const override;
    void flip() override;
    Action uci_to_action(std::string& uciStr) const override;
    TerminalType is_terminal(size_t numberLegalMoves, float& customTerminalValue) const override;
    bool gives_check(Action action) const override;
    void print(std::ostream& os) const override;
    FairyState* clone() const override;
    unsigned int number_repetitions() const override;
    string action_to_san(Action action, const std::vector<Action> &legalActions, bool leadsToWin, bool bookMove) const override;
    Tablebase::WDLScore check_for_tablebase_wdl(Tablebase::ProbeState &result) override;
    void set_auxiliary_outputs(const float* auxiliaryOutputs) override;
    void init(int variant, bool isChess960);
};

#endif // FAIRYSTATE_H
