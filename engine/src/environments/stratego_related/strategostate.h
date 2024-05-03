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
 * @file: strategostate.h
 * Created on 05.2021
 * @author: BluemlJ
 *
 * A more specialized OpenSpiel wrapper for the Stratego Implementation which is based
 * on OpenSpiel. 
 */

#ifndef STRATEGOSTATE_H
#define STRATEGOSTATE_H

#include "state.h"
#include "open_spiel/spiel.h"
#include "open_spiel/games/yorktown.h"

class StateConstantsStratego : public StateConstantsInterface<StateConstantsStratego>
{
public:
    static uint BOARD_WIDTH() {
        return open_spiel::yorktown::BoardSize();
    }
    static uint BOARD_HEIGHT() {
        return  open_spiel::yorktown::BoardSize();
    }
    static uint NB_CHANNELS_TOTAL() {
        return open_spiel::yorktown::InformationStateTensorShape().front(); 
    }
     static uint NB_VALUES_TOTAL() {
        return NB_CHANNELS_TOTAL()*open_spiel::yorktown::BoardSize()*open_spiel::yorktown::BoardSize();  // TODO
    }
    static uint NB_LABELS() {
        return open_spiel::yorktown::kNumActionDestinations*open_spiel::yorktown::BoardSize()*open_spiel::yorktown::BoardSize();  // TODO
    }
    static uint NB_LABELS_POLICY_MAP() {
        return NB_LABELS();  // TODO
    }
    static uint NB_PLAYERS() {
        return  open_spiel::yorktown::NumPlayers();
    }
     static uint NB_AUXILIARY_OUTPUTS() {
        return 0;
    }
    static std::string action_to_uci(Action action, bool is960) {
        // TODO use actual uci for this
        return std::to_string(action);
    }
    template<PolicyType p = normal, MirrorType m = notMirrored>
    static MoveIdx action_to_index(Action action) {
        return action;  // TODO
    }
    static void init(bool isPolicyMap, bool is960) {
        return; // pass
    }
    static std::vector<std::string> available_variants() {
        return {"stratego"};
    }
    static std::string start_fen(int variant) {
        switch (variant) {
        default:
            return "MBCaaaaaaaKaaaaaaaaaaaaaaDaaaaaaEaDaaaLaaa__aa__aaaa__aa__aaPaaaWNaOXaQPaaaYaaaaaaaaaaaaaaaaaaaaaaaa r 0";
        }
    }
};

class StrategoState : public State
{
private:
    std::shared_ptr<const open_spiel::Game> spielGame;
    std::unique_ptr<open_spiel::State> spielState;
public:
    StrategoState();
    StrategoState(const StrategoState& strategostate);

    // State interface
public:
    std::vector<Action> legal_actions() const;
    void set(const std::string &fenStr, bool isChess960, int variant);
    void get_state_planes(bool normalize, float *inputPlanes,  Version version) const;
    unsigned int steps_from_null() const;
    bool is_chess960() const;
    std::string fen() const;
    void do_action(Action action);
    void undo_action(Action action);
    void prepare_action();
    unsigned int number_repetitions() const;
    int side_to_move() const;
    Key hash_key() const;
    void flip();
    Action uci_to_action(std::string &uciStr) const;
    std::string action_to_san(Action action, const std::vector<Action> &legalActions, bool leadsToWin, bool bookMove) const;
    std::string action_to_string(Action action) const;
    TerminalType is_terminal(size_t numberLegalMoves, float &customTerminalValue) const;
    bool gives_check(Action action) const;
    void print(std::ostream &os) const;
    Tablebase::WDLScore check_for_tablebase_wdl(Tablebase::ProbeState &result);
    void set_auxiliary_outputs(const float* auxiliaryOutputs);
    StrategoState *clone() const;
    StrategoState *openBoard() const;
    void init(int variant, bool isChess960) override;
    GamePhase get_phase(unsigned int numPhases, GamePhaseDefinition gamePhaseDefinition) const override;
};

#endif // STRATEGOSTATE_H
