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
 * @file: openspielstate.h
 * Created on 27.12.2020
 * @author: queensgambit
 *
 * Wrapper for the OpenSpiel environment (https://github.com/deepmind/open_spiel/tree/master/open_spiel/games)
 */

#ifndef OPENSPIELSTATE_H
#define OPENSPIELSTATE_H

#include "state.h"
#include "open_spiel/spiel.h"

struct open_spiel::GameType gameType;

class StateConstantsOpenSpiel : public StateConstantsInterface<StateConstantsOpenSpiel>
{
public:
    static int BOARD_WIDTH() {
        return 0;  // TODO
    }
    static int BOARD_HEIGHT() {
        return 0;  // TODO
    }
    static int NB_CHANNELS_TOTAL() {
        return 0;  // TODO
    }
    static int NB_LABELS() {
        return 0;  // TODO
    }
    static int NB_LABELS_POLICY_MAP() {
        return 0;  // TODO
    }
    static int NB_PLAYERS() {
        assert(gameType.max_num_player == gameType.min_num_players);
        return gameType.max_num_players;
    }
    static std::string action_to_uci(Action action, bool is960) {
        // TODO
    }
    template<PolicyType p = normal, MirrorType m = notMirrored>
    static MoveIdx action_to_index(Action action) {
        return 0;  // TODO
    }
    static void init(bool isPolicyMap) {
        return; // pass
    }
};

class OpenSpielState : public State
{
private:
    open_spiel::Game* spielGame;
    open_spiel::State* spielState;
public:
    OpenSpielState();

    // State interface
public:
    std::vector<Action> legal_actions() const;
    void set(const std::string &fenStr, bool isChess960, int variant);
    void get_state_planes(bool normalize, float *inputPlanes) const;
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
    TerminalType is_terminal(size_t numberLegalMoves, bool inCheck, float &customTerminalValue) const;
    Result check_result(bool inCheck) const;
    bool gives_check(Action action) const;
    void print(std::ostream &os) const;
    Tablebase::WDLScore check_for_tablebase_wdl(Tablebase::ProbeState &result);
    State *clone() const;
};

#endif // OPENSPIELSTATE_H
