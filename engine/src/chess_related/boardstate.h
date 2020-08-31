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
  GNU General Public License fåor more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: boardstate.h
 * Created on 13.07.2020
 * @author: queensgambit
 *
 * BoardState encapsulates Board and StateInfo and inherits from the abstract State class.
 */

#ifndef BOARTSTATE_H
#define BOARTSTATE_H

#ifndef MODE_POMMERMAN
#include "uci.h"
#include "../state.h"
#include "board.h"
using namespace std;

class BoardState : public State
{
private:
    Board board;
    StateListPtr states;

public:
    BoardState();
    BoardState(const BoardState& b);

    // State interface
    vector<Action> legal_actions() const override;
    void set(const string &fenStr, bool isChess960, int variant) override;
    void get_state_planes(bool normalize, float *inputPlanes) const override;
    unsigned int steps_from_null() const override;
    bool is_chess960() const override;
    string fen() const override;
    void do_action(Action action) override;
    void undo_action(Action action) override;
    unsigned int number_repetitions() const override;
    int side_to_move() const override;
    Key hash_key() const override;
    void flip() override;
    Action uci_to_action(string& uciStr) const override;
    string action_to_san(Action action, const vector<Action>& legalActions, bool leadsToWin=false, bool bookMove=false) const override;
    TerminalType is_terminal(size_t numberLegalMoves, bool inCheck, float& customTerminalValue) const override;
    Result check_result(bool inCheck) const override;
    bool gives_check(Action action) const override;
    void print(ostream& os) const override;
    BoardState* clone() const override;
};

#endif // BOARTSTATE_H
#endif
