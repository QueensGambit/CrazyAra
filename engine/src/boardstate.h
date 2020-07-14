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

#include "uci.h"
#include "state.h"
#include "board.h"

class BoardState : public State
{
private:
    Board board;
    StateListPtr states;

public:
    BoardState();
    BoardState(const BoardState& b);

    // State interface
    vector<Action> legal_actions() const;
    State &set(const string &fenStr, bool isChess960, int variant);
    void get_state_planes(bool normalize, float *inputPlanes) const;
    unsigned int steps_from_null() const;
    bool is_chess960() const;
    string fen() const;
    void do_action(Action action);
    unsigned int number_repetitions() const;
    int side_to_move() const;
    Key hash_key() const;
    void flip();
    Action uci_to_action(string& uciStr) const;
    string action_to_san(Action action, const vector<Action>& legalActions) const;
    TerminalType is_terminal(size_t numberLegalMoves, bool inCheck) const;
    bool gives_check(Action action) const;
    unique_ptr<State> clone() const;
};

#endif // BOARTSTATE_H
