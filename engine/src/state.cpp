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
 * @file: state.cpp
 * Created on 29.01.2021
 * @author: queensgambit
 */

#include "state.h"
#include "constants.h"

TerminalType invert_terminal_type(TerminalType terminalType) {
    switch (terminalType) {
    case TERMINAL_WIN:
        return TERMINAL_LOSS;
    case TERMINAL_LOSS:
        return TERMINAL_WIN;
    default: ;
    }
    return terminalType;
}

Result State::check_result() const
{
    float customTerminalValue;
    TerminalType terminalType = this->is_terminal(legal_actions().size(), customTerminalValue);
    switch(terminalType) {
    case TERMINAL_NONE:
        return NO_RESULT;
    case TERMINAL_DRAW:
        return DRAWN;
    case TERMINAL_WIN:
        return side_to_move() == FIRST_PLAYER_IDX ? WHITE_WIN : BLACK_WIN;
    case TERMINAL_LOSS:
        return side_to_move() == FIRST_PLAYER_IDX ? BLACK_WIN : WHITE_WIN;
    case TERMINAL_CUSTOM:
        if (customTerminalValue > 0.0) {
            return side_to_move() == FIRST_PLAYER_IDX ? WHITE_WIN : BLACK_WIN;
        }
        if (customTerminalValue < 0.0) {
            return side_to_move() == FIRST_PLAYER_IDX ? BLACK_WIN : WHITE_WIN;
        }
        return DRAWN;
    }
    return NO_RESULT;
}

TerminalType State::random_rollout(float& customValueTerminal)
{
    int sideToMove = this->steps_from_null() % 2;
    while(true) {
        const std::vector<Action> actions = this->legal_actions();
        const size_t numberActions = actions.size();
        TerminalType terminalType = this->is_terminal(numberActions, customValueTerminal);
        if (terminalType != TERMINAL_NONE) {
            if (this->steps_from_null() % 2 == sideToMove) {
                return terminalType;
            }
            return invert_terminal_type(terminalType);
        }
        const size_t actionIdx = rand() % numberActions;
        const Action action = actions[actionIdx];
        this->do_action(action);
    }
    return TERMINAL_NONE;
}

float State::random_rollout()
{
    float customEval;
    TerminalType terminalType = this->random_rollout(customEval);
    switch (terminalType) {
    case TERMINAL_WIN:
        return WIN_VALUE;
        break;
    case TERMINAL_DRAW:
        return DRAW_VALUE;
        break;
    case TERMINAL_LOSS:
        return LOSS_VALUE;
        break;
    default: ; // TERMINAL_CUSTOM
    }
    return customEval;
}

bool State::mirror_policy(SideToMove sideToMove) const
{
    return sideToMove != FIRST_PLAYER_IDX;
}

bool is_win(Result res)
{
    switch (res) {
    case WHITE_WIN:
    case BLACK_WIN:
        return true;
    default:
        return false;
    }
}
