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
 * @file: boardstate.h
 * Created on 20.01.2021
 * @author: queensgambit
 */

#include "openspielstate.h"

std::vector<Action> OpenSpielState::legal_actions() const
{
    return spielState->LegalActions(spielState->CurrentPlayer());
}

unsigned int OpenSpielState::steps_from_null() const
{
    return spielState->MoveNumber();  // note: MoveNumber != PlyCount
}

std::string OpenSpielState::fen() const
{
    return spielState->ToString();
}

void OpenSpielState::do_action(Action action)
{
    spielState->ApplyAction(action);
}

void OpenSpielState::undo_action(Action action)
{
    spielState->UndoAction(!spielState->CurrentPlayer(), action); // note: this formulation assumes a two player, non-simultaneaous game
}

void OpenSpielState::prepare_action()
{
    // pass
}

int OpenSpielState::side_to_move() const
{
    return spielState->CurrentPlayer();
}

void OpenSpielState::flip()
{
    std::cerr << "flip() is unavailable" << std::endl;
}

Action OpenSpielState::uci_to_action(std::string &uciStr) const
{
    return spielState->StringToAction(uciStr);
}

TerminalType OpenSpielState::is_terminal(size_t numberLegalMoves, bool inCheck, float &customTerminalValue) const
{
    spielState->IsTerminal();
    // TODO check which terminal value
}

bool OpenSpielState::gives_check(Action action) const
{
    std::cerr << "gives_check() is unavailable" << std::endl;
}

void OpenSpielState::print(std::ostream &os) const
{
    os << spielState->ToString();
}

State *OpenSpielState::clone() const
{
    return new OpenSpielState(*this);
}
