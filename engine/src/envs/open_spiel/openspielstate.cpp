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

OpenSpielState::OpenSpielState()
{

}

OpenSpielState::OpenSpielState(const OpenSpielState &openSpielState)
{
    // todo implement copy constructor
}

std::vector<Action> OpenSpielState::legal_actions() const
{
    return spielState->LegalActions(spielState->CurrentPlayer());
}

void OpenSpielState::set(const std::string &fenStr, bool isChess960, int variant)
{
    // pass
//    open_spiel::DeserializeGameAndState(fenStr);
}

void OpenSpielState::get_state_planes(bool normalize, float *inputPlanes) const
{

}

unsigned int OpenSpielState::steps_from_null() const
{
    return spielState->MoveNumber();  // note: MoveNumber != PlyCount
}

bool OpenSpielState::is_chess960() const
{
    return false;
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

unsigned int OpenSpielState::number_repetitions() const
{

}

int OpenSpielState::side_to_move() const
{
    return spielState->CurrentPlayer();
}

Key OpenSpielState::hash_key() const
{
    return 0;
}

void OpenSpielState::flip()
{
    std::cerr << "flip() is unavailable" << std::endl;
}

Action OpenSpielState::uci_to_action(std::string &uciStr) const
{
    return spielState->StringToAction(uciStr);
}

std::string OpenSpielState::action_to_san(Action action, const std::vector<Action> &legalActions, bool leadsToWin, bool bookMove) const
{

}

TerminalType OpenSpielState::is_terminal(size_t numberLegalMoves, bool inCheck, float &customTerminalValue) const
{
    if (spielState->IsTerminal()) {
        const double reward = spielState->PlayerReward(spielState->CurrentPlayer());
        if (reward == spielGame->MaxUtility()) {
            return  TERMINAL_WIN;
            if (reward == spielGame->MinUtility() + spielGame->MaxUtility()) {
                return TERMINAL_DRAW;
            }
            if (reward == spielGame->MinUtility()) {
                return TERMINAL_LOSS;
            }
            customTerminalValue = reward;
            return TERMINAL_CUSTOM;
        }
    }
    return TERMINAL_NONE;
}

Result OpenSpielState::check_result(bool inCheck) const
{
    return NO_RESULT;
}

bool OpenSpielState::gives_check(Action action) const
{
    std::cerr << "gives_check() is unavailable" << std::endl;
}

void OpenSpielState::print(std::ostream &os) const
{
    os << spielState->ToString();
}

Tablebase::WDLScore OpenSpielState::check_for_tablebase_wdl(Tablebase::ProbeState &result)
{
    return Tablebase::WDLScoreNone;
}

State *OpenSpielState::clone() const
{
    return new OpenSpielState(*this);
}
