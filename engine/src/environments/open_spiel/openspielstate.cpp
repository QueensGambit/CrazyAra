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
#include <functional>

OpenSpielState::OpenSpielState():
    spielGame(open_spiel::LoadGame("chess()")),
//    spielGame(open_spiel::LoadGame("hex(board_size=5)")),
    spielState(spielGame->NewInitialState())
{
}

OpenSpielState::OpenSpielState(const OpenSpielState &openSpielState):
    spielGame(openSpielState.spielGame->shared_from_this()),
    spielState(openSpielState.spielState->Clone())
{
}

std::vector<Action> OpenSpielState::legal_actions() const
{
    return spielState->LegalActions(spielState->CurrentPlayer());
}

void OpenSpielState::set(const std::string &fenStr, bool isChess960, int variant)
{
    spielState = spielGame->NewInitialState(fenStr);
}

void OpenSpielState::get_state_planes(bool normalize, float *inputPlanes) const
{
    std::fill(inputPlanes, inputPlanes+StateConstantsOpenSpiel::NB_VALUES_TOTAL(), 0.0f);
    // TODO
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
    // TODO
    return 0;
}

int OpenSpielState::side_to_move() const
{
    return spielState->CurrentPlayer();
}

Key OpenSpielState::hash_key() const
{
    std::hash<std::string> hash_string;
    return hash_string(this->fen());
}

void OpenSpielState::flip()
{
    std::cerr << "flip() is unavailable" << std::endl;
}

Action OpenSpielState::uci_to_action(const std::string &uciStr) const
{
    return spielState->StringToAction(uciStr);
}

std::string OpenSpielState::action_to_san(Action action, const std::vector<Action> &legalActions, bool leadsToWin, bool bookMove) const
{
    // current use UCI move as replacement
    return spielState->ActionToString(spielState->CurrentPlayer(), action);
}

TerminalType OpenSpielState::is_terminal(size_t numberLegalMoves, bool inCheck, float &customTerminalValue) const
{
    if (spielState->IsTerminal()) {
        const double currentReturn = spielState->Returns()[spielState->MoveNumber() % 2];
        if (currentReturn == spielGame->MaxUtility()) {
            return TERMINAL_WIN;
        }
        if (currentReturn == spielGame->MinUtility() + spielGame->MaxUtility()) {
            return TERMINAL_DRAW;
        }
        if (currentReturn == spielGame->MinUtility()) {
            return TERMINAL_LOSS;
        }
        customTerminalValue = currentReturn;
        return TERMINAL_CUSTOM;
    }
    return TERMINAL_NONE;
}

Result OpenSpielState::check_result(bool inCheck) const
{
    float dummy;
    const TerminalType terminalType = is_terminal(0, 0, dummy);

    switch(terminalType) {
    case TERMINAL_WIN:
        // spielState->CurrentPlayer()) my return negative values
        // this implementation assumes a two player game with ordered turns
        switch (spielState->MoveNumber() % 2) {
        case 0:
            return WHITE_WIN;
        default:
            return BLACK_WIN;
        }
    case TERMINAL_LOSS:
        switch (spielState->MoveNumber() % 2) {
        case 0:
            return BLACK_WIN;
        default:
            return WHITE_WIN;
        }
    case TERMINAL_NONE:
        return NO_RESULT;
    default:
        return DRAWN;
    }
    return NO_RESULT;
}

bool OpenSpielState::gives_check(Action action) const
{
    // gives_check() is unavailable
    return false;
}

void OpenSpielState::print(std::ostream &os) const
{
    os << spielState->ToString();
}

Tablebase::WDLScore OpenSpielState::check_for_tablebase_wdl(Tablebase::ProbeState &result)
{
    return Tablebase::WDLScoreNone;
}

OpenSpielState* OpenSpielState::clone() const
{
    return new OpenSpielState(*this);
}
