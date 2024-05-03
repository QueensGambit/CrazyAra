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
 * @file: strategostate.cpp
 * Created on 05.2021
 * @author: BluemlJ
 */

#include "strategostate.h"
#include <functional>
#include <iostream>
#include <fstream>


StrategoState::StrategoState():
    spielGame(open_spiel::LoadGame("yorktown")),
    spielState(spielGame->NewInitialState())
{
}

StrategoState::StrategoState(const StrategoState &strategoState):
    spielGame(strategoState.spielGame->shared_from_this()),
    spielState(strategoState.spielState->Clone())
{
}


std::vector<Action> StrategoState::legal_actions() const
{
    return spielState->LegalActions(spielState->CurrentPlayer());
}

void StrategoState::set(const std::string &fenStr, bool isChess960, int variant)
{
    spielState = spielGame->NewInitialState(fenStr);
}

void StrategoState::get_state_planes(bool normalize, float *inputPlanes, Version version) const
{
    std::vector<float> v(spielGame->InformationStateTensorSize());
    spielState->InformationStateTensor(spielState->CurrentPlayer(), absl::MakeSpan(v));
    std::copy( v.begin(), v.end(), inputPlanes);
}

unsigned int StrategoState::steps_from_null() const
{
    return spielState->MoveNumber();  // note: MoveNumber != PlyCount
}

bool StrategoState::is_chess960() const
{
    return false;
}

std::string StrategoState::fen() const
{
    return spielState->ToString();
}

void StrategoState::do_action(Action action)
{
    spielState->ApplyAction(action);
    spielState->ToString();
}

void StrategoState::undo_action(Action action)
{
    spielState->UndoAction(!spielState->CurrentPlayer(), action); // note: this formulation assumes a two player, non-simultaneaous game
}

void StrategoState::prepare_action()
{
    // pass
}

unsigned int StrategoState::number_repetitions() const
{
    // TODO
    return 0;
}

int StrategoState::side_to_move() const
{
    return spielState->MoveNumber() % 2;
}

Key StrategoState::hash_key() const
{
    std::hash<std::string> hash_string;
    return hash_string(this->fen());
}

void StrategoState::flip()
{
    std::cerr << "flip() is unavailable" << std::endl;
}

Action StrategoState::uci_to_action(std::string &uciStr) const
{
    return spielState->StringToAction(uciStr);
}

std::string StrategoState::action_to_san(Action action, const std::vector<Action> &legalActions, bool leadsToWin, bool bookMove) const
{
    // current use UCI move as replacement
    return spielState->ActionToString(action);
}

std::string StrategoState::action_to_string(Action action) const
{
    // current use UCI move as replacement
    return spielState->ActionToString(action);
}
TerminalType StrategoState::is_terminal(size_t numberLegalMoves, float &customTerminalValue) const
{
    if (spielState->IsTerminal()) {
        const double currentReturn = spielState->Returns()[spielState->CurrentPlayer()];
        std::cout << "MNNNN   " << spielState->CurrentPlayer() << std::endl;
        if (currentReturn == spielGame->MaxUtility()) {
            return  TERMINAL_WIN;
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



bool StrategoState::gives_check(Action action) const
{
    // gives_check() is unavailable
    return false;
}

void StrategoState::print(std::ostream &os) const
{
    os << spielState->ToString();
}

Tablebase::WDLScore StrategoState::check_for_tablebase_wdl(Tablebase::ProbeState &result)
{
    return Tablebase::WDLScoreNone;
}

void StrategoState::set_auxiliary_outputs(const float* auxiliaryOutputs)
{
    // do nothing
}

StrategoState* StrategoState::clone() const
{
    // carefull clone will be init a random perfect information state
   return new StrategoState(*this);
}

StrategoState* StrategoState::openBoard() const
{
    // openBoard so get a corret information state which is not randomly sampled

    auto fen = this->fen();
    std::for_each(fen.begin(), fen.end(), [](char & c){
            c = ::tolower(c);
    });

    auto new_state = new StrategoState(*this);
    new_state->set(fen, false, 0);
    return new_state;
}

void StrategoState::init(int variant, bool isChess960) {
    spielState = spielGame->NewInitialState();
  
    std::string line;
    std::vector<std::string> lines;

    std::ifstream file ("positions.txt");
    if (file.is_open())
    {
        std::string str;
       
        while (std::getline(file, str))
        {
          lines.push_back(str);
        }
        file.close();

        std::string fen = lines[rand() % lines.size()];  
        fen.erase(fen.length()-1);

        spielState = spielGame->NewInitialState(fen);

    }
    else{
        //std::cout << "Unable to open position file"; 
        spielState = spielGame->NewInitialState();
    }
}

GamePhase StrategoState::get_phase(unsigned int numPhases, GamePhaseDefinition gamePhaseDefinition) const
{
    // TODO: Implement phase definition here
    return GamePhase(0);
}
