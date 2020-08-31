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
 * @file: pommermanstate.cpp
 * Created on 15.07.2020
 * @author: queensgambit
 */

#ifdef MODE_POMMERMAN
#include "pommermanstate.h"
#include "data_representation.h"
#include "agents.hpp"


PommermanState::PommermanState():
    agentToMove(0),
    plies(0)
{
    agentActions = new bboard::Move[numberAgents];
    for (size_t idx = 0; idx < numberAgents; ++idx) {
        agentActions[idx] = bboard::Move::IDLE;
    }
    agents::RandomAgent* r = new agents::RandomAgent;
    std::array<bboard::Agent*, 4> agents = {r,r,r,r};
    env.MakeGame(agents);
//    env.StartGame(1000, false, false);
}

PommermanState::~PommermanState()
{
    delete agentActions;
}

std::vector<Action> PommermanState::legal_actions() const
{
    return {Action(bboard::Move::IDLE),
            Action(bboard::Move::UP),
            Action(bboard::Move::DOWN),
            Action(bboard::Move::LEFT),
            Action(bboard::Move::RIGHT),
            Action(bboard::Move::BOMB)};
}

void PommermanState::set(const std::string &fenStr, bool isChess960, int variant)
{
    // TODO
}

void PommermanState::get_state_planes(bool normalize, float *inputPlanes) const
{
    // TODO
    StateToPlanes(&env.GetState(), 0, inputPlanes);
}

unsigned int PommermanState::steps_from_null() const
{
    return plies;
}

bool PommermanState::is_chess960() const
{
    return false;
}

std::string PommermanState::fen() const
{
    return "null";
}

void PommermanState::do_action(Action action)
{
    agentActions[agentToMove++] = bboard::Move(action);
    if (agentToMove == numberAgents) {
        bboard::Step(&env.GetState(), agentActions);
        agentToMove = 0;
    }
}

void PommermanState::undo_action(Action action) {
    // TODO
}


unsigned int PommermanState::number_repetitions() const
{
    return 0;
}

int PommermanState::side_to_move() const
{
    return agentToMove;
}

Key PommermanState::hash_key() const
{
    return 0;
}

void PommermanState::flip()
{
    // pass
}

Action PommermanState::uci_to_action(std::string &uciStr) const
{
    // TODO
    return Action(bboard::Move::IDLE);
}

std::string PommermanState::action_to_san(Action action, const std::vector<Action>& legalActions, bool leadsToWin, bool bookMove) const
{
    return "";
}

TerminalType PommermanState::is_terminal(size_t numberLegalMoves, bool inCheck, float& customTerminalValue) const
{
    // TODO
    return TERMINAL_NONE;
}

Result PommermanState::check_result(bool inCheck) const
{
    // TODO
}


bool PommermanState::gives_check(Action action) const
{
    return false;
}

PommermanState* PommermanState::clone() const
{
    // TODO
}

void PommermanState::print(std::ostream& os) const
{
    // TODO
    os << InitialStateToString(env.GetState());
}
#endif
