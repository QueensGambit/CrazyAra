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
 * @file: treemanager.h
 * Created on 24.07.2019
 * @author: queensgambit
 *
 * Contains all utility methods regarding sarch tree management
 */

#ifndef TREEMANAGER_H
#define TREEMANAGER_H

//#include "../board.h"
#include "../stateobj.h"
#include "../node.h"

/**
 * @brief pick_next_node Return the next node when doing the given move for the parent node
 * @param move Move
 * @param ownMove Boolean indicating if it was CrazyAra's move
 */
shared_ptr<Node> pick_next_node(Action move, const Node* parentNode);

/**
 * @brief same_hash_key Checks if the given node isn't a nullptr and
 *  shares the same hash key and plies from null as the position
 * @param node Node pointer
 * @param pos Position pointer
 * @return bool
 */
bool same_hash_key(Node* node, StateObj* state);

#endif // TREEMANAGER_H
