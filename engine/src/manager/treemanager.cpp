/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018  Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019  Johannes Czech

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
 * @file: treemanager.cpp
 * Created on 24.07.2019
 * @author: queensgambit
 */

#include "treemanager.h"
#include "misc.h"
#include "../node.h"

Node* pick_next_node(Move move, const Node* parentNode)
{
    if (parentNode != nullptr) {
        for (Node* node : parentNode->get_child_nodes()) {
            if (node->get_move() == move && node->is_expanded()) {
                return node;
            }
        }
    }
    return nullptr;
}

bool same_hash_key(Node* node, Board *pos)
{
    return node != nullptr && node->hash_key() == pos->hash_key();
}
