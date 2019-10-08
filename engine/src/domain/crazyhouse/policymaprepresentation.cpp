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
 * @file: policymaprepresentation.cpp
 * Created on 09.07.2019
 * @author: queensgambit
 *
 * Functionality for representing all possible moves in the policy feature maps.
 * Note, most of the entries in the policy feature map are unusable because the represent illegal moves
 * which would go beyond the board.
 * Most of the functions are 100% optimal in terms of performance, but there are only used to create a static look-up
 * table for retrieving the move probability from the policy feature maps, so this doesn't play a factor.
 *
 * The code is based on the python version:
 * CrazyAra/blob/master/DeepCrazyhouse/src/domain/crazyhouse/plane_policy_representation.py
 */

#include "policymaprepresentation.h"
