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
 * @file: main.cpp
 * Created on 17.08.2019
 * @author: queensgambit
 *
 * Main entry point for the engine CrazyAra
 */

#ifdef BUILD_TESTS
#include "../tests/tests.h"
#endif
#include "stateobj.h"
#include <iostream>
#include "crazyara.h"

#ifndef BUILD_TESTS
int main(int argc, char* argv[]) {
#ifdef XIANGQI
    variants.init();
#endif
    CrazyAra crazyara;
    crazyara.init();
    crazyara.welcome();
    crazyara.uci_loop(argc, argv);
}
#endif
