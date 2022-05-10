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
 * @file: searchsettings.cpp
 * Created on 12.06.2019
 * @author: queensgambit
 */

#include "searchsettings.h"

SearchSettings::SearchSettings():
        threads(2),
        batchSize(8),
        dirichletEpsilon(0.25f),
        dirichletAlpha(0.2f),
        nodePolicyTemperature(1.0f),
        qValueWeight(1.0f),
        qVetoDelta(0.4f),
        virtualLoss(1.0f),
        verbose(true),
        epsilonChecksCounter(100),
        useMCGS(true),
        cpuctInit(2.5f),
        cpuctBase(19652.0f),
        uInit(1.0f),
        uMin(0.25f),
        uBase(1965.0f),
        randomMoveFactor(0.0f),
        allowEarlyStopping(false),
        useNPSTimemanager(false),
        useTablebase(false),
        epsilonGreedyCounter(20),
        reuseTree(true),
        mctsSolver(false)
{

}
