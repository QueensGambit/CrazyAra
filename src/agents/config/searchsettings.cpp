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
 * @file: searchsettings.cpp
 * Created on 12.06.2019
 * @author: queensgambit
 */

#include "searchsettings.h"

SearchSettings::SearchSettings(OptionsMap &o):
    verbose(true)
{
    threads = o["Threads"];
    batchSize = o["Batch_Size"];
    useTranspositionTable = o["Use_Transposition_Table"];
    uInit = float(o["Centi_U_Init_Divisor"]) / 100.0f;
    uMin = o["Centi_U_Min"] / 100.0f;
    uBase = o["U_Base"];
    qValueWeight = o["Centi_Q_Value_Weight"] / 100.0f;
    enhanceChecks = o["Enhance_Checks"];
    enhanceCaptures = o["Enhance_Captures"];
    cpuctInit = o["Centi_CPuct_Init"] / 100.0f;
    cpuctBase = o["CPuct_Base"];
    dirichletEpsilon = o["Centi_Dirichlet_Epsilon"] / 100.0f;
    dirichletEpsilon = o["Centi_Dirichlet_Alpha"] / 100.0f;
    virtualLoss = o["Virtual_Loss"];
    qThreshInit = o["Centi_Q_Thresh_Init"] / 100.0f;
    qThreshMax = o["Centi_Q_Thresh_Max"] / 100.0f;
    qThreshBase = o["Q_Thresh_Base"];
}
