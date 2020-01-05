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
 * @file: searchsettings.h
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * Struct which stores all relevant search settings for the Search-Agents
 */

#ifndef SEARCHSETTINGS_H
#define SEARCHSETTINGS_H

#include "uci.h"

using namespace UCI;

struct SearchSettings
{
    size_t threads;
    unsigned int batchSize;
    float dirichletEpsilon;
    float dirichletAlpha;
    // policy temperature which can be applied on the every nodes' policy
    float nodePolicyTemperature;
    float qValueWeight;
    float virtualLoss;
    bool verbose;
    bool enhanceChecks;
    bool enhanceCaptures;
//    bool useFutureQValues;  currently not supported
    bool useTranspositionTable;
    float cpuctInit;
    float cpuctBase;
    float uInit;
    float uMin;
    float uBase;
    float qThreshInit;
    float qThreshMax;
    float qThreshBase;
    float randomMoveFactor;

    // adaption of checking and capture moves (currently not as UCI parameters)
    // Threshold probability for checking moves
    float threshCheck;
    // Factor based on the maximum probability with which checks will be increased
    float checkFactor;
    // Threshold probability for capture moves
    float threshCapture;
    // Factor based on the maximum probability with which captures will be increased
    float captureFactor;
    // If true, the exact given node count doesn't need to reached, but search can be stopped earlier
    bool allowEarlyStopping;

    SearchSettings();

};

#endif // SEARCHSETTINGS_H
