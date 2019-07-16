/*
 * CrazyAra, a deep learning chess variant engine
 * Copyright (C) 2018 Johannes Czech, Moritz Willig, Alena Beyer
 * Copyright (C) 2019 Johannes Czech
 *
 * CrazyAra is free software: You can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * @file: searchsettings.h
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * Struct which stores all relevant search settings for the Search-Agents
 */

#ifndef SEARCHSETTINGS_H
#define SEARCHSETTINGS_H


struct SearchSettings
{
    int threads;
    unsigned int batchSize;
    float cpuct;
    float dirichletEpsilon;
    float dirichletAlpha;
    float qValueWeight;
    float virtualLoss;
    bool verbose;
    bool enhanceChecks;
    bool enhanceCaptures;
    bool useFutureQValues;
    bool usePruning;
    float uInitDivisor;

    SearchSettings(): threads(2),
                 batchSize(2),
                 cpuct(1),
                 dirichletEpsilon(0.25),
                 dirichletAlpha(0.2),
                 qValueWeight(0.0),
                 virtualLoss(3),
                 verbose(true),
                 enhanceChecks(true),
                 enhanceCaptures(true),
                 useFutureQValues(true),
                 usePruning(false),
                 uInitDivisor(1.0) {}
};

#endif // SEARCHSETTINGS_H
