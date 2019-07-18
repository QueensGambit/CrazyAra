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
    float dirichletEpsilon;
    float dirichletAlpha;
    float qValueWeight;
    float virtualLoss;
    bool verbose;
    bool enhanceChecks;
    bool enhanceCaptures;
    bool useFutureQValues;
    bool usePruning;
    float cpuctInit;
    float cpuctBase;
    float uBase;
    float uInit;
    float uMin;

    SearchSettings(): threads(2),
        batchSize(2),
        dirichletEpsilon(0.25f),
        dirichletAlpha(0.2f),
        qValueWeight(0.0f),
        virtualLoss(3.0f),
        verbose(true),
        enhanceChecks(true),
        enhanceCaptures(true),
        useFutureQValues(true),
        usePruning(false),
        cpuctInit(2.5f),
        cpuctBase(19652.0f),
        uBase(1965.0f),
        uInit(1.0f),
        uMin(0.25f)
    {}
};

#endif // SEARCHSETTINGS_H
