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
 * @file: searchlimits.h
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#ifndef SEARCHLIMITS_H
#define SEARCHLIMITS_H


struct SearchLimits
{
public:
    int playoutsEmptyPockets;
    int playoutsFilledPockets;
    int maxSearchDepth;
    int minMovetime;

    SearchLimits():
                 playoutsEmptyPockets(256),
                 playoutsFilledPockets(512),
                 maxSearchDepth(15),
                 minMovetime(100) {}
};

#endif // SEARCHLIMITS_H
