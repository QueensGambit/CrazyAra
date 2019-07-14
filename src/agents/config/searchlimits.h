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

#include "misc.h"

#ifndef SEARCHLIMITS_H
#define SEARCHLIMITS_H


struct SearchLimits
{
public:
    TimePoint movetime;
    size_t nodes;
    int movestogo;
    int depth;
    int minMovetime;
    int time[COLOR_NB];
    int inc[COLOR_NB];
    TimePoint npmsec;
    TimePoint startTime;
    TimePoint moveOverhead;
    bool infinite;
    bool ponder;

    SearchLimits();

};

extern std::ostream& operator<<(std::ostream& os, const SearchLimits& searchLimits);

#endif // SEARCHLIMITS_H
