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
 * @file: searchlimits.cpp
 * Created on 12.06.2019
 * @author: queensgambit
 */

#include "searchlimits.h"


std::ostream &operator<<(std::ostream &os, const SearchLimits &searchLimits)
{
    os << " wtime " << searchLimits.time[WHITE]
          << " btime " << searchLimits.time[BLACK]
             << " winc "  << searchLimits.inc[WHITE]
                << " binc "  << searchLimits.inc[BLACK]
                   << "movestogo " << searchLimits.movestogo;
    return os;
}

SearchLimits::SearchLimits():
    movetime(0),
    nodes(0),
    movestogo(0),
    depth(0),
    minMovetime(0),
    npmsec(0),
    startTime(0),
    moveOverhead(0),
    infinite(false),
    ponder(false)
{
    time[WHITE] = 0;
    time[BLACK] = 0;
    inc[WHITE] = 0;
    inc[BLACK] = 0;

}
