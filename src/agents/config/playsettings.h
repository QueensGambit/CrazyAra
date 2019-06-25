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
 * @file: playsettings.h
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#ifndef PLAYSETTINGS_H
#define PLAYSETTINGS_H


struct PlaySettings
{
public:
    float temperature;
    unsigned int temperatureMoves;
    bool useTimeManagement;
    int openingGuardMoves;

    PlaySettings():
                 temperature(0.0),
                 temperatureMoves(4),
                 useTimeManagement(true),
                 openingGuardMoves(0) {}
};

#endif // PLAYSETTINGS_H
