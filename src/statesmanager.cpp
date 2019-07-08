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
 * @file: statesmanager.cpp
 * Created on 04.07.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#include "statesmanager.h"

StatesManager::StatesManager()
{

}

void StatesManager::swap_states()
{
    activeStates.swap(passiveStates);
}

void StatesManager::clear_states()
{
    if (passiveStates.size() > 0) {
        for (auto state: passiveStates) {
            delete state;
        }
        passiveStates.clear();
    }
}

