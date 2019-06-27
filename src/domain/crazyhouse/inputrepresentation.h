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
 * @file: inputrepresentation.h
 * Created on 27.05.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#ifndef INPUTREPRESENTATION_H
#define INPUTREPRESENTATION_H

#include "../../board.h"

void board_to_planes(Board pos, int board_occ, bool normalize, float *input_planes);

#endif // INPUTREPRESENTATION_H

