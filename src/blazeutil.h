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
 * @file: blazeutil.h
 * Created on 17.06.2019
 * @author: queensgambit
 *
 * Addition of missing functionality to the blaze library
 */

#ifndef BLAZEUTIL_H
#define BLAZEUTIL_H

#include <blaze/Math.h>

using blaze::StaticVector;
using blaze::DynamicVector;

// random generator used for dirchlet noise
static std::random_device r;
static std::default_random_engine generator(r());

/**
 * @brief get_dirichlet_noise Returns a vector of size length of generated dirichlet noise with value alpha
 * @param length Lenght of the vector
 * @param alpha Alpha value for the distribution
 * @return Dirchlet noise vector
 */
DynamicVector<float> get_dirichlet_noise(size_t length, const float alpha);


#endif // BLAZEUTIL_H
