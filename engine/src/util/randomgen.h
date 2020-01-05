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
 * @file: randomgen.h
 * Created on 26.11.2019
 * @author: queensgambit
 *
 * Utility methods which all random number generation from several distribution types
 */

#ifndef RANDOMGEN_H
#define RANDOMGEN_H

#include <random>

// random generator used for all sort of distributions
static std::random_device r;
static std::default_random_engine generator(r());

/**
 * @brief random_exponential Generates a random sample from a exponential distribution with a given mean.
 * @param lambda Lambda value of the distribution
 * @return Generated value
 */
template<typename T>
T random_exponential(T lambda) {
    std::exponential_distribution<T> distribution(lambda);
    return distribution(generator);
}

#endif // RANDOMGEN_H
