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
 * @file: blazeutil.h
 * Created on 17.06.2019
 * @author: queensgambit
 *
 * Addition of missing functionality to the blaze library
 */

#ifndef BLAZEUTIL_H
#define BLAZEUTIL_H

#include <cfloat>
#include <blaze/Math.h>
#include<climits>
#include "randomgen.h"

using namespace std;
using blaze::StaticVector;
using blaze::DynamicVector;

/**
 * @brief append Appends a single element to the vector. If necessary new memory will be allocated.
 * @param vec Given vector
 * @param value New value which will be appended
 */
template <typename T>
void append(DynamicVector<T>& vec, T value)
{
    vec.extend(1);
    vec[vec.size()-1] = value;
}

/**
 * @brief pick_move_idx Picks an index according to the probability distribution
 * @param distribution Probability distribution which should sum to 1
 * @return Random picked element index
 */
template <typename T>
size_t random_choice(const DynamicVector<T>& distribution)
{
    const T* prob = distribution.data();
    discrete_distribution<> d(prob, prob+distribution.size());
    return size_t(d(generator));
}

/**
 * @brief apply_temperature Applies temperature rescaling to the a given distribution by enhancing higher probability values.
 * A temperature below 0.01 relates to one hot encoding. For values greater 1 the distribution is being flattened.
 * @param distribution Arbitrary distribution
 */
template <typename T, typename U>
void apply_temperature(DynamicVector<T>& distribution, U temperature)
{
    if (temperature == 1) {
        return;
    }
    // apply exponential scaling
    distribution = pow(distribution, 1.0f / temperature);
    // re-normalize the values to probabilities again
    distribution /= sum(distribution);
}

/**
 * @brief sharpen_distribution Sets all entries blow a given threshold to 0 and renormalizes afterwards
 * @param distribution Distribution that sums to 1
 * @param thresh Threshold which is substracted
 */
template <typename T, typename U>
void sharpen_distribution(DynamicVector<T>& distribution, U thresh) {
    if (max(distribution) < thresh) {
        return;
    }
    for (auto it = distribution.begin(); it != distribution.end(); ++it) {
        if (*it < thresh) {
            *it = 0;
        }
    }
    distribution /= sum(distribution);
}

/**
 * @brief get_dirichlet_noise Returns a vector of size length of generated dirichlet noise with value alpha
 * @param length Lenght of the vector
 * @param alpha Alpha value for the distribution
 * @return Dirchlet noise vector
 */
template <typename T>
DynamicVector<T> get_dirichlet_noise(size_t length, T alpha)
{
    DynamicVector<T> dirichletNoise(length);

    for (size_t i = 0; i < length; ++i) {
        std::gamma_distribution<T> distribution(alpha, 1.0f);
        dirichletNoise[i] = distribution(generator);
    }
    dirichletNoise /= sum(dirichletNoise);
    return  dirichletNoise;
}


/**
 * @brief based on sort_indexes() by Lukasz Wiklendt https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
 */
template <typename T>
vector<size_t> argsort(const DynamicVector<T>& v)
{
    // initialize original index locations
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}


/**
 * @brief first_and_second_max Finds the first and second max entry with their corresponding index
 * @param v Vector
 * @param endIdx End index of the vector (endIdx-1 will be the last index in the loop)
 * @param firstMax Return value for max element
 * @param secondMax Return value for 2nd max element
 * @param firstArg Index for max element
 * @param secondArg Index for 2nd max element
 */
template <typename T, typename U>
void first_and_second_max(const DynamicVector<T>& v, U endIdx, T& firstMax, T& secondMax, U& firstArg, U& secondArg)
{
    firstMax = v[0];
    secondMax = -INT_MAX;
    firstArg = 0;
    secondArg = 0;
    for (size_t idx = 1; idx < endIdx; ++idx) {
        if (v[idx] > firstMax) {
            // swap with second best result
            secondMax = firstMax;
            secondArg = firstArg;

            // update first best result
            firstMax = v[idx];
            firstArg = idx;
        }
        else if (v[idx] > secondMax) {
            // update first best result
            secondMax = v[idx];
            secondArg = idx;
        }
    }

}

/**
 * @brief get_quantile Returns the value+FLT_EPSILON for the given quantil.
 * @param vec Given vector which is assumed to have only positive values and to sum up to 1.
 * @param quantile Quantil value must be in [0,1]
 * @return Value of the former element as soon as the quantile has been reached.
 * If the lowest entry already is higher than the given quantil, zero is returned.
 */
template <typename T>
T get_quantile(const DynamicVector<T>& vec_input, float quantile) {

    // quantil must be in [0,1]
    assert(quantile >= 0.0 && quantile <= 1.0);

    DynamicVector<T> vec = vec_input;
    // sort the given vector
    std::sort(vec.begin(), vec.end());
    float sum = 0;

    // fast return if first value is already greater than the given quantile
    if (vec[0] >= quantile) {
        return 0;
    }

    for (size_t idx = 1; idx < vec.size(); ++idx) {
        sum += vec[idx];
        if (sum >= quantile) {
            // return the previous value and add a small floating epsilon
            return (vec[idx-1] + FLT_EPSILON);
        }
    }

    assert(false);
    // this should never be reached
    return -1.f;
}

// https://stackoverflow.com/questions/17074324/how-can-i-sort-two-vectors-in-the-same-way-with-criteria-that-uses-only-one-of#
// Timothy Shields
template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(
    const std::vector<T>& vec,
    Compare compare)
{
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
        [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    return p;
}

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(const DynamicVector<T>& vec, Compare compare)
{
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
        [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    return p;
}

template <typename T>
void apply_permutation_in_place(DynamicVector<T>& vec, const std::vector<std::size_t>& p)
{
    std::vector<bool> done(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (done[i]) {
            continue;
        }
        done[i] = true;
        std::size_t prev_j = i;
        std::size_t j = p[i];
        while (i != j) {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

template <typename T>
std::vector<T> apply_permutation(const std::vector<T>& vec, const std::vector<std::size_t>& p)
{
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
        [&](std::size_t i){ return vec[i]; });
    return sorted_vec;
}

template <typename T>
void apply_permutation_in_place(std::vector<T>& vec, const std::vector<std::size_t>& p)
{
    std::vector<bool> done(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i)
    {
        if (done[i])
        {
            continue;
        }
        done[i] = true;
        std::size_t prev_j = i;
        std::size_t j = p[i];
        while (i != j)
        {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

/**
 * @brief fill_missing_values Resizes a given vector to a target length and fills missing values starting from startIdx with fillValue.
 * @param vec Vector to be adjusted
 * @param startIdx Starting index from which the value will be altered
 * @param targetLength New vector length after the operation
 * @param fillValue Filling value for new entries
 */
template <typename T>
void fill_missing_values(DynamicVector<T>& vec, size_t startIdx, size_t targetLength, T fillValue) {
    vec.resize(targetLength);
    for (size_t idx = startIdx; idx < targetLength; ++idx) {
        vec[idx] = fillValue;
    }
}


#endif // BLAZEUTIL_H
