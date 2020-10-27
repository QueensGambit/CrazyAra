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
 * @file: fixedvector.h
 * Created on 05.05.2020
 * @author: queensgambit
 *
 * A simple array wrapper which behaves like a fixed size vector and doesn't need to reallocate memory.
 */

#ifndef FIXEDVECTOR_H
#define FIXEDVECTOR_H

#include <cstdlib>


template <typename T>
/**
 * @brief The FixedVector class is an array with a predefined array with a fixed size.
 * It allows inserting new elements until the capacity has been reached and the reset_idx() resets
 * the array to the first element.
 */
class FixedVector
{
private:
    size_t maxCapacity;
    size_t curIdx;
    T* data;

public:
    FixedVector(size_t size):
        maxCapacity(size),
        curIdx(0)
    {
        data = new T[size];
    }

    ~FixedVector()
    {
        delete []data;
    }

    /**
     * @brief add_element Adds a new element at the current index
     * @param value Value to insert
     */
    void add_element(T value)
    {
        data[curIdx++] = value;
    }

    /**
     * @brief reset_idx Resets the input index back to the first element
     */
    void reset_idx()
    {
        curIdx = 0;
    }

    /**
     * @brief begin Returns a begin iterator
     * @return pointer
     */
    T* begin() {
        return data;
    }

    /**
     * @brief end Returns an end iterator
     * @return pointer
     */
    T* end() {
        return data + curIdx;
    }

    /**
     * @brief is_full Checks if the capacity has been reached
     * @return true if full else false
     */
    bool is_full()
    {
        return curIdx == maxCapacity;
    }

    /**
     * @brief get_element Returns the element at the given index without a bounding check
     * @param idx Elemt index to get
     * @return Value
     */
    T get_element(size_t idx) const
    {
        return data[idx];
    }

    /**
     * @brief size Returns the size of the array
     * @return size
     */
    size_t size()
    {
        return curIdx;
    }

    /**
     * @brief size Returns the capacity of the array
     * @return size_t
     */
    size_t capacity()
    {
        return maxCapacity;
    }

    /**
     * @brief current_index Returns the current index of the array
     * @return size_t
     */
    size_t current_index()
    {
        return curIdx;
    }
};

#endif // FIXEDVECTOR_H
