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
 * @file: reusablebarrier.h
 * Created on 09.12.2025
 * @author: queensgambit
 *
 * The reusable barrier is a helper class as a replacement for the barrier class available in C++ 2020.
 */


#include "reusablebarrier.h"

ReusableBarrier::ReusableBarrier(std::size_t count)
    : threshold(count), count(count), generation(0)
{

}

void ReusableBarrier::arrive_and_wait()
{
    std::unique_lock<std::mutex> lock(mtx);
    auto gen = generation;

    if (--count == 0) {
        generation++;
        count = threshold;
        cv.notify_all();
    } else {
        cv.wait(lock, [this, gen] { return gen != generation; });
    }
}
