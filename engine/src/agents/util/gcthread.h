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
 * @file: gcthread.h
 * Created on 13.05.2020
 * @author: queensgambit
 *
 * Definition of the garbage collector thread
 */

#ifndef GCTHREAD_H
#define GCTHREAD_H

#include <vector>
using namespace std;

/**
 * @brief The GCThread class is a garbage collector object which asynchronously frees memory
 */
template <class T> class GCThread
{
private:
    vector<T*> items;
public:
    void add_item_to_delete(T* item) {
        items.emplace_back(item);
    }

    void delete_elements() {
        for (size_t idx = 0; idx < items.size(); ++idx) {
            delete items[idx];
        }
        items.clear();
    }
};

template <typename T>
void run_gc_thread(GCThread<T> *t)
{
    t->delete_elements();
}

#endif // GCTHREAD_H
