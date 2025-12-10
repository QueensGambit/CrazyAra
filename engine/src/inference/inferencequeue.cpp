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
 * @file: inferencequeue.cpp
 * Created on 10.12.2025
 * @author: queensgambit
 */

#include "inferencequeue.h"


void InferenceQueue::push(InferenceRequest &&req) {
    std::unique_lock<std::mutex> lock(mutex);
    queue.push_back(std::move(req));
    conditionVariable.notify_one();
}

bool InferenceQueue::pop_blocking(InferenceRequest &out) {
    std::unique_lock<std::mutex> lock(mutex);
    conditionVariable.wait(lock, [&]{ return !queue.empty() || terminated; });
    if (terminated && queue.empty()) {
        return false;
    }
    out = std::move(queue.front()); queue.pop_front();
    return true;
}

bool InferenceQueue::try_pop(InferenceRequest &out) {
    std::unique_lock<std::mutex> lock(mutex);
    if (queue.empty()) {
        return false;
    }
    out = std::move(queue.front()); queue.pop_front();
    return true;
}

void InferenceQueue::terminate() {
    std::unique_lock<std::mutex> lock(mutex);
    terminated = true;
    conditionVariable.notify_all();
}

bool InferenceQueue::empty() const {
    std::unique_lock<std::mutex> lock(mutex);
    return queue.empty();
}

