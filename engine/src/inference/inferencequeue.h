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
 * @file: inferencequeue.h
 * Created on 10.12.2025
 * @author: queensgambit
 *
 * Provides the queue functionality as well as result and request for inference.
 */

#ifndef INFERENCEQUEUE_H
#define INFERENCEQUEUE_H

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <vector>

// A minimal InferenceRequest and InferenceResult used by MCTS threads -> worker
struct InferenceResult {
    std::vector<float> valueOutputs; // size = 1 (scalar) or batch local
    std::vector<float> probOutputs; // flattened policy vector
    std::vector<float> auxiliaryOutputs; // optional
};

struct InferenceRequest {
    // The inputPlanes must point to a caller-owned contiguous float buffer of size inputSize
    // The worker will copy from this pointer into the NN global input buffer.
    std::vector<float> inputData;
    size_t inputSize; // number of floats at inputPlanes
    size_t batchCount;  // number of local batches

    // metadata (optional): e.g. agent id, node pointer, etc.
    size_t agentID;
    size_t outputOffset;

    // result future: worker will set an InferenceResult
    std::promise<InferenceResult> promise;
};


class InferenceQueue {
public:
    InferenceQueue() = default;

    /**
     * @brief push push a job; mcts thread moves the request
     * @param req Inference request
     */
    void push(InferenceRequest&& req);

    /**
     * @brief pop_blocking blocking pop of one request; returns false if terminated
     * @param out Inference request
     * @return true or conditionVariable.wait()
     */
    bool pop_blocking(InferenceRequest& out);

    /**
     * @brief try_pop non-blocking try-pop
     * @param out Inference request
     * @return
     */
    bool try_pop(InferenceRequest& out);

    /**
     * @brief pop_with_timeout wait with timeout (milliseconds)
     * @param out Inference request
     * @param timeout true or conditionVariable.wait_for()
     * @return
     */
    template<typename Rep, typename Period>
    bool pop_with_timeout(InferenceRequest& out, std::chrono::duration<Rep, Period> timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        if (!conditionVariable.wait_for(lock, timeout, [&]{ return !queue.empty() || terminated; })) {
            return false; // timeout
        }
        if (terminated && queue.empty()) {
            return false;
        }
        out = std::move(queue.front());
        queue.pop_front();
        return true;
    }

    /**
     * @brief terminate Terminates the queue
     */
    void terminate();

    /**
     * @brief empty Checks if the queue is empty
     * @return
     */
    bool empty() const;

private:
    mutable std::mutex mutex;
    std::condition_variable conditionVariable;
    std::deque<InferenceRequest> queue;
    bool terminated = false;
};

#endif // INFERENCEQUEUE_H
