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
 * @file: inferenceworker.h
 * Created on 10.12.2025
 * @author: queensgambit
 *
 *
 */

#ifndef INFERENCEWORKER_H
#define INFERENCEWORKER_H

#include "inferencequeue.h"
#include "nn/neuralnetapiuser.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>


class InferenceWorker {
public:
    InferenceWorker(std::shared_ptr<InferenceQueue> queue,
                    std::shared_ptr<NeuralNetAPIUser> nnUser,
                    size_t maxBatchSize,
                    size_t policySize,
                    size_t auxSize = 0,
                    std::chrono::milliseconds gatherTimeout = std::chrono::milliseconds(2));


    ~InferenceWorker();

    /**
     * @brief start start worker thread
     */
    void start();

    /**
     * @brief stop stop worker and join
     */
    void stop();


private:
    /**
     * @brief run Runs the inference worker
     */
    void run();

    std::shared_ptr<InferenceQueue> queue;
    std::shared_ptr<NeuralNetAPIUser> nnUser;
    const size_t maxBatchSize;
    const size_t policySize;
    const size_t auxSize;
    const std::chrono::milliseconds gatherTimeout;

    std::thread workerThread;
    std::atomic<bool> running{false};
};
#endif // INFERENCEWORKER_H
