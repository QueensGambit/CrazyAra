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
 * @file: inferenceworker.cpp
 * Created on 10.12.2025
 * @author: queensgambit
 */

#include "inferenceworker.h"
#include <algorithm>
#include <cstring>
#include <iostream>

// Include your neural net API header (TensorRT wrapper)
// #include "neuralnetapiuser.h"

InferenceWorker::InferenceWorker(std::shared_ptr<InferenceQueue> queue,
                                 std::shared_ptr<NeuralNetAPIUser> nnUser,
                                 size_t maxBatchSize,
                                 size_t policySize,
                                 size_t auxSize,
                                 std::chrono::milliseconds gatherTimeout) :
    queue(std::move(queue)), nnUser(std::move(nnUser)), maxBatchSize(maxBatchSize),
    policySize(policySize), auxSize(auxSize), gatherTimeout(gatherTimeout) {}

InferenceWorker::~InferenceWorker() {
    stop();
}

void InferenceWorker::start() {
    if (running.exchange(true)) {
        return;
    }
    workerThread = std::thread(&InferenceWorker::run, this);
}

void InferenceWorker::stop() {
    if (!running.exchange(false)) {
        return;
    }
    if (queue) {
        queue->terminate();
    }
    if (workerThread.joinable()) {
        workerThread.join();
    }
}

void InferenceWorker::run() {
    std::vector<InferenceRequest> batches;
    batches.reserve(maxBatchSize);

    while (running) {
        batches.clear();

        InferenceRequest firstReq;
        // block until we get the first job or termination
        if (!queue->pop_blocking(firstReq)) {
            break; // terminated
        }
        batches.push_back(std::move(firstReq));

        // Gather more requests up to maxBatchSize with short non-blocking loop
        InferenceRequest req;
        while (batches.size() < maxBatchSize) {
            // try immediate pop first
            if (queue->try_pop(req)) {
                batches.push_back(std::move(req));
                continue;
            }
            // otherwise wait up to gatherTimeout to accumulate more
            if (queue->pop_with_timeout(req, gatherTimeout)) {
                batches.push_back(std::move(req));
                continue;
            }
            break; // no more requests within timeout
        }

        // Determine per-request inputSize consistency
        size_t inputSize = batches[0].inputSize;
        bool allSame = true;
        for (size_t i = 1; i < batches.size(); ++i) {
            if (batches[i].inputSize != inputSize) {
                allSame = false;
                break;
            }
        }
        if (!allSame) {
            std::cerr << "[InferenceWorker] warning: varying input sizes in batch; handling individually\n";
        }

        // Build contiguous host input buffer: batch_count * inputSize
        size_t writeIndex = 0;
        for (InferenceRequest& request : batches) {
            for (size_t i = 0; i < request.batchCount; i++) {
                memcpy(nnUser->inputPlanes + (writeIndex + i) * request.inputSize,
                       request.inputData.data() + i * request.inputSize,
                       request.inputSize * sizeof(float));
            }
            request.outputOffset = writeIndex;
            writeIndex += request.batchCount;
        }

        // --- Perform NN inference on hostInput ---
        // TODO: select_nn_index()
        nnUser->nets[0]->predict(nnUser->inputPlanes, nnUser->valueOutputs, nnUser->probOutputs, nnUser->auxiliaryOutputs);

        // -- deliver results to each requester via promise --
        for (InferenceRequest& request : batches) {
            InferenceResult res;
            res.valueOutputs.resize(request.batchCount);
            res.probOutputs.resize(request.batchCount * policySize);
            if (auxSize) {
                res.auxiliaryOutputs.resize(request.batchCount * auxSize);
            }


            for (size_t i = 0; i < request.batchCount; i++) {
                size_t gi = request.outputOffset + i;
                res.valueOutputs[i] = nnUser->valueOutputs[gi];
                memcpy(&res.probOutputs[i * policySize],
                       &nnUser->probOutputs[gi * policySize],
                       policySize * sizeof(float));
                if (auxSize) {
                    memcpy(&res.auxiliaryOutputs[i * auxSize],
                           &nnUser->auxiliaryOutputs[gi * auxSize],
                           auxSize * sizeof(float));
                }
            }

            try { request.promise.set_value(std::move(res)); } catch (...) {}
        }

    } // while running
}

