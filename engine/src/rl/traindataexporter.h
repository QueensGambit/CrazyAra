/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018  Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019  Johannes Czech

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
 * @file: traindataexporter.h
 * Created on 12.09.2019
 * @author: queensgambit
 *
 * Exporter class which saves the board position in planes (x) and the target values (y) for NN training
 */

#ifndef TRAINDATAEXPORTER_H
#define TRAINDATAEXPORTER_H

#ifdef USE_RL
#include <string>

#include "nlohmann/json.hpp"
#include "xtensor/xarray.hpp"
#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"
#include "z5/dataset.hxx"

#include "../domain/crazyhouse/inputrepresentation.h"
#include "../domain/crazyhouse/constants.h"

#include "../board.h"
#include "../domain/crazyhouse/constants.h"
#include "../node.h"
#include "../evalinfo.h"

class TrainDataExporter
{
private:
    size_t numberChunks;
    size_t chunkSize;
    size_t numberSamples;
    std::unique_ptr<z5::Dataset> dStartIndex;
    std::unique_ptr<z5::Dataset> dx;
    std::unique_ptr<z5::Dataset> dValue;
    std::unique_ptr<z5::Dataset> dPolicy;
    std::unique_ptr<z5::Dataset> dbestMoveQ;

    // current number of games - 1
    size_t gameIdx;
    // current sample index to insert
    size_t startIdx;

    string fileNameGameIdx;
    string fileNameStartIdx;

    /**
     * @brief export_planes Exports the board in plane representation (x)
     * @param pos Board position to export
     * @param idxOffset Batch-Offset where to save it in the matrix
     */
    void export_planes(const Board *pos, size_t idxOffset);

    /**
     * @brief export_policy Export the policy (e.g. mctsPolicy) to the matrix
     * @param legalMoves List of legal moves
     * @param policyProbSmall Probability for each move
     * @param sideToMove Current side to move
     * @param idxOffset Batch-Offset where to save it in the matrix
     */
    void export_policy(const vector<Move>& legalMoves, const DynamicVector<float>& policyProbSmall, Color sideToMove, size_t idxOffset);

    /**
     * @brief export_start_idx Writes the current starting index where to continue inserting in a .txt-file
     */
    void export_start_idx();

    /**
     * @brief open_dataset_from_file Reads a previously exported training set back into memory
     * @param file filesystem handle
     */
    void open_dataset_from_file(const z5::filesystem::handle::File& file);

    /**
     * @brief open_dataset_from_file Creates a new zarr data set given a filesystem handle
     * @param file filesystem handle
     */
    void create_new_dataset_file(const z5::filesystem::handle::File& file);

public:
    /**
     * @brief TrainDataExporter
     * @param fileNameExport File name of the uncompressed data to be exported in (e.g. "data.zarr")
     * @param deviceName Device name specification (e.g. gpu_0, or cpu_0)
     * @param numberChunks Defines how many chunks a single file should contain.
     * The product of the number of chunks and its chunk size yields the total number of samples of a file.
     * @param chunkSize Defines the chunk size of a single chunk
     */
    TrainDataExporter(const string& fileNameExport, const string& deviceName, size_t numberChunks=200, size_t chunkSize=128);

    /**
     * @brief export_pos Exports a given board position with result to the current training set
     * @param pos Current board position
     * @param eval Filled EvalInfo struct after mcts search
     * @param idxOffset Starting index where to start storing the training sample
     */
    void export_pos(const Board *pos, const EvalInfo& eval, size_t idxOffset);

    /**
     * @brief export_best_move_q Export the Q-value of the move which was selected after MCTS search(Optional training sample feature)
     * @param eval Filled EvalInfo struct after mcts search
     * @param idxOffset Starting index where to start storing the training sample
     */
    void export_best_move_q(const EvalInfo& eval, size_t idxOffset);

    /**
     * @brief export_game_result Assigns the game result, (Monte-Carlo value result) to every training sample.
     * The value is inversed after each step.
     * @param result Game match result: LOST, DRAW, WON
     * @param idxOffset Starting index where to start assigning values
     * @param plys Number of training samples (halfmoves/plys) for the current match
     */
    void export_game_result(const int16_t result, size_t idxOffset, size_t plys);

    size_t get_number_samples() const;

    /**
     * @brief is_file_full Returns true if the exported data set contains as many samples as initially specified, else false
     * @return bool
     */
    bool is_file_full();
};
#endif

#endif // TRAINDATAEXPORTER_H
