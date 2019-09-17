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
    size_t chunckSize;
    std::unique_ptr<z5::Dataset> dStartIndex;
    std::unique_ptr<z5::Dataset> dx;
    std::unique_ptr<z5::Dataset> dValue;
    std::unique_ptr<z5::Dataset> dPolicy;

    // current number of games - 1
    size_t gameIdx;
    // current sample index to insert
    size_t startIdx;

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

    void export_start_idx();

    void open_dataset_from_file(const z5::filesystem::handle::File& file);

    void create_new_dataset_file(const z5::filesystem::handle::File& file);

public:
    TrainDataExporter();

    /**
     * @brief export_pos Exports a given board position with result to the current training set
     * @param pos
     * @param result
     * @param idxOffset
     */
    void export_pos(const Board *pos, const EvalInfo& eval, size_t idxOffset);

    void export_game_result(const Result result, size_t idxOffset, size_t plys);

    /**
     * @brief export_positions Export a list of nodes as training data
     * @param nodes List of nodes (usually all nodes which have been played in a game)
     * @param result Game result, needed to assign the target value
     */
    void export_positions(const std::vector<Node*>& nodes, Result result);
};

#endif // TRAINDATAEXPORTER_H
