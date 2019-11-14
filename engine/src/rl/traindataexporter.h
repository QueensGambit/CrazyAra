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

    xt::xarray<int16_t> gameX;
    xt::xarray<int16_t> gameValue;
    xt::xarray<float> gamePolicy;
    xt::xarray<float> gameBestMoveQ;
    bool firstMove;

    // current number of games - 1
    size_t gameIdx;
    // current sample index to insert
    size_t startIdx;

    /**
     * @brief export_planes Exports the board in plane representation (x)
     * @param pos Board position to export
     */
    void save_planes(const Board *pos);

    /**
     * @brief save_policy Saves the policy (e.g. mctsPolicy) to the matrix
     * @param legalMoves List of legal moves
     * @param policyProbSmall Probability for each move
     * @param sideToMove Current side to move
     */
    void save_policy(const vector<Move>& legalMoves, const DynamicVector<float>& policyProbSmall, Color sideToMove);

    /**
     * @brief save_best_move_q Saves the Q-value of the move which was selected after MCTS search(Optional training sample feature)
     * @param eval Filled EvalInfo struct after mcts search
     */
    void save_best_move_q(const EvalInfo& eval);

    /**
     * @brief save_side_to_move Saves the current side to move as a +1 for WHITE and -1 for BLACK.
     * The current side to move is either WHITE(0) or BLACK(1).
     * Later if WHITE won the game the value array is inverted.
     * For a draw it will be multiplied by 0.
     * @param col current side to move
     */
    void save_side_to_move(Color col);

    /**
     * @brief save_start_idx Saves the current starting index where the next game starts to the game array
     */
    void save_start_idx();

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

    /**
     * @brief apply_result_to_value Inverts the gameValue array if WHITE lost the game.
     * In the case of a draw, all entries are set to 0.
     * @param result Possible values DRAWN, WHITE_WIN, BLACK_WIN,
     */
    void apply_result_to_value(Result result);
public:
    /**
     * @brief TrainDataExporter
     * @param fileNameExport File name of the uncompressed data to be exported in (e.g. "data.zarr")
     * @param numberChunks Defines how many chunks a single file should contain.
     * The product of the number of chunks and its chunk size yields the total number of samples of a file.
     * @param chunkSize Defines the chunk size of a single chunk
     */
    TrainDataExporter(const string& fileNameExport, size_t numberChunks=200, size_t chunkSize=128);

    /**
     * @brief export_pos Saves a given board position, policy and Q-value to the specific game arrays
     * @param pos Current board position
     * @param eval Filled EvalInfo struct after mcts search
     * @param idxOffset Starting index where to start storing the training sample
     */
    void save_sample(const Board *pos, const EvalInfo& eval, size_t idxOffset);

    /**
     * @brief export_game_samples Assigns the game result, (Monte-Carlo value result) to every training sample.
     * The value is inversed after each step and export all training samples of a single game.
     * @param result Game match result: LOST, DRAW, WON
     * @param idxOffset Starting index where to start assigning values
     * @param plys Number of training samples (halfmoves/plys) for the current match
     */
    void export_game_samples(Result result, size_t plys);

    size_t get_number_samples() const;

    /**
     * @brief is_file_full Returns true if the exported data set contains as many samples as initially specified, else false
     * @return bool
     */
    bool is_file_full();

    /**
     * @brief new_game Sets firstMove to true
     */
    void new_game();
};
#endif

#endif // TRAINDATAEXPORTER_H
