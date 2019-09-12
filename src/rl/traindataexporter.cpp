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
 * @file: traindataexporter.cpp
 * Created on 12.09.2019
 * @author: queensgambit
 */

#include "traindataexporter.h"
#include <inttypes.h>

void TrainDataExporter::export_pos(const Board *pos, Result result, size_t idxOffset)
{
    // x
    float inputPlanes[NB_VALUES_TOTAL];
    board_to_planes(pos, 0, false, inputPlanes);
    // write array to roi
    z5::types::ShapeType offset1 = { idxOffset, 0, 0, 0 };
    xt::xarray<int16_t>::shape_type shape1 = { 1, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH };
    xt::xarray<int16_t> array1(shape1);
    for (size_t i = 0; i < NB_VALUES_TOTAL; ++i) {
        array1.data()[i] = int16_t(inputPlanes[i]);
    }
    z5::multiarray::writeSubarray<int16_t>(dx, array1, offset1.begin());

    // value
    // write array to roi
    z5::types::ShapeType offset2 = { idxOffset };
    xt::xarray<int16_t>::shape_type shape2 = { 1 };
    xt::xarray<int16_t> array2(shape2, result);
    z5::multiarray::writeSubarray<int16_t>(dValue, array2, offset2.begin());

    // policy
    // write array to roi
    z5::types::ShapeType offset3 = { idxOffset, 0 };
    xt::xarray<float>::shape_type shape3 = { 1, NB_LABELS };
    xt::xarray<float> array3(shape3);

    for (size_t i = 0; i < NB_LABELS; ++i) {
        array3.data()[i] = 1.0f / float(NB_LABELS);
    }

    z5::multiarray::writeSubarray<float>(dPolicy, array3, offset3.begin());
}

TrainDataExporter::TrainDataExporter()
{
    const string fileName = "data.zr";
    // get handle to a File on the filesystem
    z5::filesystem::handle::File f(fileName);

    // create the file in zarr format
    const bool createAsZarr = true;
    z5::createFile(f, createAsZarr);

    z5::createGroup(f, "group");
    chunckSize = 128;

    // create a new zarr dataset
    const std::string dsName = "x";
    std::vector<size_t> shape = { chunckSize, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH };
    std::vector<size_t> chunks = { chunckSize, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH };
    dx = z5::createDataset(f, dsName, "int16", shape, chunks);
    dValue = z5::createDataset(f, "y_value", "int16", { chunckSize }, { chunckSize });
    dPolicy = z5::createDataset(f, "y_policy", "float32", { chunckSize, NB_LABELS }, { chunckSize, NB_LABELS });
}

void TrainDataExporter::export_positions(const std::vector<Node*>& nodes, Result result)
{
    size_t offset = 0;
    for (auto node : nodes) {
        export_pos(node->get_pos(), result, offset++);
    }
}
