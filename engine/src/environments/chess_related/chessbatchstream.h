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
 * @file: chessbatchstream.h
 * Created on 27.03.2020
 * @author: queensgambit
 *
 * Calibration stream data for INT8 quantization.
 * The calibration data is generated using sample chess positions.
 */

#ifndef CHESSBATCHSTREAM_H
#define CHESSBATCHSTREAM_H

#ifdef TENSORRT
#ifndef MODE_POMMERMAN

#include "thread.h"
#include "EntropyCalibrator.h"
#include "BatchStream.h"
#include "environments/chess_related/inputrepresentation.h"
#include "constants.h"

/**
 * @brief The ChessBatchStream class
 * Provides batches of example chess position for calibration.
 * This class is similar to the exemplary MNISTBatchStream.
 */
class ChessBatchStream : public IBatchStream
{
public:
    ChessBatchStream(int batchSize, int maxBatches);

    void reset(int firstBatch) override;

    bool next() override;

    void skip(int skipCount) override;

    float* getBatch() override;

    float* getLabels() override;

    int getBatchesRead() const override;

    int getBatchSize() const override;

    nvinfer1::Dims getDims() const override;

private:
    int mBatchSize{0};
    int mBatchCount{0};
    int mMaxBatches{0};
    nvinfer1::Dims mDims{};
    std::vector<float> mData;
    std::vector<float> mLabels{};
};

void reset_to_startpos(Board& pos, Thread* uiThread, StateListPtr& states);
#endif
#endif

#endif // CHESSBATCHSTREAM_H
