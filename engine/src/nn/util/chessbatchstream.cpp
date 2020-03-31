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

#include "chessbatchstream.h"
#include "uci.h"

ChessBatchStream::ChessBatchStream(int batchSize, int maxBatches):
    mBatchSize{batchSize},
    mMaxBatches{maxBatches},
    mDims{batchSize, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH}
{
    Bitboards::init();
    Position::init();
    Bitbases::init();
    Board pos;
    auto uiThread = make_shared<Thread>(0);

    // allocate memory
    mData.resize(NB_VALUES_TOTAL * batchSize * maxBatches);
    StateListPtr states = StateListPtr(new std::deque<StateInfo>(1));
    pos.set(StartFENs[CHESS_VARIANT], false, CHESS_VARIANT, &states->back(), uiThread.get());

    // create a vector of sample moves to create the sample data
    vector<string> uciMoves = {"e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
                               "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6", "c2c3", "e8g8",
                               "h2h3", "c6a5", "b3c2", "c7c5", "d2d4", "d8c7", "b1d2", "c5d4",
                               "c3d4", "a5c6", "d2b3", "a6a5", "c1e3", "a5a4", "b3d2", "c8d7",
                               "a1c1", "c7b7", "d2f1", "f8e8", "f1g3", "e7d8", "c2b1", "d8a5",
                               "e1e2", "a5b6", "e2d2", "a4a3",};
    for (int idx = 0; idx < batchSize * maxBatches; ++idx) {
        states->emplace_back();
        board_to_planes(&pos, pos.number_repetitions(), true, mData.data() + NB_VALUES_TOTAL * idx);
        if (idx == batchSize * maxBatches - 1) {
            // create a temporary StateInfo for the last position
            pos.do_move(UCI::to_move(pos, uciMoves[idx]), *(new StateInfo));
        }
        else {
            pos.do_move(UCI::to_move(pos, uciMoves[idx]), states->back());
        }
    }
}

void ChessBatchStream::reset(int firstBatch)
{
    mBatchCount = firstBatch;
}

bool ChessBatchStream::next()
{
    if (mBatchCount >= mMaxBatches)
    {
        return false;
    }
    ++mBatchCount;
    return true;
}

void ChessBatchStream::skip(int skipCount)
{
    mBatchCount += skipCount;
}

float* ChessBatchStream::getBatch()
{
    return mData.data() + (mBatchCount * mBatchSize * samplesCommon::volume(mDims));
}

float* ChessBatchStream::getLabels()
{
    // ignore lables
    return nullptr;
}

int ChessBatchStream::getBatchesRead() const
{
    return mBatchCount;
}

int ChessBatchStream::getBatchSize() const
{
    return mBatchSize;
}

nvinfer1::Dims ChessBatchStream::getDims() const
{
    return Dims{4, {mBatchSize, mDims.d[0], mDims.d[1], mDims.d[2]}, {}};
}
