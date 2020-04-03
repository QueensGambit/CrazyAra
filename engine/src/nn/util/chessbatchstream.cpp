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
    vector<string> uciMoves = {"e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6",
                               "b1c3", "b8c6", "c1g5", "c8d7", "d1d2", "a8c8", "d4b3", "a7a6",
                               "f1e2", "e7e6", "e1g1", "h7h6", "g5e3", "b7b5", "a2a3", "f8e7",
                               "f2f4", "e8g8", "e2f3", "e6e5", "f4f5", "a6a5", "a3a4", "b5a4",
                               "c3a4", "c6b4", "c2c3", "c8b8", "c3b4", "b8b4", "b3a5", "b4a4",
                               "a1a4", "d7a4", "f1a1", "a4b5", "d2b4", "d8d7", "a5b3", "f8b8",
                               "a1a7", "b8b7", "a7a8", "g8h7", "b3a5", "d6d5", "e3c5", "d5e4",
                               "a5b7", "e4f3", "b7d6", "e7d6", "c5d6", "d7c6", "a8a1", "f6e4",
                               "d6e5", "f3f2", "g1h1", "f2f1q", "a1f1", "b5f1", "b4e1", "f1g2",
                               "h1g2", "e4g5", "g2f1", "c6h1", "f1e2", "h1e4", "e2d1", "e4b1",
                               "d1e2", "b1e4", "e2d1", "e4f5", "e1e3", "g5f3", "e5f4", "f5d5",
                               "d1e2", "f3d4", "e2f2", "d4f5", "e3e5", "d5d3", "e5c3", "d3e4",
                               "c3e5", "e4c2", "e5e2", "c2c5", "f2f1", "c5d5", "b2b4", "f5d4",
                               "e2d3", "f7f5", "f4g3", "d5b5", "d3b5", "d4b5", "f1e2", "g7g5",
                               "e2d3", "f5f4", "g3f2", "h7g6", "d3c4", "b5a3", "c4b3", "a3b5",
                               "b3c4", "b5a3", "c4b3", "a3b5", "b3c4"};
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
