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

#ifdef TENSORRT
#ifndef MODE_POMMERMAN
#include "chessbatchstream.h"
#include "uci.h"
#include "stateobj.h"

ChessBatchStream::ChessBatchStream(int batchSize, int maxBatches):
    mBatchSize{batchSize},
    mMaxBatches{maxBatches},
    mDims{batchSize, int(StateConstants::NB_CHANNELS_TOTAL()), int(StateConstants::BOARD_HEIGHT()), int(StateConstants::BOARD_WIDTH())}
{
    Bitboards::init();
    Position::init();
    Bitbases::init();
    Board pos;
    auto uiThread = make_shared<Thread>(0);

    // allocate memory
    mData.resize(StateConstants::NB_VALUES_TOTAL() * batchSize * maxBatches);
    StateListPtr states = StateListPtr(new std::deque<StateInfo>(1));
    reset_to_startpos(pos, uiThread.get(), states);

#ifdef MODE_CHESS
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
							   
    vector<string> uciMoves2 = {};
	
#else
    vector<string> uciMoves = {"e2e4", "g8f6", "b1c3", "e7e5", "g1f3", "b8c6", "f1c4", "f8e7",
                               "d2d4", "e5d4", "f3d4", "d7d5", "d4c6", "b7c6", "e4d5", "N@h4",
                               "h1g1", "e8g8", "P@h6", "g7h6", "c1h6", "P@g3", "h2g3", "P@h2",
                               "g1h1", "h4g2", "e1f1", "c8h3", "h1h2", "g2e3", "f1g1", "e3d1",
                               "a1d1", "f6g4", "h2h3", "g4h6", "h3h6", "B@g7", "h6h2", "P@h4",
                               "B@f5", "h4g3", "f5h7", "g8h8", "f2g3", "P@f2", "g1f2", "e7h4",
                               "h2h4", "g7c3", "B@d4", "c3d4", "d1d4", "B@e3", "f2e3", "d8g5",
                               "B@f4", "N@g2", "e3f2", "g2h4", "h7e4", "R@h2", "P@g2", "h4g2",
                               "e4g2", "h2g2", "f2g2", "P@h3", "g2f1", "Q@g2", "f1e1", "a8e8",
                               "P@e7", "e8e7", "P@e6", "B@f2", "e1d1", "f2d4", "P@g7", "d4g7",
                               "P@f2", "g5h5", "P@g4", "h5g4", "N@e2", "R@f1", "B@e1", "g2f2",
                               "N@f3", "f2f3", "N@g2", "h3g2", "R@h2", "P@h3", "P@f2", "f1e1",
                               "d1e1", "g2g1q", "N@f1", "N@g2", "h2g2", "g1f1", "e1f1", "f3g2",
                               "f1e1", "N@f3", "e1d1", "R@e1"};

    vector<string> uciMoves2 = {"d2d4", "g8f6", "c1f4", "e7e6", "g1f3", "b7b6", "e2e3", "c8b7",
                                "f1d3", "f8b4", "b1d2", "b4d2", "f3d2", "N@h4", "e1g1", "h4g2",
                                "f4g5", "b8c6", "B@f3", "P@h3", "f3h5", "d8e7", "d1f3", "h7h6",
                                "g5f6", "g7f6", "f3h3", "c6d4", "c2c3", "B@f5", "d3f5", "d4f5",
                                "B@f3", "B@d5", "h3g2", "d5f3", "h5f3", "b7f3", "g2f3", "B@c6",
                                "B@e4", "c6e4", "d2e4", "B@c6", "B@d3", "P@c4", "d3c4", "B@b7",
                                "B@d3", "b6b5", "c4b5", "c6e4", "d3e4", "b7e4", "f3e4", "N@h3",
                                "g1h1", "B@d5", "N@d2", "d5e4", "d2e4", "Q@g4", "B@g2", "g4g2",
                                "h1g2", "h3g5", "b5d7", "e8f8", "B@g4", "h6h5", "B@a3", "c7c5",
                                "P@b7", "a8d8", "P@c7", "h5g4", "c7d8r", "e7d8", "a3c5", "f8g7",
                                "e4g5", "B@f3", "g5f3", "g4f3", "g2f3", "B@h5", "f3e4", "N@e5",
                                "f1g1", "P@g6", "N@d4", "e5d7", "P@h6", "g7h6", "B@f8", "d7f8",
                                "Q@c1", "B@e2", "c5f8", "d8f8", "P@g5", "f6g5", "d4e2", "h5e2",
                                "R@h3", "B@h5", "N@g4", "e2g4", "h3h5", "g4h5", "N@g4", "h5g4",
                                "B@g7", "f8g7", "e4d3", "B@e2", "d3d2", "N@e4", "d2c2", "B@d3",
                                "c2b3", "N@c5", "b3b4", "R@a4"};
#endif

    vector<string> curUciMoves = uciMoves;
    size_t offset = 0;

    for (size_t idx = 0; idx < size_t(batchSize * maxBatches); ++idx) {
        states->emplace_back();
        board_to_planes(&pos, pos.number_repetitions(), true, mData.data() + StateConstants::NB_VALUES_TOTAL() * idx, StateConstants::NB_VALUES_TOTAL());
        if (idx == curUciMoves.size()) {
            reset_to_startpos(pos, uiThread.get(), states);
            offset = curUciMoves.size();
            curUciMoves = uciMoves2;
        }
        pos.do_move(UCI::to_move(pos, curUciMoves[idx-offset]), states->back());
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
    nvinfer1::Dims dims;
    dims.nbDims = 4;
    dims.d[0] = mBatchSize;
    dims.d[1] = mDims.d[0];
    dims.d[2] = mDims.d[1];
    dims.d[3] = mDims.d[2];
    return dims;
}

void reset_to_startpos(Board& pos, Thread* uiThread, StateListPtr& states)
{
    states = StateListPtr(new std::deque<StateInfo>(1));
    pos.set(StateConstants::start_fen(StateConstants::DEFAULT_VARIANT()), false, Variant(StateConstants::DEFAULT_VARIANT()), &states->back(), uiThread);
}
#endif
#endif
