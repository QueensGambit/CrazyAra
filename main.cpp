//#define EIGEN_USE_MKL_ALL

//#define BENCHMARK

#ifdef BENCHMARK


#include <iostream>
#include <Dense>

#include "bitboard.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"
#include "syzygy/tbprobe.h"
#include "movegen.h"
#include "search.h"

#include "src/agents/mctsagent.h"
#include "src/evalinfo.h"
#include "src/domain/crazyhouse/constants.h"
#include "src/sfutil.h"
#include "uci.h"
#include "constants.h"
#include "src/board.h"
//#include "mxnet-cpp/MxNetCpp.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xinfo.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xslice.hpp"
#include "xtensor/xsort.hpp"
#include <blaze/Math.h>
#include "../blazeutil.h"

using blaze::StaticVector;
using blaze::DynamicVector;

//using namespace mxnet::cpp;
using Eigen::MatrixXd;

using namespace std;

namespace PSQT {
  void init();
}

template<bool Root>
uint64_t perft(Position& pos, Depth depth) {

  StateInfo st;
  uint64_t cnt, nodes = 0;
  const bool leaf = (depth == 2 * ONE_PLY);

  for (const auto& m : MoveList<LEGAL>(pos))
  {
      if (Root && depth <= ONE_PLY)
          cnt = 1, nodes++;
      else
      {
          pos.do_move(m, st);
          cnt = leaf ? MoveList<LEGAL>(pos).size() : perft<false>(pos, depth - ONE_PLY);
          nodes += cnt;
          pos.undo_move(m);
      }
      if (Root)
          sync_cout << UCI::move(m, pos.is_chess960()) << ": " << cnt << " " << int(m) << sync_endl;
  }
  return nodes;
}

int main() {
    xt::xarray<double> arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};


    cout << "Script start" << endl;

//    blaze::DynamicVector<int> v1( sizeUL ), v3;

    // ... Initializing the vectors
    const int size = 80; //34;

    NDArray mxnet_vec = NDArray(Shape(size), Context::cpu());
//    auto xtensor_vec =  xt::ones<float>({size}); //xt::random::rand<float>({size}); //
    xt::xarray<float> xtensor_vec{1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9,
                                 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9};

//    blaze::HybridVector<float, 1024UL> blaze_vec(size);
//    blaze::UniformVector<float> blaze_vec(sizeUL);
//    const float data[size] = {mxnet_vec.GetData()};
//    std::vector<float> data(mxnet_vec.GetData(), size);
    const float *data = mxnet_vec.GetData();
//    float result2 = *dat; // works!!

//    data = &mxnet_vec.GetData()[0];
//    blaze_vec = data; //mxnet_vec.GetData(); //99.9f;
    auto eigen_vec =  Eigen::VectorXf(size); //::Ones(size); //Eigen::VectorXf::Random(size); //
    eigen_vec << 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9,
            1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9;
    float vec[size];
    auto eigen_vec_div =  Eigen::VectorXf(size); //::Ones(size); //Eigen::VectorXf::Random(size); //
    eigen_vec_div << 1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9,1.0f/ 1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9,1.0f/ 1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9,1.0f/ 1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9,1.0f/
            1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9,1.0f/ 1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9,1.0f/ 1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9,1.0f/ 1,1.0f/2,1.0f/3,1.0f/4,1.0f/42,1.0f/5,1.0f/6,1.0f/7,1.0f/8,1.0f/9;

    float cur_max = eigen_vec[0];
    int max_idx = 0;
    for (int i = 0; i < size; ++i) {
        vec[i] = eigen_vec[i];
        if (vec[i] > cur_max) {
            max_idx = i;
            cur_max = vec[i];
        }
    }
    cout << "correct max_idx" << max_idx << endl;

//    cout << "eigen_vec" << eigen_vec << endl;
//    cout << "vec" << endl;
    for (auto i : vec) {
//        cout << i << " ";
    }
//    cout << eigen_vec << endl;
//    vector<int> vec(eigen_vec.data(), eigen_vec.data() + mat.cols());
//    blaze::StaticVector<float, size> blaze_vec;
//    blaze::HybridVector<float, 512UL> blaze_vec(size);
//    blaze::DynamicVector<float> blaze_vec(size);
//    typedef double real;
    typedef float real;

     blaze::DynamicVector<real> blaze_vec {1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9,
                                           1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9, 1,2,3,4,42,5,6,7,8,9};
//    blaze_vec = vec;
//    cout << "blaze vec" << blaze_vec << endl;

//    auto res_xtensor = xt::random::rand<float>({size});
    NDArray res_mxnet = NDArray(Shape(size), Context::cpu());
    xt::xarray<float> res_xtensor = xt::random::rand<float>({size});
//    for (int i = 0; i < size; ++i) {
//        res_xtensor[i] = vec[i];
//    }

//    xtensor_vec[42] = 99;
//    blaze_vec[42] = 99;
//    eigen_vec[42] = 99;

//    blaze::StaticVector<float, size> res_blaze;
    blaze::DynamicVector<real> res_blaze(size);
//    blaze::HybridVector<float, 512UL> res_blaze(size);

    res_blaze = 0;
    Eigen::VectorXf res_eigen = Eigen::VectorXf::Zero(size);  //Random(size);

    size_t it = 1e6; //7; //999999;

//    std::chrono::steady_clock::time_point start_mxnet = std::chrono::steady_clock::now();
//    for (size_t i = 0; i < it; ++i) {
////        res_mxnet += mxnet_vec;
//        res_mxnet *= mxnet_vec;
//        res_mxnet.WaitToWrite();
//        res_mxnet /= mxnet_vec;
//        res_mxnet.WaitToWrite();
//    }
//    std::chrono::steady_clock::time_point end_mxnet = std::chrono::steady_clock::now();
//    std::cout << "Elapsed time mxnet: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_mxnet - start_mxnet).count() << "ms" << std::endl;

//    int id = 0;
    auto id = xt::argmax(xtensor_vec);

    std::chrono::steady_clock::time_point start= std::chrono::steady_clock::now();
    for (size_t i = 0; i < it; ++i) {
//        auto sum_xtensor = xt::sum(xtensor_vec);
//          res_eigen = xtensor_vec + xtensor_vec;
            res_xtensor += xtensor_vec;
            res_xtensor *= xtensor_vec;
            res_xtensor /= xtensor_vec;
//            id = xt::argmax(xtensor_vec);
    }
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Elapsed time xtensor: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    int idx = 0;
    std::chrono::steady_clock::time_point start_blaze = std::chrono::steady_clock::now();
    for (size_t i = 0; i < it; ++i) {
        res_blaze += blaze_vec;
        res_blaze *= blaze_vec;
        res_blaze /= blaze_vec;
////        argmax(blaze_vec);
//        res_blaze = sqrt(res_blaze);
//         idx = argmax(blaze_vec); //blaze_vec.find(max(blaze_vec));
    }

    std::chrono::steady_clock::time_point end_blaze = std::chrono::steady_clock::now();
    std::cout << "Elapsed time blaze: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_blaze - start_blaze).count() << "ms" << std::endl;

    int index = 0;
    const float totalsum = sum( res_blaze );  // Results in 10
//    cout << "last number" << vec[511] * it << endl;
//    cout << "totalsum" << res_blaze[511] << endl;
    std::chrono::steady_clock::time_point start_eigen = std::chrono::steady_clock::now();
    for (size_t i = 0; i < it; ++i) {
        res_eigen += eigen_vec;
        res_eigen *= eigen_vec;
        res_eigen *= eigen_vec_div;
//        res_eigen *= (1.0f/eigen_vec);
//        eigen_vec.maxCoeff(&index);
    }
//    cout << "res einge" << res_eigen[511] << endl;


    std::chrono::steady_clock::time_point end_eigen = std::chrono::steady_clock::now();
    std::cout << "Elapsed time eigen: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_eigen - start_eigen).count() << "ms" << std::endl;

//    cout << "argmax xtensor " << id << endl;
//    cout << "argmax blaze_vec " << idx << endl;
//    cout << "argmax eigen_vec " << index << endl;

        cout << "res mxnet " << res_mxnet.At(0,7) << endl;
        cout << "res xtensor " << res_xtensor[7] << endl;
        cout << "res blaze_vec " << res_blaze[7] << endl;
        cout << "res eigen_vec " << res_eigen[7] << endl;

    // -> blaze is faster than all competitors for this matrix size by far!

    cout << endl;
}
/*
int main()
{
//    cout << "Hello World!!!!" << endl;

//    //string str_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

//    //Position pos;
//    //StateInfo st;
//    //Thread t(1024);

//    //StateListPtr states(new std::deque<StateInfo>(1));
//    //auto uiThread = std::make_shared<Thread>(0);

//    //pos.set(str_pos, false, CHESS_VARIANT, &states->back(), uiThread.get()); //Threads.main());

    // FEN strings of the initial positions
      const string StartFENs[SUBVARIANT_NB] = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #ifdef ANTI
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef ATOMIC
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef CRAZYHOUSE
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
    #endif
    #ifdef EXTINCTION
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef GRID
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef HORDE
      "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1",
    #endif
    #ifdef KOTH
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef LOSERS
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef RACE
      "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1",
    #endif
    #ifdef THREECHECK
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3+3 0 1",
    #endif
    #ifdef TWOKINGS
      "rnbqkknr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKKNR w KQkq - 0 1",
    #endif
    #ifdef SUICIDE
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
    #endif
    #ifdef BUGHOUSE
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
    #endif
    #ifdef DISPLACEDGRID
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef LOOP
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
    #endif
    #ifdef PLACEMENT
      "8/pppppppp/8/8/8/8/PPPPPPPP/8[KQRRBBNNkqrrbbnn] w - -",
    #endif
    #ifdef SLIPPEDGRID
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    #endif
    #ifdef TWOKINGSSYMMETRIC
      "rnbqkknr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKKNR w KQkq - 0 1",
    #endif
  };

//    Position pos;
//    string token, cmd;
//    //StateListPtr states(new std::deque<StateInfo>(1));
//    //auto main_thread = Threads.main();
//    //auto uiThread = std::make_shared<Thread>(0);

//    //pos.set(StartFENs[CHESS_VARIANT], false, CHESS_VARIANT, &states->back(), nullptr); //uiThread.get());

//    StateInfo st;
//    Position p;
//    p.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, CHESS_VARIANT, &st, pos.this_thread());
//    //pos.set(str_pos, WHITE, CHESS_VARIANT, &state);
//    //cout << "end" << endl;

//    return 0;


  std::cout << engine_info() << std::endl;

  std::string str_array[2];
  str_array[0] = "as";
  str_array[1] = "abb";
  std::cout << str_array[1] << std::endl;

  UCI::init(Options);
  PSQT::init();
  Bitboards::init();
  Position::init();
  Bitbases::init();
  Search::init();

//  Constants::LABELS_MIRRORED[0] = "Test";

  Constants::init();
  std::cout << "[0]" << " " << LABELS_MIRRORED[0] << std::endl;

  Threads.set(Options["Threads"]);
  Search::clear(); // After threads are up

  //UCI::loop(argc, argv);

  Board pos;
  string token, cmd;
  StateListPtr states(new std::deque<StateInfo>(1));
  auto uiThread = std::make_shared<Thread>(0);

  const string fen = "r2q1r1k/1p3pp1/1p1p1b1p/p2P1Bn1/P3bP1Q/1Bp3P1/1PP5/R3R1K1/NPNpn b - - 0 29";
//  const string fen2 = "r1b1kb1r/1pp2pPp/p1n2q2/8/8/2PB1p2/PP3PPP/R1BQK2R/PNPpnn w KQkq - 22 12";

  pos.set(fen, false, CRAZYHOUSE_VARIANT, &states->back(), uiThread.get());

  Board pos2(pos);

  //perft(pos, 2);
  std::atomic<uint64_t> nodes, tbHits, bestMoveChanges;

  nodes = perft<true>(pos2, ONE_PLY); //Limits.perft * ONE_PLY);
  sync_cout << "\nNodes searched: " << nodes << "\n" << sync_endl;

//  generate_all<CHESS_VARIANT, WHITE, Type>(pos, moveList, target);
  cout << "pos: " << pos << endl;
  Threads.set(0);

  MCTSAgent mctsagent;

//  Constants::init();
  init();
  std::cout << "LABELS_MIRRORED[0]: " << LABELS_MIRRORED[0] << std::endl;
  std::cout << "LABELS_MIRRORED[2069]: " << LABELS_MIRRORED[2069] << std::endl;
  EvalInfo eval_info = mctsagent.evalute_board_state(pos2);

  Eigen::VectorXf v(4);

  v << 5, 2, 3, 4;

  int index;
  v.maxCoeff(&index);

  cout << "argmax:" << index << endl;
  v = v * 3;

  cout << "eval_info: " << eval_info << endl;

//  std::vector<std::string> enPassentMoves;
//  fill_en_passent_moves(enPassentMoves);

//  cout << "en_passent_moves: " << endl;
//  for (auto&= m : enPassentMoves) {
//      cout << m << " ";
//  }
//  cout << endl;

//  int nbSfMoves = 0;

//  for (int mvIdx=0; mvIdx < NB_LABELS; mvIdx++) {
//      std::vector<Move> moves = make_move(LABELS[mvIdx]);
//      for (Move move : moves) {
//          nbSfMoves++;
//      }
////      string uciMove(UCI::move(mv, false));
//      //cout << "hurray" <<  mvIdx << endl; //uciMove << endl;
//  }
//  cout << "nbSfMoves: " << nbSfMoves << endl;

//  for (int color : {WHITE, BLACK}) {
////  for (int color= WHITE; color <= BLACK; ++color) {
//       cout << "color " << int(color) << " " << int(WHITE) << " " << int(BLACK) << endl;
//  }

//  cout << "mirrored moves:" << endl;
//  for (std::string move : LABELS) {
//      cout << move << " " << mirror_move(move) << endl;
//  }

  // LC-Zero
//  namespace {
//  void ApplyDirichletNoise(Node* node, float eps, double alpha) {
//    float total = 0;
//    std::vector<float> noise;

//    for (int i = 0; i < node->GetNumEdges(); ++i) {
//      float eta = Random::Get().GetGamma(alpha, 1.0);
//      noise.emplace_back(eta);
//      total += eta;
//  }
  Bitboard bitset = pos.pieces(BLACK, PAWN);

//  uint64_t bitset;
//  size_t bitmapsize = 64;

  // https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
//  for (size_t k = 0; k < bitmapsize; ++k) {
//      bitset = bitmap[k];
      size_t p = 0; // k * 64;
      while (bitset != 0) {
        if (bitset & 0x1) {
//          callback(p);
          cout << p << " " << endl;
        }
        bitset >>= 1;
        p += 1;
      }
//  }
  cout << "pawns in hand: " << pos.count_in_hand<PAWN>(WHITE) << endl;
  return 0;
}
*/
#endif
