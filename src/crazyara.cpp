/*
 * CrazyAra, a deep learning chess variant engine
 * Copyright (C) 2018 Johannes Czech, Moritz Willig, Alena Beyer
 * Copyright (C) 2019 Johannes Czech
 *
 * CrazyAra is free software: You can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * @file: crazyara.cpp
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#include "crazyara.h"

#include "bitboard.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"
#include "syzygy/tbprobe.h"
#include "movegen.h"
#include "search.h"

#include "evalinfo.h"
#include "domain/crazyhouse/constants.h"
#include "sfutil.h"
#include "uci.h"
#include "constants.h"
#include "board.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace std;

// allocate memory
std::string LABELS_MIRRORED[NB_LABELS];
std::unordered_map<Move, size_t> MV_LOOKUP = {};
std::unordered_map<Move, size_t> MV_LOOKUP_MIRRORED = {};

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

CrazyAra::CrazyAra()
{

}

void CrazyAra::welcome()
{
    start_logger("CrazyAra.log");
    std::cout << intro << std::endl;

//    Board pos;
////    string token, cmd;
//    StateListPtr states(new std::deque<StateInfo>(1));
//    auto uiThread = std::make_shared<Thread>(0);

//    const string fen = "r2q1r1k/1p3pp1/1p1p1b1p/p2P1Bn1/P3bP1Q/1Bp3P1/1PP5/R3R1K1/NPNpn w - - 0 29";
////    const string fen2 = "r1b1kb1r/1pp2pPp/p1n2q2/8/8/2PB1p2/PP3PPP/R1BQK2R/PNPpnn w KQkq - 22 12";
////    const string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"; //StartFENs[CRAZYHOUSE_VARIANT]; //variant];

//    pos.set(fen, false, CRAZYHOUSE_VARIANT, &states->back(), uiThread.get());
//    rawAgent->evalute_board_state(pos);
//    mctsAgent->evalute_board_state(pos);
}

void CrazyAra::uci_loop(int argc, char *argv[])
{
    Board pos;
    string token, cmd;
    StateListPtr states(new std::deque<StateInfo>(1));
    auto uiThread = std::make_shared<Thread>(0);

    pos.set(StartFENs[CRAZYHOUSE_VARIANT], false, CRAZYHOUSE_VARIANT, &states->back(), uiThread.get());

    for (int i = 1; i < argc; ++i)
        cmd += std::string(argv[i]) + " ";

    int it = 0;

    do {
//        if (it == 0) {
//            cmd == "uci";
//        }
//        else if (it == 1) {
//            cmd == "isready";
//        }
//        else if (it == 2) {
//            cmd = "position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 d2d3 g8f6 e1g1 e8g8 f3g5 d7d6 g1h1";
//        }
//        else if (it == 3) {
//            cmd = "go wtime 124631 btime 174951 movestogo 34";
//        }
//        else if (it == 4) {
//            cmd = "position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 d2d3 g8f6 e1g1 e8g8 f3g5 d7d6 g1h1 d8e7 b1c3";
//        }
//        else if (it == 5) {
//            cmd = "go wtime 119375 btime 174171 movestogo 33";
//        }
        if (argc == 1 && !getline(cin, cmd)) // Block here waiting for input or EOF
            cmd = "quit";

        istringstream is(cmd);

        token.clear(); // Avoid a stale if getline() returns empty or blank line
        is >> skipws >> token;

        if (    token == "quit"
            ||  token == "stop")
            Threads.stop = true;

        // The GUI sends 'ponderhit' to tell us the user has played the expected move.
        // So 'ponderhit' will be sent if we were told to ponder on the same move the
        // user has played. We should continue searching but switch from pondering to
        // normal search.
        else if (token == "ponderhit")
            Threads.main()->ponder = false; // Switch to normal search

        else if (token == "uci")
            sync_cout //<< "id name " << engine_info(true)
                      //<< "\n"       << Options
                      << "id name CrazyAra 0.5.0\n"
                      << "id author Johannes Czech, Moritz Willig, Alena Beyer\n"
                      << "option name UCI_Variant type combo default crazyhouse var crazyhouse\n"
//                      << "option name use_raw_network type check default true"
                      << "uciok"  << sync_endl;

        else if (token == "setoption")  sync_cout << "info string Updated option UCI_Variant to crazyhouse" << sync_endl;//setoption(is);
        else if (token == "go")         go(pos, is, states);
        else if (token == "position")   position(pos, is, states);
        else if (token == "ucinewgame") sync_cout << "info string newgame" << sync_endl; //setoption(is); // Search::clear();
        else if (token == "isready")    sync_cout << "readyok" << sync_endl;

        // Additional custom non-UCI commands, mainly for debugging
        else if (token == "flip")  pos.flip();
        else if (token == "bench") cout << "dummy"; //bench(pos, is, states);
        else if (token == "d")     sync_cout << pos << sync_endl;
        else if (token == "eval")  cout << "dummy"; //sync_cout << Eval::trace(pos) << sync_endl;
        else
            sync_cout << "Unknown command: " << cmd << sync_endl;

        ++it;
    } while (token != "quit" && argc == 1); // Command line args are one-shot
}

// go() is called when engine receives the "go" UCI command. The function sets
// the thinking time and other parameters from the input string, then starts
// the search.

void CrazyAra::go(Board& pos, istringstream& is, StateListPtr& states) {

  Search::LimitsType limits;
  string token;
  bool ponderMode = false;

  limits.startTime = now(); // As early as possible!

  while (is >> token)
      if (token == "searchmoves")
          while (is >> token)
              limits.searchmoves.push_back(UCI::to_move(pos, token));

      else if (token == "wtime")     is >> limits.time[WHITE];
      else if (token == "btime")     is >> limits.time[BLACK];
      else if (token == "winc")      is >> limits.inc[WHITE];
      else if (token == "binc")      is >> limits.inc[BLACK];
      else if (token == "movestogo") is >> limits.movestogo;
      else if (token == "depth")     is >> limits.depth;
      else if (token == "nodes")     is >> limits.nodes;
      else if (token == "movetime")  is >> limits.movetime;
      else if (token == "mate")      is >> limits.mate;
      else if (token == "perft")     is >> limits.perft;
      else if (token == "infinite")  limits.infinite = 1;
      else if (token == "ponder")    ponderMode = true;

//  Threads.start_thinking(pos, states, limits, ponderMode);
//  EvalInfo res = rawAgent->evalute_board_state(pos);
//  rawAgent->perform_action(pos);
  mctsAgent->perform_action(pos);

  // runtime
//  std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
//  std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

//  int elapsedTimeMS = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//  sync_cout << "info score cp " << res.centipawns << " depth " << res.depth << " nodes " << res.nodes
//            << " time " << elapsedTimeMS << " nps " << int((res.nodes / (elapsedTimeMS / 1000.0)) + 0.5) << " pv " << res.pv[0] << sync_endl;
////  sync_cout << "bestmove "<< res.pv[0] << sync_endl;
//  sync_cout << "bestmove " << UCI::move(res.pv[0], pos.is_chess960()) << sync_endl;
  //    info score cp 304 depth 6 nodes 72 time 1009 nps 71 pv b1c3 g8f6 g1f3 d7d5 e4d5 f6d5
  //    std::cout << "info score cp" << std::endl;
}

// position() is called when engine receives the "position" UCI command.
// The function sets up the position described in the given FEN string ("fen")
// or the starting position ("startpos") and then makes the moves given in the
// following move list ("moves").

void CrazyAra::position(Board& pos, istringstream& is, StateListPtr& states) {

  Move m;
  string token, fen;

  Variant variant = UCI::variant_from_name(Options["UCI_Variant"]);

  is >> token;
  if (token == "startpos")
  {
      fen = StartFENs[CRAZYHOUSE_VARIANT]; //variant];
      is >> token; // Consume "moves" token if any
  }
  else if (token == "fen")
      while (is >> token && token != "moves")
          fen += token + " ";
  else
      return;

  auto uiThread = std::make_shared<Thread>(0);

  states = StateListPtr(new std::deque<StateInfo>(1)); // Drop old and create a new one
//  sync_cout << "before pos.set()" << sync_endl;
  pos.set(fen, Options["UCI_Chess960"], CRAZYHOUSE_VARIANT, &states->back(), uiThread.get()); //Threads.main()

//      StateListPtr states(new std::deque<StateInfo>(1));

//      const string fen = "r2q1r1k/1p3pp1/1p1p1b1p/p2P1Bn1/P3bP1Q/1Bp3P1/1PP5/R3R1K1/NPNpn b - - 0 29";
//      const string fen2 = "r1b1kb1r/1pp2pPp/p1n2q2/8/8/2PB1p2/PP3PPP/R1BQK2R/PNPpnn w KQkq - 22 12";

//      pos.set(fen2, false, CRAZYHOUSE_VARIANT, &states->back(), uiThread.get());

  // Parse move list (if any)
  while (is >> token && (m = UCI::to_move(pos, token)) != MOVE_NONE)
  {
      states->emplace_back();
      pos.do_move(m, states->back());
      sync_cout << "info string consume move" << sync_endl;
  }

  sync_cout << "info string position" << sync_endl;

}


void CrazyAra::init()
{
    SearchSettings searchSettings;
    searchSettings.batchSize = 8; //8; //64;
    netSingle = new NeuralNetAPI("cpu", 1, false,
                                    "/home/queensgambit/Programming/Deep_Learning/models/risev2/json/",
                                    "/home/queensgambit/Programming/Deep_Learning/models/risev2/params/");
    rawAgent = new RawNetAgent(netSingle, PlaySettings(), 0, 0, true);
    netBatch = new NeuralNetAPI("cpu", searchSettings.batchSize, false,
                           "/home/queensgambit/Programming/Deep_Learning/models/risev2/json/",
                           "/home/queensgambit/Programming/Deep_Learning/models/risev2/params/");
    mctsAgent = new MCTSAgent(netSingle, netBatch, searchSettings, SearchLimits(), PlaySettings()); //, hashTable);

    UCI::init(Options);
//    Threads.set(1);
    Bitboards::init();
    Position::init();
    Bitbases::init();
    Search::init();

//    UCI::init(Options);
////    PSQT::init();
//    Bitboards::init();
//    Position::init();
//    Bitbases::init();
//    Search::init();
    Constants::init();
//    hashTable = new unordered_map<Key, Node*>;

}
