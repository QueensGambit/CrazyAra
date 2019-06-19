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
 * @file: planerepresentationtests.h
 * Created on 19.06.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#ifndef PLANEREPRESENTATIONTESTS_H
#define PLANEREPRESENTATIONTESTS_H

//#define BUILD_TESTS


#ifdef BUILD_TESTS
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "inputrepresentation.h"
//#include "../crazyara.h"
#include "search.h"
#include "uci.h"
#include <string>
#include "constants.h"
#include <iostream>
#include <numeric>      // std::accumulate
using namespace Catch::literals;
using namespace std;

namespace PSQT {
  void init();
}

unsigned int Factorial( unsigned int number ) {
    return number <= 1 ? number : Factorial(number-1)*number;
}

//TEST_CASE( "Factorials are computed", "[factorial]" ) {
//    REQUIRE( Factorial(1) == 1 );
//    REQUIRE( Factorial(2) == 2 );
//    REQUIRE( Factorial(3) == 6 );
//    REQUIRE( Factorial(10) == 3628800 );
//}

TEST_CASE("Board representation") {
//void test() {
    //    //StateListPtr states(new std::deque<StateInfo>(1));
    //    //auto uiThread = std::make_shared<Thread>(0);
    string str_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

//    CrazyAra crazyAra;
//    crazyAra.init();
    UCI::init(Options);
    PSQT::init();
    Bitboards::init();
    Position::init();
    Bitbases::init();
    Search::init();

    Board pos;

    StateListPtr states(new std::deque<StateInfo>(1));
//    auto uiThread = std::make_shared<Thread>(0);

    const string fen = "r2q1r1k/1p3pp1/1p1p1b1p/p2P1Bn1/P3bP1Q/1Bp3P1/1PP5/R3R1K1/NPNpn b - - 0 29";

    pos.set(fen, false, CRAZYHOUSE_VARIANT, &states->back(), nullptr); //uiThread.get()); //Threads.main()

//    float input_planes[34][8][8];
    float input_planes[NB_CHANNELS_FULL*BOARD_HEIGHT*BOARD_WIDTH];
    board_to_planes(pos, 0, false, begin(input_planes));

//    float total = 0;
//    for (auto f : input_planes)
//       total += f;
    float total = accumulate(begin(input_planes), end(input_planes), 0.0f, plus<float>());

    REQUIRE(total == 2203);

    board_to_planes(pos, 0, true, begin(input_planes));

//    for (auto f : input_planes)
//       total += f;
    total = accumulate(begin(input_planes), end(input_planes), 0.0f, plus<float>());
    REQUIRE(total == 40.71200017631054_a);

//    int a[] = {1, 3, 5, 7, 9};




    if (total > 0) {
        std::cout << "yes" << total << std::endl;
    }

}

//int main() {
//    test();
//}
class PlaneRepresentationTests
{
public:
    PlaneRepresentationTests();
};
#endif

#endif // PLANEREPRESENTATIONTESTS_H
