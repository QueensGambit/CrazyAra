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
 * @file: tests.cpp
 * Created on 18.07.2019
 * @author: queensgambit
 */

#include "tests.h"

#include <iostream>
#include "catch.hpp"
#include "../sfutil.h"
using namespace Catch::literals;
using namespace std;

TEST_CASE("En-passent moves") {
    vector<string> en_passent_moves;
    fill_en_passent_moves(en_passent_moves);

    for (auto uciMove: en_passent_moves) {
        bool returnVal = is_en_passent_candidate(get_origin_square(uciMove), get_destination_square(uciMove));
        if (!returnVal) {
            cerr << "uciMove: " << uciMove << " returned false!" << endl;
        }
        REQUIRE(returnVal == true);
    }
}

