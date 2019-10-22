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
 * @file: tournamentresult.h
 * Created on 22.10.2019
 * @author: queensgambit
 *
 * The TournamentResult struct stores the result of an arena tournament.
 * Arena tournament can be used to track the elo progress in self play mode and to decide if the current
 * NN producer weights shall be switched.
 */

#ifndef TOURNAMENTRESULT_H
#define TOURNAMENTRESULT_H

#include <cstddef>
#include <iostream>

struct TournamentResult {
    size_t numberWins;
    size_t numberDraws;
    size_t numberLosses;

    TournamentResult();

    /**
     * @brief numberGames Computes the number of total games in the tournament
     * @return number of games
     */
    size_t numberGames() const;

    /**
     * @brief score Computes the score with respect to the first player.
     * Score is a value in [0.0f, 1.0f] where 0.0f relates to 100% losses and 1.0f to 100% wins
     * @return score value
     */
     float score() const;
};

/**
 * @brief operator << Returns ostream for trounament result summary in the form
 *  "<NUMBER_WINS> - <NUMBER_DRAWS> - <NUMBER_LOSSES> [<SCORE>]"
 * @param os ostream
 * @param result Tournament result to print
 * @return osream
 */
extern std::ostream& operator<<(std::ostream& os, const TournamentResult& result);


#endif // TOURNAMENTRESULT_H
