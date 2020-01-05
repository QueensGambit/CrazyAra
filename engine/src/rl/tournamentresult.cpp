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
 * @file: tournamentresult.cpp
 * Created on 22.10.2019
 * @author: queensgambit
 */

#include "tournamentresult.h"
#include <iomanip>

TournamentResult::TournamentResult() :
    numberWins(0),
    numberDraws(0),
    numberLosses(0)
{
}

size_t TournamentResult::numberGames() const
{
    return numberWins + numberDraws + numberLosses;
}

float TournamentResult::score() const

{
    return (numberWins + numberDraws * 0.5f)/ numberGames();
}

std::ostream &operator<<(std::ostream &os, const TournamentResult &result)
{
    os << result.playerA << "-" << result.playerB << ": " << result.numberWins
       << " - " << result.numberDraws << " - " << result.numberLosses << " [" <<
          std::setprecision(2) << result.score() << "]";
    return os;
}

void write_tournament_result_to_csv(const TournamentResult &result, const string &csvFileName)
{
    ofstream csvFile;
    const char delim = ',';
    csvFile.open(csvFileName, std::ios_base::app);
    csvFile << result.playerA << delim << result.playerB << delim
         << result.numberWins << delim << result.numberDraws << delim << result.numberLosses << endl;
    csvFile.close();
}
