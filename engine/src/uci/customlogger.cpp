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
 * @file: logger.cpp
 * Created on 18.04.2021
 * @author: queensgambit
 */

#include "customlogger.h"

CustomLogger::CustomLogger() :
    in(cin.rdbuf(), file.rdbuf()),
    out(cout.rdbuf(), file.rdbuf()),
    err(cerr.rdbuf(), file.rdbuf())
{}


CustomLogger::~CustomLogger()
{
    start("", ifstream::app);
}


void CustomLogger::start(const std::string& filePath, ios_base::openmode writeMode)
{
  static CustomLogger logger;

  if (!filePath.empty() && !logger.file.is_open()) {
      logger.file.open(filePath, writeMode);

      if (!logger.file.is_open()) {
          cerr << "Unable to open debug log file " << filePath.c_str() << endl;
          exit(EXIT_FAILURE);
      }
      cin.rdbuf(&logger.in);
      cout.rdbuf(&logger.out);
      cerr.rdbuf(&logger.err);
  }
  else if (filePath.empty() && logger.file.is_open()) {
      cin.rdbuf(logger.in.buf);
      cout.rdbuf(logger.out.buf);
      cerr.rdbuf(logger.err.buf);
      logger.file.close();
  }
}
