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
 * @file: logger.h
 * Created on 18.04.2021
 * @author: queensgambit
 *
 * Extended Logger of 3rdparty/Stockfish/src/misc.cpp which allows logging in different modes of _Ios_Openmode
 * and also logging of stderr to file.
 */

#ifndef CUSTOMLOGGER_H
#define CUSTOMLOGGER_H

#include <fstream>
#include <iostream>

using namespace std;


// ---------------- struct is from 3rdparty/Stockfish/src/misc.cpp ---------------------
/// Our fancy logging facility. The trick here is to replace cin.rdbuf() and
/// cout.rdbuf() with two Tie objects that tie cin and cout to a file stream. We
/// can toggle the logging of std::cout and std:cin at runtime whilst preserving
/// usual I/O functionality, all without changing a single line of code!
/// Idea from http://groups.google.com/group/comp.lang.c++/msg/1d941c0f26ea0d81

struct Tie: public streambuf { // MSVC requires split streambuf for cin and cout

  Tie(streambuf* b, streambuf* l) : buf(b), logBuf(l) {}

  int sync() override { return logBuf->pubsync(), buf->pubsync(); }
  int overflow(int c) override { return log(buf->sputc((char)c), "<< "); }
  int underflow() override { return buf->sgetc(); }
  int uflow() override { return log(buf->sbumpc(), ">> "); }

  streambuf *buf, *logBuf;

  int log(int c, const char* prefix) {

    static int last = '\n'; // Single log file

    if (last == '\n')
        logBuf->sputn(prefix, 3);

    return last = logBuf->sputc((char)c);
  }
};
// -------------------------------------------------------------------------------------

/**
 * @brief The customLogger class prints all stdin, stdout, stderr to a file.
 * Its write mode, e.g. "w" or "a", can be adjusted.
 */
class CustomLogger {

  CustomLogger();
 ~CustomLogger();

  ofstream file;
  Tie in;
  Tie out;
  Tie err;

public:
  /**
   * @brief start Starts a logger instance.
   * @param filePath File path where the logging will be written to
   * @param writeMode ifstream::out for write mode ("w"), ifstream::app for append mode ("a")
   */
  static void start(const std::string& fileName, ios_base::openmode writeMode);
};

#endif // CUSTOMLOGGER_H
