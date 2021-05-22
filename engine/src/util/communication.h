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
 * @file: communication.h
 * Created on 12.11.2019
 * @author: queensgambit
 *
 * This document contains UCI specifc communication methods which start with UCI keywords.
 */

#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <iostream>
#include <chrono>
using namespace std;

/**
 * @brief info_string Prints a given string message to std-out in accordance with the UCI-protocol.
 * @param message String message to print
 */
template<typename T>
void info_string(const T &message) {
#ifndef DISABLE_UCI_INFO
    cout << "info string " << message << endl;
#endif
}

/**
 * @brief info_string Prints a combined message based on two arguments to std-out in accordance with the UCI-protocol.
 * @param message First message object
 * @param message Second message object
 */
template<typename T, typename U>
void info_string(const T &messageA, const U &messageB) {
#ifndef DISABLE_UCI_INFO
    cout << "info string " << messageA << ' ' << messageB << endl;
#endif
}
template<typename T, typename U, typename V>
void info_string(const T &messageA, const U &messageB, const V &messageC) {
#ifndef DISABLE_UCI_INFO
    cout << "info string " << messageA << ' ' << messageB << ' ' << messageC << endl;
#endif
}


/**
 * @brief info_string Prints a given string message to std-out in accordance with the UCI-protocol.
 * This info message can not be supressed by DISABLE_UCI_INFO
 * @param message String message to print
 */
template<typename T>
void info_string_important(const T &message) {
    cout << "info string " << message << endl;
}
template<typename T, typename U>
void info_string_important(const T &messageA, const U &messageB) {
    cout << "info string " << messageA << ' ' << messageB << endl;
}
template<typename T, typename U, typename V>
void info_string_important(const T &messageA, const U &messageB, const V &messageC) {
    cout << "info string " << messageA << ' ' << messageB << ' ' << messageC << endl;
}
template<typename T, typename U, typename V, typename W>
void info_string_important(const T &messageA, const U &messageB, const V &messageC, const W &messageD) {
    cout << "info string " << messageA << ' ' << messageB << ' ' << messageC << ' ' << messageD << endl;
}

template<typename T>
void info_msg(const T &message, bool endl=false) {
#ifndef DISABLE_UCI_INFO
    cout << "info " << message;
    if (endl) {
        cout << endl;
    }
#endif
}

template<typename T>
void info_bestmove(const T &message) {
#ifndef DISABLE_UCI_INFO
    cout << "bestmove " << message << endl;
#endif
}

template<typename T>
void info_elapsed_time(const T &message, const chrono::steady_clock::time_point& begin, const chrono::steady_clock::time_point& end) {
#ifndef DISABLE_UCI_INFO
    const int elapsedMilliSeconds = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
    const int elapsedSeconds = chrono::duration_cast<chrono::seconds>(end - begin).count();
    const int elapsedMinutes = chrono::duration_cast<chrono::minutes>(end - begin).count();
    cout << "info string " << message << ' '
         << elapsedMinutes << "[min] " << elapsedSeconds % 60 << "." << elapsedMilliSeconds % 1000 << "[s]" << endl;
#endif
}

#endif // COMMUNICATION_H
