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
  GNU General Public License fåor more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: version.h
 * Created on 30.06.2021
 * @author: queensgambit
 *
 * Version specifier as a single uint_fast32_t.
 */

#ifndef VERSION_H
#define VERSION_H

#include <memory>
#include <string>

typedef uint_fast32_t Version;
typedef uint_fast8_t VersionType;

namespace version {
#define VERSION_SEP_MAJOR 1000000
#define VERSION_SEP_MINOR 1000
const std::string SEPERATOR = ".";

inline constexpr VersionType get_major(Version version) {
    return version / VERSION_SEP_MAJOR;
}
inline constexpr VersionType get_minor(Version version) {
    return (version % VERSION_SEP_MAJOR) / VERSION_SEP_MINOR;
}
inline constexpr VersionType get_patch(Version version) {
    return version % VERSION_SEP_MINOR;
}
}

inline constexpr Version make_version(VersionType major, VersionType minor, VersionType patch) {
    return major * VERSION_SEP_MAJOR + minor * VERSION_SEP_MINOR + patch;
}

template <VersionType major, VersionType minor, VersionType patch>
inline constexpr Version make_version() {
      return make_version(major, minor, patch);
}

inline std::string version_to_string(Version version) {
  return std::to_string(version::get_major(version)) + version::SEPERATOR
         + std::to_string(version::get_minor(version)) + version::SEPERATOR
         + std::to_string(version::get_patch(version));
}

#endif // VERSION_H
