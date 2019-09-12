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
 * @file: traindata.cpp
 * Created on 11.09.2019
 * @author: queensgambit
 *
 * Please describe what the content of this file is about
 */

#include "traindata.h"

Exporter::Exporter()
{
    fout = gzopen(filename.c_str(), "wb");
}

int Exporter::export_training_sample(const TrainDataExport &trainData)
{
    auto bytes_written =
        gzwrite(fout, reinterpret_cast<const char*>(&trainData), sizeof(trainData));
    if (bytes_written != sizeof(trainData)) {
      cerr << "Unable to write into " + filename << endl;
      return -1;
    }
    return 0;
}

void Exporter::close_gz()
{
    gzclose(fout);
}
