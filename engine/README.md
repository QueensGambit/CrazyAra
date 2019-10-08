## C++ search engine

This directory contains the source code of the Monte-Carlo tree search.
The executable can be build without any python dependencies when `USE_RL` is disabled.

The file `CMakeLists.txt` provides different build options.
The engine is build with either CPU or GPU support, depending on the linked MXNet C++ library.
