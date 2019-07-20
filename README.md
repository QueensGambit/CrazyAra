# CrazyAra - A Deep Learning Chess Variant Engine


![RC_GUI WINDOWS](media/CrazyAra_Logo.png "rc")

This repository contains the source code of the engine search written in C++ based which is based on the previous [Python version](https://github.com/QueensGambit/CrazyAra).
The training scripts, preprocessing and neural network definition source files can be found in the [Python version](https://github.com/QueensGambit/CrazyAra).

### Steps to compile

Download and install the blaze library
* https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation

Build the MXNet C++ package
* https://mxnet.incubator.apache.org/versions/master/api/c++/index.html
```make -j USE_CPP_PACKAGE=1 USE_OPENCV=0 USE_MKL=1```
* _because the current dev branch broke the loading of certain models download release [MXNet-release 1.4.1](https://github.com/apache/incubator-mxnet/releases) instead_

Download & install yaml-cpp 
* https://github.com/jbeder/yaml-cpp

or 
* sudo apt-get install libyaml-cpp-dev

### Main libraries used in this project

* [Multi Variant Stockfish](https://github.com/QueensGambit/Stockfish): Stockfish fork specialized to play chess and some chess variants
	* Used for move generation and board representation as a replacement for [python-chess](https://github.com/niklasf/python-chess).
* [MXNet C++ Package](https://github.com/apache/incubator-mxnet/tree/master/cpp-package): A flexible and efficient library for deep learning
	* Used as the deep learning backend for loading and inference of the trained neural network as a replacment for the [MXNet python package](https://pypi.org/project/mxnet/)
* [Blaze](https://bitbucket.org/blaze-lib/blaze/src/master/): An open-source, high-performance C++ math library for dense and sparse arithmetic
	* Used for arethmic, numerical vector operation within the MCTS search as a replacement for [NumPy](https://numpy.org/)
* [Catch2](https://github.com/catchorg/Catch2): A multi-paradigm test framework for C++
	* Used as the testing framework as a replacmenet for [Python's unittest framework](https://docs.python.org/3/library/unittest.html)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp): (A YAML parser and emitter in C++)
	* Used for loading the .yaml configuration file

### Performance Profiling 

Install the plotting utility for gprof:
* https://github.com/jrfonseca/gprof2dot

Activate the -pg flags in `CMakeLists.txt` and rebuild.
Run the executable and generate the plot:
```
$ ./CrazyAraMCTS
$ gprof CrazyAraMCTS | gprof2dot | dot -Tpng -o output.png
```
