# CrazyAra - A Deep Learning Chess Variant Engine
![CRAZYARA_LOGO](media/crazyara_logo_medium.png "rc")

## Contents
* [Description](#description)
* [Links](#links)
* [Download](#download)
    * [Binaries](#binaries)
    * [Models](#models)
* [Variants](#variants)
* [Compilation](#compilation)
    * [Linux](#linux)
    * [Windows](#windows)
* [Libraries](#libraries)
* [Licence](#licence)
* [Publication](#publication)

<img align="right" src="media/TU_logo.png" width="128">

## Description

[_CrazyAra_](https://crazyara.org/) is an open-source neural network based engine, initially developed in pure python by [Johannes Czech](https://github.com/QueensGambit), [Moritz Willig](https://github.com/MoritzWillig) and Alena Beyer in 2018.
It started as a semester project at the [TU Darmstadt](https://www.tu-darmstadt.de/index.en.jsp) with the goal to train a neural network to play the chess variant [crazyhouse](https://en.wikipedia.org/wiki/Crazyhouse) via supervised learning on human data.
The project was part of the course [_"Deep Learning: Architectures & Methods"_](https://piazza.com/tu-darmstadt.de/summer2019/20001034iv/home) held by [Kristian Kersting](https://ml-research.github.io/people/kkersting/index.html), [Johannes Fürnkranz](http://www.ke.tu-darmstadt.de/staff/juffi) et al. in summer 2018.

The development was continued and the engine ported to C++ by [Johannes Czech](https://github.com/QueensGambit). In the course of a master thesis supervised by [Karl Stelzner](https://ml-research.github.io/people/kstelzner/) and [Kristian Kersting](https://ml-research.github.io/people/kkersting/index.html), the engine will learn crazyhouse in a reinforcement learning setting and learn to play other chess variants including classical chess.

The project is mainly inspired by the techniques described in the [Alpha-(Go)-Zero papers](https://arxiv.org/abs/1712.01815) by [David Silver](https://arxiv.org/search/cs?searchtype=author&query=Silver%2C+D), [Thomas Hubert](https://arxiv.org/search/cs?searchtype=author&query=Hubert%2C+T), [Julian Schrittwieser](https://arxiv.org/search/cs?searchtype=author&query=Schrittwieser%2C+J), [Ioannis Antonoglou](https://arxiv.org/search/cs?searchtype=author&query=Antonoglou%2C+I), [Matthew Lai](https://arxiv.org/search/cs?searchtype=author&query=Lai%2C+M), [Arthur Guez](https://arxiv.org/search/cs?searchtype=author&query=Guez%2C+A), [Marc Lanctot](https://arxiv.org/search/cs?searchtype=author&query=Lanctot%2C+M), [Laurent Sifre](https://arxiv.org/search/cs?searchtype=author&query=Sifre%2C+L), [Dharshan Kumaran](https://arxiv.org/search/cs?searchtype=author&query=Kumaran%2C+D), [Thore Graepel](https://arxiv.org/search/cs?searchtype=author&query=Graepel%2C+T), [Timothy Lillicrap](https://arxiv.org/search/cs?searchtype=author&query=Lillicrap%2C+T), [Karen Simonyan](https://arxiv.org/search/cs?searchtype=author&query=Simonyan%2C+K), [Demis Hassabis](https://arxiv.org/search/cs?searchtype=author&query=Hassabis%2C+D).

This repository contains the source code of the engine search written in C++ based on the previous [Python version](https://github.com/QueensGambit/CrazyAra).
The training scripts, preprocessing and neural network definition source files can be found in the [Python version](https://github.com/QueensGambit/CrazyAra).

_CrazyAra_ is an UCI chess engine and requires a GUI (e.g. [Cute Chess](https://github.com/cutechess/cutechess), [XBoard](https://www.gnu.org/software/xboard/), [WinBoard](http://hgm.nubati.net/)) for convinient usage.

## Links
* [:snake: Python version](https://github.com/QueensGambit/CrazyAra/)
* :notebook_with_decorative_cover: [CrazyAra paper](http://arxiv.org/abs/1908.06660)
* [:earth_africa: Project website](https://crazyara.org/)
* [♞ CrazyAra@lichess.org](https://lichess.org/@/CrazyAra)
* [♞ CrazyAraFish@lichess.org](https://lichess.org/@/CrazyAraFish)

## Download

### Binaries

We provide binary releases for the following plattforms:

Operating System | Backend | Compatible with
--- | --- | ---
Linux | [**CUDA 10.0, cuDNN v7.5.1.10, openBlas**](https://github.com/QueensGambit/CrazyAra-Engine/releases/download/0.6.0/CrazyAra_0.6.0_Linux_CUDA.zip) | NVIDIA GPUs and CPU
Linux | [**Intel MKL**](https://github.com/QueensGambit/CrazyAra-Engine/releases/download/0.6.0/CrazyAra_0.6.0_Linux_MKL.zip) | Intel CPUs
Windows | [**CUDA 10.1, cuDNN v7.5.1.10, openBlas**](https://github.com/QueensGambit/CrazyAra-Engine/releases/download/0.6.0/CrazyAra_0.6.0_Win_CUDA.zip) | NVIDIA GPUs and CPU
Windows | [**Intel MKL**](https://github.com/QueensGambit/CrazyAra-Engine/releases/download/0.6.0/CrazyAra_0.6.0_Win_MKL.zip) | Intel CPUs

_For Intel MKL the network inference is optimized when defining the following environment variable:_
```
MXNET_SUBGRAPH_BACKEND=MKLDNN
```

### Models

The following models are freely available for download:
* [4-value-8-policy](https://github.com/QueensGambit/CrazyAra-Engine/releases/download/0.6.0/4-value-8-policy.zip)
* [8-value-16-policy](https://github.com/QueensGambit/CrazyAra-Engine/releases/download/0.6.0/8-value-16-policy.zip)
* [8-value-policy-map](https://github.com/QueensGambit/CrazyAra-Engine/releases/download/0.6.0/4-value-8-policy.zip)
* [8-value-policy-map-mobile / RISEv2-mobile](https://github.com/QueensGambit/CrazyAra-Engine/releases/download/0.6.0/RISEv2-mobile.zip)
* [8-value-policy-map-preAct-relu+bn](https://github.com/QueensGambit/CrazyAra-Engine/releases/download/0.6.0/RISEv2-mobile.zip)
* [RISEv1 (CrazyAraFish weights)](https://github.com/QueensGambit/CrazyAra-Engine/releases/download/0.6.0/CrazyAraFish_RISEv1.zip)

The extracted model should be placed in the same directory as the engine executable.
The directory can be changed by adjusting the UCI-parameter `Model_Directory`.
Each model is compatible with all executables.

More information about the different models can be found in the [wiki](https://github.com/QueensGambit/CrazyAra-Engine/wiki/Model-description).

## Variants
The current list of available chess variants include:
* [Crazyhouse](https://lichess.org/variant/crazyhouse)

## Compilation

Please follow these steps to build _CrazyAra_ from source:

### Linux

1. Clone the _CrazyAra_ repository:

```$ git clone https://github.com/QueensGambit/CrazyAra-Engine.git --recursive```

2. Download and install the [**Blaze**](https://bitbucket.org/blaze-lib/blaze/src/master/) library of version **>=3.6** or current master:
	* https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation
	* https://bitbucket.org/blaze-lib/blaze/downloads/



3. Build the MXNet C++ package (e.g. IntelMKL). Building with OpenCV is not required:

   ```$ make -j USE_CPP_PACKAGE=1 USE_OPENCV=0 USE_MKL=1```
   
   Detailed build instruction can be found here:
   	* https://mxnet.incubator.apache.org/versions/master/api/c++/index.html

4. Build the _CrazyAra_ binary
```
$ export MXNET_PATH=<path_to_mxnet>/incubator-mxnet/
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
```

### Windows
Instructions can be found in the [wiki](https://github.com/QueensGambit/CrazyAra-Engine/wiki/Compile-instructions-for-Windows).

## Libraries
The following libraries are used to run _CrazyAra_:

* [**Multi Variant Stockfish**](https://github.com/ddugovic/Stockfish): Stockfish fork specialized to play chess and some chess variants
	* Used for move generation and board representation as a replacement for [python-chess](https://github.com/niklasf/python-chess).
* [**MXNet C++ Package**](https://github.com/apache/incubator-mxnet/tree/master/cpp-package): A flexible and efficient library for deep learning
	* Used as the deep learning backend for loading and inference of the trained neural network
* [**Blaze**](https://bitbucket.org/blaze-lib/blaze/src/master/): An open-source, high-performance C++ math library for dense and sparse arithmetic
	* Used for arithmetic, numerical vector operation within the MCTS search as a replacement for [NumPy](https://numpy.org/)
* [**Catch2**](https://github.com/catchorg/Catch2): A multi-paradigm test framework for C++
	* Used as the testing framework as a replacmenet for [Python's unittest framework](https://docs.python.org/3/library/unittest.html)

## Licence

_CrazyAra_ is free software, and distributed under the terms of the [GNU General Public License version 3 (GPL v3)](https://www.gnu.org/licenses/gpl-3.0.en.html).

For details about the GPL v3 license, refer to the file [LICENSE](https://github.com/QueensGambit/CrazyAraMCTS/blob/master/LICENSE).

## Publication
* J. Czech, M. Willig, A. Beyer, K. Kersting and J. Fürnkranz: **Learning to play the Chess Variant Crazyhouse above World Champion Level with Deep Neural Networks and Human Data**, [preprint](http://arxiv.org/abs/1908.06660)
```
@article{czech_learning_2019,
	title = {Learning to play the Chess Variant Crazyhouse above World Champion Level with Deep Neural Networks and Human Data},
	author = {Czech, Johannes and Willig, Moritz and Beyer, Alena and Kersting, Kristian and Fürnkranz, Johannes},
	year = {2019},
	pages = {35}
}
```

