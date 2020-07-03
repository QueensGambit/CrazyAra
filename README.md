
<div id="crazyara-logo" align="center">
    <br/>
    <img src="./etc/media/crazyara_logo_medium.png" alt="CrazyAra Logo" width="512"/>
    <h3>A Deep Learning Chess Variant Engine</h3>
</div>

<div id="badges" align="center">

[![Travis Build Status](https://travis-ci.org/queensgambit/CrazyAra.svg?branch=master)](https://travis-ci.org/queensgambit/CrazyAra)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
&nbsp; 
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/39c3329d0cea4186b5e4d32cfb6a4d5d)](https://www.codacy.com/manual/QueensGambit/CrazyAra?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=QueensGambit/CrazyAra&amp;utm_campaign=Badge_Grade)
[![ArXiv Badge](https://img.shields.io/badge/Paper-arXiv-blue.svg)](https://arxiv.org/abs/1908.06660)
[![Thesis Badge](https://img.shields.io/badge/Thesis-M.Sc.-orange.svg)](https://ml-research.github.io/papers/czech2019deep.pdf)
[![Journal Badge](https://img.shields.io/badge/Journal-Frontiers-green.svg)](https://www.frontiersin.org/articles/10.3389/frai.2020.00024/full)


</div>

## Contents
*   [Description](#description)
*   [Links](#links)
*   [Download](#download)
    *   [Binaries](#binaries)
    *   [Models](#models)
*   [Variants](#variants)
*   [Documentation](#documentation)
*   [Compilation](#compilation)
    *   [Linux](#linux)
    *   [Windows](#windows)
*   [Rating](#rating)
*   [Libraries](#libraries)
*   [Players](#players)
*   [Related](#related)
*   [Licence](#licence)
*   [Publications](#publications)

<img align="right" src="etc/media/TU_logo.png" width="128">

## Description

[_CrazyAra_](https://crazyara.org/) is an open-source neural network chess variant engine, initially developed in pure python by [Johannes Czech](https://github.com/QueensGambit), [Moritz Willig](https://github.com/MoritzWillig) and Alena Beyer in 2018.
It started as a semester project at the [TU Darmstadt](https://www.tu-darmstadt.de/index.en.jsp) with the goal to train a neural network to play the chess variant [crazyhouse](https://en.wikipedia.org/wiki/Crazyhouse) via supervised learning on human data.
The project was part of the course [_"Deep Learning: Architectures & Methods"_](https://piazza.com/tu-darmstadt.de/summer2019/20001034iv/home) held by [Kristian Kersting](https://ml-research.github.io/people/kkersting/index.html), [Johannes Fürnkranz](http://www.ke.tu-darmstadt.de/staff/juffi) et al. in summer 2018.

The development was continued and the engine ported to C++ by [Johannes Czech](https://github.com/QueensGambit). In the course of a master thesis supervised by [Karl Stelzner](https://ml-research.github.io/people/kstelzner/) and [Kristian Kersting](https://ml-research.github.io/people/kkersting/index.html), the engine learned crazyhouse in a reinforcement learning setting and was trained on other chess variants including chess960, King of the Hill and Three-Check.

The project is mainly inspired by the techniques described in the [Alpha-(Go)-Zero papers](https://arxiv.org/abs/1712.01815) by [David Silver](https://arxiv.org/search/cs?searchtype=author&query=Silver%2C+D), [Thomas Hubert](https://arxiv.org/search/cs?searchtype=author&query=Hubert%2C+T), [Julian Schrittwieser](https://arxiv.org/search/cs?searchtype=author&query=Schrittwieser%2C+J), [Ioannis Antonoglou](https://arxiv.org/search/cs?searchtype=author&query=Antonoglou%2C+I), [Matthew Lai](https://arxiv.org/search/cs?searchtype=author&query=Lai%2C+M), [Arthur Guez](https://arxiv.org/search/cs?searchtype=author&query=Guez%2C+A), [Marc Lanctot](https://arxiv.org/search/cs?searchtype=author&query=Lanctot%2C+M), [Laurent Sifre](https://arxiv.org/search/cs?searchtype=author&query=Sifre%2C+L), [Dharshan Kumaran](https://arxiv.org/search/cs?searchtype=author&query=Kumaran%2C+D), [Thore Graepel](https://arxiv.org/search/cs?searchtype=author&query=Graepel%2C+T), [Timothy Lillicrap](https://arxiv.org/search/cs?searchtype=author&query=Lillicrap%2C+T), [Karen Simonyan](https://arxiv.org/search/cs?searchtype=author&query=Simonyan%2C+K), [Demis Hassabis](https://arxiv.org/search/cs?searchtype=author&query=Hassabis%2C+D).

The training scripts, preprocessing and neural network definition source files are written in python and located at [DeepCrazyhouse/src](https://github.com/QueensGambit/CrazyAra/tree/master/DeepCrazyhouse/src).
There are two version of the search engine available:
The initial version is written in python and located at [DeepCrazyhouse/src/domain/agent](https://github.com/QueensGambit/CrazyAra/tree/master/DeepCrazyhouse/src/domain/agent).
The newer version is written in C++ and located at [engine/src](https://github.com/QueensGambit/CrazyAra/tree/master/engine/src).

_CrazyAra_ is an UCI chess engine and requires a GUI (e.g. [Cute Chess](https://github.com/cutechess/cutechess), [XBoard](https://www.gnu.org/software/xboard/), [WinBoard](http://hgm.nubati.net/)) for convinient usage.

## Links
*   [:fire: C++ engine](engine/src)
*   [:snake: Python engine](https://github.com/QueensGambit/CrazyAra/tree/master/DeepCrazyhouse/src/domain/agent)
*   :notebook_with_decorative_cover: [CrazyAra paper](https://arxiv.org/abs/1908.06660)
*   :orange_book: [Master thesis](https://ml-research.github.io/papers/czech2019deep.pdf)
*   [:earth_africa: Project website](https://crazyara.org/)
*   [♞ CrazyAra@lichess.org](https://lichess.org/@/CrazyAra)
*   [♞ CrazyAraFish@lichess.org](https://lichess.org/@/CrazyAraFish)
*   [:cyclone: Neural network](https://github.com/QueensGambit/CrazyAra/tree/master/DeepCrazyhouse/src/domain/neural_net/architectures)
*   [:wrench: Supervised learning](https://github.com/QueensGambit/CrazyAra/tree/master/DeepCrazyhouse/src/training)
*   [:hammer_and_wrench: Reinforcement learning](https://github.com/QueensGambit/CrazyAra/tree/master/engine/src/rl)

## Download

### Binaries

We provide binary releases for the following plattforms:

Operating System | Backend                                                                                                                                                               | Compatible with
---              | ---                                                                                                                                                                   | --- 
Linux            | [**CUDA 10.2, cuDNN 7.6.5, TensorRT-7.0.0.11**](https://github.com/QueensGambit/CrazyAra/releases/download/0.8.0/CrazyAra_0.8.0_Linux_TensorRT.zip) | NVIDIA GPUs
Linux            | [**Intel MKL 20190502**](https://github.com/QueensGambit/CrazyAra/releases/download/0.8.0/CrazyAra_0.8.0_Linux_MKL.zip)                                               | Intel CPUs
Windows          | [**CUDA 10.2, cuDNN 7.6.5, TensorRT-7.0.0.11**](https://github.com/QueensGambit/CrazyAra/releases/download/0.8.0/CrazyAra_0.8.0_Win_TensorRT.zip)                              | NVIDIA GPUs
Windows          | [**Intel MKL 20190502**](https://github.com/QueensGambit/CrazyAra/releases/download/0.8.0/CrazyAra_0.8.0_Win_MKL.zip)                                                 | Intel CPUs
Mac              | [**Apple Accelerate**](https://github.com/QueensGambit/CrazyAra/releases/download/0.8.0/CrazyAra_0.8.0_Mac_CPU.zip) | Mac-Books

The current _CrazyAra_ release and all its previous versions can also be found at [releases](https://github.com/QueensGambit/CrazyAra/releases).

### Models

The extracted model should be placed in the same directory as the engine executable.
The directory can be changed by adjusting the UCI-parameter `Model_Directory`.
Model-0S-96 is included in [**release 0.8.0**](https://github.com/QueensGambit/CrazyAra/releases/tag/0.8.0).

The following models are available for download:

#### Release 0.7.0

For release 0.7.0, a pre-initialized model on human games was updated by applying reinforcement learning.

The 45th model update was obtained after generating ~ 1.05 million self-play games. It was also used for the [100 evaluation games](https://github.com/QueensGambit/CrazyAra/wiki/v0.7.0) with _Multi-Variant-Stockfish_:
*   [model-os-45-risev2.zip](https://github.com/QueensGambit/CrazyAra/releases/download/0.7.0/model-os-45-risev2.zip)

The reinforcement learning loop was then continued until a total of ~ 2.37 million self-play games, resulting in the 96th model update:
*   [model-os-96-risev2.zip](https://github.com/QueensGambit/CrazyAra/releases/download/0.7.0/model-os-96-risev2.zip)

#### Release 0.6.0

For release 0.6.0 and all previous releases, models were solely based on supervised learning on the lichess.org data set as described in our [paper](https://arxiv.org/abs/1908.06660).

*   [4-value-8-policy](https://github.com/QueensGambit/CrazyAra/releases/download/0.6.0/4-value-8-policy.zip)
*   [8-value-16-policy](https://github.com/QueensGambit/CrazyAra/releases/download/0.6.0/8-value-16-policy.zip)
*   [8-value-policy-map](https://github.com/QueensGambit/CrazyAra/releases/download/0.6.0/4-value-8-policy.zip)
*   [8-value-policy-map-mobile / RISEv2-mobile](https://github.com/QueensGambit/CrazyAra/releases/download/0.6.0/RISEv2-mobile.zip)
*   [8-value-policy-map-preAct-relu+bn](https://github.com/QueensGambit/CrazyAra/releases/download/0.6.0/RISEv2-mobile.zip)
*   [RISEv1 (CrazyAraFish weights)](https://github.com/QueensGambit/CrazyAra/releases/download/0.6.0/CrazyAraFish_RISEv1.zip)

More information about the different models can be found in the [wiki](https://github.com/QueensGambit/CrazyAra-Engine/wiki/Model-description).

## Variants
The current list of available chess variants include:
*   [Crazyhouse](https://lichess.org/variant/crazyhouse)

## Documentation
For more details about the initial python version visit the wiki pages:
* [Introduction](https://github.com/QueensGambit/CrazyAra/wiki)
* [Installation guide for python MCTS](https://github.com/QueensGambit/CrazyAra/wiki/Installation-Guide)
* [Supervised-training](https://github.com/QueensGambit/CrazyAra/wiki/Supervised-training)
* [Model architecture](https://github.com/QueensGambit/CrazyAra/wiki/Model-architecture)
* [Input representation](https://github.com/QueensGambit/CrazyAra/wiki/Input-representation)
* [Output representation](https://github.com/QueensGambit/CrazyAra/wiki/Output-representation)
* [Network visualization](https://github.com/QueensGambit/CrazyAra/wiki/Network-visualization)
* [Engine settings](https://github.com/QueensGambit/CrazyAra/wiki/Engine-settings)
* [Programmer's guide](https://github.com/QueensGambit/CrazyAra/wiki/Programmer's-guide)
* [FAQ](https://github.com/QueensGambit/CrazyAra/wiki/FAQ)
* [Stockfish-10: Crazyhouse-Self-Play](https://github.com/QueensGambit/CrazyAra/wiki/Stockfish-10:-Crazyhouse-Self-Play)

## Compilation

Please follow these steps to build _CrazyAra_ from source:

### Linux

1. Clone the _CrazyAra_ repository:

   ```bash
   git clone https://github.com/QueensGambit/CrazyAra.git --recursive
   ```

2. Download and install the [**Blaze**](https://bitbucket.org/blaze-lib/blaze/src/master/) library of version **>=3.6** or current master:
	*   <https://bitbucket.org/blaze-lib/blaze/downloads/>
	```bash
	tar -xvzf blaze-3.6.tar.gz
	cd blaze-3.6
	cmake -DCMAKE_INSTALL_PREFIX=/usr/local/
	sudo make install
	```
	Detailed build instruction can be found here:

    *   <https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation>


3. Build the MXNet C++ package. Building with OpenCV is not required:

   Install the prerequesites (e.g. for IntelMKL):
   * https://github.com/intel/mkl-dnn/releases
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential git
   sudo apt-get install -y libopenblas-dev liblapack-dev
   sudo apt-get install -y graphviz
   ```
   
   ```bash
   git clone https://github.com/apache/incubator-mxnet --recursive
   ```
   
   [IntelMKL](https://github.com/intel/mkl-dnn): ```make -j $(nproc) USE_CPP_PACKAGE=1 USE_OPENCV=0 USE_MKL=1```
   
   [CUDA](https://developer.nvidia.com/cuda-zone) / [cuDNN](https://developer.nvidia.com/cudnn): ```make -j $(nproc) USE_CPP_PACKAGE=1 USE_OPENCV=0 USE_MKL=0 USE_CUDA=1 USE_CUDNN=1```
   
   _You might have to reduce the number of jobs to run in paralell e.g. `-j 4` if you run out of memory (RAM) during building. Building the MXNet-C++ examples is not required._
   
   Detailed build instruction can be found here:
   *   <https://mxnet.incubator.apache.org/get_started/ubuntu_setup.html#build-mxnet-from-source>
   *   IntelMKL: <https://github.com/apache/incubator-mxnet/blob/master/docs/python_docs/python/tutorials/performance/backend/mkldnn/mkldnn_readme.md>

4. Build the _CrazyAra_ binary
   ```bash
   export MXNET_PATH=<path_to_mxnet>/incubator-mxnet/
   cd engine
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make
   ```

### Windows
Instructions can be found in the [wiki](https://github.com/QueensGambit/CrazyAra-Engine/wiki/Compile-instructions-for-Windows).

## Rating

The playing strength of _CrazyAra_ has been evaluated over the course of development.

### Playing strength of CrazyAra 0.3.1

_CrazyAra 0.3.1_ played multiple world champion Justin Tan (LM [JannLee](https://lichess.org/@/JannLee)) at 18:00 GMT
 on 21st December in five official matches and won 4-1.
You can find a detailed report about the past event published by [okei](https://lichess.org/@/okei) here:
* https://zhchess.blogspot.com/2018/12/crazyara-plays-jannlee-for-christmas.html

_CrazyAra 0.3.1_ was also put to the test against known crazyhouse engines:
* [Strength evaluation  v0.3.1](https://github.com/QueensGambit/CrazyAra/wiki/v0.3.1)

### Playing strength of CrazyAra 0.5.0

[Matuiss2](https://github.com/Matuiss2) generated 25 games (Intel i5 8600k, 5GHz) between _CrazyAra 0.3.1_ and _CrazyAra 0.5.0_:

```python
[TimeControl "40/300"]
Score of CrazyAra 0.5.0 vs CrazyAra 0.3.1: 22 - 3 - 0 [0.88]
Elo difference: 346 +/- NaN

25 of 25 games finished.
```

### Playing strength of CrazyAra 0.6.0

_CrazyAra 0.6.0_ is the first C++ release and has been evaluated in our [paper](https://arxiv.org/abs/1908.06660) (section **11 Experimental Evaluation**).


## Libraries

These libraries are used in the python version:

*   [**python-chess**](https://python-chess.readthedocs.io/en/latest/index.html): A pure Python chess library
*   [**MXNet**](https://mxnet.incubator.apache.org/): A flexible and efficient library for deep learning
*   [**numpy**](http://www.numpy.org/): The fundamental package for scientific computing with Python
*   [**Zarr**](https://zarr.readthedocs.io/en/stable/): An implementation of chunked, compressed, N-dimensional arrays

The following libraries are used to run the C++ version of _CrazyAra_:

*   [**Multi Variant Stockfish**](https://github.com/ddugovic/Stockfish): Stockfish fork specialized to play chess and some chess variants
    *   Used for move generation and board representation as a replacement for [python-chess](https://github.com/niklasf/python-chess).
*   [**MXNet C++ Package**](https://github.com/apache/incubator-mxnet/tree/master/cpp-package): A flexible and efficient library for deep learning
	*   Used as the deep learning backend for loading and inference of the trained neural network
*   [**Blaze**](https://bitbucket.org/blaze-lib/blaze/src/master/): An open-source, high-performance C++ math library for dense and sparse arithmetic
    *   Used for arithmetic, numerical vector operation within the MCTS search as a replacement for [NumPy](https://numpy.org/)
*   [**Catch2**](https://github.com/catchorg/Catch2): A multi-paradigm test framework for C++
	*   Used as the testing framework as a replacement for [Python's unittest framework](https://docs.python.org/3/library/unittest.html)

When _CrazyAra_ is also built with reinforcement learning functionality the following libraries are used:
*   [**z5**](https://github.com/constantinpape/z5): Lighweight C++ and Python interface for datasets in zarr and N5 format 
	*   Used for exporting generated self-play data in C++ in the **Zarr** data format
*   [**xtensor**](https://github.com/xtensor-stack/xtensor): C++ tensors with broadcasting and lazy computing
	*   Used as the internal matrix format within **z5**

## Players
_CrazyAra's_ knowledge in the game of crazhyouse for supervised neural networks is based on human played games of the
[lichess.org database](https://database.lichess.org/).

The most active players which influenced the play-style of CrazyAra the most are:
1. [**mathace**](https://lichess.org/@/mathace)
2. [**ciw**](https://lichess.org/@/ciw)
3. [**retardedplatypus123**](https://lichess.org/@/retardedplatypus123)
4. [**xuanet**](https://lichess.org/@/xuanet)
5. [**dovijanic**](https://lichess.org/@/dovijanic)
6. [KyleLegion](https://lichess.org/@/KyleLegion)
7. [LM JannLee](https://lichess.org/@/JannLee)
8. [crosky](https://lichess.org/@/crosky)
9. [mariorton](https://lichess.org/@/mariorton)
10. [IM opperwezen](https://lichess.org/@/opperwezen)

Please have a look at [Supervised training](https://github.com/QueensGambit/CrazyAra/wiki/Supervised-training) or our paper for more detailed information.

## Related

Similar open source neural network chess projects are listed below:

### chess-alpha-zero
In CrazyAra v.0.1.0 the Monte-Carlo-Tree-Search (MCTS) was imported and adapted from the following project: 
* https://github.com/Zeta36/chess-alpha-zero

For CrazyAra v.0.2.0 the MCTS was rewritten from scratch adding new functionality:
* Reusing the old search tree for future positions
* Node and child-nodes structure using numpy-arrays
* Always using mate-in-one connection if possible in the current search tree

### SixtyFour crazyhouse engine
* https://github.com/FTdiscovery/64CrazyhouseDeepLearning

### Leela-Chess-Zero chess engine
* http://lczero.org/
* https://github.com/LeelaChessZero/lc0

### Allie(Stein) chess engine
* https://github.com/manyoso/allie

### Scorpio chess engine 
* https://github.com/dshawul/Scorpio

## Research

The following is a collection of useful research links

AlphaGo Zero paper:
https://arxiv.org/pdf/1712.01815.pdf

Journal Nature:
https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf

CrazyAra paper:
https://arxiv.org/abs/1908.06660

SixtyFour engine paper:
https://arxiv.org/abs/1908.09296v1

Hyper-Parameter Sweep on AlphaZero General:
https://arxiv.org/abs/1903.08129

DeepMind Blogpost:
https://deepmind.com/blog/alphago-zero-learning-scratch/

How AlphaGo Zero works - Google DeepMind
https://www.youtube.com/watch?v=MgowR4pq3e8

Deep Mind's AlphaGo Zero - EXPLAINED
https://www.youtube.com/watch?v=NJBLx29JuHs

A Simple Alpha(Go) Zero Tutorial
https://web.stanford.edu/~surag/posts/alphazero.html

AlphaGo Zero - How and Why it Works:
http://tim.hibal.org/blog/alpha-zero-how-and-why-it-works/

Simple Chess AI implementation
https://github.com/mnahinkhan/Chess/blob/master/Chess/chess.py

## Licence

_CrazyAra_ is free software, and distributed under the terms of the [GNU General Public License version 3 (GPL v3)](https://www.gnu.org/licenses/gpl-3.0.en.html).
The source-code including all project files is licensed under the GPLv3-License if not stated otherwise.

For details about the GPL v3 license, refer to the file [LICENSE](https://github.com/QueensGambit/CrazyAra/blob/master/LICENSE).

## Publications
*   J. Czech, M. Willig, A. Beyer, K. Kersting and J. Fürnkranz: **Learning to play the Chess Variant Crazyhouse above World Champion Level with Deep Neural Networks and Human Data**, [link](https://www.frontiersin.org/article/10.3389/frai.2020.00024)
```latex
@ARTICLE{10.3389/frai.2020.00024,
	 AUTHOR={Czech, Johannes and Willig, Moritz and Beyer, Alena and Kersting, Kristian and Fürnkranz, Johannes},   
	 TITLE={Learning to Play the Chess Variant Crazyhouse Above World Champion Level With Deep Neural Networks and Human Data},      
	 JOURNAL={Frontiers in Artificial Intelligence},      
	 VOLUME={3},      
	 PAGES={24},     
	 YEAR={2020},      
	 URL={https://www.frontiersin.org/article/10.3389/frai.2020.00024},       
	 DOI={10.3389/frai.2020.00024},      
	 ISSN={2624-8212},   
	 ABSTRACT={Deep neural networks have been successfully applied in learning the board games Go, chess, and shogi without prior knowledge by making use of reinforcement learning. Although starting from zero knowledge has been shown to yield impressive results, it is associated with high computationally costs especially for complex games. With this paper, we present CrazyAra which is a neural network based engine solely trained in supervised manner for the chess variant crazyhouse. Crazyhouse is a game with a higher branching factor than chess and there is only limited data of lower quality available compared to AlphaGo. Therefore, we focus on improving efficiency in multiple aspects while relying on low computational resources. These improvements include modifications in the neural network design and training configuration, the introduction of a data normalization step and a more sample efficient Monte-Carlo tree search which has a lower chance to blunder. After training on 569537 human games for 1.5 days we achieve a move prediction accuracy of 60.4%. During development, versions of CrazyAra played professional human players. Most notably, CrazyAra achieved a four to one win over 2017 crazyhouse world champion Justin Tan (aka LM Jann Lee) who is more than 400 Elo higher rated compared to the average player in our training set. Furthermore, we test the playing strength of CrazyAra on CPU against all participants of the second Crazyhouse Computer Championships 2017, winning against twelve of the thirteen participants. Finally, for CrazyAraFish we continue training our model on generated engine games. In 10 long-time control matches playing Stockfish 10, CrazyAraFish wins three games and draws one out of 10 matches.}
}
```

* J. Czech: **Deep Reinforcement Learning for Crazyhouse**, [pdf](https://ml-research.github.io/papers/czech2019deep.pdf)
```latex
@mastersthesis{czech2019deep,
	       title = { Deep Reinforcement Learning for Crazyhouse },
	       author = { Johannes Czech },
               year = { 2019 },
               type = { M.Sc. },
	       crossref = { https://github.com/QueensGambit/CrazyAra },
	       school = { TU Darmstadt },
	       pages = { 54 },
	       month = { dec }
	       }
```
