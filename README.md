
# CrazyAra - Deep Learning for Crazyhouse <img src="https://raw.githubusercontent.com/QueensGambit/CrazyAra/master/etc/media/CrazyAra_Logo.png" width="64">


**The new C++ version for the MCTS can be found at: https://github.com/QueensGambit/CrazyAra-Engine**


                                      _                                                                      
                       _..           /   ._   _.  _        /\   ._   _.                                      
                     .' _ `\         \_  |   (_|  /_  \/  /--\  |   (_|                                      
                    /  /e)-,\                         /                                                      
                   /  |  ,_ |                                                                                
                  /   '-(-.)/       An open-source neural network based engine for the chess variant         
                .'--.   \  `        Crazyhouse. The project is mainly inspired by the techniques described   
               /    `\   |          in the Alpha-(Go)-Zero papers by Silver, Hubert, Schrittwieser et al.    
             /`       |  / /`\.-.   It started as a semester project at the Technische Universität Darmstadt 
           .'        ;  /  \_/__/   as part of the course "Deep Learning: Architectures & Methods" held by   
         .'`-'_     /_.'))).-` \    Prof. Kristian Kersting, Prof. Johannes Fürnkranz et al. in summer 2018. 
        / -'_.'---;`'-))).-'`\_/                                                                             
       (__.'/   /` .'`              Developers:           Johannes Czech, Moritz Willig, Alena Beyer  
        (_.'/ /` /`                 Source Code (GitHub): QueensGambit/CrazyAra-AI (GPLv3-License)            
          _|.' /`                   Website:              http://www.crazyara.org/                           
    jgs.-` __.'|                    Lichess:              https://lichess.org/@/CrazyAra                           
        .-'||  |                    ASCII-Art:            Joan G. Stark (http://www.oocities.org/spunk1111/)                  
           \_`/                     ASCII-Font:           Mini by Glenn Chappell 4/93, Paul Burton           
                                               
                                    
Official Repository of the Crazyhouse-Bot CrazyAra which is powered by a Deep Convolutional Neural Network and is compatible with the Universial-Chess-Interface (UCI).

## Installation Guide
Please follow the instructions in the wiki-page at:
* [Installation guide](https://github.com/QueensGambit/CrazyAra/wiki/Installation-Guide)

## Documentation
For more details about the training procedure visit the wiki pages:
* [Introduction](https://github.com/QueensGambit/CrazyAra/wiki)
* [Supervised-training](https://github.com/QueensGambit/CrazyAra/wiki/Supervised-training)
* [Model architecture](https://github.com/QueensGambit/CrazyAra/wiki/Model-architecture)
* [Input representation](https://github.com/QueensGambit/CrazyAra/wiki/Input-representation)
* [Output representation](https://github.com/QueensGambit/CrazyAra/wiki/Output-representation)
* [Network visualization](https://github.com/QueensGambit/CrazyAra/wiki/Network-visualization)
* [Engine settings](https://github.com/QueensGambit/CrazyAra/wiki/Engine-settings)
* [Programmer's guide](https://github.com/QueensGambit/CrazyAra/wiki/Programmer's-guide)
* [FAQ](https://github.com/QueensGambit/CrazyAra/wiki/FAQ)
* [Stockfish-10:-Crazyhouse-Self-Play](https://github.com/QueensGambit/CrazyAra/wiki/Stockfish-10:-Crazyhouse-Self-Play)

You can also find our original project proposal document as well as our presentation about CrazyAra 0.1:
* https://github.com/QueensGambit/CrazyAra/tree/master/etc/doc

## Links
* [:fire: C++ version](https://github.com/QueensGambit/CrazyAra-Engine/)
* :notebook_with_decorative_cover: [CrazyAra paper](https://arxiv.org/abs/1908.06660)
* [:earth_africa: Project website](https://crazyara.org/)
* [♞ CrazyAra@lichess.org](https://lichess.org/@/CrazyAra)
* [♞ CrazyAraFish@lichess.org](https://lichess.org/@/CrazyAraFish)

## Playing strength evaluation

### Strength of CrazyAra 0.3.1

CrazyAra 0.3.1 played multiple world champion Justin Tan (LM [JannLee](https://lichess.org/@/JannLee)) at 18:00 GMT
 on 21st December in five official matches and won 4-1.
You can find a detailed report about the past event published by [okei](https://lichess.org/@/okei) here:
* https://zhchess.blogspot.com/2018/12/crazyara-plays-jannlee-for-christmas.html

CrazyAra 0.3.1 was also put to the test against known crazyhouse engines:
* [Strength evaluation  v0.3.1](https://github.com/QueensGambit/CrazyAra/wiki/v0.3.1)

### Strength of CrazyAra 0.5.0

[Matuiss2](https://github.com/Matuiss2) generated 25 games (Intel i5 8600k, 5GHz) between CrazyAra 0.3.1 and CrazyAra 0.5.0:

```python
[TimeControl "40/300"]
Score of CrazyAra 0.5.0 vs CrazyAra 0.3.1: 22 - 3 - 0 [0.88]
Elo difference: 346 +/- NaN

25 of 25 games finished.
```

## License
This source-code including all project files is licensed under the GPLv3-License if not stated otherwise.

See [LICENSE](https://github.com/QueensGambit/CrazyAra/blob/master/LICENSE) for more details.

## Project Links:
* [Project website](http://www.crazyara.org/)
* [CrazyAra's lichess-org account](https://lichess.org/@/CrazyAra)
* [Project management plattform on taiga.io](https://tree.taiga.io/project/queensgambit-deep-learning-project-crazyhouse/)


## Main libraries used in this project
* [python-chess](https://python-chess.readthedocs.io/en/latest/index.html): A pure Python chess library
* [MXNet](https://mxnet.incubator.apache.org/): A flexible and efficient library for deep learning
* [numpy](http://www.numpy.org/): The fundamental package for scientific computing with Python
* [zarr](https://zarr.readthedocs.io/en/stable/): An implementation of chunked, compressed, N-dimensional arrays

## Human influence
CrazyAra's knowledge in the game of crazhyouse is only based on human played games of
[lichess.org database](https://database.lichess.org/).

The most active players which influence the playstyle of CrazyAra the most are:
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

Please have a look at [Supervised training](https://github.com/QueensGambit/CrazyAra/wiki/Supervised-training)
for more detailed information.

## Links to other similar projects

### chess-alpha-zero
In CrazyAra v.0.1.0 the Monte-Carlo-Tree-Search (MCTS) was imported and adapted from the following project: 
* https://github.com/Zeta36/chess-alpha-zero

For CrazyAra v.0.2.0 the MCTS was rewritten from scratch adding new functionality:
* Reusing the old search tree for future positions
* Node and child-nodes structure using numpy-arrays
* Always using mate-in-one connection if possible in the current search tree

### 64CrazyhouseDeepLearning
* https://github.com/FTdiscovery/64CrazyhouseDeepLearning

### Leela-Chess-Zero
* http://lczero.org/
* https://github.com/LeelaChessZero/lczero

## Research links:
AlphaGo Zero paper:
https://arxiv.org/pdf/1712.01815.pdf

Journal Nature:
https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf

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

## Publication
* J. Czech, M. Willig, A. Beyer, K. Kersting and J. Fürnkranz: **Learning to play the Chess Variant Crazyhouse above World Champion Level with Deep Neural Networks and Human Data**, [preprint](https://arxiv.org/abs/1908.06660)
```
@article{czech_learning_2019,
	title = {Learning to play the Chess Variant Crazyhouse above World Champion Level with Deep Neural Networks and Human Data},
	author = {Czech, Johannes and Willig, Moritz and Beyer, Alena and Kersting, Kristian and Fürnkranz, Johannes},
	year = {2019},
	pages = {35}
}
```
