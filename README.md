
# CrazyAra - Deep Learning for Crazyhouse <img src="https://raw.githubusercontent.com/QueensGambit/CrazyAra/master/etc/media/CrazyAra_Logo.png" width="64">

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
       (__.'/   /` .'`              Developers:           Johannes Czech, Moritz Willig, Alena Beyer et al.  
        (_.'/ /` /`                 Source Code (GitHub): QueensGambit/CrazyAra-AI (GPLv3-License)            
          _|.' /`                   Website:              http://www.crazyara.org/                           
    jgs.-` __.'|                    Lichess:              https://lichess.org/@/CrazyAra                           
        .-'||  |                    ASCII-Art:            Joan G. Stark (http://www.oocities.org/spunk1111/)                  
           \_`/                     ASCII-Font:           Mini by Glenn Chappell 4/93, Paul Burton           
                                               
                                    
Official Repository of the Crazyhouse-Bot CrazyAra which is powered by a Deep Convolutional Neural Network and is compatible with the Universial-Chess-Interface (UCI).


## Installation Guide
Please follow the instructions in the wiki-page at:
* 

## License
This source-code including all project files is licensed under the GPLv3-License if not stated otherwise.
See LICENSE for more details.

## Project Links:
Project website:
* http://www.crazyara.org/ 

Link to CrazyAra's lichess-org account: 
* https://lichess.org/@/CrazyAra

Project management plattform:
* https://tree.taiga.io/project/queensgambit-deep-learning-project-crazyhouse/


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
