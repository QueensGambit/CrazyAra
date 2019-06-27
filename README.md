# CrazyAra-MCTS
Port of the CrazyAra written in Python to C++


### Steps to compile

Download and install the blaze library
* https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation

Build the MXNet C++ package
* https://mxnet.incubator.apache.org/versions/master/api/c++/index.html

Clone the Sf library
* put the `src/` folder in `libs/sf`
* add `friend class Board;` in lib/sf/position.h under `private:`
* uncomment main() in libs/sf

Download Catch
* https://github.com/catchorg/Catch2/releases
