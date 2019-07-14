# CrazyAra-MCTS
CrazyAra written in C++ based on the Python version


### Steps to compile

Download and install the blaze library
* https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation

Build the MXNet C++ package
* https://mxnet.incubator.apache.org/versions/master/api/c++/index.html
```make -j USE_CPP_PACKAGE=1 USE_OPENCV=0 USE_MKL=1```
* _because the current dev branch broke the loading of certain models download release [MXNet-release 1.4.1](https://github.com/apache/incubator-mxnet/releases) instead_


Clone the Sf library
* put the `src/` folder in `libs/sf`
* add `friend class Board;` in lib/sf/position.h under `private:`
* uncomment main() in libs/sf

Download Catch
* https://github.com/catchorg/Catch2/releases

Downlaod & install yaml-cpp 
* https://github.com/jbeder/yaml-cpp

or 
* sudo apt-get install libyaml-cpp-dev


### Performance Profiling

Install the plotting utility for gprof:
* https://github.com/jrfonseca/gprof2dot

Activate the -pg flags in `CMakeLists.txt` and rebuild.
Run the executable and generate the plot:
```
$ ./CrazyAraMCTS
$ gprof CrazyAraMCTS | gprof2dot | dot -Tpng -o output.png
```
