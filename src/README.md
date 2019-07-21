# Source Code Directory

`CrazyAra.cpp` is the main entry point of the engine and handles the UCI-communication.

## Folder Structure
* **`agents`**: Contains the specification for different search agent regimes
* **`nn`**: Contains the functionality methods for loading the neural network and predicting the policy and value evaluation
* **`domain`**: Contains conversion methods of the board into plane representation and constant definition for chess variants
* **`util`**: Contains additional utility methods for the blaze library and stockfish backend

## Performance Profiling 

Install the plotting utility for [gprof](https://ftp.gnu.org/old-gnu/Manuals/gprof-2.9.1/html_mono/gprof.html):
* https://github.com/jrfonseca/gprof2dot

Activate the -pg flags in `CMakeLists.txt` and rebuild.
Run the executable and generate the plot:
```
$ ./CrazyAraMCTS
$ gprof CrazyAraMCTS | gprof2dot | dot -Tpng -o output.png
```
