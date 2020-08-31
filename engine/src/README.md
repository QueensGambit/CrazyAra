# Source Code Directory

`main.cpp` is the main entry point of the engine and creates the CrazyAra object which handles the UCI-communication.

## Folder Structure
*   **`agents`**: Contains the specification for different search agent regimes
*   **`chess_related`**: Contains conversion methods of the board into plane representation and constant definition for chess variants. You don't need to include this directory when using CrazyAra as a library for a different game.
*   **`manager`**: Contains manager classes for different aspects e.g. the search tree, the time and the states list
*   **`nn`**: Contains the functionality methods for loading the neural network and predicting the policy and value evaluation
*   **`util`**: Contains additional utility methods for the blaze library and stockfish backend
*   **`rl`**: Contains functionality methods for applying reinforcmenet learning which is only used for `USE_RL=ON`

## Performance Profiling 

Install the plotting utility for [gprof](https://ftp.gnu.org/old-gnu/Manuals/gprof-2.9.1/html_mono/gprof.html):
* https://github.com/jrfonseca/gprof2dot

Activate the -pg flags in `CMakeLists.txt` and rebuild.
Run the executable and generate the plot:
```bash
./CrazyAra
gprof CrazyAra | gprof2dot | dot -Tpng -o output.png
```
