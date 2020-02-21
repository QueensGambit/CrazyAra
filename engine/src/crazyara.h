/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018       Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019-2020  Johannes Czech

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: crazyara.h
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * Main entry point for the executable which manages the UCI communication
 */

#ifndef CRAZYARA_H
#define CRAZYARA_H

#include <iostream>

#include "agents/rawnetagent.h"
#include "agents/mctsagent.h"
#include "nn/neuralnetapi.h"
#include "agents/config/searchsettings.h"
#include "agents/config/searchlimits.h"
#include "agents/config/playsettings.h"
#include "node.h"
#include "manager/statesmanager.h"
#ifdef USE_RL
#include "rl/selfplay.h"
#include "agents/config/rlsettings.h"
#endif

class CrazyAra
{
private:
    const string intro =  string("\n") +
                    string("                                  _                                           \n") +
                    string("                   _..           /   ._   _.  _        /\\   ._   _.           \n") +
                    string("                 .' _ `\\         \\_  |   (_|  /_  \\/  /--\\  |   (_|           \n") +
                    string("                /  /e)-,\\                         /                           \n") +
                    string("               /  |  ,_ |                    __    __    __    __             \n") +
                    string("              /   '-(-.)/          bw     8 /__////__////__////__////         \n") +
                    string("            .'--.   \\  `                 7 ////__////__////__////__/          \n") +
                    string("           /    `\\   |                  6 /__////__////__////__////           \n") +
                    string("         /`       |  / /`\\.-.          5 ////__////__////__////__/            \n") +
                    string("       .'        ;  /  \\_/__/         4 /__////__////__////__////             \n") +
                    string("     .'`-'_     /_.'))).-` \\         3 ////__////__////__////__/              \n") +
                    string("    / -'_.'---;`'-))).-'`\\_/        2 /__////__////__////__////               \n") +
                    string("   (__.'/   /` .'`                 1 ////__////__////__////__/                \n") +
                    string("    (_.'/ /` /`                       a  b  c  d  e  f  g  h                  \n") +
                    string("      _|.' /`                                                                 \n") +
                    string("jgs.-` __.'|  Developers: Johannes Czech, Moritz Willig, Alena Beyer          \n") +
                    string("    .-'||  |  Source-Code: QueensGambit/CrazyAra-Engine (GPLv3-License)       \n") +
                    string("       \\_`/   Inspiration: A0-paper by Silver, Hubert, Schrittwieser et al.   \n") +
                    string("              ASCII-Art: Joan G. Stark, Chappell, Burton                      \n");
    RawNetAgent* rawAgent;
    MCTSAgent* mctsAgent;
    NeuralNetAPI* netSingle;
    SearchSettings* searchSettings;
    PlaySettings* playSettings;
    bool networkLoaded = false;
    StatesManager* states;
    Variant variant;

#ifdef USE_RL
    MCTSAgent* mctsAgentContender;
    RLSettings* rlSettings;
#endif
    /**
     * @brief engine_info Returns a string about the engine version and authors
     * @return string
     */
    string engine_info();

    /**
     * @brief create_new_mcts_agent Factory method to create a new MCTSAgent when loading new neural network weights
     * @param modelDirectory Directory where the .params and .json files are stored
     * @param states State-Manager, needed to keep track of 3-fold-repetition
     * @param netSingle Neural net with batch-size 1. It will be loaded from file.
     * @param netBatches Neural net handes with a batch-size defined by the uci options. It will be loaded from file.
     * @return Pointer to the new MCTSAgent object
     */
    MCTSAgent* create_new_mcts_agent(NeuralNetAPI* netSingle, NeuralNetAPI** netBatches, StatesManager* states);

    /**
     * @brief create_new_net_single Factory to create and load a new model from a given directory
     * @param modelDirectory Model directory where the .params and .json files are stored
     * @return Pointer to the newly created object
     */
    NeuralNetAPI* create_new_net_single(const string& modelDirectory);

    /**
     * @brief create_new_net_batches Factory to create and load a new model for batch-size access
     * @param modelDirectory Model directory where the .params and .json files are stored
     * @return Pointer to the newly createded objects. For every thread a sepreate net
     */
    NeuralNetAPI** create_new_net_batches(const string& modelDirectory);

public:
    CrazyAra();
    ~CrazyAra();

    /**
     * @brief welcome Prints a welcome message to std-out
     */
    void welcome();

    /**
     * @brief uci_loop Runs the uci-loop which reads std-in UCI-messages
     * @param argc Number of arguments
     * @param argv Argument values
     */
    void uci_loop(int argc, char* argv[]);

    /**
     * @brief init Initializes all needed backend-types
     */
    void init();

    /**
     * @brief is_ready Loads the neural network weights and creates the agent object in case there haven't loaded already
     * @return True, if everything isReady
     */
    bool is_ready();

    /**
     * @brief new_game Handles the request of starting a new game
     */
    void new_game();

    /**
     * @brief go Main method which starts the search after receiving the UCI "go" command
     * @param pos Current board position
     * @param is List of command line arguments for the search
     * @param evalInfo Returns the evalutation information
     * @param applyMoveToTree Tells if the given move shall be applied to the tree for future reusage
     */
    void go(Board* pos, istringstream& is, EvalInfo& evalInfo, bool applyMoveToTree=true);

    /**
     * @brief go Wrapper function for go() which accepts a FEN string
     * @param fen FEN string
     * @param goCommand Go command (such as "go movetime 5000")
     * @param evalInfo Returns the evalutation information
     */
    void go(const string& fen, string goCommand, EvalInfo& evalInfo);

    /**
     * @brief position Method which is called from the UCI command-line when a new position is described.
     * This can be a FEN string or the starting position followed by a list of moves
     * @param pos Position object which will be set
     * @param is List of command line arguments which describe the position
     */
    void position(Board* pos, istringstream& is);

    /**
     * @brief benchmark Runs a list of benchmark position for a given time
     * @param is Movetime in ms
     */
    void benchmark(istringstream& is);

#ifdef USE_RL
    /**
     * @brief selfplay Starts self play for a given number of games
     * @param is Number of games to generate
     */
    void selfplay(istringstream &is);

    /**
     * @brief arena Starts the arena comparision between two different NN weights.
     * The score can be used for logging and to decide if the current weights shall be replaced.
     * The arena ends with either the keywords "keep" or "replace".
     * "keep": Signals that the current generator should be kept
     * "replace": Signals that the current generator should be replaced by the contender
     * @param is Number of games to generate
     */
    void arena(istringstream &is);

    /**
     * @brief init_rl_settings Initializes the rl settings used for the mcts agent with the current UCI parameters
     */
    void init_rl_settings();
#endif

    /**
     * @brief init_search_settings Initializes the search settings with the current UCI parameters
     */
    void init_search_settings();

    /**
     * @brief init_play_settings Initializes the play settings with the current UCI parameters
     */
    void init_play_settings();
};

#endif // CRAZYARA_H
