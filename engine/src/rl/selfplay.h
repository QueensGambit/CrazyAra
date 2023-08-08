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
 * @file: selfplay.h
 * Created on 16.09.2019
 * @author: queensgambit
 *
 * Functionality for running CrazyAra in self play mode
 */

#ifndef SELFPLAY_H
#define SELFPLAY_H

#include "../agents/mctsagent.h"
#include "../agents/rawnetagent.h"
#include "gamepgn.h"
#include "tournamentresult.h"
#include "../agents/config/rlsettings.h"
#include "../stateobj.h"
#ifdef SF_DEPENDENCY
#include "uci.h"
using namespace UCI;
#else
#include "uci/customuci.h"
using namespace CUSTOM_UCI;
#endif

#ifdef USE_RL
/**
 * @brief update_states_after_move Plays the best move of evalInfo and updates the relevant set of variables
 * @param evalInfo Struct which contains the best move and all legal moves
 * @param state Current game board position
 * @param states States list
 * @param gamePGN PGN of the current game
 * @param gameResult Current game result (usually NO_RESULT after move was played)
 */
void play_move_and_update(const EvalInfo& evalInfo, StateObj* state, GamePGN& gamePGN, Result& gameResult);


/**
 * @brief load_random_fen Returns a random fen string from a epd file defined by its file path.
 * If the filepath is "" or "<empty>" an empty string will be returned.
 * @param filepath Points to the location of the epd file.
 * @return fen string
 */
string load_random_fen(string filepath);


class SelfPlay
{
private:
    RawNetAgent* rawAgent;
    MCTSAgent* mctsAgent;
    SearchLimits* searchLimits;
    PlaySettings* playSettings;
    RLSettings* rlSettings;
    OptionsMap& options;
    GamePGN gamePGN;
    TrainDataExporter* exporter;
    string filenamePGNSelfplay;
    string filenamePGNArena;
    string fileNameGameIdx;
    size_t gameIdx;
    float gamesPerMin;
    float samplesPerMin;
    size_t backupNodes;
    float backupDirichletEpsilon;
    float backupQValueWeight;
    bool is960;

public:
    /**
     * @brief SelfPlay
     * @param rawAgent Raw network agent which uses the raw network policy for e.g. game initiliation
     * @param mctsAgent MCTSAgent which is used during selfplay for game generation
     * @param searchLimits Search limit configuration struct
     * @param playSettings Playing setting configuration struct
     * @param RLSettings Additional settings for reinforcement learning usage
     * @param options Object holding all UCI options
     */
    SelfPlay(RawNetAgent* rawAgent, MCTSAgent* mctsAgent,  SearchLimits* searchLimits, PlaySettings* playSettings,
        RLSettings* rlSettings, OptionsMap& options);
    ~SelfPlay();

    /**
     * @brief go Starts the self play game generation for a given number of games
     * @param numberOfGames Number of games to generate
     * @param int variant to generate games for
     */
    void go(size_t numberOfGames, int variant);

    /**
     * @brief go_arena Starts comparision matches between the original mctsAgent with the old NN weights and
     * the mctsContender which uses the new updated wieghts
     * @param mctsContender MCTSAgent using different NN weights
     * @param numberOfGames Number of games to compare
     * @param int variant to generate games for
     * @return Score in respect to the contender, as floating point number.
     *  Wins give 1.0 points, 0.5 for draw, 0.0 for loss.
     */
    TournamentResult go_arena(MCTSAgent *mctsContender, size_t numberOfGames, int variant);

private:
    /**
     * @brief generate_game Generates a new game in self play mode
     * @param variant Current chess variant
     */
    void generate_game(int variant, bool verbose);

    /**
     * @brief generate_arena_game Generates a game of the current NN weights vs the new acquired weights
     * @param whitePlayer MCTSAgent which will play with the white pieces
     * @param blackPlayer MCTSAgent which will play with the black pieces
     * @param variant Current chess variant
     * @param fen Starting position. If empty, the standard starting or a random position (for 960 games) will be used.
     * The fen will be stored in gamePGN.fen.
     * @param verbose If true the games will printed to stdout
     */
    Result generate_arena_game(MCTSAgent *whitePlayer, MCTSAgent *blackPlayer, int variant, bool verbose, const string& fen);

    /**
     * @brief write_game_to_pgn Writes the game log to a pgn file
     * @param pngFileName Filename to export
     * @param verbose If true, game will also be printed to stdout
     */
    void write_game_to_pgn(const std::string& pngFileName, bool verbose);

    /**
     * @brief set_game_result Sets the game result to the gamePGN object
     * @param res Game result
     */
    void set_game_result_to_pgn(Result res);

    /**
     * @brief reset_speed_statistics Resets the interal measurements for gameIdx, gamesPerMin and samplesPerMin
     */
    void reset_speed_statistics();

    /**
     * @brief speed_statistic_report Updates the speed statistics and prints a summary to std-out
     */
    void speed_statistic_report(float elapsedTimeMin, size_t generatedSamples);

    /**
     * @brief export_number_generated_games Creates a file which describes how many games have been generated in the newly created .zip-file
     */
    void export_number_generated_games() const;

    /**
     * @brief adjust_node_count Adjusts the amount of nodes to search based on rlSettings->nodeRandomFactor to increase playing variety
     * @param searchLimits searchLimit struct to be modified
     * @param randInt Randomly generated integer
     */
    void adjust_node_count(SearchLimits* searchLimits, int randInt);

    /**
     * @brief is_quick_search Checks if a quick search sould be done
     * @return True for quick search else false
     */
    bool is_quick_search();

    /**
     * @brief is_quick_search Checks if resignation is allowed
     * @return True if resignation is allowed else false
     */
    bool is_resignation_allowed();

    /**
     * @brief check_for_resignation Modifies gameResult to WHITE_WIN or BLACK_WIN if resignation is allowed and Q-value threshold had been reached
     * @param allowResingation True if resignation is allowed
     * @param evalInfo Evaluation struct
     * @param position Board position. It is expected that the evalBestMove has already been applied.
     * @param gameResult Game result which may be modified
     */
    void check_for_resignation(const bool allowResignation, const EvalInfo& evalInfo, const StateObj* position, Result& gameResult);

    /**
     * @brief reset_search_params Resets all search parameters to their initial values
     * @param Signals if a quick search was done
     */
    void reset_search_params(bool isQuickSearch);
};
#endif

/**
 * @brief clean_up Applies a clean-up operation after a generated game.
 * Deletes the position, clears the current active states of the StatesManager, clears the game history of the MCTSAgent,
 * calls new_game() for the gamePGN struct
 * @param gamePGN gamePGN struct
 * @param mctsAgent mctsAgent object
 * @param states StatesManager
 */
void clean_up(GamePGN& gamePGN, MCTSAgent* mctsAgent);

/**
 * @brief init_games_from_raw_policy Inits a new starting position by sampling from the raw policy with temperature 1.
 * The iteration stops either when the number of plys is reached or when the next move would lead to a terminal state.
 * @param rawAgent Handle to the raw agent
 * @param plys Number of plys to generate
 * @param gamePGN Game pgn struct where the moves will be stored
 * @param variant Game variant
 * @param is960 If we are in a 960 game or not
 * @param rawPolicyProbTemp Probability for which a temperature scaling > 1.0f is applied
 * @param fen Starting fen. If "" then standard starting position will be used.
 * @return New state object
 */
unique_ptr<StateObj> init_starting_state_from_raw_policy(RawNetAgent& rawAgent, size_t plys, GamePGN& gamePGN, int variant, bool is960, float rawPolicyProbTemp, const string& fen);

/**
 * @brief init_starting_state_from_fixed_move Initializes a starting position using a vector of actions
 * @param gamePGN Game pgn struct where the moves will be stored
 * @param variant Game variant
 * @param is960 If we are in a 960 game or not
 * @param actions Vector of actions
 * @return New state object
 */
unique_ptr<StateObj> init_starting_state_from_fixed_move(GamePGN& gamePGN, int variant, bool is960, const vector<Action>& actions);

/**
 * @brief apply_raw_policy_temp Applies a temperature scaling to the policyProbSmall of the eval struct.
 * The temperature is applied given a certain probability.
 * If this is the case in 5% of the times a temperature of 10 and in 20% a temperature of 5, and for the remaining a temperature of 2 is applied.
 * This is meant to flatten the policy distribution.
 * @param eval Evaluation struct
 * @param rawPolicyProbTemp Probability for which a temperature scaling > 1.0f is applied
 */
void apply_raw_policy_temp(EvalInfo& eval, float rawPolicyProbTemp);

/**
 * @brief clip_ply Clips a given ply touse maxPly as the maximum value. In case of an entry greater than max it is uniformly sampled from
 * [0, maxPly] instead.
 * @param ply Given ply which might be greater than maxPly
 * @param maxPly Maximum ply value
 * @return
 */
size_t clip_ply(size_t ply, size_t maxPly);

#endif // SELFPLAY_H
