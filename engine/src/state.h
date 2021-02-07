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
  GNU General Public License fåor more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: state.h
 * Created on 13.07.2020
 * @author: queensgambit
 *
 * State is an abstract class which is used in the MCTS as a generic interface for various environments.
 * It uses the curiously recurring template pattern (CRTP) idiom to reduce the number of virtual methods.
 */

#ifndef GAMESTATE_H
#define GAMESTATE_H

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

typedef uint64_t Key;
typedef int64_t Action;
typedef uint16_t MoveIdx;
typedef unsigned int uint;
typedef int SideToMove;
#define FIRST_PLAYER_IDX 0
const int ACTION_NONE = 0;

enum PolicyType {
    normal,
    classic
};

enum MirrorType {
    notMirrored,
    mirrored
};

enum TerminalType {
    TERMINAL_LOSS,
    TERMINAL_DRAW,
    TERMINAL_WIN,
    TERMINAL_CUSTOM,
    TERMINAL_NONE
};

enum Result {
    DRAWN = 0,
    WHITE_WIN,
    BLACK_WIN,
    NO_RESULT,
};

/**
 * @brief is_win Return true if the given result is a win, else false
 * @param res Result
 * @return Bool
 */
bool is_win(Result res);

// -------------------------------------------------------------------------
// from Stockfish/src/syzygy
namespace Tablebase {

enum WDLScore {
    WDLLoss        = -2, // Loss
    WDLBlessedLoss = -1, // Loss, but draw under 50-move rule
    WDLDraw        =  0, // Draw
    WDLCursedWin   =  1, // Win, but draw under 50-move rule
    WDLWin         =  2, // Win

    WDLScoreNone  = -1000
};

// Possible states after a probing operation
enum ProbeState {
    FAIL              =  0, // Probe failed (missing file table)
    OK                =  1, // Probe succesful
    CHANGE_STM        = -1, // DTZ should check the other side
    ZEROING_BEST_MOVE =  2, // Best move zeroes DTZ (capture or pawn move)
    THREAT            =  3  // Threatening to force capture in giveaway
};
}
// -------------------------------------------------------------------------

template<typename T>
class StateConstantsInterface
{
public:
    static uint BOARD_WIDTH() {
        return T::BOARD_WIDTH();
    }
    static uint BOARD_HEIGHT() {
        return T::BOARD_HEIGHT();
    }
    static uint NB_CHANNELS_TOTAL() {
        return T::NB_CHANNELS_TOTAL();
    }
    static uint NB_SQUARES() {
        return BOARD_WIDTH() * BOARD_HEIGHT();
    }
    static uint NB_VALUES_TOTAL() {
        return NB_CHANNELS_TOTAL() * NB_SQUARES();
    }
    static uint NB_LABELS() {
        return T::NB_LABELS();
    }
    static uint NB_LABELS_POLICY_MAP() {
        return T::NB_LABELS_POLICY_MAP();
    }
    static uint NB_PLAYERS() {
        return T::NB_PLAYERS();
    }
    static std::string action_to_uci(Action action, bool is960) {
        return T::action_to_uci(action, is960);
    }
    template<PolicyType p, MirrorType m>
    static MoveIdx action_to_index(Action action) {
        return T::action_to_index<p, m>(action);
    }
    static void init(bool isPolicyMap) {
        return T::init(isPolicyMap);
    }
};

class State
{
public:
    virtual ~State() = default;

    /**
     * @brief leads_to_terminal Checks if a given action leads to a terminal state
     * @param a Given action
     * @return true if leads to terminal, else false
     */
    bool leads_to_terminal(Action a)
    {
        std::unique_ptr<State> posCheckTerminal = std::unique_ptr<State>(this->clone());
        bool givesCheck = posCheckTerminal->gives_check(a);
        posCheckTerminal->do_action(a);
        return posCheckTerminal->check_result(givesCheck) != NO_RESULT;
    }

    /**
     * @brief random_rollout Does a random rollout until it reaches a terminal node.
     * This functions modifies the current state and returns the terminal type.
     * @return Terminal type
     */
    TerminalType random_rollout(float& customValueTerminal);

    /**
     * @brief random_rollout Does a random rollout until it reaches a terminal node.
     * This functions modifies the current state and returns the corresponding value evaluation of the terminal type.
     * @return Terminal type
     */
    float random_rollout();

    /**
     * @brief legal_actions Returns all legal actions as a vector list
     * @return vector of legal actions
     */
    virtual std::vector<Action> legal_actions() const = 0;

    /**
     * @brief set Sets a new states and modifies the current state.
     * @param fenStr String description about the state
     * @param isChess960 If true 960 mode will be active
     * @param variant Variant which the position corresponds to.
     * @return An alias to the updated state
     */
    virtual void set(const std::string& fenStr, bool isChess960, int variant) = 0;

    /**
     * @brief get_state_planes Returns the state plane representation of the current state which can be used for NN inference.
     * @param normalize If true thw normalized represnetation should be returned, otherwise the raw representation
     * @param inputPlanes Pointer to the memory array where to set the state plane representation. It is assumed that the memory has already been allocated
     */
    virtual void get_state_planes(bool normalize, float* inputPlanes) const = 0;

    /**
     * @brief steps_from_null Number of steps form the initial position (e.g. starting position)
     * @return number of steps
     */
    virtual unsigned int steps_from_null() const = 0;

    /**
     * @brief is_chess960 Returns true if the position is a 960 random position, else false
     * @return bool
     */
    virtual bool is_chess960() const = 0;

    /**
     * @brief fen Returns the fen or string description of the current state
     * @return string
     */
    virtual std::string fen() const = 0;

    /**
     * @brief do_action Applies a given action to the current state
     * @param action Type of action to apply. It is assumed that the action is discrete and integer format
     */
    virtual void do_action(Action action) = 0;

    /**
     * @brief undo_action Undos a given action
     * @param action Type of action to apply. It is assumed that the action is discrete and integer format
     */
    virtual void undo_action(Action action) = 0;

    /**
     * @brief prepare_action Function which is called once in case of MCTS_STORE_STATES before a new action is applied in a leaf node.
     * It can be used to store e.g. action buffers in the state which can then be used for all other legal actions.
     * By default keep this method empty.
     */
    virtual void prepare_action() = 0;

    /**
     * @brief number_repetitions Returns the number of times this state has already occured in the current episode
     * @return int
     */
    virtual unsigned int number_repetitions() const = 0;

    /**
     * @brief side_to_move Returns the side to move (e.g. Color: WHITE or BLACK) in chess
     * @return int
     */
    virtual int side_to_move() const = 0;

    /**
     * @brief hash_key Returns a uique identifier for the current position which can be used for accessing the hash table
     * @return
     */
    virtual Key hash_key() const = 0;

    /**
     * @brief flip Flips the state along the x-axis
     */
    virtual void flip() = 0;

    /**
     * @brief uci_to_action Converts the given action in uci notation to an action object
     * Note: The "const" modifier had to be dropped for "uciStr" because Stockfish's UCI::to_move() method does not allow "const".
     * @param uciStr uci specification for the action
     * @return Action
     */
    virtual Action uci_to_action(std::string& uciStr) const = 0;

    /**
     * @brief action_to_san Converts a given action to SAN (pgn move notation) usign the current position and legal moves
     * @param action Given action
     * @param legalActions List of legal moves for the current position
     * @param leadsToWin Indicator which marks action as a terminating action (usually indicated with suffix #).
     * @param bookMove Indicator which marks action as book move
     * @return SAN string
     */
    virtual std::string action_to_san(Action action, const std::vector<Action>& legalActions, bool leadsToWin=false, bool bookMove=false) const = 0;

    /**
     * @brief is_terminal Returns the terminal type for the current state. If the state is a non terminal state,
     * then TERMINAL_NONE should be returned.
     * @param numberLegalMoves Number of legal moves in the current position
     * @param inCheck Boolean which defines if there is a check in the current position
     * @param customTerminalValue Value which will be assigned to the node value evaluation. You need to return TERMINAL_CUSTOM in this case;
     * otherwise the value will later be overwritten. In the default case, this parameter can be ignored.
     * @return TerminalType
     */
    virtual TerminalType is_terminal(size_t numberLegalMoves, bool inCheck, float& customTerminalValue) const = 0;

    /**
     * @brief check_result Returns the current game result. In case a normal position is given NO_RESULT is returned.
     * @param inCheck Determines if a king in the current position is in check (needed to differ between checkmate and stalemate).
     * It can be computed by `gives_check(<last-move-before-current-position>)`.
     * @return value in [DRAWN, WHITE_WIN, BLACK_WIN, NO_RESULT]
     */
    virtual Result check_result(bool inCheck) const = 0;

    /**
     * @brief gives_check Checks if the current action is a checking move
     * @param action Action
     * @return bool
     */
    virtual bool gives_check(Action action) const = 0;

    /**
     * @brief print Print method used for the operator <<
     * @param os OS stream object
     */
    virtual void print(std::ostream& os) const = 0;

    /**
     * @brief check_for_tablebase_wdl Checks the current state for a table base entry.
     * Return Tablebase::WDLScoreNone and Tablebase::FAIL if your state doesn't support tablebases.
     * @param result ProbeState result
     * @return WDLScore
     */
    virtual Tablebase::WDLScore check_for_tablebase_wdl(Tablebase::ProbeState& result) = 0;

    /**
     * @brief operator << Operator overload for <<
     * @param os ostream object
     * @param state state object
     * @return ostream
     */
    friend std::ostream& operator<<(std::ostream& os, const State& state)
    {
        state.print(os);
        return os;
    }

    /**
     * @brief clone Clones the current state as a deep copy.
     * Returning a unique_ptr instead is possible but becomes messy:
     * https://github.com/CppCodeReviewers/Covariant-Return-Types-and-Smart-Pointers
     * @return deep copy
     */
    virtual State* clone() const = 0;
};

#endif // GAMESTATE_H
