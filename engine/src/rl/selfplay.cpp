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
 * @file: selfplay.cpp
 * Created on 16.09.2019
 * @author: queensgambit
 *
 */

#ifdef USE_RL
#include "selfplay.h"

#include "thread.h"
#include <iostream>
#include <fstream>
#include "uci.h"
#include "../domain/variants.h"
#include "../util/blazeutil.h"
#include "../util/randomgen.h"
#include "../util/chess960position.h"

void play_move_and_update(const EvalInfo& evalInfo, Board* position, StateListPtr& states, GamePGN& gamePGN, Result& gameResult)
{
    states->emplace_back();
    bool givesCheck = position->gives_check(evalInfo.bestMove);
    position->do_move(evalInfo.bestMove, states->back(), givesCheck);
    gameResult = get_result(*position, givesCheck);
    position->undo_move(evalInfo.bestMove);  // undo and later redo move to get PGN move with result
    gamePGN.gameMoves.push_back(pgn_move(evalInfo.bestMove,
                                        position->is_chess960(),
                                        *position,
                                        evalInfo.legalMoves,
                                        is_win(gameResult)));
    states->emplace_back();
    position->do_move(evalInfo.bestMove, states->back(), givesCheck);
}


SelfPlay::SelfPlay(RawNetAgent* rawAgent, MCTSAgent* mctsAgent, SearchLimits* searchLimits, PlaySettings* playSettings, RLSettings* rlSettings):
    rawAgent(rawAgent), mctsAgent(mctsAgent), searchLimits(searchLimits), playSettings(playSettings), rlSettings(rlSettings),
    gameIdx(0), gamesPerMin(0), samplesPerMin(0)
{
    bool is960 = true;
#ifdef MODE_CRAZYHOUSE
    gamePGN.variant = "crazyhouse";
#elif defined MODE_CHESS
    if (is960) {
        gamePGN.variant = "chess960";
    }
    else {
        gamePGN.variant = "standard";
    }
#endif
    gamePGN.event = "SelfPlay";
    gamePGN.site = "Darmstadt, GER";
    gamePGN.date = "?";  // TODO: Change this later
    gamePGN.round = "?";
    gamePGN.is960 = false;
    this->exporter = new TrainDataExporter(string("data_") + mctsAgent->get_device_name() + string(".zarr"),
                                           rlSettings->numberChunks, rlSettings->chunkSize);
    filenamePGNSelfplay = string("games_") + mctsAgent->get_device_name() + string(".pgn");
    filenamePGNArena = string("arena_games_")+ mctsAgent->get_device_name() + string(".pgn");
    fileNameGameIdx = string("gameIdx_") + mctsAgent->get_device_name() + string(".txt");

    backupNodes = searchLimits->nodes;
    backupQValueWeight = mctsAgent->get_q_value_weight();
    backupDirichletEpsilon = mctsAgent->get_dirichlet_noise();
}

SelfPlay::~SelfPlay()
{
    delete exporter;
}

void SelfPlay::adjust_node_count(SearchLimits* searchLimits, int randInt)
{
    size_t maxRandomNodes = size_t(searchLimits->nodes * rlSettings->nodeRandomFactor);
    if (maxRandomNodes != 0) {
        searchLimits->nodes += (size_t(randInt) % maxRandomNodes) - maxRandomNodes / 2;
    }
}

bool SelfPlay::is_quick_search() {
    if (rlSettings->quickSearchProbability < 0.01f) {
        return false;
    }
    return float(rand()) / RAND_MAX < rlSettings->quickSearchProbability;
}

bool SelfPlay::is_resignation_allowed() {
    if (rlSettings->resignProbability < 0.01f) {
        return false;
    }
    return float(rand()) / RAND_MAX < rlSettings->resignProbability;
}

void SelfPlay::check_for_resignation(const bool allowResingation, const EvalInfo &evalInfo, const Position *position, Result &gameResult)
{
    if (!allowResingation) {
        return;
    }
    if (evalInfo.bestMoveQ < rlSettings->resignThreshold) {
        if (position->side_to_move() == WHITE) {
            gameResult = WHITE_WIN;
        }
        else {
            gameResult = BLACK_WIN;
        }
    }
}

void SelfPlay::reset_search_params(bool isQuickSearch)
{
    searchLimits->nodes = backupNodes;
    if (isQuickSearch) {
        mctsAgent->update_q_value_weight(backupQValueWeight);
        mctsAgent->update_dirichlet_epsilon(backupDirichletEpsilon);
    }
}

void SelfPlay::generate_game(Variant variant, bool verbose)
{
    states = StateListPtr(new std::deque<StateInfo>(0));
    chrono::steady_clock::time_point gameStartTime = chrono::steady_clock::now();

    size_t ply = size_t(random_exponential<float>(1.0f/playSettings->meanInitPly) + 0.5f);
    ply = clip_ply(ply, playSettings->maxInitPly);

    srand(unsigned(int(time(nullptr))));
    Board* position = init_starting_pos_from_raw_policy(*rawAgent, ply, gamePGN, variant, states,
                                                        rlSettings->rawPolicyProbabilityTemperature);
    EvalInfo evalInfo;
    Result gameResult;
    exporter->new_game();

    size_t generatedSamples = 0;
    const bool allowResignation = is_resignation_allowed();
    do {
        searchLimits->startTime = now();
        const int randInt = rand();
        const bool isQuickSearch = is_quick_search();

        if (isQuickSearch) {
            searchLimits->nodes = rlSettings->quickSearchNodes;
            mctsAgent->update_q_value_weight(rlSettings->quickSearchQValueWeight);
            mctsAgent->update_dirichlet_epsilon(rlSettings->quickDirichletEpsilon);
        }
        adjust_node_count(searchLimits, randInt);
        mctsAgent->set_search_settings(position, searchLimits, &evalInfo);
        mctsAgent->perform_action();
        if (rlSettings->reuseTreeForSelpay) {
            mctsAgent->apply_move_to_tree(evalInfo.bestMove, true);
        }

        if (!isQuickSearch && !exporter->is_file_full()) {
            if (rlSettings->lowPolicyClipThreshold > 0) {
                sharpen_distribution(evalInfo.policyProbSmall, rlSettings->lowPolicyClipThreshold);
            }
            exporter->save_sample(position, evalInfo);
            ++generatedSamples;
        }
        play_move_and_update(evalInfo, position, states, gamePGN, gameResult);
        reset_search_params(isQuickSearch);
        check_for_resignation(allowResignation, evalInfo, position, gameResult);
    }
    while(gameResult == NO_RESULT);

    // export all training samples of the generated game
    exporter->export_game_samples(gameResult);

    set_game_result_to_pgn(gameResult);
    write_game_to_pgn(filenamePGNSelfplay, verbose);
    clean_up(gamePGN, mctsAgent, position);

    // measure time statistics
    if (verbose) {
        const float elapsedTimeMin = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - gameStartTime).count() / 60000.f;
        speed_statistic_report(elapsedTimeMin, generatedSamples);
    }
    ++gameIdx;
}

Result SelfPlay::generate_arena_game(MCTSAgent* whitePlayer, MCTSAgent* blackPlayer, Variant variant, bool verbose)
{
    states = StateListPtr(new std::deque<StateInfo>(0));
    gamePGN.white = whitePlayer->get_name();
    gamePGN.black = blackPlayer->get_name();
    Board* position = init_board(variant, true, gamePGN, states);
    EvalInfo evalInfo;

    MCTSAgent* activePlayer;
    MCTSAgent* passivePlayer;
    // preserve the current active states
    Result gameResult;
    do {
        searchLimits->startTime = now();
        if (position->side_to_move() == WHITE) {
            activePlayer = whitePlayer;
            passivePlayer = blackPlayer;
        }
        else {
            activePlayer = blackPlayer;
            passivePlayer = whitePlayer;
        }
        activePlayer->set_search_settings(position, searchLimits, &evalInfo);
        activePlayer->perform_action();
        activePlayer->apply_move_to_tree(evalInfo.bestMove, true);
        if (position->plies_from_null() != 0) {
            passivePlayer->apply_move_to_tree(evalInfo.bestMove, false);
        }
        play_move_and_update(evalInfo, position, states, gamePGN, gameResult);
    }
    while(gameResult == NO_RESULT);
    set_game_result_to_pgn(gameResult);
    write_game_to_pgn(filenamePGNArena, verbose);
    clean_up(gamePGN, whitePlayer, position);
    blackPlayer->clear_game_history();
    return gameResult;
}

void clean_up(GamePGN& gamePGN, MCTSAgent* mctsAgent, Board* position)
{
    gamePGN.new_game();
    mctsAgent->clear_game_history();
    delete position;
}

void SelfPlay::write_game_to_pgn(const std::string& pngFileName, bool verbose)
{
    ofstream pgnFile;
    pgnFile.open(pngFileName, std::ios_base::app);
    if (verbose) {
        cout << endl << gamePGN << endl;
    }
    pgnFile << gamePGN << endl;
    pgnFile.close();
}

void SelfPlay::set_game_result_to_pgn(Result res)
{
    gamePGN.result = result[res];
}

void SelfPlay::reset_speed_statistics()
{
    gameIdx = 0;
    gamesPerMin = 0;
    samplesPerMin = 0;
}

void SelfPlay::speed_statistic_report(float elapsedTimeMin, size_t generatedSamples)
{
    // compute running cummulative average
    gamesPerMin = (gameIdx * gamesPerMin + (1 / elapsedTimeMin)) / (gameIdx + 1);
    samplesPerMin = (gameIdx * samplesPerMin + (generatedSamples / elapsedTimeMin)) / (gameIdx + 1);

    cout << "    games    |  games/min  | samples/min " << endl
         << "-------------+-------------+-------------" << endl
         << std::setprecision(5)
         << setw(13) << gameIdx << '|'
         << setw(13) << gamesPerMin << '|'
         << setw(13) << samplesPerMin << endl << endl;
}

void SelfPlay::export_number_generated_games() const
{
    ofstream gameIdxFile;
    gameIdxFile.open(fileNameGameIdx);
    gameIdxFile << gameIdx;
    gameIdxFile.close();
}


void SelfPlay::go(size_t numberOfGames, Variant variant)
{
    reset_speed_statistics();
    gamePGN.white = mctsAgent->get_name();
    gamePGN.black = mctsAgent->get_name();

    if (numberOfGames == 0) {
        while(!exporter->is_file_full()) {
            generate_game(variant, true);
        }
    }
    else {
        for (size_t idx = 0; idx < numberOfGames; ++idx) {
            generate_game(variant, true);
        }
    }
    export_number_generated_games();
}

TournamentResult SelfPlay::go_arena(MCTSAgent *mctsContender, size_t numberOfGames, Variant variant)
{
    TournamentResult tournamentResult;
    tournamentResult.playerA = mctsContender->get_name();
    tournamentResult.playerB = mctsAgent->get_name();
    Result gameResult;
    for (size_t idx = 0; idx < numberOfGames; ++idx) {
        if (idx % 2 == 0) {
            gameResult = generate_arena_game(mctsContender, mctsAgent, variant, true);
            if (gameResult == WHITE_WIN) {
                ++tournamentResult.numberWins;
            }
            else {
                ++tournamentResult.numberLosses;
            }
        }
        else {
            gameResult = generate_arena_game(mctsAgent, mctsContender, variant, true);
            if (gameResult == BLACK_WIN) {
                ++tournamentResult.numberWins;
            }
            else {
                ++tournamentResult.numberLosses;
            }
        }
        if (gameResult == DRAWN) {
            ++tournamentResult.numberDraws;
        }
    }
    return tournamentResult;
}

Board* init_board(Variant variant, bool is960, GamePGN& gamePGN, StateListPtr& states)
{
    Board* position = new Board();
    auto uiThread = make_shared<Thread>(0);

    states->emplace_back();
    if (is960) {
        string firstRank = startPos();
        string lastRank = string(firstRank);
        std::transform(firstRank.begin(), firstRank.end(), firstRank.begin(), ::tolower);
        const string fen = firstRank + "/pppppppp/8/8/8/8/PPPPPPPP/" + lastRank +  " w KQkq - 0 1";
        gamePGN.fen = fen;
        position->set(fen, true, variant, &states->back(), uiThread.get());
    }
    else {
        position->set(StartFENs[variant], false, variant, &states->back(), uiThread.get());
    }
    return position;
}

Board* init_starting_pos_from_raw_policy(RawNetAgent &rawAgent, size_t plys, GamePGN &gamePGN, Variant variant, StateListPtr& states,
                                         float rawPolicyProbTemp)
{
    Board* position = init_board(variant, true, gamePGN, states);

    for (size_t ply = 0; ply < plys; ++ply) {
        EvalInfo eval;
        rawAgent.set_search_settings(position, nullptr, &eval);
        rawAgent.evaluate_board_state();
        apply_raw_policy_temp(eval, rawPolicyProbTemp);
        const size_t moveIdx = random_choice(eval.policyProbSmall);
        eval.bestMove = eval.legalMoves[moveIdx];

        if (leads_to_terminal(*position, eval.bestMove, states)) {
            break;
        }
        else {
            gamePGN.gameMoves.push_back(pgn_move(eval.legalMoves[moveIdx],
                                                 false,
                                                 *position,
                                                 eval.legalMoves,
                                                 false,
                                                 true));
            states->emplace_back();
            position->do_move(eval.bestMove, states->back());
        }
    }
    return position;
}

size_t clip_ply(size_t ply, size_t maxPly)
{
    if (ply > maxPly) {
        return size_t(rand()) % maxPly;
    }
    return ply;
}

void apply_raw_policy_temp(EvalInfo &eval, float rawPolicyProbTemp)
{
    if (float(rand()) / RAND_MAX < rawPolicyProbTemp) {
        float temp = 2.0f;
        const float prob = float(rand()) / INT_MAX;
        if (prob < 0.05f) {
            temp = 10.0f;
        }
        else if (prob < 0.25f) {
            temp = 5.0f;
        }
        apply_temperature(eval.policyProbSmall, temp);
    }
}
#endif
