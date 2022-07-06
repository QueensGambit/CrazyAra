"""
@file: rl_config.py
Created on 01.04.2021
@project: CrazyAra
@author: queensgambit, maxalexger

Configuration file for Reinforcement Learning
"""
from dataclasses import dataclass


@dataclass
class RLConfig:
    """Dataclass storing the options (except UCI options) for executing reinforcement learning."""
    # How many arena games will be done to judge the quality of the new network
    arena_games: int = 100
    # Directory where the executable is located and where the selfplay data will be stored
    binary_dir: str = f'/data/RL/'
    binary_name: str = f'BoardAra'
    # How many times to train the NN, create a model contender or generate nn_update_files games
    nb_nn_updates: int = 10
    # How many new generated training files are needed to apply an update to the NN
    nn_update_files: int = 10
    precision: str = f'float16'
    # Replay Memory
    rm_nb_files: int = 5  # how many data packages/files shall be randomly taken from memory
    rm_fraction_for_selection: float = 0.05  # which percentage of the most recent memory shall be taken into account
    # The UCI_Variant. Must be in ["3check", "atomic", "chess", "crazyhouse",
    # "giveaway" (= antichess), "horde", "kingofthehill", "racingkings"]
    uci_variant: str = f'tictactoe' #f'tictactoe'


@dataclass
class UCIConfig:
    """
    Dataclass which contains the UCI Options that are used during Reinforcement Learning.
    The options will be passed to the binary before game generation starts.
    """
    Allow_Early_Stopping: bool = False
    Batch_Size: int = 8
    Centi_Dirichlet_Alpha: int = 30  # default: 20
    Centi_Dirichlet_Epsilon: int = 25
    Centi_Epsilon_Checks: int = 0
    Centi_Epsilon_Greedy: int = 0
    Centi_Node_Temperature: int = 100
    Centi_Resign_Probability: int = 90
    Centi_Q_Value_Weight: int = 0
    Centi_Quick_Probability: int = 0
    Centi_Temperature: int = 80
    MaxInitPly: int = 0  # default: 30
    MCTS_Solver: bool = False
    MeanInitPly: int = 0  # default: 15
    Milli_Policy_Clip_Thresh: int = 10
    Nodes: int = 200
    Reuse_Tree: str = False
    Search_Type: str = f'mcts'
    Selfplay_Chunk_Size: int = 128  # default: 128
    Selfplay_Number_Chunks: int = 32  # default: 640
    Simulations: int = 3200
    SyzygyPath: str = f''
    Temperature_Moves: int = 15  # CZ: 500
    Timeout_MS: int = 0


@dataclass
class UCIConfigArena:
    """
    This class overrides the UCI options from the UCIConfig class for the arena tournament.
    All other options will be taken from the UCIConfig class.
    """
    Centi_Temperature: int = 60


