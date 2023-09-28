"""
@file: generate_cutechess_command
Created on 21.09.2023
@project: CrazyAra
@author: Felix

Generates the command for running a cutechess experiment based on the given parameters
"""

device = 4
batch_size = 16
nodes = 800
movetime = 0
player_a = "specific_opening"
player_b = "no_phases"

pgnout = f"/data/cutechess_results/{player_a}_vs_{player_b}_movetime{movetime}_nodes{nodes}_bs{batch_size}.pgn"
command = f"./cutechess-cli -variant standard -openings file=chess.epd format=epd order=random -pgnout {pgnout} -resign movecount=5 score=600 -draw movenumber=30 movecount=4 score=20 -concurrency 1 " \
          f"-engine name=ClassicAra_{player_a} cmd=./ClassicAra dir=~/CrazyAra/engine/build option.Model_Directory=/data/model/ClassicAra/chess/{player_a} proto=uci " \
          f"-engine name=ClassicAra_{player_b} cmd=./ClassicAra dir=~/CrazyAra/engine/build option.Model_Directory=/data/model/ClassicAra/chess/{player_b} proto=uci " \
          f"-each option.First_Device_ID={device} option.Batch_Size={batch_size} option.Fixed_Movetime={movetime} tc=0/6000+0.1 option.Nodes={nodes} option.Simulations={nodes*2} option.Search_Type=mcts -games 2 -rounds 500 -repeat " \
          f"|& tee {pgnout[:-4]}_output.txt"
print(command)
