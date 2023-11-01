"""
@file: run_cutechess_experiments
Created on 03.10.2023
@project: CrazyAra
@author: Felix

Executes cutechess commands in the shell
"""

import subprocess
from datetime import datetime

device = 0
player_a = "correct_phases"
player_b = "no_phases"

batch_size_options = [64]
use960 = True
movetime_options = [100, 200, 400, 800, 1600]
nodes_options = [100, 200, 400, 800, 1600, 3200]

variant = "fischerandom" if use960 else "standard"
openings_file = "960_openings.epd" if use960 else "chess.epd"
out_mode = "960_" if use960 else ""


def generate_and_run_command(batch_size, movetime, nodes):
    start_time = datetime.now()
    pgnout = f"/data/cutechess_results/{out_mode}{player_a}_vs_{player_b}_movetime{movetime}_nodes{nodes}_bs{batch_size}.pgn"
    command = f"./cutechess-cli -variant {variant} -openings file={openings_file} format=epd order=random -pgnout {pgnout} -resign movecount=5 score=600 -draw movenumber=30 movecount=4 score=20 -concurrency 1 " \
              f"-engine name={out_mode}ClassicAra_{player_a} cmd=./FH_ClassicAra dir=~/CrazyAra/engine/build option.Model_Directory=/data/model/ClassicAra/chess/{player_a} proto=uci " \
              f"-engine name={out_mode}ClassicAra_{player_b} cmd=./FH_ClassicAra dir=~/CrazyAra/engine/build option.Model_Directory=/data/model/ClassicAra/chess/{player_b} proto=uci " \
              f"-each option.First_Device_ID={device} option.Batch_Size={batch_size} option.Fixed_Movetime={movetime} tc=0/6000+0.1 option.Nodes={nodes} option.Simulations={nodes * 2} option.Search_Type=mcts -games 2 -rounds 500 -repeat "

    with open(f'{pgnout[:-4]}_output.txt', 'w') as f:
        print("=====================================")
        new_experiment_info = f"New Experiment: BS: {batch_size}, Movetime: {movetime}, Nodes: {nodes}\n"
        f.write(new_experiment_info)
        print(new_experiment_info)
        f.write(f"Start Time: {start_time.strftime('%m/%d/%Y, %H:%M:%S')}\n")
        print(f"Start Time: {start_time.strftime('%m/%d/%Y, %H:%M:%S')}")
        f.write(f"{command}\n")
        print(command)
        f.flush()
        result = subprocess.run(command, text=True, capture_output=False, check=True, shell=True, stdout=f, stderr=subprocess.STDOUT)
        #print(result.stdout, result.stderr)
        end_time = datetime.now()
        f.write(f"End Time: {end_time.strftime('%m/%d/%Y, %H:%M:%S')}\n")
        print(f"End Time: {end_time.strftime('%m/%d/%Y, %H:%M:%S')}")
        runtime_hours = (end_time-start_time)/60/60
        print(f"Runtime in hours: {runtime_hours}")
        f.write(f"Runtime in hours: {runtime_hours}\n")


for batch_size in batch_size_options:
    for nodes in nodes_options:
        movetime = 0
        generate_and_run_command(batch_size, movetime, nodes)
    for movetime in movetime_options:
        nodes = 0
        generate_and_run_command(batch_size, movetime, nodes)


