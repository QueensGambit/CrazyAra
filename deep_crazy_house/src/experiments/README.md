## Training Runs

This directory contains all experiments that have been conducted in the course of the project.
Have a look at the file `experiments_summary.csv` for details about the history of the different experiments.
The past and early experiments are stored for completeness reasons.

The model (CrazyAra v0.1.0) which went live on lichess.org on 10th september was was trained via
`train_all_games_over_2000_elo/SGD/lr_0.001_mxnet` `train_all_games_over_2000_elo/SGD/lr_0.01_mxnet` 
and `train_all_games_over_2000_elo/SGD/lr_0.1_keras`.

The latest deeper model (CrazyAra v0.2.0) using the RISE architecture was trained using `train_all_games_over_2000_elo/gluon/Rise_1_cycle_training`,
`train_all_games_over_2000_elo/gluon/Rise_1_cycle_training/final_cooldown_1` and 
`train_all_games_over_2000_elo/gluon/Rise_1_cycle_training/final_cooldown_2`.

The more recent training runs store various metrics and the form of a tensorboard log-file.
You can inspect the logfile using the command:
`tensorboard --logdir logs` on your terminal aka. command line window.
--- 
The export was done using `mxboard`. For more details and installation instructions please visit:
* https://github.com/awslabs/mxboard



