### Configuration files for CrazyAra

This is the configuration directory of CrazyAra for both supervised training and reinforcement learning.

If you want to test sample MCTS predictions in [MCTS_eval_demo.ipynb](https://github.com/QueensGambit/CrazyAra/blob/master/DeepCrazyhouse/src/samples/MCTS_eval_demo.ipynb),
 then follow these steps:

*   Specify the fields `main_config["model_architecture_dir"]` and `main_config["model_weights_dir"]` in the file
    `main_config.py` to the appropriate paths of your system. Make sure that the path has a "/" at the end of the path.
