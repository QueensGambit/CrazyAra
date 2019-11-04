### Configuration files for CrazyAra

If you want to test sample MCTS predictions in [MCTS_eval_demo.ipynb](https://github.com/QueensGambit/CrazyAra/blob/master/DeepCrazyhouse/src/samples/MCTS_eval_demo.ipynb),
 then follow these steps:
 
*   First rename `main_config_template.py` into `main_config.py`

*   Specify the fields `main_config["model_architecture_dir"]` and `main_config["model_weights_dir"]` in the file
    `main_config.py` to the appropriate paths of your system. Make sure that the path has a "/" at the end of the path.
