### Steps to run the jupyter notebook [MCTS_eval_demo.ipynb](https://github.com/QueensGambit/CrazyAra/blob/master/DeepCrazyhouse/src/samples/MCTS_eval_demo.ipynb)

1. Rename `CrazyAra/DeepCrazyhouse//configs/main_config_template.py` into `CrazyAra/configs/main_config.py`.

2. Specify the directories `model_architecture_dir` and `model_weights_dir` for the neural network
 in the file `main_config.py`.

You can find the `.sym` file and `.params` file in the
 [CrazyAra 0.5.0](https://github.com/QueensGambit/CrazyAra/releases) release in the `model/` directory.

---

The specification for the neural network file is:

`model-<combined_validation_loss>-<validation_move_accuracy>-<k-number of steps trained>.params`
