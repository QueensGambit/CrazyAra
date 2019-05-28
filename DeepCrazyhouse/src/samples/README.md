### Steps to run the jupyter notebook `MCTS_eval_demo.ipynb`

1. Rename `CrazyAra/configs/main_config_sample.py` into `CrazyAra/configs/main_config_sample.py`.

2. Specify the directories `model_architecture_dir` and `model_weights_dir` for the neural network.

You find the `.sym` file at releases in `CrazyAra_0.X.X/sym`.

You can download the `.params` file e.g. `model-1.25948-0.589-0246.params` under assests of release CrazyAra_0.3.1.

---

The specification for the neural network file is:

`model-<combined_validation_loss>-<validation_move_accuracy>-<k-number of steps trained>.params`
