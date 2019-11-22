## Supervised learning

### Prerequisites

Make sure to have a recent [MXNet](https://mxnet.incubator.apache.org/index.html) version with CUDA support installed:
 ```bash
 pip install mxnet-cu<cuda_version>==<version_id>
```

For supervised training you need the following [additional libraries](https://github.com/QueensGambit/CrazyAra/blob/master/DeepCrazyhouse/src/training/requirements.txt):

```bash
pip install -r requirements.txt
```

* zarr (chunked, compressed, N-dimensional array library)
* numcodecs (compression codec library)
* tqdm: (progress bar library)
* MXBoard (logging MXNet data for visualization in TensorBoard)

### Training data specification
Specify the directories `"planes_train_dir"`, `"planes_val_dir"`, `"planes_test_dir"`, `"planes_mate_in_one_dir"` at
[main_config_template.py](https://github.com/QueensGambit/CrazyAra/blob/master/DeepCrazyhouse/configs/main_config_template.py)
to the directories of the preprocessed training data on your system. All directories should end with a `/`.
Then copy the configuration file and rename it to `main_config.py`.

You can create the plane representation from chess pgn-files using [convert_pgn_to_planes.ipynb](https://github.com/QueensGambit/CrazyAra/blob/master/DeepCrazyhouse/src/preprocessing/convert_pgn_to_planes.ipynb).

### Jupyter notebooks
Use `train_cnn.ipynb` to conduct a training run.
`train_cnn.ipynb` is a jupyter notebook file which can be opened with jupyter and installed using `pip` or anaconda: 
* <https://jupyter.org/install.html>

Jupyter notebooks are displayed in a web-browser and can be launched with `jupyter notebook` from the command line. 
After a successfull training run you can export the outputs as a html-file:  `File->Download as->Html(.html)`.

### Tensorboard
The [tensorboard](https://github.com/tensorflow/tensorboard) log files will be exported in `./logs` which can be viewed with tensorboard during training.
Use `tensorboard --logdir logs` to launch tensorboard from the command line.
Every time progress was made on the validation data a checkpoint weight file will be export in `./weights`.

### Training configuration

The training settings are defined in the first block of `train_cnn.ipynb`.
If you experience a cuda memory error at the start of training, increase the `div_factor` variable to e.g. `2` or more.
This will reduce the batch-size and learning rate accordingly.
