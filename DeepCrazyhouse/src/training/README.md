## Supervised learning

### Prerequisites

Make sure to have a recent [Pytorch](https://pytorch.org/get-started/locally/) version with CUDA support installed.

For supervised training you need the following [additional libraries](https://github.com/QueensGambit/CrazyAra/blob/master/DeepCrazyhouse/src/training/requirements.txt):

```bash
pip install -r requirements.txt
```

#### Training with MXNet or Gluon (deprecated)
Make sure to have a recent [MXNet](https://mxnet.incubator.apache.org/index.html) version with CUDA support installed:
 ```bash
 pip install mxnet-cu<cuda_version>==<version_id>
```

You need to install the following libraries when training with MXNet:
```bash
    pip install -y mxboard
    pip uninstall -y onnx
    pip install onnx==1.3.0
```

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
After a successful training run you can export the outputs as a html-file:  `File->Download as->Html(.html)`.

### Tensorboard
The [tensorboard](https://github.com/tensorflow/tensorboard) log files will be exported in `./logs` which can be viewed with tensorboard during training.
Use `tensorboard --logdir logs` to launch tensorboard from the command line.
Every time progress was made on the validation data a checkpoint weight file will be export in `./weights`.

### Training configuration

The training settings are defined in the first block of `train_cnn.ipynb`.
If you experience a cuda memory error at the start of training, increase the `div_factor` variable to e.g. `2` or more.
This will reduce the batch-size and learning rate accordingly.

### Start training from a Docker container

The training can also be started from the [crazyara docker container](https://github.com/QueensGambit/CrazyAra/blob/master/engine/src/rl/Dockerfile)
or the [official NVIDIA MXNet Docker container](https://docs.nvidia.com/deeplearning/frameworks/mxnet-release-notes/overview.html#overview)
and installing the packages in [requirements.txt](https://github.com/QueensGambit/CrazyAra/blob/master/DeepCrazyhouse/src/training/requirements.txt). 

```bash
docker run --gpus '"device=0"' --shm-size 16G --memory 64G -it \
 --rm -v ~/data:/data/SL -p "8888:8888" -p "6006:6006" \
 --name crazyara_training crazyara_docker:latest
```

Next, you can access the jupyter notebook in your browser:

`<IP-Address of server>:8888`

and the Tensorboard on:

`<IP-Address of server>:6006`


For older docker versions use:
```bash
nvidia-docker run --shm-size 16G --memory 64G -it \
 --rm -v ~/data:/data/SL -p "8888:8888" -p "6006:6006" \
 --name crazyara_training crazyara_docker:latest
```

Then you can start a notebook-server within the NVIDIA-docker container:
```bash
jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --no-browser .
```
and access the notebook by replacing `127.0.0.1` with the respective IP-address of the server in the URL.
