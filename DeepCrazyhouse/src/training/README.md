## Supervised learning

### Prerequisites

Make sure to have a recent [Pytorch](https://pytorch.org/get-started/locally/) version with CUDA support installed.

For supervised training you need the following [additional libraries](https://github.com/QueensGambit/CrazyAra/blob/master/DeepCrazyhouse/src/training/requirements.txt):

```bash
pip install -r requirements.txt
```

### Training data specification
Specify the directories `"planes_train_dir"`, `"planes_val_dir"`, `"planes_test_dir"`, `"planes_mate_in_one_dir"` at
[main_config_template.py](https://github.com/QueensGambit/CrazyAra/blob/master/DeepCrazyhouse/configs/main_config_template.py)
to the directories of the preprocessed training data on your system. All directories should end with a `/`.
Then copy the configuration file and rename it to `main_config.py`.

You can create the plane representation from chess pgn-files using [convert_pgn_to_planes.ipynb](https://github.com/QueensGambit/CrazyAra/blob/master/DeepCrazyhouse/src/preprocessing/convert_pgn_to_planes.ipynb).

### Command line script
Use `train_cli.py` to conduct a training run.

An example call is:
```
$ train_cli.py --model-type resnet --name-initials XY --use-custom-architecture False --export-dir /data/training_run
```

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
that is also used for reinforcement learning
or the [official NVIDIA MXNet Docker container](https://docs.nvidia.com/deeplearning/frameworks/mxnet-release-notes/overview.html#overview)
and installing the packages in [requirements.txt](https://github.com/QueensGambit/CrazyAra/blob/master/DeepCrazyhouse/src/training/requirements.txt). 

After building the container
```bash
docker build -t crazyara_docker .
```
you can start it via the `docker run` command:
```bash
docker run --gpus all --privileged --shm-size 16G --memory 128G -it \
 --rm -v ~/data:/data/SL -p "8888:8888" -p "6006:6006" \
 --name crazyara_training crazyara_docker:latest
```

`--privileged` is required to run the Linux-init process to be able to use the apport service for generating core dumps.

Next, you need to detach from the container using `ctrl+p+q` and start a new docker-session:
```shell script
docker exec -it crazyara_training bash
```

Then you can start a notebook-server within the NVIDIA-docker container:
```bash
jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --no-browser .
```
and access the notebook using `127.0.0.1` or `localhost` on your local machine.

You also need to make sure to open the ssh session with `-L 8888:localhost:8888` or to add `LocalForward 8888 127.0.0.1:8888` in your `~/.ssh/config` file.
