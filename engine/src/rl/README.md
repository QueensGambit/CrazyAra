## Reinforcement learning

This directory contains the main files which are needed to run CrazyAra in selfplay mode.

<img align="right" src="https://www.docker.com/sites/default/files/d8/2019-07/horizontal-logo-monochromatic-white.png" width="128">

### Docker Image

For a convenient setup installation, we provide a
[Dockerfile](https://github.com/QueensGambit/CrazyAra/blob/master/engine/src/rl/Dockerfile).
The dockerfile is based on the [official NVIDIA 
MXNet Docker container](https://docs.nvidia.com/deeplearning/frameworks/mxnet-release-notes/overview.html#overview) and
installs all additional libraries for reinforcement learning.
Lastly, it compiles the CrazyAra executable from the C++ source code using the current repository state.
:warning: NVIDIA Docker [does not work on Windows](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#is-microsoft-windows-supported).

In order to build the docker container with pytorch support, use the following command:
 
```shell script
 docker build -t crazyara_docker .
```
if you want to use mxnet instead, uncomment the following lines in the Dockerfile before building:
```shell script
# FROM nvcr.io/nvidia/mxnet:20.09-py3
# ENV FRAMEWORK="mxnet"
```
and comment the line:
```shell script
FROM nvcr.io/nvidia/pytorch:22.05-py3
```

Afterwards you can start the container using a specified list of GPUs:
```shell script
docker run --gpus all --shm-size 16G --memory 64G -it --privileged \
 --rm -v ~/mnt:/data/RL --name crazyara_rl crazyara_docker:latest
```
If you want to launch the docker using only a subset of available you can specify them by e.g. `--gpus '"device=10,11,12"'` instead.

The parameter `-v` describes the mount directory, where the selfplay data will be stored.
`--privileged` is required to run the Linux-init process to be able to use the apport service for generating core dumps.

Next, you need to detach from the container using `ctrl+p+q` and start a new docker-session:
```shell script
docker exec -it crazyara_rl bash
```

---


#### CrazyAra binary

The Dockerfile builds the _CrazyAra_ binary from source with reinforcement learning support at `root/CrazyAra/engine/build/`.
Now, you can move the binary to the main reinforcement learning directory where the selfplay games are generated:
```shell script
mv /root/CrazyAra/engine/build/CrazyAra /data/RL
```

#### Network file
You can download a network which was trained via
 supervised learning as a starting point:

```shell script
cd /data/RL
wget https://github.com/QueensGambit/CrazyAra/releases/download/0.9.5/ClassicAra-sl-model-wdlp-rise3.3-input3.0.zip
unzip ClassicAra-sl-model-wdlp-rise3.3-input3.0.zip
```

Alternatively, if a model file is already available on the host machine, you can move the model directory into the mounted docker directory.

#### Selfplay

After all premilirary action have been done, you can finally start selfplay from a given checkpoint file, which is stored in the directory `/data/RL/model/`.
If you want to start learning from zero knowledge, you may use a set of weights which have initialized randomly.

If the program cannot find a model inside `/data/RL/model/` it will look at `/data/RL/model/<variant>/`, where `<variant>` is the selected chess variant.

The python script [**rl_loop.py**](https://github.com/QueensGambit/CrazyAra/blob/master/engine/src/rl/rl_loop.py) is the main script for managing the reinforcement learning loop.
It can be started in two different modes: a generator mode, and a generator+training mode.

```
cd /root/CrazyAra/engine/src/rl
```

##### Trainer
You need to specify at least one gpu to also update the current neural network weights.
The gpu trainer will stop generating games and update the network as soon as enough training samples have been acquired.

:warning: There can only be one trainer and it must be started before starting any generators to ensure correct indexing.

```shell script
python rl_loop.py --device-id 0 --trainer &
```

##### Generator
The other gpu's can be used to generate games.
```shell script
python rl_loop.py --device-id 1 &
```

#### Configuration
The main configuration files for reinforcement learning can be found at `/root/CrazyAra/DeepCrazyhouse/configs/`:
*   https://github.com/QueensGambit/CrazyAra/tree/master/DeepCrazyhouse/configs


#### Trouble Shooting

The docker container comes with automatic core dump generation by default.
If you encounter a crash of the executable, you can find the corresponding core dump in `/var/lib/apport/coredump/`.
In order to analyze the core dump you can use `gdb`:
`gdb path/to/the/binary path/to/the/core/dump/file`


---

#### Useful commands

*   `nvidia-smi`: Shows GPU utilization
*   `docker images`: Lists all availabe docker images
*   `docker ps`: List all running docker containers
*   `Ctrl-p + Ctrl-q`: To detach the tty without exiting the shell. Processes will continue running in daemon mode.
*   `docker attach [container-id]`: Attach to a running docker container session
*   `docker exec -it [container-id] bash`: Enter a running docker container in shell mode and create a new session
*   `docker kill [OPTIONS] CONTAINER [CONTAINER...]`: Kill one or more running containers
*   `docker image rm [OPTIONS] IMAGE [IMAGE...]`: Remove one or more images

* `tmux`: Start a new tmux session
* `tmux detach`: Detach from a tmux session
* `tmux attach -t <id>`: Attach to a tmux session
