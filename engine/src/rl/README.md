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

In order to build the docker container, use the following command:
 
```shell script
 docker build -t crazyara_docker .
```

Afterwards you can start the container using a specified list of GPUs:
```shell script
docker run --gpus '"device=10,11,12"' -it \
 --rm -v local_dir:/data/RL crazyara_docker:latest
```
If you want to use all available gpu, use `-gpus all` instead.

The parameter `-v` describes the mount directory, where the selfplay data will be stored.

For older docker version you can use the `nvidia-docker run` command instead:
```shell script
nvidia-docker run -it --rm \
 -v <local_dir>:/data/RL crazyara_docker:latest
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
wget https://github.com/QueensGambit/CrazyAra/releases/download/0.6.0/RISEv2-mobile.zip
unzip RISEv2-mobile.zip
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

<img class="emoji" title=":warning:" alt=":warning:" src="https://camo.githubusercontent.com/163bb588effffc0b9d07dac9331ace26e508806732e5f0a66bad309c3a2a5784/68747470733a2f2f6769746875622e6769746875626173736574732e636f6d2f696d616765732f69636f6e732f656d6f6a692f756e69636f64652f323661302e706e67" height="20" width="20" align="absmiddle" data-canonical-src="https://github.githubassets.com/images/icons/emoji/unicode/26a0.png">
There can only be one trainer and it must be started before starting any generators to ensure correct indexing.

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

---

#### Useful commands

*   `nvidia-smi`: Shows GPU utilization
*   `docker images`: Lists all availabe docker images
*   `docker ps`: List all running docker containers
*   `Ctrl-p + Ctrl-q`: To detach the tty without exiting the shell. Processes will continue running in daemon mode.
*   `docker exec -it [container-id] bash`: Enter a running docker container in shell mode
*   `docker kill [OPTIONS] CONTAINER [CONTAINER...]`: Kill one or more running containers
*   `docker image rm [OPTIONS] IMAGE [IMAGE...]`: Remove one or more images
