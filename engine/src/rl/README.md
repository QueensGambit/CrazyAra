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

#### Selfplay

After all, premilirary action have been done, you can finally start selfplay from a given checkpoint file, which is stored in the directory `/data/RL/model/`.
If you want to start learning from zero knowledge, you may use a set of weights which have initialized randomly.

The python script [**rl_loop.py**](https://github.com/QueensGambit/CrazyAra/blob/master/engine/src/rl/rl_loop.py) is the main script for managing the reinforcement learning loop.
It can be started in two different modes: a generator mode, and a generator+training mode.

```
cd mv /root/CrazyAra/engine/src/rl
```


##### Generator
```shell script
python rl_loop.py --device-id 0&
```

##### Trainer
You need to specify at least one gpu to also update the current neural network weights.
The gpu trainer will stop generating games and update the network as soon as enough training samples have been acquired.

```shell script
python rl_loop.py --device-id 1 --trainer&
```

---

#### Useful commands

*   `nvidia-smi`: Shows GPU utilization
*   `docker images`: Lists all availabe docker images
*   `docker ps`: List all running docker containers
*   `Ctrl-p + Ctrl-q`: To detach the tty without exiting the shell. Processes will continue running in daemon mode.
*   `docker exec -it [container-id] bash`: Enter a running docker container in shell mode
*   `docker kill [OPTIONS] CONTAINER [CONTAINER...]`: Kill one or more running containers
*   `docker image rm [OPTIONS] IMAGE [IMAGE...]`: Remove one or more images
