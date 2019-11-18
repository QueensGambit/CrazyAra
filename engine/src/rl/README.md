## Reinforcement learning

This directory contains the main files which are needed to run CrazyAra in selfplay mode.

<img align="right" src="https://www.docker.com/sites/default/files/d8/2019-07/horizontal-logo-monochromatic-white.png" width="256">

### Docker Image

For a convenient setup installation, we provide a
[docker installation file](https://github.com/QueensGambit/CrazyAra/blob/master/engine/src/rl/Dockerfile).
The docker image is based on the [official NVIDIA 
MXNet Docker image](https://docs.nvidia.com/deeplearning/frameworks/mxnet-release-notes/overview.html#overview) and
installs all additional libraries for reinforcement learning.
Lastly, it builds the CrazyAra executable from the C++ source code using the current repository state.

In order to build the docker image, use the following command:
 
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

For older docker version you can use `nvidia-docker run` command instead:
```shell script
nvidia-docker run -it --rm \
 -v local_dir:/data/RL crazyara_docker:latest
```

---

#### Network file
You can download a network which was trained via
 supervised learning as a starting point:

```shell script
cd /data/RL
wget https://github.com/QueensGambit/CrazyAra/releases/download/0.6.0/RISEv2-mobile.zip
unzip RISEv2-mobile.zip
```

#### Selfplay

Next you can start selplay from a given checkpoint file, which is stored in `model/`:

##### Generator
```shell script
python rl_loop.py --gpu_idx 0
```

##### Trainer
You need to specify at least one gpu to also update the current neural network weights.
The gpu trainer will stop generation and update the network as soon as enough training samples have been acquired.

```shell script
python rl_loop.py --gpu_idx 1 --trainer
```
