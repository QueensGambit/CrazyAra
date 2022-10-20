#  CrazyAra, a deep learning chess variant engine
#  Copyright (C) 2018  Johannes Czech, Moritz Willig, Alena Beyer
#  Copyright (C) 2019  Johannes Czech

#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Dockerfile for installing all dependencies to run CrazyAra in a reinforcement learning setting.
# This dockerfile is build based on the official MXNet-NVIDIA-Docker image.
# @file: Dockerfile
# Created on 08.11.2019
# @author: queensgambit

# Base this Dockerfile from the official NVIDIA-MXNet docker contaienr
# see release page for  all current available releases:
# https://docs.nvidia.com/deeplearning/frameworks/mxnet-release-notes/running.html
FROM nvcr.io/nvidia/pytorch:22.05-py3
ENV FRAMEWORK="pytorch"
# alternative arguments: Uncomment the following two lines if you want to use mxnet instead
# FROM nvcr.io/nvidia/mxnet:20.09-py3
# ENV FRAMEWORK="mxnet"

MAINTAINER QueensGambit

# Info message before building
RUN if [ "${FRAMEWORK}" = "pytorch" ]; then \
echo "choose pytorch" ; \
else \
echo "choose mxnet" ; \
fi

# General dependencies to run CrazyAra
# -------------------------------------
# Install blaze
RUN cd /root \
    && wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.6.tar.gz \
    && tar -xvzf blaze-3.6.tar.gz \
    && cd blaze-3.6 \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    && make install

# Clone TensorRT repository for TensorRT backend
RUN cd /root \
    && git clone https://github.com/NVIDIA/TensorRT \
    && cd TensorRT

RUN if [ "${FRAMEWORK}" = "pytorch" ]; then \
# checkout commit for tag 22.05
cd /root/TensorRT \
&& git checkout 99a11a5fcdd1f184739bb20a8c4a473262c8ecc8 ; \
else \
# checkout commit for tag 20.09 ("git checkout tags/20.09" is currently not working)
cd /root/TensorRT \
&& git checkout f693a6d723ef2766be36deb5e7987cd50159973a ; \
fi

ENV TENSORRT_PATH /root/TensorRT/
ENV CUDA_PATH /usr/local/cuda/
                                                                                                                                                                                                                                                                
# Additional dependencies for reinforcement learning                                                                                                                                                                                                                               
# ---------------------------------------------------                                                                                                                                                                                                                              
# Install boost library (dependecy of z5 library, preinstalled on normal Debian systems)
RUN cd /root \
    && wget https://sourceforge.net/projects/boost/files/boost/1.70.0/boost_1_70_0.tar.bz2 \
    && tar --bzip2 -xf boost_1_70_0.tar.bz2 \
    && cd boost_1_70_0 \
    && ./bootstrap.sh \
    && ./b2 install

# Install XTL (dependency for XTensor)
RUN cd /root \
    && git clone https://github.com/xtensor-stack/xtl.git \
    && cd xtl \
    && git checkout tags/0.6.5 \
    && cmake CMakeLists.txt \
    && make install

# Install XTensor (used in z5 library)
RUN cd /root \
    && git clone https://github.com/xtensor-stack/xtensor.git \
    && cd xtensor \
    && git checkout tags/0.20.8 \
    && cmake CMakeLists.txt \
    && make install

# Install z5-library (C++ interface for zarr data format)
RUN cd /root \
    && git clone https://github.com/constantinpape/z5.git \
    && cd z5 \
    && git checkout tags/2.0.5
ENV Z5_PATH /root/z5/

# Clone the CrazyAra repository
# Install python dependencies for training the neural network
RUN pwd
RUN cd /root \
    && git clone https://github.com/Chooga/CrazyAra.git --recursive \
    && cd /root/CrazyAra/ \
    && git checkout tictactoe \
    && cd /root/CrazyAra/DeepCrazyhouse/src/training/ \
    && /usr/bin/python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# prepare setup for pytorch
RUN if [ "${FRAMEWORK}" = "pytorch" ]; then \
    pip uninstall -y mxboard \
    && pip install --upgrade onnx ; \
fi

# Install graphviz for plotting NN architectures
RUN apt-get update -y \
    && apt-get install -y graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build CrazyAra executable with reinforcement learning support
# --------------------------------------------------------------
RUN cd /root/CrazyAra/engine \
    && mkdir build \
    && cd build
RUN if [ "${FRAMEWORK}" = "pytorch" ]; then \
    cd /root/CrazyAra/engine/build \
    && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_RL=ON -DMODE_CRAZYHOUSE=0 -DMODE_CHESS=1 -DMODE_LICHESS=0 .. \
    && make -j8 ; \
else \
    cd /root/CrazyAra/engine/build \
    && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_RL=ON -DMODE_CRAZYHOUSE=0 -DMODE_CHESS=1 -DMODE_LICHESS=0 -DBACKEND_TENSORRT_7=ON .. \
    && make -j8 ; \
fi

#-DBACKEND_TENSORRT_7=ON
# Rename the config files
RUN mv /root/CrazyAra/DeepCrazyhouse/configs/main_config_template.py /root/CrazyAra/DeepCrazyhouse/configs/main_config.py 

# Install debugging utilities for "coredumpctl gdb"
RUN apt-get update -y \
     && apt-get install -y systemd-coredump \
     && apt-get install -y gdb \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/*

CMD cd /root/CrazyAra/engine/src/rl/ \
    && /bin/bash
