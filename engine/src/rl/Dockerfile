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

FROM nvcr.io/nvidia/mxnet:20.01-py3

MAINTAINER QueensGambit

# General dependencies to run CrazyAra
# -------------------------------------
# Install blaze
RUN cd /root \
    && wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.6.tar.gz \
    && tar -xvzf blaze-3.6.tar.gz \
    && cd blaze-3.6 \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    && make install

# Clone MXNet repository
RUN cd /root \
    && git clone https://github.com/apache/incubator-mxnet --recursive \
    && cd incubator-mxnet/cpp-package/include/mxnet-cpp/ \
    && wget https://raw.githubusercontent.com/dmlc/MXNet.cpp/master/include/mxnet-cpp/op.h
ENV MXNET_PATH /root/incubator-mxnet/

# Clone TensorRT repository for TensorRT backend
RUN cd /root \
    && git clone https://github.com/NVIDIA/TensorRT
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
    && cmake CMakeLists.txt \
    && make install

# Install XTensor (used in z5 library)
RUN cd /root \
    && git clone https://github.com/xtensor-stack/xtensor.git \
    && cd xtensor \
    && cmake CMakeLists.txt \
    && make install

# Install z5-library (C++ interface for zarr data format)
RUN cd /root \
    && git clone https://github.com/constantinpape/z5.git
ENV Z5_PATH /root/z5/

# Clone the CrazyAra repository
RUN cd /root \
    && git clone https://github.com/QueensGambit/CrazyAra.git --recursive

# Install python dependencies for training the neural network
RUN cd /root/CrazyAra/DeepCrazyhouse/src/training/ \
    && pip install -r requirements.txt

# Install graphviz for plotting NN architectures
RUN apt-get update && apt-get install -y --no-install-recommends graphviz=0.8.4 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build CrazyAra executable
RUN export MXNET_PATH=/root/incubator-mxnet/ \
    && cd /root/CrazyAra/engine \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release ..

CMD make -j8

CMD cd /root
CMD /bin/bash
