# Crazy House Project Source

## Setup
The former python MCTS version of this project uses MXNet. To install MXNet look at
[https://mxnet.apache.org/](https://mxnet.apache.org/)

 Adjust the paths in `DeepCrazyhouse/configs/main_config.py` to the correct location on your file system.
 The most important entries to change are `model_architecture_dir` and `model_weights_dir`.

## General project structure
The source code is divided into four main parts: model, training, experiments and
samples. The different parts are located in identically named folders.

The `domain` folder contains domain classes, which present the problem domain of Crazyhouse.
We define 'domain' in the sense of a data class (as it is commonly defined as
e.g. in the Model-View-Controller pattern).
This includes classes to load, store and (pre)process game data, as well as
classes to analyse, evaluate, or modify board positions.

The `training` folder contains classes to train, evaluate and test
neural networks. Each class typically represents a single Trainer, which
later can be setup and run in the different experiments.

The `experiments` folder stores all experiments run for this project. For each
experiment a new jupyter notebook has to be created. The notebook should briefly
describe the experiment setup. The experiments code as well as the resulting
output should be saved directly within the notebook.

The `samples` folder contains a collection of arbitrary python files or
jupyter notebooks, to demonstrate some code or test functionality.


All python code should (more or less) follow the PEP8 standard.


## Folder structure
```
|- domain
|    Classes to which describe the Crazyhouse domain for neural networks 
|- training
|    Setup scripts to be using in training.
|   (loading data, setting up learners, ...)
|- experiments
 |   Folder containing the results of the run experiments.
 |- mw000_sample.ipynb
 |   Demo template to start with
 |- {user_id}{id}_{experiment_name}.ipynb
     Naming should follow this naming pattern to prevent
     merge conficts.
```

---

Avoid committing the outputs of jupyter notebook files by using the script `ipynb_output_filter.py` on the repository
 * https://github.com/toobaz/ipynb_output_filter

The repository includes instructions on how to set-up an automatic workflow.

