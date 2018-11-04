"""
@file: setup.py
Created on 02.11.18
@project: CrazyAra
@author: queensgambit

Setup scripting for creating Cython binaries
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("DeepCrazyhouse/src/domain/agent/player/MCTSAgent.pyx", annotate=True)
)
