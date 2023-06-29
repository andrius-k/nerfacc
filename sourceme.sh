#!/bin/bash

conda activate venv

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/project/tang/anpei/Enviroment/anaconda3/envs/venv/x86_64-conda-linux-gnu/lib/
export CUDA_HOME=$CONDA_PREFIX 
export CXX=g++ 
export CC=gcc
