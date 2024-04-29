#!/usr/bin/env bash

export CONDA_ENV_NAME=vibe-env
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.7

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

which python
which pip

pip install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0
pip install git+https://github.com/giacaglia/pytube.git --upgrade
pip install -r requirements.txt

#TO setup environment: conda activate vibe-env

#in case of error:
##pip install --force-reinstall torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/
#pip install --upgrade torch torchvision