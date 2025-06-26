#!/usr/bin/env bash

apt-get update
apt-get upgrade -y
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get -y install cuda-toolkit-12-9
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash ./Anaconda3-2024.10-1-Linux-x86_64.sh
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda create -n tetris python=3.11.11 -y
conda activate tetris
conda env config vars set TF_FORCE_GPU_ALLOW_GROWTH=true TF_NUM_INTEROP_THREADS=32 TF_NUM_INTRAOP_THREADS=32
git clone https://github.com/m-sher/QTris.git
git clone https://github.com/m-sher/TFTetrisEnv.git
cd QTris
pip install -r requirements.txt
pip install ../TFTetrisEnv
