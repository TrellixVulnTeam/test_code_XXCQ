#!/bin/sh

## -- git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
# Do the following in the directory where you want to put your repo
git lfs install

## common tools
sudo apt-get install ffmpeg zip unzip


## -- Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# Run the following and then follow the instructions on your screen. Basically say yes to 
# all questions and use default installation location.
bash Miniconda3-latest-Linux-x86_64.sh

## -- Install pytorch (choose the version you want)
# pytorch 1.6
conda install -c pytorch pytorch==1.6.0 torchvision cudatoolkit=10.1 cudnn=7.6.0
# pytorch 1.7
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 cudnn=7.6.0 -c pytorch



