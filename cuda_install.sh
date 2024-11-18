#!/bin/bash
# https://developer.nvidia.com/cuda-downloads
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# NVIDIA Driver Instructions (choose one option)
#sudo apt-get install -y nvidia-open
sudo apt-get install -y cuda-drivers

### ---- 
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

## test 
## (base) ➜  ~ nvcc -V
## nvcc: NVIDIA (R) Cuda compiler driver
## Copyright (c) 2005-2024 NVIDIA Corporation
## Built on Thu_Sep_12_02:18:05_PDT_2024
## Cuda compilation tools, release 12.6, V12.6.77
## Build cuda_12.6.r12.6/compiler.34841621_0
## (base) ➜  ~ pwd
## /home/xlisp
## (base) ➜  ~
## 
