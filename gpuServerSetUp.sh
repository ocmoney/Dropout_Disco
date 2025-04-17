#!/bin/bash

# -- Instructions-
# chmod u+x ./gpuServerSetUp.sh 
# source ./gpuServerSetUp.sh 
# ----------------

git clone https://github.com/ocmoney/Dropout_Disco.git
git config --global user.name "GPU Server"
git config --global user.email "jimmeylove@gmail.com"

cd ./Dropout_Disco
git checkout feature/model
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirementsLinux.txt 