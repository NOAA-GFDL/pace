#!/bin/bash

source /root/.bashrc

echo "y" | conda create --name pace python=3.8
conda init bash
conda activate pace

pip3 install --upgrade pip setuptools wheel
pip3 install -r requirements_dev.txt -c constraints.txt
