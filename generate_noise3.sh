#!/bin/bash
source /home/xizh00005/.bashrc
conda activate nemo
ROOT='/home/xizh00005/project/DMSR/'
cd $ROOT

python generate_noisy_set_h6f.py --corruption fog
