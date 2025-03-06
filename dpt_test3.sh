#!/bin/bash
source /home/xizh00005/.bashrc
conda activate nemo
ROOT='/home/xizh00005/project/DMSR/'
cd $ROOT


python generate_sketches.py --corruption speckle_noise --result_dir /share_chairilg/data/REAL275/dpt_output/ND/speckle_noise