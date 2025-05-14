#!/bin/bash
source /home/xizh00005/.bashrc
conda activate nemo
ROOT='/home/xizh00005/project/DMSR/'
cd $ROOT

python generate_sketches.py --dataset HouseCat6D --split train --result_dir /share_chairilg/data/HouseCat6D/dpt_output/train
