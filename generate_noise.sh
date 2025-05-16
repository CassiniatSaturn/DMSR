#!/bin/bash
source /home/xizh00005/.bashrc
conda activate nemo
ROOT='/home/xizh00005/project/DMSR/'
cd $ROOT

pip install imagecorruptions
python generate_noisy_set_h6d.py --corruption defocus_blur