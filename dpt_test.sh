#!/bin/bash
source /home/xizh00005/.bashrc
conda activate oldnet
ROOT='/home/xizh00005/project/DMSR/'
cd $ROOT

python generate_sketches.py --corruption defocus_blur --result_dir /share_chairilg/data/REAL275/dpt_output/defocus_blur
python generate_sketches.py --corruption frost --result_dir /share_chairilg/data/REAL275/dpt_output/frost
python generate_sketches.py --corruption fog --result_dir /share_chairilg/data/REAL275/dpt_output/fog
python generate_sketches.py --corruption jpeg_compression --result_dir /share_chairilg/data/REAL275/dpt_output/jpeg_compression
python generate_sketches.py --corruption elastic_transform --result_dir /share_chairilg/data/REAL275/dpt_output/elastic_transform