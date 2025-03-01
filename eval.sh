conda activate dmsr
python evaluate.py --corruption speckle_noise --result_dir ./results/speckle_noise
python evaluate.py --corruption gaussian_blur --result_dir ./results/gaussian_blur
python evaluate.py --corruption gaussian_noise --result_dir ./results/gaussian_noise
