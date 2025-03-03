!/bin/bash
source /home/xizh00005/.bashrc
conda activate oldnet
ROOT='/home/xizh00005/project/DMSR/'
cd $ROOT

python evaluate.py --corrupt_roi 1 --corruption gaussian_noise --result_dir ./results/NDNP/gaussian_noise
python evaluate.py --corrupt_roi 1 --corruption speckle_noise --result_dir ./results/NDNP/speckle_noise
python evaluate.py --corrupt_roi 1 --corruption gaussian_blur --result_dir ./results/NDNP/gaussian_blur

# python evaluate.py --corrupt_roi 1 --corruption defocus_blur --result_dir ./results/NDNP/defocus_blur
# python evaluate.py --corrupt_roi 1 --corruption frost --result_dir ./results/NDNP/frost
# python evaluate.py --corrupt_roi 1 --corruption fog --result_dir ./results/NDNP/fog
# python evaluate.py --corrupt_roi 1 --corruption jpeg_compression --result_dir ./results/NDNP/jpeg_compression
# python evaluate.py --corrupt_roi 1 --corruption elastic_transform --result_dir ./results/NDNP/elastic_transform


# python evaluate.py --corrupt_roi 0 --corruption defocus_blur --result_dir ./results/NFNP/defocus_blur
# python evaluate.py --corrupt_roi 0 --corruption frost --result_dir ./results/NFNP/frost
# python evaluate.py --corrupt_roi 0 --corruption fog --result_dir ./results/NFNP/fog
# python evaluate.py --corrupt_roi 0 --corruption jpeg_compression --result_dir ./results/NFNP/jpeg_compression
# python evaluate.py --corrupt_roi 0 --corruption elastic_transform --result_dir ./results/NFNP/elastic_transform