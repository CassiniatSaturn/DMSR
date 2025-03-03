import os
import numpy
import cv2
import time
import glob
import argparse
import numpy as np
from tqdm import tqdm
import pickle as cPickle
import matplotlib.pyplot as plt

from imagecorruptions import corrupt, get_corruption_names
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument(
    "--corruption",
    type=str,
    default="motion_blur",
    help="['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']",
)

parser.add_argument("--data", type=str, default="real_test", help="val, real_test")

parser.add_argument(
    "--data_dir",
    type=str,
    default="/share_chairilg/data/REAL275",
    help="data directory",
)

parser.add_argument(
    "--result_dir",
    type=str,
    default="/share_chairilg/data/REAL275/NoiseReal",
    help="result directory",
)
opt = parser.parse_args()

result_dir = os.path.join(opt.result_dir, opt.corruption)

print("Corruption: ", get_corruption_names("validation"))

os.makedirs(result_dir, exist_ok=True)

file_path = "Real/test_list.txt"

img_list = [
    os.path.join(file_path.split("/")[0], line.rstrip("\n"))
    for line in open(os.path.join(opt.data_dir, file_path))
]

for img_id, path in tqdm(enumerate(img_list), total=len(img_list)):
    img_path = os.path.join(opt.data_dir, path + "_color.png")
    scene_idx = path.split("/")[1:3]
    os.makedirs(os.path.join(result_dir, *scene_idx), exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    img = np.array(img)

    img = corrupt(img, corruption_name=opt.corruption)

    path = path.split("/")[1:]
    path = "/".join(path)

    Image.fromarray(img).save(os.path.join(result_dir, path + "_color.png"))
