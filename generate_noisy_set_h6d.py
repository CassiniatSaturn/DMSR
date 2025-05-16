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
    default="fog",
    help="['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']",
)

parser.add_argument("--data", type=str, default="real_test", help="val, real_test")

parser.add_argument(
    "--data_dir",
    type=str,
    default="/share_chairilg/data/HouseCat6D",
    help="data directory",
)

parser.add_argument(
    "--result_dir",
    type=str,
    default="/share_chairilg/data/REAL275/noisy_test",
    help="result directory",
)
opt = parser.parse_args()

result_dir = os.path.join(opt.result_dir, opt.corruption)

os.makedirs(result_dir, exist_ok=True)

# train_scenes_rgb = glob.glob(os.path.join(opt.data_dir,'test','test_scene*','rgb'))
# train_scenes_rgb.sort()

train_scenes_rgb = ['/share_chairilg/data/HouseCat6D/test/test_scene5/rgb']

img_list = []
for scene in train_scenes_rgb:
            img_paths = glob.glob(os.path.join(scene, '*.png'))
            img_paths.sort()
            for img_path in img_paths:
                img_list.append(img_path)

print(len(img_list))

for img_id, path in tqdm(enumerate(img_list), total=len(img_list)):
    raw_rgb = cv2.imread(path)
    img = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)
        
    # img_path = os.path.join(opt.data_dir, path + "_color.png")
    # scene_idx = path.split("/")[1:3]
    # os.makedirs(os.path.join(result_dir, *scene_idx), exist_ok=True)

    # img = Image.open(img_path).convert("RGB")
    # img = np.array(img)

    corrupted_img = corrupt(img, corruption_name=opt.corruption)
    
    save_path = path.replace('/test/',f'/noisy_test/{opt.corruption}/')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    Image.fromarray(corrupted_img).save(save_path)