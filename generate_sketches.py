import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
import pickle as cPickle

import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.utils import get_bbox

print(torch.cuda.is_available())
parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default="HouseCat6D",
    help="HouseCat6D, Real",
)


parser.add_argument(
    "--split",
    type=str,
    default="train",
    help="train, test",
)

parser.add_argument(
    "--corruption",
    type=str,
    default="",
    help="['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']",
)
parser.add_argument(
    "--result_dir",
    type=str,
    default="/share_chairilg/data/HouseCat6D/dpt_output/train",
    help="result directory",
)
parser.add_argument("--gpu", type=str, default="1", help="GPU to use")

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

if opt.dataset == "HouseCat6D":
    opt.data_dir = "/share_chairilg/data/HouseCat6D"
    opt.corruption = "gaussian_noise"
    img_list=[]
    if opt.split == "train":
        train_scenes_rgb = glob.glob(os.path.join(opt.data_dir,'scene*','rgb'))
        train_scenes_rgb.sort()
        detection_dir = "/share_chairilg/data/HouseCat6D/{scene}/labels/{img_id}_label.pkl" # use GT detection
    else:
        train_scenes_rgb = glob.glob(os.path.join(opt.data_dir,"test",'test_scene*','rgb'))
        train_scenes_rgb.sort()
        detection_dir = "/share_chairilg/data/HouseCat6D/groundedsam_segmentation"


    for scene in train_scenes_rgb:
        img_paths = glob.glob(os.path.join(scene, '*.png'))
        img_paths.sort()
        img_paths = img_paths[:-1]
        for img_path in img_paths:
            img_list.append(img_path)

    print(f"Found {len(img_list)} images in {opt.dataset}")

else:
    opt.data_dir = "/share_chairilg/data/REAL275"
    file_path = "Real/test_list.txt"
    img_list = [
        os.path.join(file_path.split("/")[0], line.rstrip("\n"))
        for line in open(os.path.join(opt.data_dir, file_path))
    ]

    print(f"Found {len(img_list)} images in {opt.dataset}")


save_dpt_dir = opt.result_dir

# Depth estimation model: expects input images 384x384 normalized [-1,1]
# image = (image - mean) / std
depth_norm_color = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)

# Surface normal estimation model: expects input images 384x384 normalized [0,1]
normal_norm_color = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Surface normal estimation model: expects input images 384x384 normalized [0,1]
model_normal = torch.hub.load(
    "alexsax/omnidata_models", "surface_normal_dpt_hybrid_384"
)

# Depth estimation model: expects input images 384x384 normalized [-1,1]
model_depth = torch.hub.load("alexsax/omnidata_models", "depth_dpt_hybrid_384")


model_depth.cuda()
model_depth.eval()
model_normal.cuda()
model_normal.eval()

from PIL import Image


def detect():
    # get test data list
    for img_id, path in tqdm(enumerate(img_list), total=len(img_list)):
        if opt.dataset == "HouseCat6D":
            raw_rgb = cv2.imread(path)[:, :, :3]
            img_path_parsing = path.split("/")
            scene_name = img_path_parsing[-3]
            img_name = img_path_parsing[-1][:-4] # remove .png
            if opt.split == "train":
                detection_file = detection_dir.format(scene=scene_name,img_id=img_name)
            else:
                detection_file = os.path.join(
                    detection_dir,
                    "results_{}_{}.pkl".format(scene_name, img_name),
                )

            pred_depth_path = os.path.join(save_dpt_dir, scene_name, "{}_depth.pkl".format(img_name))
            pred_normal_path = os.path.join(save_dpt_dir,scene_name, "{}_normal.pkl".format(img_name))

            os.makedirs(
                os.path.join(save_dpt_dir,scene_name), exist_ok=True
            )
        else:
            tmp_path = path.split("/")[1:]
            tmp_path = "/".join(tmp_path)
            tmp_path = os.path.join(opt.data_dir, "NoiseReal", opt.corruption, tmp_path)

            raw_rgb = cv2.imread(tmp_path + "_color.png")[:, :, :3]
            raw_rgb = raw_rgb[:, :, ::-1]

            img_path = os.path.join(opt.data_dir, path)
            # load mask-rcnn detection results
            img_path_parsing = img_path.split("/")

            detection_file = os.path.join(
                    f"/share_chairilg/data/REAL275/NoiseReal/{opt.corruption}/detections",
                    "results_{}_{}_{}.pkl".format(
                        opt.data.split("_")[-1], img_path_parsing[-2], img_path_parsing[-1]
                    ),
                )
            pred_depth_path = os.path.join(save_dpt_dir, path + "_depth.pkl")
            pred_normal_path = os.path.join(save_dpt_dir, path + "_normal.pkl")

            os.makedirs(
                os.path.join(save_dpt_dir, "/".join(path.split("/")[:-1])), exist_ok=True
            )

        # load mask-rcnn detection results
        with open(detection_file, "rb") as f:
            mrcnn_result = cPickle.load(f)

        num_insts = len(mrcnn_result["class_ids"])

        # Handle the different dict keys in GT detection dict
        if "rois" not in mrcnn_result:
            mrcnn_result["rois"] = mrcnn_result["bboxes"]
            
        # load dpt depth predictions
        depths = []
        normals = []

        for i in range(num_insts):
            rmin, rmax, cmin, cmax = get_bbox(mrcnn_result["rois"][i], raw_rgb.shape[0], raw_rgb.shape[1])
            rgb = raw_rgb[rmin:rmax, cmin:cmax, :]
            rgb = cv2.resize(rgb, (384, 384), interpolation=cv2.INTER_LINEAR)

            with torch.no_grad():
                rgb_d = depth_norm_color(rgb).unsqueeze(0).cuda()
                rgb_n = normal_norm_color(rgb).unsqueeze(0).cuda()
                pred_normal = model_normal(rgb_n)[0]
                pred_depth = model_depth(rgb_d)[0]

                pred_normal = pred_normal * 2 - 1
                ds_normal = torch.nn.functional.interpolate(
                    pred_normal.unsqueeze(0),
                    size=(192, 192),
                    mode="bilinear",
                ).squeeze()

                # between [0,1]
                ds_depth = torch.nn.functional.interpolate(
                    pred_depth.unsqueeze(0).unsqueeze(0),
                    size=(192, 192),
                    mode="bilinear",
                ).squeeze()

                depths.append(ds_depth.cpu().numpy())
                normals.append(ds_normal.permute(1, 2, 0).cpu().numpy())

                # map normal from [0,1] to [-1, 1]
                # ds_normal = ds_normal * 2 - 1

                # plt.subplot(2, 2, 1)
                # plt.imshow(depths[i])
                # plt.subplot(2, 2, 2)
                # plt.imshow(ds_depth.detach().cpu().numpy())
                # plt.subplot(2, 2, 3)
                # plt.imshow(normals[i])
                # plt.subplot(2, 2, 4)
                # plt.imshow(ds_normal.permute(1, 2, 0).detach().cpu().numpy())
                # plt.show()

        # [num_insts, 192,192]
        packed_depth = np.stack(depths, axis=0)
        # [num_insts, 192,192,3]
        packed_normal = np.stack(normals, axis=0)

        with open(pred_depth_path, "wb") as f:
            # [num_insts, 192,192]
            cPickle.dump(packed_depth, f)

        with open(pred_normal_path, "wb") as f:
            # [num_insts, 192,192,3]
            cPickle.dump(packed_normal, f)


if __name__ == "__main__":
    detect()
