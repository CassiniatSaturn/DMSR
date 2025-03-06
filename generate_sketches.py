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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.sgpa import SPGANet
from lib.align import ransacPnP_LM
from lib.utils import load_depth, get_bbox, draw_detections, compute_mAP

print(torch.cuda.is_available())
parser = argparse.ArgumentParser()

parser.add_argument(
    "--use_gt", type=int, default=0, help="use GT mask as detection results"
)
parser.add_argument(
    "--corruption",
    type=str,
    default="gaussian_noise",
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
    default="/share_chairilg/data/REAL275/dpt_output/gaussian_noise",
    help="result directory",
)
parser.add_argument("--gpu", type=str, default="1", help="GPU to use")

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

assert opt.data in ["val", "real_test"]
if opt.data == "val":
    cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
    file_path = "CAMERA/val_list.txt"
else:
    cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084
    file_path = "Real/test_list.txt"

K = np.eye(3)
K[0, 0] = cam_fx
K[1, 1] = cam_fy
K[0, 2] = cam_cx
K[1, 2] = cam_cy

save_dpt_dir = opt.result_dir

# path for shape & scale prior
mean_shapes = np.load("assets/mean_points_emb.npy")
with open("assets/mean_scale.pkl", "rb") as f:
    mean_scale = cPickle.load(f)

xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
norm_scale = 1000.0

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

from PIL import Image


def detect():
    # get test data list
    img_list = [
        os.path.join(file_path.split("/")[0], line.rstrip("\n"))
        for line in open(os.path.join(opt.data_dir, file_path))
    ]

    for img_id, path in tqdm(enumerate(img_list), total=len(img_list)):
        tmp_path = path.split("/")[1:]
        tmp_path = "/".join(tmp_path)
        tmp_path = os.path.join(opt.data_dir, "NoiseReal", opt.corruption, tmp_path)

        raw_rgb = cv2.imread(tmp_path + "_color.png")[:, :, :3]
        raw_rgb = raw_rgb[:, :, ::-1]

        img_path = os.path.join(opt.data_dir, path)
        # load mask-rcnn detection results
        img_path_parsing = img_path.split("/")

        mrcnn_path = os.path.join(
                f"/share_chairilg/data/REAL275/NoiseReal/{opt.corruption}/detections",
                "results_{}_{}_{}.pkl".format(
                    opt.data.split("_")[-1], img_path_parsing[-2], img_path_parsing[-1]
                ),
            )

        # mrcnn_path = os.path.join(
        #     f"/share_chairilg/data/REAL275/deformnet_eval/mrcnn_results/{opt.data}",
        #     "results_{}_{}_{}.pkl".format(
        #         opt.data.split("_")[-1], img_path_parsing[-2], img_path_parsing[-1]
        #     ),
        # )

        with open(mrcnn_path, "rb") as f:
            mrcnn_result = cPickle.load(f)

        if opt.use_gt:
            with open(img_path + "_label.pkl", "rb") as f:
                gts = cPickle.load(f)
            # Read the gt detection
            mrcnn_result["rois"] = gts["bboxes"]
            mrcnn_result["class_ids"] = gts["class_ids"]

            mask_packed = cv2.imread(img_path + "_mask.png")[..., 0]

            gt_masks = np.stack(
                [
                    (mask_packed == (i + 1)).astype(bool)
                    for i in range(len(gts["class_ids"]))
                ],
                axis=0,
            )
            mrcnn_result["masks"] = gt_masks.transpose(1, 2, 0)

        num_insts = len(mrcnn_result["class_ids"])
        # load dpt depth predictions
        depths = []
        normals = []

        # if num_insts != 0:
        #     dpt_dir = "/share_chairilg/data/REAL275/dpt_output"
        #     pred_depth_path = os.path.join(dpt_dir, path + "_depth.pkl")

        #     if not os.path.exists(pred_depth_path):
        #         print(f"File {pred_depth_path} does not exist")
        #         continue
        #     with open(pred_depth_path, "rb") as f:
        #         # [num_insts, 192,192]
        #         pred_depth_all = cPickle.load(f)
        #     pred_normal_path = os.path.join(dpt_dir, path + "_normal.pkl")
        #     with open(pred_normal_path, "rb") as f:
        #         # [num_insts, 192,192,3]
        #         pred_normal_all = cPickle.load(f)

        for i in range(num_insts):
            rmin, rmax, cmin, cmax = get_bbox(mrcnn_result["rois"][i])
            rgb = raw_rgb[rmin:rmax, cmin:cmax, :]
            rgb = cv2.resize(rgb, (384, 384), interpolation=cv2.INTER_LINEAR)

            model_normal.cuda()
            model_depth.cuda()
            model_normal.eval()
            model_depth.eval()

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
                # plt.imshow(pred_depth_all[i])
                # plt.subplot(2, 2, 2)
                # plt.imshow(ds_depth.detach().cpu().numpy())
                # plt.subplot(2, 2, 3)
                # plt.imshow(pred_normal_all[i])
                # plt.subplot(2, 2, 4)
                # plt.imshow(ds_normal.permute(1, 2, 0).detach().cpu().numpy())
                # plt.show()

        # [num_insts, 192,192]
        packed_depth = np.stack(depths, axis=0)
        # [num_insts, 192,192,3]
        packed_normal = np.stack(normals, axis=0)

        pred_depth_path = os.path.join(save_dpt_dir, path + "_depth.pkl")
        pred_normal_path = os.path.join(save_dpt_dir, path + "_normal.pkl")

        os.makedirs(
            os.path.join(save_dpt_dir, "/".join(path.split("/")[:-1])), exist_ok=True
        )

        with open(pred_depth_path, "wb") as f:
            # [num_insts, 192,192]
            cPickle.dump(packed_depth, f)

        with open(pred_normal_path, "wb") as f:
            # [num_insts, 192,192,3]
            cPickle.dump(packed_normal, f)


if __name__ == "__main__":
    detect()
