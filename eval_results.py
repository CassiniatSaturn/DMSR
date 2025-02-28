import glob
import os
import pickle as cPickle

import numpy as np
import matplotlib.pyplot as plt

from lib.utils import plot_mAP, compute_mAP
from lib.lapose_eval import compute_degree_cm_mAP


def plot_mAP(
    iou_aps,
    pose_aps,
    out_dir,
    iou_thres_list,
    degree_thres_list,
    shift_thres_list,
    metric="",
):
    """Draw iou 3d AP vs. iou thresholds."""

    labels = ["bottle", "bowl", "camera", "can", "laptop", "mug", "mean", "nocs"]
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:pink",
        "tab:olive",
        "tab:purple",
        "tab:red",
        "tab:gray",
    ]
    styles = ["-", "-", "-", "-", "-", "-", "--", ":"]

    fig, (ax_iou, ax_degree, ax_shift) = plt.subplots(1, 3, figsize=(8, 3.5))
    # IoU subplot
    ax_iou.set_title("3D IoU", fontsize=10)
    ax_iou.set_ylabel("Average Precision")
    ax_iou.set_ylim(0, 100)
    ax_iou.set_xlabel("Percent")
    ax_iou.set_xlim(0, 100)
    ax_iou.xaxis.set_ticks([0, 25, 50, 75, 100])
    ax_iou.grid()
    for i in range(1, iou_aps.shape[0]):
        ax_iou.plot(
            100 * np.array(iou_thres_list),
            100 * iou_aps[i, :],
            color=colors[i - 1],
            linestyle=styles[i - 1],
            label=labels[i - 1],
        )
    # rotation subplot
    ax_degree.set_title("Rotation", fontsize=10)
    ax_degree.set_ylim(0, 100)
    ax_degree.yaxis.set_ticklabels([])
    ax_degree.set_xlabel("Degree")
    ax_degree.set_xlim(0, 60)
    ax_degree.xaxis.set_ticks([0, 20, 40, 60])
    ax_degree.grid()
    for i in range(1, pose_aps.shape[0]):
        ax_degree.plot(
            np.array(degree_thres_list),
            100 * pose_aps[i, : len(degree_thres_list), -1],
            color=colors[i - 1],
            linestyle=styles[i - 1],
            label=labels[i - 1],
        )
    # translation subplot
    ax_shift.set_title("Translation", fontsize=10)
    ax_shift.set_ylim(0, 100)
    ax_shift.yaxis.set_ticklabels([])
    ax_shift.set_xlabel("Centimeter")
    ax_shift.set_xlim(0, 10)
    ax_shift.xaxis.set_ticks([0, 5, 10])
    ax_shift.grid()
    for i in range(1, pose_aps.shape[0]):
        ax_shift.plot(
            np.array(shift_thres_list),
            100 * pose_aps[i, -1, : len(shift_thres_list)],
            color=colors[i - 1],
            linestyle=styles[i - 1],
            label=labels[i - 1],
        )
    ax_shift.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(out_dir, f"{metric}_mAP.png"))
    plt.close(fig)
    return


def real275_evaluator(result_dir, data_source="real_test"):
    assert data_source in ["val", "real_test"]
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions
    result_pkl_list = glob.glob(os.path.join(result_dir, "results_*.pkl"))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list)
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, "rb") as f:
            result = cPickle.load(f)
            if "gt_handle_visibility" not in result:
                result["gt_handle_visibility"] = np.ones_like(result["gt_class_ids"])
            else:
                assert len(result["gt_handle_visibility"]) == len(
                    result["gt_class_ids"]
                ), "{} {}".format(
                    result["gt_handle_visibility"], result["gt_class_ids"]
                )
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False

    la_iou_aps, la_pose_aps = compute_degree_cm_mAP(
        pred_results,
        result_dir,
        degree_thres_list,
        shift_thres_list,
        iou_thres_list,
        iou_pose_thres=0.1,
        use_matches_for_pose=True,
    )
    write_eval_logs(result_dir, la_iou_aps, la_pose_aps, metric="LaPose")

    # iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(
    #     pred_results,
    #     result_dir,
    #     degree_thres_list,
    #     shift_thres_list,
    #     iou_thres_list,
    #     iou_pose_thres=0.1,
    #     use_matches_for_pose=True,
    # )

    # write_eval_logs(
    #     result_dir,
    #     iou_aps,
    #     pose_aps,
    #     iou_acc=iou_acc,
    #     pose_acc=pose_acc,
    #     metric="Real275",
    # )

    degree_thresholds = list(range(0, 61, 1))
    shift_thresholds = [i for i in range(51)]
    iou_3d_thresholds = [i / 100 for i in range(101)]
    # Following LaPose, evaluated on the scale-agonistic metrics
    norm_results = []
    for curr_result in pred_results:
        gt_rts = curr_result["gt_RTs"].copy()
        gt_scale = np.cbrt(np.linalg.det(gt_rts[:, :3, :3]))
        gt_rts[:, :3, :] = gt_rts[:, :3, :] / gt_scale[:, None, None]
        curr_result["gt_RTs"] = gt_rts
        pred_rts = curr_result["pred_RTs"].copy()
        pred_scale = np.cbrt(np.linalg.det(pred_rts[:, :3, :3]))
        pred_rts[:, :3, :] = pred_rts[:, :3, :] / pred_scale[:, None, None]
        curr_result["pred_RTs"] = pred_rts
        norm_results.append(curr_result)

    norm_iou_aps, norm_pose_aps = compute_degree_cm_mAP(
        norm_results,
        result_dir,
        degree_thresholds=degree_thresholds,
        shift_thresholds=shift_thresholds,
        iou_3d_thresholds=iou_3d_thresholds,
        iou_pose_thres=0.1,
        use_matches_for_pose=True,
    )

    write_eval_logs(
        result_dir,
        norm_iou_aps,
        norm_pose_aps,
        degree_thres_list=degree_thresholds,
        shift_thres_list=shift_thresholds,
        iou_thres_list=iou_3d_thresholds,
        metric="Normalized",
    )

    # plot
    # plot_mAP(
    #     iou_aps,
    #     pose_aps,
    #     result_dir,
    #     iou_thres_list,
    #     degree_thres_list,
    #     shift_thres_list,
    #     metric="Real275",
    # )

    plot_mAP(
        la_iou_aps,
        la_pose_aps,
        result_dir,
        iou_thres_list,
        degree_thres_list,
        shift_thres_list,
        metric="LaPose",
    )

    plot_mAP(
        norm_iou_aps,
        norm_pose_aps,
        result_dir,
        iou_thres_list,
        degree_thres_list,
        shift_thres_list,
        metric="Normalized",
    )


def write_eval_logs(
    result_dir,
    iou_aps,
    pose_aps,
    degree_thres_list=None,
    shift_thres_list=None,
    iou_thres_list=None,
    iou_acc=None,
    pose_acc=None,
    metric="",
):

    assert metric in ["Real275", "LaPose", "Normalized"]

    print(f"Evaluated on Metrics:", metric)
    if degree_thres_list is None:
        degree_thres_list = list(range(0, 61, 1))
    if shift_thres_list is None:
        shift_thres_list = [i / 2 for i in range(21)]
    if iou_thres_list is None:
        iou_thres_list = [i / 100 for i in range(101)]

    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)

    # metric
    fw = open("{0}/eval_logs.txt".format(result_dir), "a")
    messages = []
    messages.append(f"{metric} mAP:")

    if metric == "Normalized":
        shift_05_idx = shift_thres_list.index(20)
        shift_10_idx = shift_thres_list.index(50)

        messages.append("3D IoU at 25: {:.1f}".format(iou_aps[-1, iou_25_idx] * 100))
        messages.append("3D IoU at 50: {:.1f}".format(iou_aps[-1, iou_50_idx] * 100))
        messages.append("3D IoU at 75: {:.1f}".format(iou_aps[-1, iou_75_idx] * 100))
        messages.append(
            "5 degree, 20%: {:.1f}".format(
                pose_aps[-1, degree_05_idx, shift_05_idx] * 100
            )
        )
        messages.append(
            "5 degree, 50%: {:.1f}".format(
                pose_aps[-1, degree_05_idx, shift_10_idx] * 100
            )
        )
        messages.append(
            "10 degree, 20%: {:.1f}".format(
                pose_aps[-1, degree_10_idx, shift_05_idx] * 100
            )
        )
        messages.append(
            "10 degree, 50%: {:.1f}".format(
                pose_aps[-1, degree_10_idx, shift_10_idx] * 100
            )
        )
        messages.append(
            "5 degree: {:.1f}".format(pose_aps[-1, degree_05_idx, -1] * 100)
        )
        messages.append(
            "10 degree: {:.1f}".format(pose_aps[-1, degree_10_idx, -1] * 100)
        )
        messages.append("20%: {:.1f}".format(pose_aps[-1, -1, shift_05_idx] * 100))
        messages.append("50%: {:.1f}".format(pose_aps[-1, -1, shift_10_idx] * 100))
    else:
        shift_05_idx = shift_thres_list.index(5)
        shift_10_idx = shift_thres_list.index(10)

        messages.append("3D IoU at 25: {:.1f}".format(iou_aps[-1, iou_25_idx] * 100))
        messages.append("3D IoU at 50: {:.1f}".format(iou_aps[-1, iou_50_idx] * 100))
        messages.append("3D IoU at 75: {:.1f}".format(iou_aps[-1, iou_75_idx] * 100))
        messages.append(
            "5 degree: {:.1f}".format(pose_aps[-1, degree_05_idx, -1] * 100)
        )
        messages.append(
            "10 degree: {:.1f}".format(pose_aps[-1, degree_10_idx, -1] * 100)
        )
        messages.append("10cm: {:.1f}".format(pose_aps[-1, -1, shift_10_idx] * 100))
        messages.append(
            "10 degree, 10cm: {:.1f}".format(
                pose_aps[-1, degree_10_idx, shift_10_idx] * 100
            )
        )

        messages.append(
            "5 degree, 5cm: {:.1f}".format(
                pose_aps[-1, degree_05_idx, shift_05_idx] * 100
            )
        )
        messages.append(
            "10 degree, 5cm: {:.1f}".format(
                pose_aps[-1, degree_10_idx, shift_05_idx] * 100
            )
        )
        messages.append(
            "10 degree, 5cm: {:.1f}".format(
                pose_aps[-1, degree_10_idx, shift_05_idx] * 100
            )
        )

    if metric == "Real275":
        messages.append(f"{metric} Acc:")
        messages.append("3D IoU at 25: {:.1f}".format(iou_acc[-1, iou_25_idx] * 100))
        messages.append("3D IoU at 50: {:.1f}".format(iou_acc[-1, iou_50_idx] * 100))
        messages.append("3D IoU at 75: {:.1f}".format(iou_acc[-1, iou_75_idx] * 100))

        messages.append(
            "5 degree: {:.1f}".format(pose_acc[-1, degree_05_idx, -1] * 100)
        )
        messages.append(
            "10 degree: {:.1f}".format(pose_acc[-1, degree_10_idx, -1] * 100)
        )
        messages.append("10cm: {:.1f}".format(pose_acc[-1, -1, degree_10_idx] * 100))
        messages.append(
            "10 degree, 10cm: {:.1f}".format(
                pose_acc[-1, degree_10_idx, shift_10_idx] * 100
            )
        )

        messages.append(
            "5 degree, 5cm: {:.1f}".format(
                pose_acc[-1, degree_05_idx, shift_05_idx] * 100
            )
        )

    for msg in messages:
        print(msg)
        fw.write(msg + "\n")
    fw.close()


if __name__ == "__main__":
    real275_evaluator("results/eval_camera")
