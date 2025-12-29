# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from mmdeploy_runtime import PoseTracker
from hydra import initialize_config_dir, compose

sys.path.append(os.path.join(os.path.expanduser("~"), "Collab_AI"))
sys.path.append(os.path.join(os.path.expanduser("~"), "Collab_AI", "dataloaders", "variables"))
from fit3d_variables import coco_wholebody

sys.path.append(os.path.expanduser('~/datasets/mocap/my_scripts/imar_vision_datasets_tools/'))
from util.smplx_util import SMPLXHelper

def load_hydra_cfg(config_path, config_name, overrides):
    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg

def build_kpts2smpl_model(cfg, weights_path, device="cpu"):
    import importlib

    module = importlib.import_module(cfg.architecture.module_path)
    net = module.model(cfg).to(device)
    net.eval()

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt  # raw state_dict

    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"[WEIGHTS] loaded {weights_path}")
    print(f"[WEIGHTS] missing={len(missing)} unexpected={len(unexpected)}")

    return net

def parse_args():
    parser = argparse.ArgumentParser(description='show how to use SDK Python API')

    ##### original arguments
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument('det_model', help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('pose_model', help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('video', help='video path or camera index')
    parser.add_argument('--output_dir', help='output directory', default=None)
    parser.add_argument(
        '--skeleton',
        default='coco',
        choices=['coco', 'coco_wholebody', 'coco_wholebody_truncated_hand'],
        help='skeleton for keypoints')

    ##### 2D to 3D model arguments    
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--weights", required=True)

    args = parser.parse_args()
    if args.video.isnumeric():
        args.video = int(args.video)

    args.config_path    = os.path.expanduser(args.config_path)
    args.weights        = os.path.expanduser(args.weights)

    return args

#print(coco_wholebody["truncated_hand_skeleton_links"])
#sys.exit()

VISUALIZATION_CFG = dict(
    coco=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
        palette=[(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
                 (255, 153, 255), (153, 204, 255), (255, 102, 255),
                 (255, 51, 255), (102, 178, 255), (51, 153, 255),
                 (255, 153, 153), (255, 102, 102), (255, 51, 51),
                 (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
                 (0, 0, 255), (255, 0, 0), (255, 255, 255)],
        link_color=[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ],
        point_color=[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ]),
    coco_wholebody=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
                  (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
                  (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
                  (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
                  (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
                  (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
                  (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
                  (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
                  (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
                  (129, 130), (130, 131), (131, 132)],
        palette=[(51, 153, 255), (0, 255, 0), (255, 128, 0), (255, 255, 255),
                 (255, 153, 255), (102, 178, 255), (255, 51, 51)],
        link_color=[
            1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1,
            1, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        point_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2,
            2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1,
            1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068,
            0.066, 0.066, 0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043,
            0.040, 0.035, 0.031, 0.025, 0.020, 0.023, 0.029, 0.032, 0.037,
            0.038, 0.043, 0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012,
            0.012, 0.011, 0.011, 0.013, 0.015, 0.009, 0.007, 0.007, 0.007,
            0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011, 0.009, 0.011,
            0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010, 0.034, 0.008,
            0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009, 0.009,
            0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01,
            0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
            0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032,
            0.02, 0.019, 0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047,
            0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
            0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031
        ]))
VISUALIZATION_CFG["coco_wholebody_truncated_hand"] = dict(
    skeleton=coco_wholebody["truncated_hand_skeleton_links"],

    # reuse palette from coco_wholebody
    palette=VISUALIZATION_CFG["coco_wholebody"]["palette"],

    # link_color: one entry per link
    # strategy:
    #   body links → normal colors
    #   left truncated hand → color 4
    #   right truncated hand → color 5
    link_color=(
        # body + upper body (first 25 links)
        VISUALIZATION_CFG["coco_wholebody"]["link_color"][:25]
        +
        # left truncated hand (3 links)
        [4, 4, 4]
        +
        # right truncated hand (3 links)
        [5, 5, 5]
    ),

    # start from wholebody point colors
    point_color=list(VISUALIZATION_CFG["coco_wholebody"]["point_color"]),

    # IMPORTANT: keep sigmas identical so tracking / OKS stays correct
    sigmas=VISUALIZATION_CFG["coco_wholebody"]["sigmas"],
)

def visualize(frame,
              results,
              output_dir,
              frame_id,
              thr=0.5,
              resize=1280,
              skeleton_type='coco'):

    skeleton = VISUALIZATION_CFG[skeleton_type]['skeleton']
    palette = VISUALIZATION_CFG[skeleton_type]['palette']
    link_color = VISUALIZATION_CFG[skeleton_type]['link_color']
    point_color = VISUALIZATION_CFG[skeleton_type]['point_color']

    scale = resize / max(frame.shape[0], frame.shape[1])
    keypoints, bboxes, _ = results
    scores = keypoints[..., 2]
    keypoints = (keypoints[..., :2] * scale).astype(int)
    bboxes *= scale
    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    for kpts, score, bbox in zip(keypoints, scores, bboxes):
        
        used = set()
        for u, v in skeleton:
            used.add(u); used.add(v)

        show = [0] * len(kpts)
        for j in used:
            show[j] = 1

        for (u, v), color in zip(skeleton, link_color):
            if score[u] > thr and score[v] > thr:
                cv2.line(img, kpts[u], tuple(kpts[v]), palette[color], 1,
                         cv2.LINE_AA)
            else:
                show[u] = show[v] = 0
        for kpt, show, color in zip(kpts, show, point_color):
            if show:
                cv2.circle(img, kpt, 1, palette[color], 2, cv2.LINE_AA)
    if output_dir:
        cv2.imwrite(f'{output_dir}/{str(frame_id).zfill(6)}.jpg', img)
    else:
        cv2.imshow('pose_tracker', img)
        return cv2.waitKey(1) != 'q'
    return True


def main():
    args = parse_args()
    np.set_printoptions(precision=4, suppress=True)
    
    ##### initialize video capture
    video   = cv2.VideoCapture(args.video)

    ##### initialize 2D pose tracker
    tracker = PoseTracker(det_model=args.det_model, pose_model=args.pose_model, device_name=args.device_name)
    relevant_joint_idxs = sorted(set(i for pair in coco_wholebody["truncated_hand_skeleton_links"] for i in pair))

    # optionally use OKS for keypoints similarity comparison
    sigmas  = VISUALIZATION_CFG[args.skeleton]['sigmas']
    state   = tracker.create_state(det_interval=10, det_min_bbox_size=100, keypoint_sigmas=sigmas)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    ##### initialize 2D to 3D model
    cfg = load_hydra_cfg(args.config_path, args.config_name, args.override)
    input_key = cfg.params.kpts2smpl.input  # "kpts_normalized_filtered"
    net = build_kpts2smpl_model(cfg, args.weights)
    smplx_helper = SMPLXHelper(os.path.expanduser('~/datasets/mocap/data/models_smplx_v1_1/models/'))
    smplx_helper.smplx_model = smplx_helper.smplx_model.to("cpu")
    faces       = smplx_helper.smplx_model.faces
    faces       = np.asarray(faces)    
    
    t_start = time.time()
    frame_id = 0
    while True:
        success, frame = video.read()
        if not success:
            break

        ##### get 2d pose
        results = tracker(state, frame, detect=-1)

        ##### visualize 2d pose
        if not visualize(
                frame,
                results,
                args.output_dir,
                frame_id,
                skeleton_type=args.skeleton):
            break

        ##### get 3d pose
        keypoints, bboxes, _ = results

        if keypoints is None or len(keypoints) == 0:
            frame_id += 1
            continue

        kpts    = keypoints[0, :, :2].astype(np.float32)    # [k, 2]
        kpts    = kpts[relevant_joint_idxs]                 # [k', 2]
        scores  = keypoints[0, :, 2].astype(np.float32)     # [k]
        scores  = scores[relevant_joint_idxs]               # [k']
        bbox    = bboxes[0].astype(np.float32)              # [4]
            
        frame_id += 1
        if frame_id % 30 == 0:
            elapsed = time.time() - t_start
            fps = frame_id / elapsed
            print(f"[FPS] {fps:.2f} ({frame_id} frames)")


if __name__ == '__main__':
    main()
