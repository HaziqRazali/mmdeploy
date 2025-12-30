# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
import time
import torch

import cv2
import numpy as np
import open3d as o3d

from mmdeploy_runtime import PoseTracker
from hydra import initialize_config_dir, compose

sys.path.append(os.path.join(os.path.expanduser("~"), "Collab_AI"))
sys.path.append(os.path.join(os.path.expanduser("~"), "Collab_AI", "dataloaders", "variables"))
from fit3d_variables import coco_wholebody

sys.path.append(os.path.expanduser('~/datasets/mocap/my_scripts/imar_vision_datasets_tools/'))
from util.smplx_util import SMPLXHelper

sys.path.append(os.path.expanduser('~/datasets/mocap/my_scripts/imar_vision_datasets_tools/util'))
from dataset_util import rot6d_to_matrix

sys.path.append(os.path.expanduser('~/datasets/mocap/my_scripts/'))
from utils_draw import render_simple_pyrender

#"""
class StickFigureO3D:
    def __init__(self, edges, window_name="SMPL Stick", width=800, height=800):
        self.edges = np.asarray(edges, dtype=np.int32)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height, visible=True)

        self.lines = o3d.geometry.LineSet()
        self.lines.lines = o3d.utility.Vector2iVector(self.edges)

        n_pts = int(self.edges.max()) + 1 if len(self.edges) > 0 else 22
        pts0 = np.zeros((n_pts, 3), dtype=np.float64)
        self.lines.points = o3d.utility.Vector3dVector(pts0)

        colors = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (len(self.edges), 1))
        self.lines.colors = o3d.utility.Vector3dVector(colors)

        self.vis.add_geometry(self.lines)

        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        opt.line_width = 2.0

        self._inited = False

    def update(self, J_pos):
        pts = np.asarray(J_pos, dtype=np.float64)

        # optional: center on root for stable viewing
        pts = pts - pts[0:1]

        self.lines.points = o3d.utility.Vector3dVector(pts)
        self.vis.update_geometry(self.lines)

        if not self._inited:
            self.vis.reset_view_point(True)
            self._inited = True

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()
#"""

def load_hydra_cfg(config_path, config_name, overrides):
    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg

def _ms(x):  # seconds -> milliseconds
    return 1000.0 * x

def normalize_kpts_bbox_inplace(kpts_xy_f32, bbox_xyxy_f32):
    x1, y1, x2, y2 = bbox_xyxy_f32
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    kpts_xy_f32[:, 0] = (kpts_xy_f32[:, 0] - x1) / w
    kpts_xy_f32[:, 1] = (kpts_xy_f32[:, 1] - y1) / h
    return kpts_xy_f32

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
        state = ckpt

    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"[WEIGHTS] loaded {weights_path}")
    print(f"[WEIGHTS] missing={len(missing)} unexpected={len(unexpected)}")

    return net

def fk_joints_from_offsets(R, parents, J_OFF):
    """
    R: (22,3,3) rotation matrices
    parents: (22,) int32 with parents[0] = -1
    J_OFF: (22,3) rest offsets (root is absolute rest position)
    returns: J_pos (22,3)
    """
    T = np.zeros((22, 4, 4), dtype=np.float32)
    for j in range(22):
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R[j]
        M[:3, 3] = J_OFF[j]
        p = parents[j]
        if p == -1:
            T[j] = M
        else:
            T[j] = T[p] @ M
    return T[:, :3, 3]

def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    R: (..., 3, 3) rotation matrices
    returns q: (..., 4) as (w, x, y, z), normalized
    """
    # Based on standard robust conversion
    r00 = R[..., 0, 0]
    r11 = R[..., 1, 1]
    r22 = R[..., 2, 2]

    qw = torch.sqrt(torch.clamp(1.0 + r00 + r11 + r22, min=0.0)) * 0.5
    qx = torch.sqrt(torch.clamp(1.0 + r00 - r11 - r22, min=0.0)) * 0.5
    qy = torch.sqrt(torch.clamp(1.0 - r00 + r11 - r22, min=0.0)) * 0.5
    qz = torch.sqrt(torch.clamp(1.0 - r00 - r11 + r22, min=0.0)) * 0.5

    # Pick signs using off-diagonal terms
    qx = torch.copysign(qx, R[..., 2, 1] - R[..., 1, 2])
    qy = torch.copysign(qy, R[..., 0, 2] - R[..., 2, 0])
    qz = torch.copysign(qz, R[..., 1, 0] - R[..., 0, 1])

    q = torch.stack([qw, qx, qy, qz], dim=-1)
    q = q / (q.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    return q


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: (..., 4) as (w, x, y, z), assumed normalized
    returns R: (..., 3, 3)
    """
    q = q / (q.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    R = torch.stack(
        [
            ww + xx - yy - zz, 2 * (xy - wz),       2 * (xz + wy),
            2 * (xy + wz),       ww - xx + yy - zz, 2 * (yz - wx),
            2 * (xz - wy),       2 * (yz + wx),     ww - xx - yy + zz,
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return R


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """
    q0, q1: (..., 4) (w,x,y,z), normalized
    t: scalar in [0,1]
    """
    # Ensure shortest path (sign-fix)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs()

    # If very close, use lerp
    if not isinstance(t, torch.Tensor):
        t_tensor = torch.tensor(t, dtype=q0.dtype, device=q0.device)
    else:
        t_tensor = t

    close = dot > 0.9995
    q = torch.where(
        close,
        (1.0 - t_tensor) * q0 + t_tensor * q1,
        torch.zeros_like(q0),
    )

    # SLERP for the rest
    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))  # (...,1)
    sin_theta_0 = torch.sin(theta_0).clamp_min(1e-8)
    theta = theta_0 * t_tensor
    sin_theta = torch.sin(theta)

    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q_slerp = s0 * q0 + s1 * q1

    q = torch.where(close, q, q_slerp)
    q = q / (q.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    return q


class QuatSlerpEmaFilter:
    """
    Filters per-joint rotations using SLERP-EMA in quaternion space.
    - update only when valid=True
    - hold-last when valid=False
    """
    def __init__(self, alpha: float = 0.2, warmup_frames: int = 0):
        self.alpha = float(alpha)
        self.warmup_frames = int(warmup_frames)
        self._q_state = None  # (J,4)
        self._valid_count = 0

    @property
    def has_state(self) -> bool:
        return self._q_state is not None

    def reset(self):
        self._q_state = None
        self._valid_count = 0

    def get_rotmats(self) -> torch.Tensor:
        if self._q_state is None:
            raise RuntimeError("Filter has no state yet.")
        return quat_to_rotmat(self._q_state)

    def update(self, R: torch.Tensor, valid: bool = True) -> torch.Tensor:
        """
        R: (J,3,3)
        returns filtered R_f: (J,3,3)
        """
        if (not valid):
            # hold-last if we can
            if self._q_state is not None:
                return quat_to_rotmat(self._q_state)
            return R

        q = rotmat_to_quat(R)  # (J,4)

        if self._q_state is None:
            self._q_state = q
            self._valid_count = 1
            return R

        # warmup: snap state to measurement to avoid initial lag
        if self._valid_count < self.warmup_frames:
            self._q_state = q
            self._valid_count += 1
            return R

        # slerp-ema
        self._q_state = quat_slerp(self._q_state, q, self.alpha)
        self._valid_count += 1
        return quat_to_rotmat(self._q_state)

def parse_args():
    parser = argparse.ArgumentParser(description='show how to use SDK Python API')

    ##### original arguments
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument('det_model', help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('pose_model', help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('video', help='video path or camera index')
    parser.add_argument("--save-2d", type=str, default=None, help="Path to save visualization (directory or video)")
    parser.add_argument("--show-2d", type=int, default=0, help="Show visualization window")
    parser.add_argument("--pred-3d", type=int, default=1)
    parser.add_argument("--show-3d-skel", type=int, default=0, help="Show skel visualization window")
    parser.add_argument("--show-3d-mesh", type=int, default=0, help="Show mesh window")
    parser.add_argument('--skeleton', default='coco', choices=['coco', 'coco_wholebody', 'coco_wholebody_truncated_hand'], help='skeleton for keypoints')

    ##### 2D to 3D model arguments    
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--weights", required=True)

    ##### smoothing op
    parser.add_argument("--smooth", type=str, default="none",
                        choices=["none", "ema_quat"],
                        help="Smoothing applied at output of kpts2smpl (body_pose only).")
    parser.add_argument("--ema-alpha", type=float, default=0.2,
                        help="EMA strength for SLERP (0=no update, 1=no smoothing).")
    parser.add_argument("--warmup-frames", type=int, default=0,
                        help="Snap to measurements for first N valid frames (reduces initial lag).")
    
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
              frame_id,
              skeleton_type='coco',
              save_vis=None,
              show_vis=False,
              thr=0.5,
              resize=400,
              ):

    skeleton    = VISUALIZATION_CFG[skeleton_type]['skeleton']
    palette     = VISUALIZATION_CFG[skeleton_type]['palette']
    link_color  = VISUALIZATION_CFG[skeleton_type]['link_color']
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

    if save_vis is not None:
        os.makedirs(save_vis, exist_ok=True)
        cv2.imwrite(
            os.path.join(save_vis, f"{frame_id:06d}.jpg"),
            img
        )

    if show_vis:
        cv2.imshow("pose_tracker", img)
        return cv2.waitKey(1) != ord('q')

    return True


def main():
    args = parse_args()
    np.set_printoptions(precision=4, suppress=True)
    
    ##### initialize video capture
    video   = cv2.VideoCapture(args.video)

    ##### initialize 2D pose tracker
    tracker = PoseTracker(det_model=args.det_model, pose_model=args.pose_model, device_name=args.device_name)
    relevant_joint_idxs = sorted(set(i for pair in coco_wholebody["truncated_hand_skeleton_links"] for i in pair))
    sigmas  = VISUALIZATION_CFG[args.skeleton]['sigmas']
    state   = tracker.create_state(det_interval=15, det_min_bbox_size=50, keypoint_sigmas=sigmas)

    ##### initialize kpts2smpl

    # configs
    cfg         = load_hydra_cfg(args.config_path, args.config_name, args.override)
    input_key   = "kpts_normalized_filtered"

    # build net
    net = build_kpts2smpl_model(cfg, args.weights).float()
    mask = torch.ones((1, 22, 1), device="cpu", dtype=torch.float32)

    # initialize smpl
    smplx_helper = SMPLXHelper(os.path.expanduser('~/datasets/mocap/data/models_smplx_v1_1/models/'))
    smplx_helper.smplx_model = smplx_helper.smplx_model.to("cpu")
    faces       = smplx_helper.smplx_model.faces
    faces       = np.asarray(faces)    
    
    # for manual fk computation
    with torch.inference_mode():
        I = torch.eye(3, dtype=torch.float32)

        # identity rotations for rest pose
        global_orient_rest = I[None, None]                  # (1,1,3,3) matches your later usage
        body_pose_rest     = I[None, None].repeat(1, 21, 1, 1)  # (1,21,3,3)

        out0 = smplx_helper.smplx_model(
            transl=torch.zeros((1, 3), dtype=torch.float32),
            global_orient=global_orient_rest,
            body_pose=body_pose_rest,
            pose2rot=False,   # IMPORTANT when providing rotmats
        )

        J_rest = out0.joints[0].detach().cpu().numpy()[:22].astype(np.float32)
        parents = smplx_helper.smplx_model.parents[:22].detach().cpu().numpy().astype(np.int32)

        # offsets
        J_OFF = np.zeros_like(J_rest, dtype=np.float32)
        for j in range(22):
            p = int(parents[j])
            if p == -1:
                J_OFF[j] = J_rest[j]
            else:
                J_OFF[j] = J_rest[j] - J_rest[p]

        edges = [(int(p), j) for j, p in enumerate(parents) if int(p) != -1]
        stick_vis = None
        if args.show_3d_skel:
            stick_vis = StickFigureO3D(edges, window_name="SMPL Stick", width=800, height=800)

    ##### begin 

    t_start = time.time()
    frame_id = 0

    # timer
    prof_sum = {
        "read_frame": 0.0,
        "pose2d": 0.0,
        "preprocess_2d": 0.0,
        "mlp_forward": 0.0,
        "smpl_convert": 0.0,
        "smpl_render": 0.0,
        "smpl_forward_pass": 0.0,
        "total_frame": 0.0,
    }
    prof_n = 0

    # we always set the global orientation to 0
    GLOBAL_ORIENT = torch.tensor([
                [1., 0.,  0.],
                [0., 0., -1.],
                [0., 1.,  0.]
                ], device="cpu")[None, None]
    
    # smoothing op
    pose_filter = None
    missing_streak = 0
    if args.smooth == "ema_quat":
        pose_filter = QuatSlerpEmaFilter(alpha=args.ema_alpha, warmup_frames=args.warmup_frames)

    # for the 3d rendering function
    ref_center, ref_radius = None, None

    while True:
        t_total0 = time.perf_counter()

        ##### read video
        t0 = time.perf_counter()
        success, frame = video.read()
        if not success or frame is None:
            break
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
        t1 = time.perf_counter()
        prof_sum["read_frame"] += (t1 - t0)

        ##### get 2d pose
        t0 = time.perf_counter()
        results = tracker(state, frame, detect=-1)
        t1 = time.perf_counter()
        prof_sum["pose2d"] += (t1 - t0)

        ##### visualize 2d pose
        if args.save_2d or args.show_2d:
            if not visualize(
                    frame,
                    results,
                    frame_id,
                    skeleton_type=args.skeleton,
                    save_vis=args.save_2d,
                    show_vis=args.show_2d):
                break

        ##### get 3d pose
        if args.pred_3d:

            ##### preprocessing
            t0 = time.perf_counter()
            keypoints, bboxes, _ = results

            if keypoints is None or len(keypoints) == 0:
                if pose_filter is not None:
                    pose_filter.reset()
                frame_id += 1
                continue

            # normalize kpts
            kpts = keypoints[0, :, :2].astype(np.float32, copy=False)
            bbox = bboxes[0].astype(np.float32, copy=False)
            normalize_kpts_bbox_inplace(kpts, bbox)

            # retrieve relevant kpts
            kpts    = kpts[relevant_joint_idxs]                     # [k', 2]
            scores  = keypoints[0, relevant_joint_idxs, 2].astype(np.float32, copy=False)
            kpts[scores < 0.1] = 0.0
            kpts    = torch.from_numpy(kpts).unsqueeze(0)           # [k']
            t1 = time.perf_counter()
            prof_sum["preprocess_2d"] += (t1 - t0)
                
            ##### kpts2smpl forward pass
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = net({"kpts_normalized_filtered": kpts, "mask": mask}, mode="val")
            pred_smpl = out["pred_smpl"][0]  # [22,6]
            t1 = time.perf_counter()
            prof_sum["mlp_forward"] += (t1 - t0)

            ##### smooth
            t0 = time.perf_counter()
            global_orient = GLOBAL_ORIENT
            body_pose = pred_smpl[1:]                     # [21, 6]
            body_pose = rot6d_to_matrix(body_pose)        # [21, 3, 3]

            if pose_filter is not None:
                body_pose = pose_filter.update(body_pose, valid=True)

            body_pose = body_pose[None]                   # [1, 21, 3, 3]
            t1 = time.perf_counter()
            prof_sum["smpl_convert"] += (t1 - t0)

            ##### 3d skel visualization
            if args.show_3d_skel:

                t0 = time.perf_counter()
                R_root = global_orient[0, 0].detach().cpu().numpy().astype(np.float32)  # (3,3)
                R_body = body_pose[0].detach().cpu().numpy().astype(np.float32)         # (21,3,3)

                R       = np.zeros((22, 3, 3), dtype=np.float32)
                R[0]    = R_root
                R[1:]   = R_body
                J_pos = fk_joints_from_offsets(R, parents, J_OFF)
                t1 = time.perf_counter()
                prof_sum["smpl_forward_pass"] += (t1 - t0)

                if stick_vis is not None:
                    stick_vis.update(J_pos)

            ##### 3d mesh visualization
            if args.show_3d_mesh:
                
                t0 = time.perf_counter()
                world_smplx_params = {
                    "transl": torch.zeros((1, 3), device="cpu", dtype=torch.float32),
                    "global_orient": global_orient.to(device="cpu", dtype=torch.float32),
                    "body_pose": body_pose.to(device="cpu", dtype=torch.float32),
                }
                world_posed_data    = smplx_helper.smplx_model(**world_smplx_params)
                joints              = world_posed_data.joints[0,:22]
                vertices            = world_posed_data.vertices[0]   # [10475, 3]
                vertices            = vertices.detach().cpu().numpy()
                t1 = time.perf_counter()
                prof_sum["smpl_forward_pass"] += (t1 - t0)

                t0 = time.perf_counter()
                ref_center, ref_radius, image = render_simple_pyrender(
                    vertices,
                    faces,
                    "temp.png",
                    img_width=300,
                    img_height=300,
                    ref_center=ref_center,
                    ref_radius=ref_radius,
                    views=((0, 0, 0),),
                    label=None,
                    save=False,
                )
                cv2.imshow('smplx', image)
                cv2.waitKey(1)
                t1 = time.perf_counter()
                prof_sum["smpl_render"] += (t1 - t0)

        ##### compute time taken for frame
        prof_sum["total_frame"] += (time.perf_counter() - t_total0)
        prof_n += 1
        frame_id += 1

        ##### print timers
        if frame_id % 30 == 0:
            elapsed = time.time() - t_start
            fps = frame_id / elapsed
            print(f"[FPS] {fps:.2f} ({frame_id} frames)")

            # averages over the *current profiling window* (prof_n frames)
            denom = max(prof_n, 1)
            print(
                "[TIMING ms/frame] "
                f"read={_ms(prof_sum['read_frame']/denom):.2f} | "
                f"pose2d={_ms(prof_sum['pose2d']/denom):.2f} | "
                f"prep={_ms(prof_sum['preprocess_2d']/denom):.2f} | "
                f"mlp={_ms(prof_sum['mlp_forward']/denom):.2f} | "
                f"smpl_conv={_ms(prof_sum['smpl_convert']/denom):.2f} | "
                f"smpl_forward={_ms(prof_sum['smpl_forward_pass']/denom):.2f} | "
                f"render={_ms(prof_sum['smpl_render']/denom):.2f} | "
                f"total={_ms(prof_sum['total_frame']/denom):.2f}"
            )

            # reset window so each print is for the last ~30 frames
            for k in prof_sum:
                prof_sum[k] = 0.0
            prof_n = 0

    if stick_vis is not None:
        stick_vis.close()

if __name__ == '__main__':
    main()
