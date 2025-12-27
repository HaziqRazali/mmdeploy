import os
import sys
import cv2
import time
import torch
import numpy as np

from omegaconf import OmegaConf
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import load_config, get_input_shape

# allow CPU parallelism
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)
torch.set_num_interop_threads(1)

# architecture definition
sys.path.append(os.path.expanduser("~/Collab_AI/models/synthium/"))
from kpts2smpl import model

# keypoints we want
sys.path.append(os.path.expanduser("~/Collab_AI/dataloaders/variables"))
from fit3d_variables import coco_wholebody

############ load kpts2smpl
cfg_v1  = OmegaConf.load(os.path.expanduser("~/Collab_AI/experiments/synthium/kpts2smpl/v1.yaml"))
cfg_mlp = OmegaConf.load(os.path.expanduser("~/Collab_AI/experiments/synthium/kpts2smpl/decoder/mlp.yaml"))
cfg = OmegaConf.merge(cfg_v1, cfg_mlp)
cfg.optimization.batch_size = 1
smpl_net = model(cfg)
state = torch.load(os.path.expanduser("~/Collab_AI/weights/synthium/kpts2smpl/trainedonall_mmposesmall_mlp_lr1e-3_thr0_kptsmask0_orient0/all_epoch_0308_best_0308_state_dict.pt"), map_location="cpu")
smpl_net.load_state_dict(state, strict=True)
smpl_net.eval()
print("kpts2smpl")
print(smpl_net)
print()

# sanity check run
dummy = torch.zeros((1, 31, 2))
data = {
    cfg.params.kpts2smpl.input: dummy,
    "mask": torch.ones((1, 22, 6))
}
with torch.no_grad():
    out = smpl_net(data, mode="val")
print(out["pred_smpl"].shape)

# the joint idxs as input to the model
relevant_joint_idxs = sorted(set(i for pair in coco_wholebody["truncated_hand_skeleton_links"] for i in pair))

#################### load mmpose
deploy_cfg_path = 'configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py'
model_cfg_path  = os.path.expanduser('~/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py')
backend_model   = ['rtmw-ort-simcc/rtmw-m/end2end.onnx']

deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
tp = build_task_processor(model_cfg, deploy_cfg, device='cpu')
model = tp.build_backend_model(backend_model)
input_shape = get_input_shape(deploy_cfg)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)          # Linux: use V4L2 backend
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
codec = "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])
print("Camera FOURCC:", codec)

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Resolution:", int(w), "x", int(h))

# warm-up (important)
ok, frame = cap.read()
model_inputs, _ = tp.create_input(frame, input_shape)
_ = model.test_step(model_inputs)

print("Running (CPU optimized)")

mask = torch.ones((1, 22, 6), dtype=torch.float32)
xy = np.empty((2,), dtype=np.float32)
wh = np.empty((2,), dtype=np.float32)

t0 = time.perf_counter()
frame_count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
        
    with torch.no_grad():
        model_inputs, _ = tp.create_input(frame, input_shape)
        result = model.test_step(model_inputs)
        
    inst = result[0].pred_instances
    if len(inst) == 0:
        continue
        
    # retreive data
    kpts    = inst.keypoints[0]                     # [133, 2]
    scores  = inst.keypoint_scores[0]               # [133]
    bbox    = inst.bboxes[0]                        # [4]    
    kpts    = kpts.astype(np.float32, copy=False)   # [133, 2]
    bbox    = bbox.astype(np.float32, copy=False)   # [4]
    
    # select joints
    kpts    = kpts[relevant_joint_idxs]             # [31, 2]    
    scores  = scores[relevant_joint_idxs]           # [31]
        
    # normalize wrt bbox
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    xy[0], xy[1] = x1, y1
    wh[0], wh[1] = w, h
    kpts_norm = (kpts - xy) / wh # [31, 2]
    kpts_norm = torch.from_numpy(kpts_norm).unsqueeze(0)
    
    # forward pass
    data = {
        "kpts_normalized_filtered": kpts_norm,
        "mask": mask,
    }    
    with torch.inference_mode():
        out = smpl_net(data, mode="val")
    pred_smpl = out["pred_smpl"]
        
    # OPTIONAL visualization (comment out for speed)
    #tp.visualize(
    #    image=frame,
    #    model=model,
    #    result=result[0],
    #    show=True,
    #    wait_time=1
    #)

    frame_count += 1
    if frame_count % 30 == 0:  # print every 30 frames
        t1 = time.perf_counter()
        fps = frame_count / (t1 - t0)
        print(f"FPS: {fps:.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
os._exit(0)
