import os
import cv2
import time
import torch

# ✅ allow CPU parallelism
os.environ.pop("OMP_NUM_THREADS", None)
os.environ.pop("MKL_NUM_THREADS", None)
os.environ.pop("OPENBLAS_NUM_THREADS", None)

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import load_config, get_input_shape

deploy_cfg_path = 'configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py'
model_cfg_path  = os.path.expanduser('~/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py')
backend_model   = ['rtmw-ort-simcc/rtmw-m/end2end.onnx']

deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
tp = build_task_processor(model_cfg, deploy_cfg, device='cpu')
model = tp.build_backend_model(backend_model)
input_shape = get_input_shape(deploy_cfg)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
codec = "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])
print("Camera FOURCC:", codec)

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Resolution:", int(w), "x", int(h))

# 🔥 warm-up (important)
ok, frame = cap.read()
model_inputs, _ = tp.create_input(frame, input_shape)
_ = model.test_step(model_inputs)

print("Running (CPU optimized)")

t0 = time.perf_counter()
frame_count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
        
    with torch.no_grad():
        model_inputs, _ = tp.create_input(frame, input_shape)
        result = model.test_step(model_inputs)
        
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
        print(f"FPS (end-to-end): {fps:.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
os._exit(0)
