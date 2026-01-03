

python pose_tracker_synthium.py cpu \
$HOME/mmdeploy/rtmpose-ort/rtmdet-nano/ \
$HOME/mmdeploy/rtmpose-ort/rtmw-dw-l-m/ \
$HOME/datasets/mocap/data/fit3d/train/s07/videos/60457274/dumbbell_biceps_curls.mp4 \
--config-path "~/Collab_AI/experiments/synthium/kpts2smpl" \
--config-name "v1" \
--override "run_name=fit3d_mmposelarge_mlp_lr1e-3_thr0_kptsmask0" \
--override "optimization.batch_size=1" \
--override "dataset_settings.kpts_config=whole_body" \
--weights "~/Collab_AI/weights/synthium/kpts2smpl/trainedonall_mmposesmall_mlp_lr1e-3_thr0_kptsmask0_orient0/all_epoch_0308_best_0308_state_dict.pt" \
--show-2d 1 --pred-3d 1 --show-3d-mesh 1

##### upper body

python pose_tracker_synthium.py cpu \
$HOME/mmdeploy/rtmpose-ort/rtmdet-nano/ \
$HOME/mmdeploy/rtmpose-ort/rtmw-dw-l-m/ \
$HOME/datasets/mocap/data/fit3d/train/s07/videos/60457274/dumbbell_biceps_curls.mp4 \
--config-path "~/Collab_AI/experiments/synthium/kpts2smpl" \
--config-name "v1" \
--override "optimization.batch_size=1" \
--override "dataset_settings.kpts_config=upper_body" \
--override "params.kpts2smpl.units=[38, 256, 256, 132]" \
--weights "~/Collab_AI/weights/synthium/kpts2smpl/trainedonall_mmposesmall_mlp_lr1e-3_thr0.0_kptsmask0.0_orient0_upperbody/all_epoch_0976_best_0976_state_dict.pt" \
--show-2d 1 --pred-3d 0 --show-3d-mesh 0 \
--smooth ema_quat --ema-alpha 0.10 --warmup-frames 10

python pose_tracker_synthium.py cpu \
$HOME/mmdeploy/rtmpose-ort/rtmdet-nano/ \
$HOME/mmdeploy/rtmpose-ort/rtmw-dw-l-m/ \
0 \
--config-path "~/Collab_AI/experiments/synthium/kpts2smpl" \
--config-name "v1" \
--override "optimization.batch_size=1" \
--override "dataset_settings.kpts_config=upper_body" \
--override "params.kpts2smpl.units=[38, 256, 256, 132]" \
--weights "~/Collab_AI/weights/synthium/kpts2smpl/trainedonall_mmposesmall_mlp_lr1e-3_thr0.0_kptsmask0.0_orient0_upperbody/all_epoch_0976_best_0976_state_dict.pt" \
--show-2d 1 --pred-3d 1 --show-3d-mesh 1 \
--smooth ema_quat --ema-alpha 0.10 --warmup-frames 10

#####

python pose_tracker.py cpu \
$HOME/mmdeploy/rtmpose-ort/rtmdet-nano/ \
$HOME/mmdeploy/rtmpose-ort/rtmw-dw-l-m/ \
$HOME/datasets/telept/data/ipad/rgb_1764569430654.mp4 \
--skeleton "coco_wholebody"

./pose_tracker \
$HOME/mmdeploy/rtmpose-ort/rtmdet-nano/ \
$HOME/mmdeploy/rtmpose-ort/rtmw-dw-l-m/ \
$HOME/datasets/telept/data/ipad/rgb_1764569430654.mp4 \
--device cpu --skeleton "coco-wholebody" --det_interval 10