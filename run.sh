

python pose_tracker_synthium.py cpu \
$HOME/mmdeploy/rtmpose-ort/rtmdet-nano/ \
$HOME/mmdeploy/rtmpose-ort/rtmw-dw-l-m/ \
$HOME/datasets/mocap/data/fit3d/train/s07/videos/60457274/dumbbell_biceps_curls.mp4 \
--skeleton "whole_body_skeleton" \
--config-path "~/Collab_AI/experiments/synthium/kpts2smpl" \
--config-name "v1" \
--override "run_name=fit3d_mmposelarge_mlp_lr1e-3_thr0_kptsmask0" \
--override "optimization.batch_size=1" \
--weights "~/Collab_AI/weights/synthium/kpts2smpl/trainedonall_mmposesmall_mlp_lr1e-3_thr0_kptsmask0_orient0/all_epoch_0308_best_0308_state_dict.pt" \
--show-2d 1 --pred-3d 1 --show-3d-mesh 1

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