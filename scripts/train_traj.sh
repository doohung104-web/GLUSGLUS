#!/bin/bash
set -euo pipefail

# 1. 设置随机端口，防止冲突
MASTER_PORT=$((25000 + $RANDOM % 100))

# 2. 定义路径 (根据你的环境)
DIR_PATH='/2025900123/yrr/GLUS-main'
PATH_TO_CHECKPOINTS=$DIR_PATH/checkpoints
PATH_TO_DATA=$DIR_PATH/DATASET
SAVE_DIR=$DIR_PATH/outputs_traj/NEWS1

# ⭐ 关键：设置实验名称 (这将是 log 文件夹的名字)
EXP_NAME="glus_s_traj_run"

echo "启动训练..."
echo "MASTER_PORT = $MASTER_PORT"
echo "EXP_NAME = $EXP_NAME"

echo "当前GPU占用:"
nvidia-smi

# 避免继承外部 CUDA_VISIBLE_DEVICES 导致槽位重映射（只剩 slot 0）
unset CUDA_VISIBLE_DEVICES

# 3. 切换目录
cd "$DIR_PATH"

# 4. 启动 DeepSpeed (固定 1号 A100)
deepspeed --include localhost:1 --master_port=$MASTER_PORT train_ds.py \
  --version "$PATH_TO_CHECKPOINTS/LISA-7B-v1" \
  --dataset_dir "$PATH_TO_DATA" \
  --sam_config "sam2_hiera_l.yaml" \
  --vision_pretrained "$PATH_TO_CHECKPOINTS/sam2_hiera_large.pt" \
  --dataset "refer_video_seg" \
  --refer_video_seg_data="mevis||refyoutube_vos" \
  --sample_rates="1,1" \
  --no_eval \
  --steps_per_epoch 100 \
  --log_base_dir "$SAVE_DIR/logs" \
  --exp_name "$EXP_NAME" \
  --epochs 30 \
  --context_frame_num 4 \
  --question_frame_num 4 \
  --total_question_frame_num 4 \
  --use_contrastive_loss \
  --batch_size 4 \
  --grad_accumulation_steps 20

# 训练完成后退出，不自动合并权重
echo "训练结束"
