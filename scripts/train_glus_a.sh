#!/bin/bash

MASTER_PORT=$((25000 + $RANDOM % 100))

# Feel free to modify these

DIR_PATH='/2025900123/yrr/GLUS-main'

PATH_TO_CHECKPOINTS=$DIR_PATH/checkpoints
PATH_TO_DATA=$DIR_PATH/DATASET  ##只指向 DATASET 这一层，剩下的交给 Python 代码去拼接,因为refyoutube_vos.py代码中会自动加一层文件名
SAVE_DIR=$DIR_PATH/outputs/NEWA1

# 新增

EXP_NAME="glus_a"

echo "MASTER_PORT = $MASTER_PORT"
echo "DIR_PATH = $DIR_PATH"
echo "PATH_TO_CHECKPOINTS = $PATH_TO_CHECKPOINTS"
echo "PATH_TO_DATA = $PATH_TO_DATA"
echo "SAVE_DIR = $SAVE_DIR"
echo "EXP_NAME = $EXP_NAME"


deepspeed --master_port=$MASTER_PORT train_ds.py \
  --version "$PATH_TO_CHECKPOINTS/LISA-7B-v1" \
  --dataset_dir "$PATH_TO_DATA" \
  --sam_config "sam2_hiera_l.yaml" \
  --vision_pretrained "$PATH_TO_CHECKPOINTS/sam2_hiera_large.pt" \
  --dataset "refer_video_seg" \
  --refer_video_seg_data="mevis||refyoutube_vos||davis17||revos||lvvis" \
  --sample_rates "40,150,4,30,30" \
  --exp_name "lisa-7b" \
  --no_eval \
  --steps_per_epoch 100 \
  --log_base_dir "$SAVE_DIR/logs" \
  --exp_name "glus_a" \
  --epochs 50 \
  --context_frame_num 4 \
  --question_frame_num 4 \
  --total_question_frame_num 4 

# When modifying 'context_frame_num' and 'question_frame_num', dont forget to modify utils.utils at the same time.
# Make sure total_question_frame_num == question_frame_num.
# 注意：如果你改了 context_frame_num / question_frame_num
# 记得同时改 utils/utils.py 里的相关配置

