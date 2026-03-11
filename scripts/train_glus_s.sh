#!/bin/bash



MASTER_PORT=$((25000 + $RANDOM % 100))

# Feel free to modify these

DIR_PATH='/2025900123/yrr/GLUS-main'

PATH_TO_CHECKPOINTS=$DIR_PATH/checkpoints
PATH_TO_DATA=$DIR_PATH/DATASET
SAVE_DIR=$DIR_PATH/outputs/NEWS1

EXP_NAME="glus_s0208"


echo "MASTER_PORT = $MASTER_PORT"
echo "DIR_PATH = $DIR_PATH"
echo "PATH_TO_CHECKPOINTS = $PATH_TO_CHECKPOINTS"
echo "PATH_TO_DATA = $PATH_TO_DATA"
echo "SAVE_DIR = $SAVE_DIR"
echo "EXP_NAME = $EXP_NAME"

# ⭐ 关键：先切到项目根目录
cd "$DIR_PATH"

deepspeed --master_port=$MASTER_PORT train_ds.py \
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
  --grad_accumulation_steps 10 
# When modifying 'context_frame_num' and 'question_frame_num', dont forget to modify utils.utils at the same time.
# Make sure total_question_frame_num == question_frame_num.


exit 0  # 在这里加上 exit，让脚本跑完训练就停下，不要去跑后面的合并
# Merge lora weights and save hf model 
# 合并 ZeRO 权重
cd $SAVE_DIR/logs/glus_s/ckpt_model && python $DIR_PATH/zero_to_fp32.py . ../pytorch_model.bin

#合并 LoRA 权重并保存 HF 模型
cd $DIR_PATH

python merge_lora_weights_and_save_hf_model.py \
  --version "$PATH_TO_CHECKPOINTS/LISA-7B-v1" \
  --weight "$SAVE_DIR/logs/glus_s/pytorch_model.bin" \
  --save_path="$SAVE_DIR/model"