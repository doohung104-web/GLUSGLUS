#!/bin/bash

# Feel free to modify these. Take mevis valid_u benchmark as an example.

DIR_PATH='/2025900123/yrr/GLUS-main'

# ^-^增！设置数据集根目录的绝对路径（根据您之前的成功运行，DATASET是您的数据文件夹名）
PATH_TO_DATA=$DIR_PATH/DATASET/mevis

MODEL_PATH=$DIR_PATH/outputs/model/GLUS-S
VIS_SAVE_PATH=$DIR_PATH/generated
KF_PATH=$DIR_PATH/Key_frame/mevis_u.json


NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

for (( GPU_ID=0; GPU_ID<$NUM_GPUS; GPU_ID++ ))
do
    echo "Launching process on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python inference_iter.py \
        --version "$MODEL_PATH" \
        --use_kf \
        --score_json "$KF_PATH" \
        --vis_save_path "$VIS_SAVE_PATH" \
        --dataset_dir "$PATH_TO_DATA" \
        --val_set='mevis' \
        --set_name='valid_u' \
        --context_frame_num 4 \
        --question_frame_num 4 &
done

echo "All processes started in background"
wait