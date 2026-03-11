# #!/bin/bash

# # Feel free to modify these. Take mevis valid_u benchmark as an example.

# DIR_PATH='/2025900123/yrr/GLUS-main'

# MODEL_PATH=$DIR_PATH/outputs/NEWS1/infer/glus_s_step3000_hf
# VIS_SAVE_PATH=$DIR_PATH/generated_Refer-YouTube-VOS
# PATH_TO_DATA=$DIR_PATH/DATASET #只指向 DATASET 这一层，剩下的交给 Python 代码去拼接,因为refyoutube_vos.py代码中会自动加一层文件名
# NUM_GPUS= $(nvidia-smi -L | wc -l)
# echo "Detected $NUM_GPUS GPUs"

# for (( GPU_ID=0; GPU_ID<$NUM_GPUS; GPU_ID++ ))
# do
#     echo "Launching process on GPU $GPU_ID"
#     CUDA_VISIBLE_DEVICES=$GPU_ID python inference_iter.py \
#         --version "$MODEL_PATH" \
#         --vis_save_path "$VIS_SAVE_PATH" \
#         --dataset_dir "$PATH_TO_DATA" \
#         --val_set='refyoutube_vos' \
#         --set_name='valid' \
#         --context_frame_num 4 \
#         --question_frame_num 4 \
#         --precision bf16 &
# done

# echo "All processes started in background"
# wait

#!/bin/bash


DIR_PATH='/2025900123/yrr/GLUS-main'

MODEL_PATH=$DIR_PATH/outputs_final/NEWS1/model_final
VIS_SAVE_PATH=$DIR_PATH/outputs_final/NEWS1/inference_results_mevis
# 只指向 DATASET 这一层，剩下的交给 Python 代码去拼接
PATH_TO_DATA=$DIR_PATH/DATASET

NUM_GPUS=2 #双卡运行

for (( GPU_ID=0; GPU_ID<$NUM_GPUS; GPU_ID++ ))
do
    echo "Launching process on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python inference_iter.py \
        --version "$MODEL_PATH" \
        --vis_save_path "$VIS_SAVE_PATH" \
        --dataset_dir "$PATH_TO_DATA" \
        --val_set='mevis' \
        --set_name='valid' \
        --context_frame_num 4 \
        --question_frame_num 4 \
        --precision bf16 &
done

echo "All processes started in background"
wait