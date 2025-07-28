#!/bin/bash
BATCH_SIZE=2

DEVICE_IDS=(0 1 2 3)
MODEL_NAME_OR_PATH="/root/private_data/Reason-RFT/checkpoints/resume_finetune_qwen2vl_2b_task1_only_rl-2025-07-09-12-36-57/checkpoint-200"


BENCHMARK_LIST="pointLabel13"
STRATAGE_LIST="cot-sft"
CITY_LIST=("roma" "sanfrancisco" "chengdu" "porto")

for i in "${!DEVICE_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES=${DEVICE_IDS[$i]} python eval/eval_by_vllm_for_open_source.py \
        --batch_size $BATCH_SIZE \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --benchmark_list $BENCHMARK_LIST \
        --stratage_list $STRATAGE_LIST \
        --city ${CITY_LIST[$i]} &
done

wait
echo "All task finish."