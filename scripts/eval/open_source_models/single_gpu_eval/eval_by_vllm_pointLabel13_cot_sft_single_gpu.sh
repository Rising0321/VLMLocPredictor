#!/bin/bash

DEVICE_ID=2
BATCH_SIZE=2

MODEL_NAME_OR_PATH=/root/private_data/Reason-RFT/checkpoints/resume_finetune_qwen2vl_2b_task1_only_rl-2025-07-08-18-38-41/checkpoint-1500
BENCHMARK_LIST="pointLabel13"
STRATAGE_LIST="cot-sft"
CITY="chengdu"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python eval/eval_by_vllm_for_open_source.py \
    --batch_size $BATCH_SIZE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --benchmark_list $BENCHMARK_LIST \
    --city $CITY \
    --stratage_list $STRATAGE_LIST