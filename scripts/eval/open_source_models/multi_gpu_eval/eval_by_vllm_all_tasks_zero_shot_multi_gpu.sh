#!/bin/bash
BATCH_SIZE=1

#DEVICE_IDS=(0)
#MODEL_NAME_OR_PATH_LIST=(
#    "/root/private_data/Reason-RFT/checkpoints/resume_finetune_qwen2vl_2b_task1_only_rl-2025-06-18-00-26-16/checkpoint-1900"
#    "/root/private_data/Reason-RFT/checkpoints/resume_finetune_qwen2vl_2b_task1_only_rl-2025-06-18-00-26-16/checkpoint-1900"
#    "/root/private_data/Reason-RFT/checkpoints/resume_finetune_qwen2vl_2b_task1_only_rl-2025-06-18-00-26-16/checkpoint-1900"
#    "/root/private_data/Reason-RFT/checkpoints/resume_finetune_qwen2vl_2b_task1_only_rl-2025-06-18-00-26-16/checkpoint-1900"
#)
#
#BENCHMARK_LIST=("pointLabel13" "pointLabel13" "pointLabel13" "pointLabel13")
#STRATAGE_LIST=("cot-sft" "cot-sft" "cot-sft" "cot-sft")
#CITY_LIST=("roma" "chengdu" "sanfrancisco" "porto")
#
#for i in "${!DEVICE_IDS[@]}"; do
#    CUDA_VISIBLE_DEVICES=${DEVICE_IDS[$i]} python eval/eval_by_vllm_for_open_source.py \
#        --batch_size $BATCH_SIZE \
#        --model_name_or_path ${MODEL_NAME_OR_PATH_LIST[$i]} \
#        --benchmark_list ${BENCHMARK_LIST[$i]} \
#        --stratage_list ${STRATAGE_LIST[$i]} \
#        --city ${CITY_LIST[$i]} &
#done
#
#wait
#echo "All task finish."

DEVICE_IDS=(0)
MODEL_NAME_OR_PATH_LIST=(
    "/root/private_data/Reason-RFT/checkpoints/resume_finetune_qwen2vl_2b_task1_only_rl-2025-06-18-00-26-16/checkpoint-1900"
)

BENCHMARK_LIST=("pointLabel13")
STRATAGE_LIST=("cot-sft")
CITY_LIST=("chengdu")

for i in "${!DEVICE_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES=${DEVICE_IDS[$i]} python eval/eval_by_vllm_for_open_source.py \
        --batch_size $BATCH_SIZE \
        --model_name_or_path ${MODEL_NAME_OR_PATH_LIST[$i]} \
        --benchmark_list ${BENCHMARK_LIST[$i]} \
        --stratage_list ${STRATAGE_LIST[$i]} \
        --city ${CITY_LIST[$i]} &
done

wait
echo "All task finish."