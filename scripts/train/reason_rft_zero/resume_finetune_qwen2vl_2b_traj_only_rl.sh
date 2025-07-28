#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=$(pwd)/train/
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export OMP_NUM_THREADS=4
export NCCL_IB_HCA=mlx5_0,mlx5_1

export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

# Wandb
export WANDB_MODE=offline
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=trajectory-reasoning
# export WANDB_API_KEY="ef8c75b02ac66d8f3b1d2c76774079d2c360e4e3"
# export WANDB_RUN_NAME=resume_finetune_qwen2vl_2b_task1_only_rl-$(date +%Y-%m-%d-%H-%M-%S)
# wandb login $WANDB_API_KEY

export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export WANDB_RUN_NAME=resume_finetune_qwen2vl_2b_task1_only_rl-$(date +%Y-%m-%d-%H-%M-%S)
swanlab login -k $WANDB_API_KEY

# Dataset
export TASK_NAME=trajectory
export DATASET_NAME=/root/private_data/Reason-RFT/trajDataJsonsDirty/sft/chengdu/pointLabel13/train-ours.json
export IMAGE_PATH=/root/private_data/Reason-RFT/VLM/
export MODEL_NAME_OR_PATH=/root/private_data/Reason-RFT/checkpoints/resume_finetune_qwen2vl_2b_task1_only_rl-2025-07-08-18-38-41/checkpoint-1600
# export MODEL_NAME_OR_PATH=/root/private_data/Reason-RFT/checkpoints/cotsft_qwen2vl_2b_task1_only_rl-2025-06-17-20-18-33/checkpoint-1400
# export MODEL_NAME_OR_PATH=/root/private_data/Reason-RFT/checkpoints/cotsft_qwen2vl_2b_task1_only_rl-2025-06-16-22-43-49/checkpoint-1000

export OUTPUT_DIR=$(pwd)/checkpoints/${WANDB_RUN_NAME}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir "$OUTPUT_DIR"
fi

# Debug
export DEBUG_MODE="True"
export LOG_PATH=${OUTPUT_DIR}/reward.log

torchrun --nproc_per_node=3 --nnodes=1 --master_port=29515 \
  train/stage_rl/grpo.py \
  --deepspeed scripts/train/zero3.json \
  --output_dir ${OUTPUT_DIR} \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --dataset_name ${DATASET_NAME} \
  --image_path ${IMAGE_PATH} \
  --task_name ${TASK_NAME} \
  --use_vllm_for_gen true \
  --use_system_prompt false \
  --max_prompt_length 2380 \
  --max_completion_length 512 \
  --num_generations 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --logging_steps 1 \
  --bf16 \
  --report_to wandb \
  --gradient_checkpointing false \
  --attn_implementation flash_attention_2 \
  --max_pixels 480000 \
  --save_steps 100 \
  --num_train_epochs 16 \
  --run_name ${WANDB_RUN_NAME} \
  2>&1 | tee ${OUTPUT_DIR}/train.log
