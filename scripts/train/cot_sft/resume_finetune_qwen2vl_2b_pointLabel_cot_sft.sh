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

#export WANDB_MODE=offline
# Wandb
export WANDB_MODE=disabled
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=trajectory-reasoning
export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export WANDB_RUN_NAME=cotsft_qwen2vl_2b_task1_only_rl-$(date +%Y-%m-%d-%H-%M-%S)
wandb login $WANDB_API_KEY

export ACCELERATE_CPU_AFFINITY=1

export IMAGE_DIR=/root/private_data/Reason-RFT/VLM
# export PRETRAIN_MODEL_PATH=/root/private_data/Reason-RFT/models/Qwen2-1.5B-Instruct
export PRETRAIN_MODEL_PATH=/root/private_data/Reason-RFT/checkpoints/cotsft_qwen2vl_2b_task1_only_rl-2025-06-16-22-43-49/checkpoint-1000
export OUTPUT_PATH=$(pwd)/checkpoints/${WANDB_RUN_NAME}
export DATASET=pointLabel,pointLabelPorto,pointLabelSanfrancisco,pointLabelRome,pointLogic,pointLogicPorto,pointLogicSanfrancisco,pointLogicRome
# export DATASET=pointLabelLLMSanfrancisco
# pointLabelLLM, pointLabelLLMPorto, pointLabelLLMRome, pointLabelLLMSanfrancisco


export DISABLE_VERSION_CHECK=1

if [ ! -d "$OUTPUT_PATH" ]; then
  mkdir "$OUTPUT_PATH"
fi
# finetuning_type lora  / full
torchrun --nproc_per_node=4 --nnodes=1 --master_port=29514 \
  train/stage_sft/train.py \
  --freeze_multi_modal_projector false \
  --train_mm_proj_only false \
  --deepspeed scripts/train/zero3.json \
  --stage sft \
  --do_train \
  --model_name_or_path $PRETRAIN_MODEL_PATH \
  --dataset $DATASET \
  --image_dir $IMAGE_DIR \
  --template qwen2_vl \
  --finetuning_type full \
  --output_dir $OUTPUT_PATH \
  --overwrite_cache \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --weight_decay 0.1 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --ddp_timeout 90000 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --cutoff_len 8192 \
  --save_steps 200 \
  --plot_loss \
  --num_train_epochs 15 \
  --bf16 \
  2>&1 | tee ${OUTPUT_DIR}/train.log