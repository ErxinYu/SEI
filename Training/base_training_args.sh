#!/bin/bash

ID=$RANDOM
export CUDA_VISIBLE_DEVICES="2,5" 
export header="NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node 2 --nnodes 1 \
--rdzv-id=$ID --rdzv_backend c10d \
-m less.train.train"

export base_training_args="--do_train True \
--max_seq_length 512 \
--use_fast_tokenizer True \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--evaluation_strategy no \
--logging_steps 1 \
--save_strategy no \
--num_train_epochs 3 \
--bf16 False \
--tf32 False \
--fp16 True \
--overwrite_output_dir True \
--report_to wandb \
--optim adamw_torch \
--seed 0 \
--percentage 1 \
--save_strategy epoch \
--lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--learning_rate 2e-05 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1"