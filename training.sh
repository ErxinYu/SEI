#!/bin/bash

# Set environment
export PYTHONPATH="[PROJECT_ROOT]"
export CUDA_VISIBLE_DEVICES="4"

# Configure paths
PROJECT_ROOT="[PROJECT_ROOT]"
DATA_DIR="${PROJECT_ROOT}/output/combined"
OUTPUT_DIR="${PROJECT_ROOT}/sft/gsm8k"

# Training configuration
TRAIN_FILE="${DATA_DIR}/llama3_iter2_gsm8k_15k_data.json"
JOB_NAME="llama3_iter2_gsm8k_15k_data.json"
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"

# Define training arguments
training_args="
--percentage 1 
--output_dir ${OUTPUT_DIR}/${JOB_NAME} 
--model_name_or_path ${MODEL_PATH}
--train_files ${TRAIN_FILE}

# Model configuration
--lora True
--lora_r 8 
--lora_alpha 32
--lora_dropout 0.1
--lora_target_modules q_proj k_proj v_proj o_proj

# Training parameters
--num_train_epochs 3
--learning_rate 2e-05
--per_device_train_batch_size 1
--gradient_accumulation_steps 1
--max_seq_length 512
--warmup_ratio 0.03
--weight_decay 0.0

# Training settings
--do_train True
--use_fast_tokenizer True
--lr_scheduler_type linear
--evaluation_strategy no
--logging_steps 1
--save_strategy epoch
--overwrite_output_dir True

# Hardware optimization
--bf16 False
--tf32 False
--fp16 True

# Misc
--report_to wandb
--optim adamw_torch
--seed 0
"

# Execute training with logging
LOG_FILE="${OUTPUT_DIR}/${JOB_NAME}/train.log"
mkdir -p "$(dirname "$LOG_FILE")"
python -m less.train.single_train ${training_args} 2>&1 | tee "$LOG_FILE"
