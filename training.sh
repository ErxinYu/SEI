
export PYTHONPATH="/home/yex/LESS"

train_files="/home/yex/LESS/output/combined/llama3_iter2_gsm8k_15k_data.json"
job_name="llama3_iter2_gsm8k_15k_data.json"
export CUDA_VISIBLE_DEVICES="4"

model_path="meta-llama/Meta-Llama-3-8B-Instruct"

# Training arguments
#--filter_em_score 0.05 \
training_args="\
--percentage 1 \
--output_dir /home/yex/LESS/sft/gsm8k/${job_name} \
--lora_r 8 \
--lora_alpha 32 \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--do_train True \
--max_seq_length 512 \
--use_fast_tokenizer True \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--evaluation_strategy no \
--logging_steps 1 \
--save_strategy no \
--bf16 False \
--tf32 False \
--fp16 True \
--overwrite_output_dir True \
--report_to wandb \
--optim adamw_torch \
--seed 0 \
--save_strategy epoch \
--lora True \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--learning_rate 2e-05 \
--model_name_or_path $model_path \
--train_files $train_files"


# Execute training
python -m less.train.single_train $training_args 2>&1 | tee "$output_dir/train.log"