# main evaluation function
# Model options:
#Qwen/Qwen2.5-Math-7B
#meta-llama/Meta-Llama-3-8B-Instruct

export PYTHONPATH="[PROJECT_ROOT]"
model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
lora_path=[PROJECT_ROOT]/sft/gsm8k/iter-1/Llama3_top0.2_syntheticData_alpaca/checkpoint-11841
export CUDA_VISIBLE_DEVICES="[GPU_ID]" 

python -m evaluation.run_eval \
    --eval_dataset train\
    --data_dir [DATA_ROOT]/gsm8k/gsm8k_train.json\
    --n_shot 0 \
    --save_dir [PROJECT_ROOT]/output/gsm8k/iter-3 \
    --model_name_or_path $model_name_or_path\
    --tokenizer $model_name_or_path \
    --use_chat_format \
    --eval_batch_size 16 \
    --max_num_examples 10000\
    --lora_path $lora_path\
