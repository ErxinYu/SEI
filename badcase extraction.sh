# main evaluation function
#Qwen/Qwen2.5-Math-7B
#meta-llama/Meta-Llama-3-8B-Instruct
export PYTHONPATH="/home/yex/LESS"
model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
lora_path=/home/yex/LESS/sft/gsm8k/iter-1/Llama3_top0.2_syntheticData_alpaca/checkpoint-11841
export CUDA_VISIBLE_DEVICES="4" 
python -m evaluation.eval.gsm.run_eval \
    --eval_dataset train\
    --data_dir /home/yex/data/gsm8k/gsm8k_train.json\
    --n_shot 0 \
    --save_dir /home/yex/LESS/output/gsm8k/iter-3 \
    --model_name_or_path  $model_name_or_path\
    --tokenizer $model_name_or_path \
    --use_chat_format \
    --eval_batch_size 16 \
    --max_num_examples 10000\
    --lora_path $lora_path\