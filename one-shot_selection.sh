# main evaluation function
model_name_or_path=Qwen/Qwen2.5-Math-7B
export CUDA_VISIBLE_DEVICES="5" 
export PYTHONPATH="/home/yex/LESS"
#Qwen/Qwen2.5-Math-7B
#meta-llama/Meta-Llama-3-8B-Instruct
python -m evaluation.eval.gsm.run_one_shot_eval \
    --test_data /home/yex/LESS/output/gsm8k/qwen/6dev_bad_good20.json\
    --instruct_data /home/yex/LESS/output/gsm8k/qwen/7generated_data.json\
    --data_start 13000 \
    --data_end 15000 \
    --mode select \
    --save_dir /home/yex/LESS/output/gsm8k/qwen/8temp \
    --model_name_or_path  $model_name_or_path\
    --tokenizer $model_name_or_path \
    --use_chat_format \
    --eval_batch_size 16 \
