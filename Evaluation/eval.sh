export CUDA_VISIBLE_DEVICES="5" 
#--target_tasks "gsm8k,MATH.Algebra,MATH.Counting_&_Probability,MATH.Geometry,MATH.Intermediate_Algebra,MATH.Number_Theory,MATH.Prealgebra,MATH.Precalculus,college_math.algebra,college_math.precalculus,college_math.calculus,college_math.vector_calculus,college_math.probability,college_math.linear_algebra,college_math.differential_equation,tal,gaokao_bench_math_en,math23k_en,agieval.gaokao-math-en,agieval.math,agieval.sat-math"
# "MATH.Algebra,MATH.Counting_&_Probability,MATH.Geometry,MATH.Intermediate_Algebra,MATH.Number_Theory,MATH.Prealgebra,MATH.Precalculus"
#--model_name_or_path Qwen/Qwen2.5-Math-7B\
#deepseek-ai/deepseek-math-7b-instruct
#meta-llama/Meta-Llama-3-8B-Instruct
#--target_tasks "gsm8k,MATH.Algebra,MATH.Counting_&_Probability,MATH.Geometry,MATH.Intermediate_Algebra,MATH.Number_Theory,MATH.Prealgebra,MATH.Precalculus"
python -m driver \
    --batch_size 16 \
    --prompt_template alpaca \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct\
    --save_dir ./result/combined/llama3_qwen_3w/ \
    --lora_path /home/yex/LESS/sft/combined/llama3_qwen_3w/checkpoint-90000\
    --target_tasks "gsm8k"