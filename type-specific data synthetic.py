from openai import OpenAI
import json
from tqdm import tqdm
import os
from random import sample
import re

from rouge_score import rouge_scorer, scoring
import numpy as np
from multiprocessing import Pool
from functools import partial
# 初始化ROUGE评分器
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

client = OpenAI(api_key = "<KEY>")
client.base_url = "https://open.xiaoai.one/v1" # 
client.api_key = ""

def cal_similarity(seed_tokens_list, generated_data_):
    # 并行计算每个生成数据的相似度
    with Pool(os.cpu_count()) as pool:
        # 分词
        try:
            inst_tokens = scorer._tokenizer.tokenize(generated_data_["question"])
            # 计算ROUGE分数
            rouge_scores = pool.map(partial(rouge_scorer._score_lcs, inst_tokens), seed_tokens_list)
            # 如果任何一个相似度大于0.7，则丢弃
            rouge_scores = [score.fmeasure for score in rouge_scores]
            print("max rouge_scores",max(rouge_scores),len(rouge_scores))
            if max(rouge_scores) > 0.7:
                return seed_tokens_list, 0
            # 否则，加入种子指令中
            seed_tokens_list.append(inst_tokens)
        except Exception as e:
            print(f"Error: {e}",generated_data_)
    return seed_tokens_list,1


def generate_prompt_new(cluster_name, sampled_seed, sampled_generated):
    # print("sampled_seed",sampled_seed)
    # print("sampled_generated",sampled_generated)
    
    content = f'''
Based on the given examples, which make the {cluster_name}, create 20 difficult math problems that are likely to cause the same error in the model.

Requirement:
1. Identify the commonality in the given examples and consider what issues in these examples might cause the model to make mistakes.
2. Make the new problems more challenging and diverse.
3. Format the output strictly as a string in this structure: [{{"question":,"solution":}}, {{"question":,"solution":,}},...]. Ensure no additional output beyond the specified structure. Output in JSON format.
4. The reasoning process for each step should be provided in the solution.
5. Ensure the final answer is a number and place it on a new line, denoted by \n#### num.

Provided Questions:
'''
    for i, case in enumerate(sampled_seed + sampled_generated):
        content += f"Question{i + 1}: {case['question']}\n"
        content += f"Solution{i + 1}: {case['solution']}\n"
    content += '''
Generated Questions:
'''
    return content

# 读取输入文件
data = []
input_file = '/home/yex/LESS/output/gsm8k/qwen/6seed_badcases.json'
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        bad_case = json.loads(line)
        data.append(bad_case)

# 构建从簇名到错误案例列表的映射
cluster_bad_cases = {}
for bad_case in data:
    cluster_names = bad_case.get('cluster_name', [])
    for cluster_name in cluster_names:
        if cluster_name not in cluster_bad_cases:
            cluster_bad_cases[cluster_name] = []
        cluster_bad_cases[cluster_name].append(bad_case)
print("cluster_bad_cases",cluster_bad_cases.keys())
# 计算总的目标案例数量
total_target_cases = sum(len(bad_cases) for bad_cases in cluster_bad_cases.values())
print("total_target_cases",total_target_cases)
# 根据比例确定每个簇需要生成的新问题数量
cluster_target_num = {}
for cluster_name, bad_cases in cluster_bad_cases.items():
    proportion = len(bad_cases) / total_target_cases
    cluster_target_num[cluster_name] = int(15000 * proportion)  # 根据需要调整倍数
print("cluster_target_num",cluster_target_num)


#已经生成的数据
# input_file_1 = f'/home/yex/LESS/output/gsm8k/iter-1/8_15k_data.json'
# previous_generated_data = []
# with open(input_file_1, 'r') as f:
#     for line in f:
#         line = line.strip()
#         generate_case = json.loads(line)
#         previous_generated_data.append(generate_case)  
# print(len(previous_generated_data))

#seed 指令构建
seed_data = []
for bad_case in data:
    seed_data.append({"question":bad_case['question'], "solution":bad_case["solution"]})



# 读取测试文件，要和测试文件保持不同
test_data = []
input_file = '/home/yex/data/gsm8k/gsm8k_test.json'
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        bad_case = json.loads(line)
        test_data.append(bad_case)
seed_tokens_list = [scorer._tokenizer.tokenize(seed["question"]) for seed in seed_data  + test_data]
print(len(seed_tokens_list),len(seed_data),len(test_data))
cluster_generated_data = {}
all_generated_data = []
for cluster_name, target_num in cluster_target_num.items():
    print(f"Processing {cluster_name}",cluster_target_num)
    # if cluster_name == "Multiplication and Division Errors":
    #     continue
    cat_generate_data = []
    output_file = f'/home/yex/LESS/output/gsm8k/iter-2/7temp/7{cluster_name}_generated_data.json'
    print("cat_generate_data",len(cat_generate_data))
    with open(output_file, 'w') as f:
        while len(cat_generate_data) < target_num:  # 需要生成5个新问题
            print(f"\n\nWe have generated {len(cat_generate_data)} data of {cluster_name}")
            if len(cat_generate_data) < 3:
                sampled_seed = sample(seed_data, min(8,len(seed_data)))
                sampled_generated = []
            else:
                sampled_seed = sample(seed_data, 5)
                sampled_generated = sample(cat_generate_data, 3)
            try:
                prompt = generate_prompt_new(cluster_name, sampled_seed, sampled_generated)
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )
                new_questions_response = resp.choices[0].message.content    
                json_pattern = r"```json(.*?)```"
                match = re.search(json_pattern, new_questions_response, re.DOTALL)
                if match == None:
                    content = new_questions_response
                else:
                    content = match.group(1)
                new_questions_response = json.loads(content)
                for questions_response in new_questions_response:
                    seed_tokens_list, flag_keep = cal_similarity(seed_tokens_list, questions_response)
                    if flag_keep:
                        json.dump(questions_response, f)
                        questions_response["cluster"] = cluster_name
                        cat_generate_data.append(questions_response)
                        all_generated_data.append(questions_response)
                        f.write('\n')
                    print("final:",len(seed_tokens_list),len(all_generated_data),len(cat_generate_data))
            except Exception as e:
                print(f"Error generating new questions: {e}")

    cluster_generated_data[cluster_name] = cat_generate_data

# # 保存生成的数据到输出文件
output_file = f'/home/yex/LESS/output/gsm8k/qwen/7generated_data.json'
with open(output_file, 'w') as f:
    for cluster_name in cluster_generated_data:
        for data in cluster_generated_data[cluster_name]:
            json.dump(data, f)
            f.write('\n')
print(f"Generated data saved to {output_file}")