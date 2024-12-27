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

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

# Initialize OpenAI client with anonymized credentials
client = OpenAI()
client.base_url = "[API_ENDPOINT]/v1"
client.api_key = "[API_KEY]"

def cal_similarity(seed_tokens_list, generated_data_):
    with Pool(os.cpu_count()) as pool:
        try:
            inst_tokens = scorer._tokenizer.tokenize(generated_data_["question"])
            rouge_scores = pool.map(partial(rouge_scorer._score_lcs, inst_tokens), seed_tokens_list)
            rouge_scores = [score.fmeasure for score in rouge_scores]
            print("max rouge_scores", max(rouge_scores), len(rouge_scores))
            if max(rouge_scores) > 0.7:
                return seed_tokens_list, 0
            seed_tokens_list.append(inst_tokens)
        except Exception as e:
            print(f"Error: {e}", generated_data_)
    return seed_tokens_list, 1

def generate_prompt_new(cluster_name, sampled_seed, sampled_generated):
    content = f'''
Based on the given examples and error type, create 20 difficult math problems that are likely to cause errors in the model.
Requirement:
1. Identify the commonality in the given examples and consider what issues in these examples might cause the model to make mistakes.
2. Make the new problems more challenging and diverse.
3. Format the output strictly as a string in this structure: [{{"question":,"solution":}}, {{"question":,"solution":,}},...].
Ensure no additional output beyond the specified structure. Output in JSON format.
4. The reasoning process for each step should be provided in the solution.
5. Ensure the final answer is a number and place it on a new line, denoted by \n#### num.
6. Don’t make any mathematical mistakes of your own!

Provided Questions:
'''
    for i, case in enumerate(sampled_seed + sampled_generated):
        content += f"Question{i + 1}: {case['question']}\n"
        content += f"Solution{i + 1}: {case['solution']}\n"
    content += '''
Generated Questions:
'''
    return content

# File paths
PROJECT_ROOT = "[PROJECT_ROOT]"
input_file = f"{PROJECT_ROOT}/output/gsm8k/qwen/6seed_badcases.json"
test_file = f"{PROJECT_ROOT}/data/gsm8k/gsm8k_test.json"
output_dir = f"{PROJECT_ROOT}/output/gsm8k/iter-2/7temp"
final_output = f"{PROJECT_ROOT}/output/gsm8k/qwen/7generated_data.json"

# Read seed bad cases
data = []
with open(input_file, 'r') as f:
    for line in f:
        data.append(json.loads(line.strip()))

# Build cluster mapping
cluster_bad_cases = {}
for bad_case in data:
    cluster_names = bad_case.get('cluster_name', [])
    for cluster_name in cluster_names:
        if cluster_name not in cluster_bad_cases:
            cluster_bad_cases[cluster_name] = []
        cluster_bad_cases[cluster_name].append(bad_case)

# Calculate target numbers
total_target_cases = sum(len(bad_cases) for bad_cases in cluster_bad_cases.values())
cluster_target_num = {
    cluster_name: int(15000 * len(bad_cases) / total_target_cases)
    for cluster_name, bad_cases in cluster_bad_cases.items()
}

# Prepare seed data
seed_data = [{"question": case['question'], "solution": case["solution"]} for case in data]

# Read test data
test_data = []
with open(test_file, 'r') as f:
    for line in f:
        test_data.append(json.loads(line.strip()))

# Initialize token list
seed_tokens_list = [scorer._tokenizer.tokenize(seed["question"]) for seed in seed_data + test_data]

# Generate data for each cluster
cluster_generated_data = {}
all_generated_data = []

for cluster_name, target_num in cluster_target_num.items():
    print(f"Processing {cluster_name}", cluster_target_num)
    cat_generate_data = []
    output_file = f"{output_dir}/7{cluster_name}_generated_data.json"
    
    with open(output_file, 'w') as f:
        while len(cat_generate_data) < target_num:
            print(f"\n\nWe have generated {len(cat_generate_data)} data of {cluster_name}")
            
            # Sample seed and generated data
            if len(cat_generate_data) < 3:
                sampled_seed = sample(seed_data, min(8, len(seed_data)))
                sampled_generated = []
            else:
                sampled_seed = sample(seed_data, 5)
                sampled_generated = sample(cat_generate_data, 3)
            
            try:
                # Generate new questions
                prompt = generate_prompt_new(cluster_name, sampled_seed, sampled_generated)
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Process response
                new_questions_response = resp.choices[0].message.content    
                json_pattern = r"```json(.*?)```"
                match = re.search(json_pattern, new_questions_response, re.DOTALL)
                content = match.group(1) if match else new_questions_response
                new_questions_response = json.loads(content)
                
                # Process each generated question
                for questions_response in new_questions_response:
                    seed_tokens_list, flag_keep = cal_similarity(seed_tokens_list, questions_response)
                    if flag_keep:
                        questions_response["cluster"] = cluster_name
                        json.dump(questions_response, f)
                        f.write('\n')
                        cat_generate_data.append(questions_response)
                        all_generated_data.append(questions_response)
                    print("final:", len(seed_tokens_list), len(all_generated_data), len(cat_generate_data))
                    
            except Exception as e:
                print(f"Error generating new questions: {e}")

    cluster_generated_data[cluster_name] = cat_generate_data

# Save final results
with open(final_output, 'w') as f:
    for cluster_name in cluster_generated_data:
        for data in cluster_generated_data[cluster_name]:
            json.dump(data, f)
            f.write('\n')

print(f"Generated data saved to {final_output}")
