from openai import OpenAI
import json
from tqdm import tqdm
import os
import re

client = OpenAI(api_key="<KEY>")
client.base_url = "https://open.xiaoai.one/v1" 
client.api_key = ""

def generate_prompt_new(row, existing_clusters):
    content = f'''
You are an expert in error analysis and categorization. You will be given a list of error keyword phrases and a set of existing clusters. Your task is to:

1. Cluster these keyword phrasess into meaningful groups.
2. Provide the clustering results.
3. Explain the reasoning behind each cluster.
4. Assign a descriptive name to each cluster.

Please follow these steps:

1. Analyze the given error keyword phrases and the existing clusters to identify common themes or patterns.
2. For each keyword phrase, determine if it fits into any existing cluster based on likely causes, effects, or areas of occurrence.
3. For each existing cluster:
   a. List the keyword phrases that have been assigned to it.
   b. Explain why these keyword phrases belong to this cluster.
4. If a keyword phrase does not fit into any existing cluster, create a cluster:
   a. List the keyword phrases in this cluster.
   b. Explain why these keyword phrases are grouped together.
   c. Assign a concise but descriptive name to the cluster that captures its essence.
5. Ensure all keyword phrases are covered by clusters.

Strictly output in plain text according to the following format:
[
{{"Cluster name":, "Keyword phrases":[], "explanation":,}},
{{"Cluster name":, "Keyword phrases":[], "explanation":,}}
...
] 
Ensure no additional output beyond the specified structure. Output in JSON format.

Your clustering should aim to provide meaningful insights that can help in understanding and addressing the errors more effectively.

Here is the list of error keywords: 
{row}

And here are the existing clusters:
{existing_clusters}
'''
    return content

# 读取输入文件
input_file1 = '/home/yex/LESS/output/gsm8k/qwen/2unique_error_keywords.txt'
input_file2 = '/home/yex/LESS/output/gsm8k/qwen/3seedErrorCluster.json'
output_file = '/home/yex/LESS/output/gsm8k/qwen/4extendErrorCluster.json'

# 读取 row 数据
rows = []
with open(input_file1, 'r') as f:
    next(f)  # 跳过第一行
    for line in f:
        rows.append(line.strip())  # 假设每行是一个关键词列表
        # print(len(line.split(",")))
# 读取 existing_clusters 数据
data = []
existing_clusters = []
with open(input_file2, 'r') as f:
    for line in f:
        entry = json.loads(line)
        data.append(entry)
        existing_clusters.append(entry["Cluster name"])

# 处理每一行数据

for i, row in enumerate(rows, start=1):
    prompt = generate_prompt_new(row, existing_clusters)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        error_cluster = resp.choices[0].message.content
        
        json_pattern = r"```json(.*?)```"
        match = re.search(json_pattern, error_cluster, re.DOTALL)
        if match == None:
            content = error_cluster
        else:
            content = match.group(1)
        error_cluster = json.loads(content)

        num = 0
        for cluster in error_cluster:
            num += len(cluster['Keyword phrases'])
        if isinstance(error_cluster, list):
            print(i,num)
            for error in error_cluster:
                cluster_name = error["Cluster name"]
                keywords = error["Keyword phrases"]
                # 检查 cluster_name 是否在 existing_clusters 中
                if cluster_name in existing_clusters:
                    # 找到对应的 data 条目并更新其关键词列表
                    for entry in data:
                        if entry["Cluster name"] == cluster_name:
                            entry["Keyword phrases"].extend(keywords)
                else:
                    # 如果是新的 cluster，则直接添加到 data
                    data.append(error)
                    # error["Cluster name"] = error["Cluster name"].replace("New Cluster: ", "", 1)
                    existing_clusters.append(cluster_name)
                    print(existing_clusters)
    except Exception as e:
        print(f"Error processing row: {e}")
# 将更新后的 data 写入文件
with open(output_file, 'w') as f:
    for entry in data:
        json.dump(entry, f)
        f.write('\n')

print(f"Processing complete. Results saved to {output_file}")
