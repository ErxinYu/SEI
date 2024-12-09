from openai import OpenAI
import json
from tqdm import tqdm
import os

client = OpenAI(api_key="<KEY>")
client.base_url = "https://open.xiaoai.one/v1" 
client.api_key = ""

def generate_prompt_new(row):
    content = f'''
You are an expert in error analysis and categorization. You will be given a list of error keyword phrases. Your task is to:

1. Analyze the given error keyword phrases and identify common themes or patterns.
2. Group similar keyword phrases together based on their likely causes, effects, or areas of occurrence.
3. For each cluster:
   a. List the keyword phrases in the cluster.
   b. Explain why these keyword phrases are grouped together.
   c. Assign a concise but descriptive name to the cluster that captures its essence.
4. Clusters should cover all the keyword phrases.
5. Present your results in a clear, structured format.

Strictly output in plain text according to the following format, do not output in other formats or with extra symbols:
[
{{"Cluster name":, "Keyword phrases":[], "explanation":,}}, 
{{"Cluster name":, "Keyword phrases":[], "explanation":,}} ...
]

Your clustering should aim to provide meaningful insights that can help in understanding and addressing the errors more effectively.
Here is the list of error keywords: 
{row}
'''
    return content

# 读取输入文件
input_file = '/home/yex/LESS/output/gsm8k/qwen/2unique_error_keywords.txt'
output_file = '/home/yex/LESS/output/gsm8k/qwen/3seedErrorCluster.json'

data = []
with open(input_file, 'r') as f:
    for line in f:
        data.append(line.strip())

# 就处理每一行数据
with open(output_file, 'w') as f:
    for i, row in enumerate(data, start=1):
        print(i)
        prompt = generate_prompt_new(row)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        error_cluster = resp.choices[0].message.content
        # print("error_keywords", error_cluster)

        # 使用 eval() 解析字符串为 Python 对象
        error_cluster = eval(error_cluster)
        print(error_cluster)
        num = 0
        for cluster in error_cluster:
            num += len(cluster['Keyword phrases'])
        print(num)
        # 确保 error_cluster 是一个列表
        if isinstance(error_cluster, list):
            for error in error_cluster:
                json.dump(error, f)
                f.write('\n')
        break
print(f"Processing complete. Results saved to {output_file}")
