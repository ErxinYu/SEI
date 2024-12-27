from openai import OpenAI
import json
from tqdm import tqdm
import os

client = OpenAI(api_key = "<KEY>")
client.base_url = "https://open.xiaoai.one/v1" # 换成代理，一定要加v1
client.api_key = "sk-b4VB6HsCGierg3d2623158C77bB84aA68087C3B920A46d4c"


def generate_prompt_new(row):
    content = f'''
Based on the given mathematical problem, identify the step where the model first made an error in its reasoning process. Analyze the reason for this error and summarize it using a keywords phrase. The input consists of a math question, the correct answer, and the model's incorrect answer. Please output the result in the following format:

["keywords phrase"]

Ensure that your analysis focuses on the initial mistake in the model's problem-solving process. The keywords phrase should be concise yet descriptive, effectively summarizing the primary reason for the model's first mistake. Strictly adhere to the list format output without any additional information.

Math Question: {row['question']}
Solution: {row['solution']}
Model Output: {row['model_output']}
'''
    return content

# 读取输入文件
input_file = '/Users/erxinyu/Desktop/LESS____/0train_predictions_gsm_iter_3.json'
output_file = '/Users/erxinyu/Desktop/LESS____/1train_badCase_errorKeywords.json'

data = []
with open(input_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

# 处理每一行数据
i=1
for row in data:
    i+=1
    print(i)
    # if i ==10:
    #     break
    prompt = generate_prompt_new(row)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        # print("error_keywords",resp.choices[0].message.content)
        error_keywords = resp.choices[0].message.content
        # print("error_keywords",error_keywords)
        error_keywords = eval(error_keywords)
        row["error_Keywords"] = error_keywords
        print(error_keywords)
        # exit()
    except Exception as e:
        print(f"Error processing row: {e}")
        # row["error keywords"] = ["processing_error"]
        # exit()

# 将结果写入输出文件
# 将结果写入输出文件，每行一个JSON对象
with open(output_file, 'w') as f:
    for row in data:
        json.dump(row, f)
        f.write('\n')

print(f"Processing complete. Results saved to {output_file}")
