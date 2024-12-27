from openai import OpenAI
import json
from tqdm import tqdm
import os

# Initialize OpenAI client with anonymized credentials
client = OpenAI()
client.base_url = "[API_ENDPOINT]/v1"  # Proxy endpoint with required v1 suffix
client.api_key = "[API_KEY]"

def generate_prompt_new(row):
    content = f'''
Based on the given mathematical problem, identify the step where the model made an error in its reasoning process. Analyze the reason for this error and summarize it using a keyphrase. The input consists of a math question, the correct answer, and the model's incorrect answer. Please output the result in the following format:

["Error keyphrase"]

Ensure that your analysis focuses on the mistake in the model's problem-solving process. The keyphrases should be concise yet descriptive, effectively summarizing the primary reason for the model's mistake. Strictly adhere to the list format output without any additional information.

Math Question: {row['question']}
Solution: {row['solution']}
Model Output: {row['model_output']}
'''
    return content

# File paths
input_file = '[PROJECT_ROOT]/0train_predictions_gsm_iter_3.json'
output_file = '[PROJECT_ROOT]/1train_badCase_errorKeywords.json'

# Read input data
data = []
with open(input_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Process each row
for i, row in enumerate(data, start=1):
    print(i)
    prompt = generate_prompt_new(row)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        error_keywords = eval(resp.choices[0].message.content)
        row["error_Keywords"] = error_keywords
        print(error_keywords)
    except Exception as e:
        print(f"Error processing row: {e}")

# Write results to output file
with open(output_file, 'w') as f:
    for row in data:
        json.dump(row, f)
        f.write('\n')

print(f"Processing complete. Results saved to {output_file}")
