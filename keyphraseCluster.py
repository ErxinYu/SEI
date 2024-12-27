from openai import OpenAI
import json
from tqdm import tqdm
import os

# Initialize OpenAI client with anonymized credentials
client = OpenAI()
client.base_url = "[API_ENDPOINT]"
client.api_key = "[API_KEY]"

def generate_prompt_new(row):
    content = f'''

You are an expert in error analysis and categorization. You will be given a list of error keyphrases. Your task is to:
1. Analyze the given error keyphrases and identify common themes or patterns.
2. Group similar keyphrases together based on their likely causes, effects, or areas of occurrence. 3. For each cluster:
a. List the keyphrases in the cluster.
b. Explain why these keyphrases are grouped together.
c. Assign a concise but descriptive name to the cluster that captures its essence. 4. Clusters should cover all the keyphrases.
5. Present your results in a clear, structured format.
Strictly output in plain text according to the following format, do not output in other formats or with extra symbols:

[
{{"Cluster name":, "Keyphrases":[], "explanation":,}},
{{"Cluster name":, "Keyphrases":[], "explanation":,}} ... 
]

Your clustering should aim to provide meaningful insights that can help in understanding and addressing the errors more effectively.
Here is the list of error keyphrases: {Error Keyphrases Set}
'''
    return content

# File paths
input_file = '[PROJECT_ROOT]/output/gsm8k/qwen/2unique_error_keywords.txt'
output_file = '[PROJECT_ROOT]/output/gsm8k/qwen/3seedErrorCluster.json'

# Read input data
data = []
with open(input_file, 'r') as f:
    for line in f:
        data.append(line.strip())

# Process each line
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

        # Parse response to Python object
        error_cluster = eval(error_cluster)
        print(error_cluster)
        
        # Count total phrases
        num = sum(len(cluster['Keyword phrases']) for cluster in error_cluster)
        print(num)
        
        # Write results if valid
        if isinstance(error_cluster, list):
            for error in error_cluster:
                json.dump(error, f)
                f.write('\n')
        break

print(f"Processing complete. Results saved to {output_file}")
