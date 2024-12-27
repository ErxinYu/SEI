import argparse
import os
import re
import json
import random
import torch
import vllm
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers 
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import sys

sys.path.append('[PROJECT_ROOT]/evaluation')
from eval.utils import (
    generate_completions,
    query_openai_chat_model,
    dynamic_import_function,
)
from eval.gsm.examplars import EXAMPLARS as GSM_EXAMPLARS

exact_match = evaluate.load("exact_match")

def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    with open(args.test_data) as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append({
                "question": example["question"],
                "solution": example["solution"],
            })

    # Process number formats
    for example in test_data:
        example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["solution"])
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", example["answer"])
        example["answer"] = numbers[-1]
        assert isinstance(float(example["answer"]), float), f"answer is not a valid number: {example['answer']}"

    instruction_data = []
    with open(args.instruct_data) as fin:
        for line in fin:
            data = json.loads(line)
            instruction_data.append(data)
    instruction_data = instruction_data[args.data_start:args.data_end]

    if args.use_chat_format:
        def apply_chat_format(instruction, example, tokenizer):
            instruct_question = instruction["question"]
            instruct_solution = instruction["solution"]
            demonstration = f"### Instruction:\n{instruct_question}\n\n### Response:\n{instruct_solution}"
            question = example["question"]
            prompt = "Below is an instruction that describes a task. "+ \
            "Write a response that appropriately completes the request.\n\n"+ \
            f"Here is an example:\n{demonstration}\n\n" + \
            f"### Instruction:\n{question}\n\n### Response:"
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return prompt

    if args.model_name_or_path:
        print("Loading model and tokenizer from ...", args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).half().cuda()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.bos_token
            tokenizer.padding_side = "left"

        prompts = []
        for instruct in instruction_data:
            for test_case in test_data:
                prompt_test_case = apply_chat_format(instruct, test_case, tokenizer)
                prompts.append(prompt_test_case)
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=512,
            batch_size=args.eval_batch_size,
            do_sample=False,
        )

    predictions = []
    for output in outputs:
        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            predictions.append(numbers[-1])
        else:
            predictions.append(output)
    targets = [example["answer"] for example in test_data]
    
    # Calculate scores
    em_scores = []
    em_scores_good = []
    em_scores_bad = []
    for i in range(0, len(instruction_data)):
        batch_predictions = predictions[i * 20:(i + 1) * 20]
        batch_badcases_outputs = batch_predictions[:10]
        batch_goodcases_outputs = batch_predictions[10:]

        em_score_bad = exact_match.compute(
            predictions=batch_badcases_outputs, 
            references=targets[:10], 
            ignore_case=True, 
            ignore_punctuation=True
        )["exact_match"]

        em_score_good = exact_match.compute(
            predictions=batch_goodcases_outputs, 
            references=targets[10:], 
            ignore_case=True, 
            ignore_punctuation=True
        )["exact_match"]  

        em_scores.append((em_score_good + em_score_bad) / 2)
        em_scores_good.append(em_score_good)
        em_scores_bad.append(em_score_bad)

    output_all = [{
        "question": example["question"],
        "solution": example["solution"],
        "em_score": em_score,
    } for example, em_score, em_score_bad, em_score_good in zip(instruction_data, em_scores, em_scores_bad, em_scores_good)]

    print(f"Exact match : {em_scores, em_scores_bad, em_scores_good}")

[Rest of the argument parser code remains unchanged...]
