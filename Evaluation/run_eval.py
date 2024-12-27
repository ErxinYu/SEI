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
from evaluation.eval.utils import (
    generate_completions,
    query_openai_chat_model,
    dynamic_import_function,
)
from evaluation.eval.gsm.examplars import EXAMPLARS as GSM_EXAMPLARS
import sys
sys.path.append('[PROJECT_ROOT]/evaluation')
exact_match = evaluate.load("exact_match")


def main(args):
    random.seed(42)

    print("Loading data...", args.data_dir, args.lora_path)
    test_data = []
    with open(args.data_dir) as fin:
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

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)
    print("data number:", len(test_data))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    if args.use_chat_format:
        def apply_chat_format(example, tokenizer):
            question = example["question"]
            prompt = "Below is an instruction that describes a task. "+ \
            "Write a response that appropriately completes the request.\n\n"+ \
            f"### Instruction:\n{question}\n\n### Response:"
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return prompt

    if args.model_name_or_path:
        print("Loading model and tokenizer from ...", args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).half().cuda()
        if args.lora_path:
            model = PeftModel.from_pretrained(model, args.lora_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.bos_token
        tokenizer.padding_side = "left"
        prompts = [apply_chat_format(example, tokenizer) for example in test_data]          
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
    em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match : {em_score}")

    predictions = [{
        "question": example["question"],
        "solution": example["solution"],
        "answer": example["answer"],
        "model_output": output,
        "prediction": pred
    } for example, output, pred in zip(test_data, outputs, predictions)]

    with open(os.path.join(args.save_dir, f"0{args.eval_dataset}_predictions.json"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n") 

# ArgumentParser configuration
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [Original argument definitions remain unchanged]
    # ...
    args = parser.parse_args()

    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
