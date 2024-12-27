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
# 添加路径到 sys.path
sys.path.append('/home/yex/LESS/evaluation')
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
    # some numbers are in the `x,xxx` format, and we want to remove the comma
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
        def apply_chat_format(instruction,example, tokenizer):
            instruct_question = instruction["question"]
            instruct_solution = instruction["solution"]
            demonstration = f"### Instruction:\n{instruct_question}\n\n### Response:\n{instruct_solution}"
            question = example["question"]
            prompt =  "Below is an instruction that describes a task. "+ \
            "Write a response that appropriately completes the request.\n\n"+ \
            f"Here is an example:\n{demonstration}\n\n" + \
            f"### Instruction:\n{question}\n\n### Response:"
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)#add_generation_prompt for adding “<|start_header_id|>assistant<|end_header_id|>” 
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
        # replace numbers like `x,xxx` with `xxxx`
        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            predictions.append(numbers[-1])
        else:
            predictions.append(output)
    targets = [example["answer"] for example in test_data]
    
    # 将计算结果添加到em_scores列表中
    em_scores = []
    em_scores_good = []
    em_scores_bad = []
    for i in range(0, len(instruction_data)):
        batch_predictions = predictions[i * 20 :(i + 1) * 20]
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

        # 将计算结果添加到em_scores列表中
        em_scores.append((em_score_good+em_score_bad)/2)
        em_scores_good.append(em_score_good)
        em_scores_bad.append(em_score_bad)
    output_all = [{
        "question": example["question"],
        "solution": example["solution"],
        "em_score": em_score,
    } for example, em_score,em_score_bad,em_score_good in zip(instruction_data, em_scores,em_scores_bad,em_scores_good)]

    print(f"Exact match : {em_scores,em_scores_bad,em_scores_good}")


    # with open(os.path.join(args.save_dir, f"8insturction_score_{args.data_start}_{args.data_end}_temp.json"), "w") as fout:
    #     for prediction in output_all:
    #         fout.write(json.dumps(prediction) + "\n") 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_dataset", 
        type=str, 
        default="train"
    )
    parser.add_argument(
        "--test_data", 
        type=str, 
        default="data/gsm"
    )
    parser.add_argument(
        "--instruct_data", 
        type=str, 
        default="data/gsm"
    )
    parser.add_argument(
        "--max_num_examples", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default=None,
        help="if specified, we will load the model from a revision of the model in the hub"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine", 
        type=str, 
        default=None, help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--data_start", 
        type=int, 
        default=8, 
        help="max number of examples to use for demonstration."
    )
    parser.add_argument(
        "--data_end", 
        type=int, 
        default=8, 
        help="max number of examples to use for demonstration."
    )
    parser.add_argument(
        "--no_cot", 
        action="store_true", 
        help="If given, we're evaluating a model without chain-of-thought."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq", 
        action="store_true", 
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true", 
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--stop_at_double_newline",
        action="store_true",
        help="If given, will stop generation at double newline instead of single."
    )
    parser.add_argument(
        '--additional_stop_sequence',
        type=str,
        nargs="+",
        default=[],
        help="Additional stop sequences to use when generating completions. Useful for e.g. llama-3-instruct."
    )
    parser.add_argument(
        "--upload_to_hf",
        type=str,
        default=None,
        help="If specified, we will upload the results to Hugging Face Datasets. "
             "This should be the name of the dataset to upload to."
    )
    parser.add_argument(
        "--hf_upload_name",
        type=str,
        default=None,
        help="If uploading to hf, this is the model name"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)