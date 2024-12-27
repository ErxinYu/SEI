#!/usr/bin/env python
# coding=utf-8
import logging
import os
import random
import sys
import time
import numpy as np
import datasets
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser, Trainer,
                          set_seed
                          )
import transformers
from less.data_selection.get_training_dataset import get_training_dataset_llama3,get_training_dataset,get_training_dataset_llama3_filter
from less.train.data_arguments import DataArguments, get_data_statistics
from less.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from less.train.training_arguments import TrainingArguments
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Device: {training_args.device}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Load training dataset
    
    train_dataset = get_training_dataset_llama3(data_args.train_files,
                                         tokenizer=tokenizer,
                                         max_seq_length=data_args.max_seq_length,
                                         sample_percentage=data_args.percentage,
                                         seed=data_args.sample_data_seed)
    # if data_args.filter_em_score:
    #     def filter_em_score(example):
    #         return example['em_score'].item() > data_args.filter_em_score
    #     train_dataset = train_dataset.filter(filter_em_score)

    if data_args.filter_em_score:
        # 提取所有 em_score 的值
        em_scores = np.array(train_dataset['em_score'])  # 获取 em_score 列作为 NumPy 数组

        # 计算需要保留的样本数量（前 5%）
        top_5_percent_count = max(1, int(len(em_scores) * data_args.filter_em_score))  # 至少保留一个样本

        # 对 em_scores 排序，获取前 5% 的样本的 indices
        sorted_indices = np.argsort(em_scores)[::-1]  # 按降序排序
        top_indices = sorted_indices[:top_5_percent_count]  # 获取前 5% 的样本 indices

        # 根据前 5% 的 indices 提取对应的样本
        train_dataset = train_dataset.select(top_indices)




    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype="auto").half().cuda()

    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            f"Applied LoRA to model."
        )
        model.print_trainable_parameters()
        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)



    get_data_statistics(train_dataset)
    # train_dataset = train_dataset.remove_columns(
    #     ["cluster"])
            
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding=True)
    )

    # Training
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()