#!/usr/bin/env python
# coding=utf-8
import logging
import os
import random
import sys
from typing import Dict, Any

import numpy as np
import torch
import datasets
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    set_seed
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from less.data_selection.get_training_dataset import get_training_dataset_llama3
from less.train.data_arguments import DataArguments, get_data_statistics
from less.train.model_arguments import ModelArguments
from less.train.training_arguments import TrainingArguments

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_logging(training_args: TrainingArguments) -> None:
    """Configure logging system"""
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

def prepare_tokenizer(model_path: str) -> AutoTokenizer:
    """Prepare and configure tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

def filter_dataset_by_em_score(dataset: datasets.Dataset, filter_ratio: float) -> datasets.Dataset:
    """Filter dataset based on em_score"""
    if not filter_ratio:
        return dataset
        
    em_scores = np.array(dataset['em_score'])
    top_count = max(1, int(len(em_scores) * filter_ratio))
    sorted_indices = np.argsort(em_scores)[::-1]
    top_indices = sorted_indices[:top_count]
    return dataset.select(top_indices)

def setup_lora_model(model: AutoModelForCausalLM, config: Dict[str, Any]) -> PeftModel:
    """Configure LoRA model"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['lora_target_modules'],
    )
    
    model = get_peft_model(model, lora_config)
    logger.info("Applied LoRA to model.")
    model.print_trainable_parameters()
    
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    return model

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    setup_logging(training_args)
    logger.info(f"Device: {training_args.device}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")
    
    set_seed(training_args.seed)
    
    tokenizer = prepare_tokenizer(model_args.model_name_or_path)
    
    train_dataset = get_training_dataset_llama3(
        data_args.train_files,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        sample_percentage=data_args.percentage,
        seed=data_args.sample_data_seed
    )
    
    train_dataset = filter_dataset_by_em_score(train_dataset, data_args.filter_em_score)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype="auto"
    ).half().cuda()
    
    if not isinstance(model, PeftModel) and model_args.lora:
        model = setup_lora_model(model, vars(model_args))
    
    get_data_statistics(train_dataset)
    
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
            tokenizer=tokenizer, 
            model=model, 
            padding=True
        )
    )
    
    train_result = trainer.train()
    trainer.save_model()
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()
