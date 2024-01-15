import json
import os
import fire
from typing import Dict, List, Optional, Sequence, Union
from rich.console import Console
from src.rich_utils import rich_theme
from rich.markdown import Markdown

import numpy as np
import torch

from datasets import concatenate_datasets, load_dataset
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
)

from src.trainer import SPIN_Trainer


def main(
    batch_size: int = 2,
    model_name: str = "Upstage/SOLAR-10.7B-v1.0",
    ds_names: List[str] = ["Open-Orca/OpenOrca"],
    accelerate_mix_precision: str = "bf16",
    deepspeed_stage: int = 2,
    deepspeed_gradient_clipping: float = 1.0,
    deepspeed_grad_accum_steps: int = 1,
    offload_optimizer_device: str = "cpu",
    offload_param_device: str = "cpu",
    peft: bool = True,
    lora_alpha: int = 256,
    lora_dropout: float = 0.05,
    lora_r: int = 128,
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
    reg_term: float = 0.5,
):
    # for progress print
    console = Console(theme=rich_theme)

    # Setting up Accelerator
    console.print("Setting up Accelerator...", style="info")
    accelerator = Accelerator(
        mixed_precision=accelerate_mix_precision,
        deepspeed_plugin=DeepSpeedPlugin(
            zero_stage=deepspeed_stage,
            gradient_clipping=deepspeed_gradient_clipping,
            gradient_accumulation_steps=deepspeed_grad_accum_steps,
            offload_optimizer_device=offload_optimizer_device,
            offload_param_device=offload_param_device,
        ),
    )

    # Load the model and tokenizer
    console.print(
        Markdown(f"Loading model and tokenizer from `{model_name}` ..."), style="info"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Training with LORA
    if peft:
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )
        console.print(f"Configuring LORA...", style="info")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Load Dataset
    console.print(Markdown(f"Loading dataset from `{ds_names}` ..."), style="info")
    datasets = [load_dataset(ds_name) for ds_name in ds_names]
    if len(datasets) > 1:
        datasets = concatenate_datasets(datasets)
    else:
        datasets = datasets[0]

    # Train
    console.print("Start training...", style="info")
    trainer = SPIN_Trainer(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        batch_size=batch_size,
        accelerator=accelerator,
        reg_term=reg_term,
    )

    trainer.train_one_iter(self_play_iter=0, epochs=4)


if __name__ == "__main__":
    fire.Fire(main)
