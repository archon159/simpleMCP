from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class HFModel:
    tokenizer: Any
    model: Any


def load_hf_model(model_id: str, dtype: str = "auto", device: Any = None) -> HFModel:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    torch_dtype = None
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16

    if not device:
        device="auto",
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch_dtype,
    )
    return HFModel(tokenizer=tokenizer, model=model)


def generate_from_messages(
    hf,
    messages,
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int = 0,
    logger: Any = None,
) -> str:
    if seed:
        torch.manual_seed(seed)

    prompt = hf.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    logger.info("Actual Prompt")
    logger.info(prompt)

    inputs = hf.tokenizer(prompt, return_tensors="pt").to(hf.model.device)
    input_len = inputs["input_ids"].shape[1]  

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=hf.tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        out = hf.model.generate(**inputs, **gen_kwargs)

    gen_ids = out[0][input_len:]
    text = hf.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text