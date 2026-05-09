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
    tools=None,
    max_new_tokens: int,
    temperature: float,
    seed: int = 0,
    logger: Any = None,
    writing_mode: bool = False,
    enable_thinking: bool = False,
) -> str:
    if seed:
        torch.manual_seed(seed)

    prompt = hf.tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
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

    # Decode the generated ids twice for different purposes:
    #   - skip_special_tokens=True  : returned for the agent. Control tokens
    #     such as <|python_tag|> and <|eom_id|> are removed so the JSON inside
    #     a tool call parses cleanly.
    #   - skip_special_tokens=False : logged only when writing_mode is on, so
    #     the raw token-level output (including the special tokens that drive
    #     tool calling) is visible for inspection. Agent behavior is identical
    #     whether writing_mode is on or off.
    text = hf.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if writing_mode:
        raw_with_tokens = hf.tokenizer.decode(gen_ids, skip_special_tokens=False)
        logger.info("Raw output with special tokens (writing mode):")
        logger.info(raw_with_tokens)

    return text