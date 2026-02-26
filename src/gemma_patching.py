"""
Helpers for patching hidden states in Gemma (and similar) causal LM models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PatchConfig:
    target_layer: int
    max_new_tokens: int = 10
    do_sample: bool = False
    num_beams: int = 1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0


def load_model_and_tokenizer(model_name: str, device: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if resolved_device == "cuda" else None,
    )
    model.eval()
    return model, tokenizer


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not find transformer layers on the model.")


def find_token_indices_for_word(tokenizer, text: str, word: str):
    """
    Return (start_idx, end_idx) token indices that correspond to `word`
    in the tokenized representation. Both indices are inclusive.

    Notes:
      - Uses detokenization to map tokens back to character spans.
      - Works with BPE/wordpiece tokenizers (handles leading spaces).
      - If `word` is not found, returns (None, None, enc).
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]

    offsets = []
    for i in range(len(ids)):
        piece = tokenizer.decode(ids[: i + 1])
        offsets.append(len(piece))

    full = tokenizer.decode(ids)
    start_char = full.find(word)
    if start_char < 0:
        return None, None, enc
    end_char = start_char + len(word)

    start_idx, end_idx = None, None
    for i, off in enumerate(offsets):
        prev_off = offsets[i - 1] if i > 0 else 0
        if start_idx is None and prev_off <= start_char < off:
            start_idx = i
        if prev_off < end_char <= off:
            end_idx = i
            break

    return start_idx, end_idx + 1, enc


def replace_pre_hook(target_start_pos: int, target_end_pos: int, source_reprs: torch.Tensor):
    def hook(_module, inputs):
        hidden_states = inputs[0]
        if hidden_states.shape[1] > target_start_pos:
            hidden_states = hidden_states.clone()
            hidden_states[0, target_start_pos:target_end_pos, :] = source_reprs.to(
                hidden_states.device, dtype=hidden_states.dtype
            )
            return (hidden_states,)
        return None

    return hook


def prepare_source_hidden_states(
    model,
    tokenizer,
    source_prompt: str,
    source_substring: str,
    layers_to_cache: Optional[Iterable[int]] = None,
) -> Dict[int, torch.Tensor]:
    start_pos, end_pos, source_inputs = find_token_indices_for_word(
        tokenizer, source_prompt, source_substring
    )
    if start_pos is None or end_pos is None:
        raise ValueError(
            "Could not locate token position for '{}' in '{}'".format(
                source_substring, source_prompt
            )
        )

    device = next(model.parameters()).device
    source_inputs = {k: v.to(device) for k, v in source_inputs.items()}

    with torch.no_grad():
        outputs = model(**source_inputs, output_hidden_states=True)

    layers = (
        list(range(model.config.num_hidden_layers + 1))
        if layers_to_cache is None
        else list(layers_to_cache)
    )
    hs_cache: Dict[int, torch.Tensor] = {}
    for layer in layers:
        hs_cache[layer] = outputs["hidden_states"][layer][0][start_pos:end_pos]

    return hs_cache


def prepare_soft_prompt_source_inputs(model, tokenizer, continuous_prompt: torch.Tensor):
    """
    Prepares and caches hidden states for a continuous prompt.

    Args:
        continuous_prompt (torch.Tensor):
            The trained continuous prompt used as the source prompt.

    Returns:
        dict[int, torch.Tensor]:
            A dictionary mapping each layer index to its hidden state slice
            corresponding to the soft prompt tokens at that layer.
    """
    prompt_len = continuous_prompt.shape[0]
    source_prompt = " ?" * prompt_len
    source_substring = source_prompt

    start_pos, end_pos, source_inputs = find_token_indices_for_word(
        tokenizer, source_prompt, source_substring
    )
    if start_pos is None or end_pos is None:
        raise ValueError(
            "Could not locate token position for '{}' in '{}'".format(
                source_substring, source_prompt
            )
        )

    device = next(model.parameters()).device
    source_inputs = {k: v.to(device) for k, v in source_inputs.items()}

    layers = get_transformer_layers(model)
    hook = layers[0].register_forward_pre_hook(
        replace_pre_hook(start_pos, end_pos, continuous_prompt)
    )

    try:
        with torch.no_grad():
            outputs = model(**source_inputs, output_hidden_states=True)
    finally:
        hook.remove()

    hs_cache: Dict[int, torch.Tensor] = {}
    layers_to_cache = list(range(model.config.num_hidden_layers + 1))
    for layer in layers_to_cache:
        hs_cache[layer] = outputs["hidden_states"][layer][0][start_pos:end_pos]

    return hs_cache


def patchscopes(
    model,
    tokenizer,
    source_reprs: torch.Tensor,
    target_prompt: str,
    target_placeholder: str,
    patch_config: PatchConfig,
    end_phrase: str,
) -> str:
    layers = get_transformer_layers(model)
    if patch_config.target_layer < 0 or patch_config.target_layer >= len(layers):
        raise ValueError(
            "target_layer {} out of range (0..{}).".format(
                patch_config.target_layer, len(layers) - 1
            )
        )

    target_start_pos, target_end_pos, target_inputs = find_token_indices_for_word(
        tokenizer, target_prompt, target_placeholder
    )
    if target_start_pos is None or target_end_pos is None:
        raise ValueError(
            "Could not locate token position for '{}' in '{}'".format(
                target_placeholder, target_prompt
            )
        )

    device = next(model.parameters()).device
    target_inputs = {k: v.to(device) for k, v in target_inputs.items()}

    hook = layers[patch_config.target_layer].register_forward_pre_hook(
        replace_pre_hook(target_start_pos, target_end_pos, source_reprs)
    )

    try:
        with torch.no_grad():
            out_ids = model.generate(
                **target_inputs,
                max_new_tokens=patch_config.max_new_tokens,
                do_sample=patch_config.do_sample,
                num_beams=patch_config.num_beams,
                temperature=patch_config.temperature,
                top_k=patch_config.top_k,
                top_p=patch_config.top_p,
                use_cache=True,
            )
    finally:
        hook.remove()

    prompt_len = target_inputs["input_ids"].shape[1]
    gen_ids = out_ids[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.split(end_phrase)[0]
