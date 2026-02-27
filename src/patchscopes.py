# -*- coding: utf-8 -*-
"""Patchscopes utilities extracted for reuse."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str, *, token: str | None = None, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(model_name, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, token=token)
    model.eval()
    return model, tok


def find_token_indices_for_word(tokenizer, text: str, word: str) -> Tuple[int | None, int | None, dict]:
    """
    Return (start_idx, end_idx) token indices that correspond to `word`
    in the tokenized representation. Both indices are inclusive.

    Notes:
      - Uses detokenization to map tokens back to character spans.
      - Works with BPE/wordpiece tokenizers (handles leading spaces).
      - If `word` is not found, returns (None, None).
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]

    # Incrementally decode to track char offsets
    offsets = []
    for i in range(len(ids)):
        piece = tokenizer.decode(ids[: i + 1])
        offsets.append(len(piece))

    # Locate word in the fully decoded string
    full = tokenizer.decode(ids)
    start_char = full.find(word)
    if start_char < 0:
        return None, None, enc
    end_char = start_char + len(word)

    start_idx, end_idx = None, None
    for i, off in enumerate(offsets):
        prev_off = offsets[i - 1] if i > 0 else 0
        # token i covers [prev_off, off)
        if start_idx is None and prev_off <= start_char < off:
            start_idx = i
        if prev_off < end_char <= off:
            end_idx = i
            break

    return start_idx, end_idx + 1, enc


def prepare_source_inputs(model, tokenizer, source_prompt: str, source_substring: str, device: torch.device):
    source_start_pos, source_end_pos, source_inputs = find_token_indices_for_word(
        tokenizer, source_prompt, source_substring
    )
    assert source_start_pos is not None and source_end_pos is not None, (
        f"Could not locate token position for '{source_substring}' in '{source_prompt}'"
    )

    source_inputs = {k: v.to(device) for k, v in source_inputs.items()}

    hs_cache: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        outputs = model(**source_inputs, output_hidden_states=True)

    layers_to_cache = list(range(model.config.num_hidden_layers + 1))
    for layer in layers_to_cache:
        hs_cache[layer] = outputs["hidden_states"][layer][0][source_start_pos:source_end_pos]

    return hs_cache


def replace_pre_hook(target_start_pos: int, target_end_pos: int, source_reprs: torch.Tensor):
    def hook(_module, inputs):
        hs = inputs[0]
        # If this is the full prompt pass (hs.shape[1] > target_start_pos), do the replacement.
        # On incremental decoding steps, hs.shape[1] == 1, so this guard skips them.
        if hs.shape[1] > target_start_pos:
            hs = hs.clone()  # avoid in-place on shared tensor
            hs[0, target_start_pos:target_end_pos, :] = source_reprs.to(hs.device, dtype=hs.dtype)
            return (hs,)
        return None

    return hook


def patchscopes(
    model,
    tokenizer,
    source_reprs: torch.Tensor,
    target_prompt: str,
    target_placeholder: str,
    source_layer: int,
    target_layer: int,
    end_phrase: str,
    *,
    device: torch.device,
    max_new_tokens: int = 10,
    do_sample: bool = False,
    num_beams: int = 1,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
):
    target_block = model.model.layers[target_layer]

    target_start_pos, target_end_pos, target_inputs = find_token_indices_for_word(
        tokenizer, target_prompt, target_placeholder
    )
    if target_start_pos is None or target_end_pos is None:
        raise RuntimeError(
            f"Could not locate token position for '{target_placeholder}' in '{target_prompt}'"
        )
    target_inputs = {k: v.to(device) for k, v in target_inputs.items()}

    hook = target_block.register_forward_pre_hook(
        replace_pre_hook(target_start_pos, target_end_pos, source_reprs)
    )

    with torch.no_grad():
        out_ids = model.generate(
            **target_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=True,
        )

    hook.remove()

    prompt_len = target_inputs["input_ids"].shape[1]
    gen_ids = out_ids[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.split(end_phrase)[0]


def prepare_soft_prompt_source_inputs(
    model,
    tokenizer,
    continuous_prompt: torch.Tensor,
    *,
    device: torch.device,
):
    """
    Prepares and caches hidden states for a soft prompt.

    Args:
        continuous_prompt (torch.Tensor):
            The trained continuous prompt used as the source prompt.

    Returns:
        dict[int, torch.Tensor]:
            A dictionary mapping each layer index to its hidden state slice
            corresponding to the soft prompt tokens at that layer.
    """
    # Normalize shape to [1, seq_len, hidden_size].
    if continuous_prompt.dim() == 2:
        continuous_prompt = continuous_prompt.unsqueeze(0)
    elif continuous_prompt.dim() != 3:
        raise ValueError(
            f"continuous_prompt must be 2D or 3D, got shape {tuple(continuous_prompt.shape)}"
        )

    # If the prompt was trained for a different model, its hidden size may not match.
    # In that case, slice or pad to the current model hidden size to avoid shape errors.
    model_hidden_size = model.config.hidden_size
    prompt_hidden_size = continuous_prompt.shape[-1]
    if prompt_hidden_size != model_hidden_size:
        if prompt_hidden_size > model_hidden_size:
            continuous_prompt = continuous_prompt[..., :model_hidden_size]
        else:
            pad = model_hidden_size - prompt_hidden_size
            continuous_prompt = F.pad(continuous_prompt, (0, pad))

    continuous_prompt = continuous_prompt.to(device)

    source_prompt = " ?" * continuous_prompt.shape[1]
    source_substring = source_prompt

    source_start_pos, source_end_pos, source_inputs = find_token_indices_for_word(
        tokenizer, source_prompt, source_substring
    )
    assert source_start_pos is not None and source_end_pos is not None, (
        f"Could not locate token position for '{source_substring}' in '{source_prompt}'"
    )

    source_inputs = {k: v.to(device) for k, v in source_inputs.items()}

    hook = model.model.layers[0].register_forward_pre_hook(
        replace_pre_hook(source_start_pos, source_end_pos, continuous_prompt)
    )

    with torch.no_grad():
        outputs = model(**source_inputs, output_hidden_states=True)

    hook.remove()

    hs_cache: Dict[int, torch.Tensor] = {}
    layers_to_cache = list(range(model.config.num_hidden_layers + 1))
    for layer in layers_to_cache:
        hs_cache[layer] = outputs["hidden_states"][layer][0][source_start_pos:source_end_pos]

    return hs_cache
