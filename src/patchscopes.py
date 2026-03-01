"""
    THE FUNCTIONS IN THIS FILE ARE TAKEN FROM COURSE MATERIAL BY DANIELLA GOTTESMAN, AND MOR GEVA
    ALL RIGHTS ARE RESERVED TO THE ORIGINAL AUTHORS, AND THIS USAGE IS UNDER THEIR EXPRESS PERMISSION.
    LINK TO ORIGINAL MATERIAL: https://colab.research.google.com/drive/1Be42DpybrBxO3I-vZVyVuwo_acgIJucQ?usp=sharing

Patchscopes: A framework for inspecting hidden representations in language models.

This module implements Patchscopes, an interpretability tool that uses a model's
own generative abilities to translate hidden representations into natural language.
It works by patching representations from a source layer/prompt into a target
layer/prompt to decode encoded information.

References:
    [1] Patchscopes: A Unifying Framework for Inspecting Hidden Representations
        of Language Models (https://openreview.net/pdf?id=5uwBzcn885)
    
    [2] Eliciting Textual Descriptions from Representations of Continuous Prompts
        (https://aclanthology.org/2025.findings-acl.849.pdf)
"""

from typing import Optional, Tuple, Dict, List
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


def find_token_indices_for_word(
    tokenizer: PreTrainedTokenizer,
    text: str,
    word: str
) -> Tuple[Optional[int], Optional[int], Dict]:
    """
    THE FUNCTIONS IN THIS FILE ARE TAKEN FROM COURSE MATERIAL BY DANIELLA GOTTESMAN, AND MOR GEVA
    ALL RIGHTS ARE RESERVED TO THE ORIGINAL AUTHORS, AND THIS USAGE IS UNDER THEIR EXPRESS PERMISSION.
    LINK TO ORIGINAL MATERIAL: https://colab.research.google.com/drive/1Be42DpybrBxO3I-vZVyVuwo_acgIJucQ?usp=sharing
    Find token indices corresponding to a word in tokenized text.
    
    Uses detokenization to map tokens back to character spans. Works with
    BPE/wordpiece tokenizers (handles leading spaces).
    
    Args:
        tokenizer: HuggingFace tokenizer
        text: Full text string
        word: Substring to locate
    
    Returns:
        Tuple of (start_idx, end_idx, encoded_inputs) where indices are inclusive.
        Returns (None, None, encoded_inputs) if word is not found.
    
    Example:
        >>> start, end, enc = find_token_indices_for_word(tok, "Hello world", "world")
        >>> # start=1, end=2 (exclusive) for typical tokenizer
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


def prepare_source_inputs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    source_prompt: str,
    source_substring: str,
    device: torch.device = None
) -> Dict[int, torch.Tensor]:
    """
    THE FUNCTIONS IN THIS FILE ARE TAKEN FROM COURSE MATERIAL BY DANIELLA GOTTESMAN, AND MOR GEVA
    ALL RIGHTS ARE RESERVED TO THE ORIGINAL AUTHORS, AND THIS USAGE IS UNDER THEIR EXPRESS PERMISSION.
    LINK TO ORIGINAL MATERIAL: https://colab.research.google.com/drive/1Be42DpybrBxO3I-vZVyVuwo_acgIJucQ?usp=sharing
    Extract and cache hidden states from source prompt.
    
    Runs the model on the source prompt and caches hidden states from all layers
    for the tokens corresponding to the source substring.
    
    Args:
        model: HuggingFace causal language model
        tokenizer: Corresponding tokenizer
        source_prompt: Full prompt text
        source_substring: Substring to extract representations for
        device: Device to run on (defaults to model's device)
    
    Returns:
        Dictionary mapping layer index -> hidden states tensor of shape
        (n_tokens, hidden_dim) where n_tokens is the number of tokens in
        source_substring.
    
    Raises:
        AssertionError: If source_substring is not found in source_prompt
    """
    if device is None:
        device = next(model.parameters()).device
    
    source_start_pos, source_end_pos, source_inputs = find_token_indices_for_word(
        tokenizer, source_prompt, source_substring
    )
    assert source_start_pos is not None and source_end_pos is not None, \
        f"Could not locate token position for '{source_substring}' in '{source_prompt}'"

    source_inputs = {k: v.to(device) for k, v in source_inputs.items()}

    hs_cache = {}
    with torch.no_grad():
        outputs = model(**source_inputs, output_hidden_states=True)

    layers_to_cache = list(range(model.config.num_hidden_layers + 1))
    for layer in layers_to_cache:
        hs_cache[layer] = outputs["hidden_states"][layer][0][source_start_pos:source_end_pos]

    return hs_cache


def prepare_soft_prompt_source_inputs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    continuous_prompt: torch.Tensor,
    device: torch.device = None
) -> Dict[int, torch.Tensor]:
    """Extract hidden states from a continuous (soft) prompt.
    THE FUNCTIONS IN THIS FILE ARE TAKEN FROM COURSE MATERIAL BY DANIELLA GOTTESMAN, AND MOR GEVA
    ALL RIGHTS ARE RESERVED TO THE ORIGINAL AUTHORS, AND THIS USAGE IS UNDER THEIR EXPRESS PERMISSION.
    LINK TO ORIGINAL MATERIAL: https://colab.research.google.com/drive/1Be42DpybrBxO3I-vZVyVuwo_acgIJucQ?usp=sharing
    
    Continuous prompts are embedding vectors optimized via prompt tuning.
    This function injects the continuous prompt into the model's first layer
    and extracts the resulting hidden states across all layers.
    
    Args:
        model: HuggingFace causal language model
        tokenizer: Corresponding tokenizer
        continuous_prompt: Tensor of shape (n_tokens, hidden_dim) containing
            the optimized continuous prompt embeddings
        device: Device to run on (defaults to model's device)
    
    Returns:
        Dictionary mapping layer index -> hidden states tensor of shape
        (n_tokens, hidden_dim)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Create placeholder tokens for the continuous prompt
    source_prompt = " ?" * continuous_prompt.shape[0]
    source_substring = source_prompt

    source_start_pos, source_end_pos, source_inputs = find_token_indices_for_word(
        tokenizer, source_prompt, source_substring
    )
    assert source_start_pos is not None and source_end_pos is not None, \
        f"Could not locate token position for '{source_substring}' in '{source_prompt}'"

    source_inputs = {k: v.to(device) for k, v in source_inputs.items()}

    # Create hook to replace placeholder embeddings with continuous prompt
    def replace_pre_hook(target_start_pos, target_end_pos, source_reprs):
        def hook(_module, inputs):
            hs = inputs[0]
            if hs.shape[1] > target_start_pos:
                hs = hs.clone()
                hs[0, target_start_pos:target_end_pos, :] = source_reprs.to(hs.device, dtype=hs.dtype)
                return (hs,)
            return None
        return hook

    hook = model.model.layers[0].register_forward_pre_hook(
        replace_pre_hook(source_start_pos, source_end_pos, continuous_prompt)
    )

    with torch.no_grad():
        outputs = model(**source_inputs, output_hidden_states=True)

    hook.remove()

    hs_cache = {}
    layers_to_cache = list(range(model.config.num_hidden_layers + 1))
    for layer in layers_to_cache:
        hs_cache[layer] = outputs["hidden_states"][layer][0][source_start_pos:source_end_pos]

    return hs_cache


def patchscopes(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    source_reprs: torch.Tensor,
    target_prompt: str,
    target_placeholder: str,
    source_layer: int,
    target_layer: int,
    end_phrase: str,
    max_new_tokens: int = 10,
    do_sample: bool = False,
    num_beams: int = 1,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    device: torch.device = None,
    bad_words_ids: list = None,
) -> str:
    """
    THE FUNCTIONS IN THIS FILE ARE TAKEN FROM COURSE MATERIAL BY DANIELLA GOTTESMAN, AND MOR GEVA
    ALL RIGHTS ARE RESERVED TO THE ORIGINAL AUTHORS, AND THIS USAGE IS UNDER THEIR EXPRESS PERMISSION.
    LINK TO ORIGINAL MATERIAL: https://colab.research.google.com/drive/1Be42DpybrBxO3I-vZVyVuwo_acgIJucQ?usp=sharing
    Apply Patchscopes to decode hidden representations into text.
    Patches source representations into target prompt at specified layers,
    then generates text to decode the information in the representations.
    
    Args:
        model: HuggingFace causal language model
        tokenizer: Corresponding tokenizer
        source_reprs: Hidden state tensor of shape (n_tokens, hidden_dim) from
            source layer, typically obtained from prepare_source_inputs()
        target_prompt: Template prompt containing placeholder
        target_placeholder: Substring in target_prompt to replace with source_reprs
        source_layer: Layer index where source_reprs were extracted (for reference)
        target_layer: Layer index where patching occurs
        end_phrase: Stop generation when this phrase is encountered
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to use sampling (vs greedy)
        num_beams: Number of beams for beam search
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        device: Device to run on (defaults to model's device)
        bad_words_ids: List of token ids that should not be generated (e.g., [[token1], [token2]])
    
    Returns:
        Generated text up to (but not including) end_phrase
    
    Raises:
        RuntimeError: If target_placeholder is not found in target_prompt
    
    Example:
        >>> source_hs = prepare_source_inputs(model, tok, "Paris, France", "France")
        >>> result = patchscopes(
        ...     model, tok,
        ...     source_reprs=source_hs[10],
        ...     target_prompt="Country: ?",
        ...     target_placeholder="?",
        ...     source_layer=10,
        ...     target_layer=5,
        ...     end_phrase="\\n"
        ... )
        >>> print(result)  # e.g., "France"
    """
    if device is None:
        device = next(model.parameters()).device
    
    target_block = model.model.layers[target_layer]

    target_start_pos, target_end_pos, target_inputs = find_token_indices_for_word(
        tokenizer, target_prompt, target_placeholder
    )
    if target_start_pos is None or target_end_pos is None:
        raise RuntimeError(
            f"Could not locate token position for '{target_placeholder}' in '{target_prompt}'"
        )
    target_inputs = {k: v.to(device) for k, v in target_inputs.items()}

    # Create hook to replace target representations with source representations
    def replace_pre_hook(target_start_pos, target_end_pos, source_reprs):
        def hook(_module, inputs):
            hs = inputs[0]
            # Only patch during full prompt pass, not incremental decoding
            if hs.shape[1] > target_start_pos:
                hs = hs.clone()
                hs[0, target_start_pos:target_end_pos, :] = source_reprs.to(hs.device, dtype=hs.dtype)
                return (hs,)
            return None
        return hook

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
            bad_words_ids=bad_words_ids
        )

    hook.remove()

    prompt_len = target_inputs["input_ids"].shape[1]
    gen_ids = out_ids[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.split(end_phrase)[0]


def logit_lens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    hidden_states: List[torch.Tensor],
    tok_pos: int,
    layer: int,
    k: int = 20,
) -> Dict[str, any]:
    """
    THE FUNCTIONS IN THIS FILE ARE TAKEN FROM COURSE MATERIAL BY DANIELLA GOTTESMAN, AND MOR GEVA
    ALL RIGHTS ARE RESERVED TO THE ORIGINAL AUTHORS, AND THIS USAGE IS UNDER THEIR EXPRESS PERMISSION.
    LINK TO ORIGINAL MATERIAL: https://colab.research.google.com/drive/1Be42DpybrBxO3I-vZVyVuwo_acgIJucQ?usp=sharing
    Apply logit lens to decode hidden states via vocabulary projection.
    
    Projects a hidden state from an intermediate layer through the model's
    unembedding matrix to see what tokens are most strongly represented.
    This is a specific case of Patchscopes where source and target are the same.
    
    Args:
        model: HuggingFace causal language model with lm_head
        tokenizer: Corresponding tokenizer
        hidden_states: List of hidden state tensors from model output, one per layer.
            Shape: [(batch, seq, hidden), ...] for each layer (including embeddings)
        tok_pos: Sequence position to inspect
        layer: Which hidden layer to use (0 to num_layers)
        k: Number of top candidates to return
    
    Returns:
        Dictionary with keys:
            - topk_tokens: List of top-k token strings
            - topk_ids: List of top-k token IDs
            - topk_probs: List of top-k probabilities
    
    Example:
        >>> inputs = tokenizer("The capital of France is", return_tensors="pt")
        >>> with torch.no_grad():
        ...     output = model(**inputs, output_hidden_states=True)
        >>> result = logit_lens(model, tokenizer, output['hidden_states'], -1, 10)
        >>> print(result['topk_tokens'][:3])  # ['Paris', 'France', 'the']
    """
    # Extract hidden state for requested layer and position
    h = hidden_states[layer][0, tok_pos, :]  # shape: (hidden_size,)

    # Apply final layer norm if not at the final layer
    is_final_layer = (layer == len(hidden_states) - 1)
    if not is_final_layer and hasattr(model, "model") and hasattr(model.model, "norm"):
        h = model.model.norm(h)

    # Project to logits and compute probabilities
    logits = model.lm_head(h)  # shape: (vocab_size,)
    probs = F.softmax(logits, dim=-1)  # shape: (vocab_size,)

    # Get top-k
    topk_probs, topk_ids = torch.topk(probs, k=min(k, probs.numel()), dim=-1)

    # Convert to lists and tokens
    topk_ids_list = topk_ids.tolist()
    topk_tokens = tokenizer.convert_ids_to_tokens(topk_ids_list)

    return {
        "topk_tokens": topk_tokens,
        "topk_ids": topk_ids_list,
        "topk_probs": topk_probs.tolist(),
    }


class PatchscopesWrapper:
    """Convenience wrapper for Patchscopes with a fixed model and tokenizer.
    
    Simplifies repeated calls by caching model and tokenizer references.
    
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> ps = PatchscopesWrapper(model, tokenizer)
        >>> 
        >>> # Extract source representations
        >>> source_hs = ps.prepare_source_inputs("Paris, France", "France")
        >>> 
        >>> # Apply patchscopes
        >>> result = ps.patchscopes(
        ...     source_reprs=source_hs[10],
        ...     target_prompt="Country: ?",
        ...     target_placeholder="?",
        ...     source_layer=10,
        ...     target_layer=5,
        ...     end_phrase="\\n"
        ... )
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None
    ):
        """Initialize wrapper with model and tokenizer.
        
        Args:
            model: HuggingFace causal language model
            tokenizer: Corresponding tokenizer
            device: Device to run on (defaults to model's device)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
    
    def prepare_source_inputs(
        self,
        source_prompt: str,
        source_substring: str
    ) -> Dict[int, torch.Tensor]:
        """Extract and cache hidden states from source prompt."""
        return prepare_source_inputs(
            self.model, self.tokenizer, source_prompt, source_substring, self.device
        )
    
    def prepare_soft_prompt_source_inputs(
        self,
        continuous_prompt: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Extract hidden states from a continuous (soft) prompt."""
        return prepare_soft_prompt_source_inputs(
            self.model, self.tokenizer, continuous_prompt, self.device
        )
    
    def patchscopes(
        self,
        source_reprs: torch.Tensor,
        target_prompt: str,
        target_placeholder: str,
        source_layer: int,
        target_layer: int,
        end_phrase: str,
        **kwargs
    ) -> str:
        """Apply Patchscopes to decode hidden representations."""
        return patchscopes(
            self.model,
            self.tokenizer,
            source_reprs,
            target_prompt,
            target_placeholder,
            source_layer,
            target_layer,
            end_phrase,
            device=self.device,
            **kwargs
        )
    
    def logit_lens(
        self,
        hidden_states: List[torch.Tensor],
        tok_pos: int,
        layer: int,
        k: int = 20
    ) -> Dict[str, any]:
        """Apply logit lens to decode hidden states."""
        return logit_lens(
            self.model, self.tokenizer, hidden_states, tok_pos, layer, k
        )
