# Patchscopes Module

A reusable Python module for applying **Patchscopes**, an interpretability technique that decodes hidden representations in language models by patching them into carefully designed prompts.

## Overview

Patchscopes uses a model's own generative abilities to translate hidden representations into natural language. It works by:
1. Extracting hidden states from a **source prompt** at some **source layer**
2. Patching those representations into a **target prompt** at some **target layer**
3. Generating text to decode the information encoded in those representations

This module also includes **Logit Lens**, a specific case of Patchscopes that projects intermediate hidden states through the unembedding matrix to see what tokens they represent.

## Installation

The module requires:
```bash
pip install torch transformers
```

## Quick Start

```python
from patchscopes import PatchscopesWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create wrapper
ps = PatchscopesWrapper(model, tokenizer, device)

# Extract representations from source
source_hs = ps.prepare_source_inputs("Paris, France", "France")

# Apply patchscopes with a target prompt
result = ps.patchscopes(
    source_reprs=source_hs[10],  # Use layer 10
    target_prompt="Country: ?",
    target_placeholder="?",
    source_layer=10,
    target_layer=5,
    end_phrase="\n"
)

print(result)  # e.g., "France"
```

## Core Functions

### `prepare_source_inputs(model, tokenizer, source_prompt, source_substring, device=None)`

Extract and cache hidden states from all layers for a given substring in a prompt.

**Returns:** Dictionary mapping layer index → hidden states tensor

**Example:**
```python
source_hs = prepare_source_inputs(
    model, tokenizer, 
    "Diana, Princess of Wales", 
    "Wales"
)
# source_hs[10] contains hidden states from layer 10 for "Wales"
```

### `patchscopes(model, tokenizer, source_reprs, target_prompt, target_placeholder, source_layer, target_layer, end_phrase, ...)`

Apply Patchscopes to decode hidden representations into text.

**Key Parameters:**
- `source_reprs`: Hidden state tensor (from `prepare_source_inputs`)
- `target_prompt`: Template containing placeholder
- `target_placeholder`: Substring to replace with source_reprs
- `source_layer` / `target_layer`: Layer indices for patching
- `end_phrase`: Stop generation at this phrase
- `do_sample`: Enable sampling (vs greedy decoding)
- `temperature`, `top_k`, `top_p`: Sampling parameters

**Example:**
```python
target_prompt = (
    "Syria: Country in the Middle East\n|\n"
    "Leonardo DiCaprio: American actor\n|\n"
    "?"
)

result = patchscopes(
    model, tokenizer,
    source_reprs=source_hs[5],
    target_prompt=target_prompt,
    target_placeholder="?",
    source_layer=5,
    target_layer=3,
    end_phrase="\n|\n",
    max_new_tokens=20
)
```

### `logit_lens(model, tokenizer, hidden_states, tok_pos, layer, k=20)`

Apply Logit Lens to project hidden states onto vocabulary space.

**Returns:** Dictionary with `topk_tokens`, `topk_ids`, `topk_probs`

**Example:**
```python
inputs = tokenizer("The capital of France is", return_tensors="pt")
with torch.no_grad():
    output = model(**inputs, output_hidden_states=True)

result = logit_lens(model, tokenizer, output['hidden_states'], -1, 20)
print(result['topk_tokens'][:5])  # ['Paris', ' Paris', 'France', ...]
```

### `prepare_soft_prompt_source_inputs(model, tokenizer, continuous_prompt, device=None)`

Extract hidden states from continuous (soft) prompt embeddings.

**Use case:** Interpreting prompt-tuned embeddings

**Example:**
```python
# continuous_prompt is a tensor of shape (n_tokens, hidden_dim)
# from prompt tuning training
source_hs = prepare_soft_prompt_source_inputs(
    model, tokenizer, continuous_prompt
)

# Now use patchscopes to decode what the prompt encodes
target_prompt = (
    "Classify sentiment: positive or negative | "
    "Classify topics: sports, politics, weather | "
    + "?" * continuous_prompt.shape[0]
)
result = patchscopes(...)
```

## Using the Wrapper Class

The `PatchscopesWrapper` class simplifies repeated operations:

```python
ps = PatchscopesWrapper(model, tokenizer, device)

# All methods automatically use the stored model/tokenizer
source_hs = ps.prepare_source_inputs("Text here", "substring")
result = ps.patchscopes(source_hs[10], "Target: ?", "?", 10, 5, "\n")
```

## Advanced Examples

### Multi-Token Patching

```python
# Extract multiple tokens
source_hs = ps.prepare_source_inputs(
    "Diana, Princess of Wales",
    "Diana, Princess of Wales"  # Full phrase
)

# Use multiple placeholder tokens
target_prompt = "Entity description: ? ? ? ? ? ?"
target_placeholder = "? ? ? ? ? ?"

result = ps.patchscopes(
    source_hs[4], 
    target_prompt,
    target_placeholder,
    source_layer=4,
    target_layer=4,
    end_phrase="\n"
)
```

### Sampling for Diversity

```python
result = ps.patchscopes(
    source_reprs,
    target_prompt,
    target_placeholder,
    source_layer=5,
    target_layer=3,
    end_phrase="\n",
    do_sample=True,
    temperature=0.8,
    top_p=0.9
)
```

### Comparing Layers

```python
# Compare how information evolves across layers
for layer in [5, 10, 15, 20]:
    result = ps.patchscopes(
        source_hs[layer],
        target_prompt="Meaning: ?",
        target_placeholder="?",
        source_layer=layer,
        target_layer=5,
        end_phrase="\n"
    )
    print(f"Layer {layer}: {result}")
```

## Target Prompt Design

The effectiveness of Patchscopes depends heavily on target prompt design:

✅ **Good target prompts:**
- Provide clear context via few-shot examples
- Use consistent formatting
- Have clear stopping criteria (end_phrase)

❌ **Poor target prompts:**
- Ambiguous or open-ended
- No examples or context
- Unclear where to stop generation

**Example of a good target prompt:**
```python
target_prompt = (
    "Syria: Country in the Middle East\n|\n"
    "Leonardo DiCaprio: American actor\n|\n"
    "Samsung: Electronics company from South Korea\n|\n"
    "?"
)
```

## Use Cases

1. **Decoding entity information**: What does the model know about "Paris"?
2. **Tracking information flow**: How does information propagate across layers?
3. **Interpreting continuous prompts**: What task is a prompt-tuned embedding solving?
4. **Vocabulary projections**: What tokens are encoded at intermediate layers?
5. **Probing representations**: What attributes are encoded (sentiment, topic, etc.)?

## References

- **Patchscopes Paper**: [A Unifying Framework for Inspecting Hidden Representations](https://openreview.net/pdf?id=5uwBzcn885)
- **Continuous Prompts**: [Eliciting Textual Descriptions from Representations of Continuous Prompts](https://aclanthology.org/2025.findings-acl.849.pdf)
- **Logit Lens**: [Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

## Module Location

```
src/patchscopes.py
```

Import with:
```python
from patchscopes import (
    PatchscopesWrapper,
    patchscopes,
    logit_lens,
    prepare_source_inputs,
    prepare_soft_prompt_source_inputs,
    find_token_indices_for_word
)
```

## License

Copyright (c) 2025 Mor Geva and Daniela Gottesman (original notebook)
Module implementation for Manifolds in LLMs project.
