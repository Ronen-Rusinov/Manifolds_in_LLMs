# Patchscopes Module Integration - Summary

## Overview

The Patchscopes functionality from the Jupyter notebook has been successfully extracted into a reusable Python module that can be imported by other scripts in the project.

## Files Created

### 1. Core Module: `src/patchscopes.py` (456 lines)

The main module containing all Patchscopes functionality:

**Core Functions:**
- `find_token_indices_for_word()` - Locate token positions for substrings
- `prepare_source_inputs()` - Extract hidden states from text prompts
- `prepare_soft_prompt_source_inputs()` - Extract hidden states from continuous prompts
- `patchscopes()` - Main patching function to decode representations
- `logit_lens()` - Project hidden states onto vocabulary space

**Wrapper Class:**
- `PatchscopesWrapper` - Convenience class for repeated operations with same model

**Key Features:**
- ✅ Fully typed with type hints
- ✅ Comprehensive docstrings with examples
- ✅ Device-agnostic (CPU/GPU)
- ✅ Supports sampling and beam search
- ✅ Handles both discrete and continuous prompts
- ✅ Compatible with HuggingFace transformers

### 2. Documentation: `src/PATCHSCOPES_README.md`

Complete documentation including:
- Quick start guide
- API reference for all functions
- Advanced usage examples (multi-token patching, sampling, layer comparison)
- Target prompt design best practices
- Use cases and references

### 3. Example Script: `examples/patchscopes_example.py`

Demonstrates all major features:
- Single-token patching
- Multi-token patching
- Logit lens analysis
- Sampling-based decoding

### 4. Tests: `tests/test_patchscopes.py`

Unit tests verifying:
- Module imports
- Function signatures
- Class structure
- Docstring presence

### 5. Updated: `README.md`

Added section explaining the Patchscopes module with quick example.

## Usage Examples

### Basic Usage

```python
from src.patchscopes import PatchscopesWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Create wrapper
ps = PatchscopesWrapper(model, tokenizer)

# Extract representations
source_hs = ps.prepare_source_inputs("Paris, France", "France")

# Decode with patchscopes
result = ps.patchscopes(
    source_reprs=source_hs[10],
    target_prompt="Country: ?",
    target_placeholder="?",
    source_layer=10,
    target_layer=5,
    end_phrase="\n"
)
```

### Continuous Prompt Interpretation

```python
# For prompt-tuned embeddings
source_hs = ps.prepare_soft_prompt_source_inputs(continuous_prompt)

target_prompt = (
    "Classify sentiment: positive or negative | "
    "Classify topics: sports, politics, weather | "
    + "?" * continuous_prompt.shape[0]
)

result = ps.patchscopes(
    source_hs[4], 
    target_prompt,
    "?" * continuous_prompt.shape[0],
    source_layer=4,
    target_layer=5,
    end_phrase="|"
)
```

### Logit Lens

```python
inputs = ps.tokenizer("The capital of France is", return_tensors="pt")
with torch.no_grad():
    output = ps.model(**inputs, output_hidden_states=True)

result = ps.logit_lens(output['hidden_states'], -1, 20)
print(result['topk_tokens'][:5])  # ['Paris', ' Paris', ...]
```

## Integration with Existing Code

The module follows the same patterns as existing modules in `src/`:
- Similar import style to `config_manager.py` and `paths.py`
- Can be imported with `from src.patchscopes import ...`
- Device handling follows PyTorch conventions
- Compatible with the project's config system

## Testing

Run the unit tests:
```bash
python tests/test_patchscopes.py
```

Run the full examples (requires model download):
```bash
python examples/patchscopes_example.py
```

## Key Improvements Over Notebook

1. **Reusability**: Functions can be imported and used in any script
2. **Type Safety**: Full type hints for better IDE support
3. **Documentation**: Comprehensive docstrings and separate README
4. **Error Handling**: Clear error messages and assertions
5. **Flexibility**: Device-agnostic, works with any HF model
6. **Maintainability**: Single source of truth for Patchscopes logic

## Dependencies

Required packages (already in project):
- `torch`
- `transformers`

Optional for examples:
- HuggingFace account for gated models (Llama)

## References

Based on:
- Original notebook: `Solution_Advanced_NLP_Course_3 (1).ipynb`
- Patchscopes paper: https://openreview.net/pdf?id=5uwBzcn885
- Continuous prompts paper: https://aclanthology.org/2025.findings-acl.849.pdf

## Next Steps

The module is ready to use! You can:
1. Import it in experiment scripts
2. Use it for activation analysis
3. Integrate with manifold learning experiments
4. Extend with custom target prompts for specific tasks

## File Locations

```
src/
  patchscopes.py              # Main module
  PATCHSCOPES_README.md       # Full documentation
  
examples/
  patchscopes_example.py      # Usage examples
  
tests/
  test_patchscopes.py         # Unit tests

README.md                     # Updated with Patchscopes section
```
