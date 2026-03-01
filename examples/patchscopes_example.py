"""Example usage of the Patchscopes module.

This script demonstrates how to use the patchscopes module for:
1. Single-token patching
2. Multi-token patching
3. Logit lens analysis
4. Continuous prompt interpretation
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from patchscopes import PatchscopesWrapper, patchscopes, logit_lens





def load_model_and_tokenizer(model_name: str, device: torch.device):
    """Load model and tokenizer.
    THE EXAMPLES IN THIS FILE ARE TAKEN FROM COURSE MATERIAL BY DANIELLA GOTTESMAN, AND MOR GEVA
    ALL RIGHTS ARE RESERVED TO THE ORIGINAL AUTHORS, AND THIS USAGE IS UNDER THEIR EXPRESS PERMISSION.
    LINK TO ORIGINAL MATERIAL: https://colab.research.google.com/drive/1Be42DpybrBxO3I-vZVyVuwo_acgIJucQ?usp=sharing"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer


def example_single_token_patching(ps: PatchscopesWrapper):
    """Example of single token patching.
    THE EXAMPLES IN THIS FILE ARE TAKEN FROM COURSE MATERIAL BY DANIELLA GOTTESMAN, AND MOR GEVA
    ALL RIGHTS ARE RESERVED TO THE ORIGINAL AUTHORS, AND THIS USAGE IS UNDER THEIR EXPRESS PERMISSION.
    LINK TO ORIGINAL MATERIAL: https://colab.research.google.com/drive/1Be42DpybrBxO3I-vZVyVuwo_acgIJucQ?usp=sharing"""
    print("\n=== Single Token Patching Example ===")
    
    source_prompt = "Diana, Princess of Wales"
    source_substring = "Wales"
    
    target_prompt = (
        "Syria: Country in the Middle East\n|\n"
        "Leonardo DiCaprio: American actor\n|\n"
        "Samsung: South Korean multinational major appliance and consumer electronics corporation\n|\n"
        "?"
    )
    target_placeholder = "?"
    
    source_layer = 2
    target_layer = 5
    
    # Extract source representations
    source_hs_cache = ps.prepare_source_inputs(source_prompt, source_substring)
    source_reprs = source_hs_cache[source_layer]
    
    # Apply patchscopes
    result = ps.patchscopes(
        source_reprs,
        target_prompt,
        target_placeholder,
        source_layer,
        target_layer,
        end_phrase="\n|\n",
        max_new_tokens=20
    )
    
    print(f"Source: '{source_substring}' from '{source_prompt}'")
    print(f"Result: {result}")


def example_multi_token_patching(ps: PatchscopesWrapper):
    """Example of multi-token patching."""
    print("\n=== Multi-Token Patching Example ===")
    
    source_prompt = "Diana, Princess of Wales"
    source_substring = "Diana, Princess of Wales"
    
    target_prompt = (
        "Syria: Country in the Middle East\n|\n"
        "Leonardo DiCaprio: American actor\n|\n"
        "Samsung: South Korean multinational major appliance and consumer electronics corporation\n|\n"
        "? ? ? ? ?"
    )
    target_placeholder = "? ? ? ? ?"
    
    source_layer = 4
    target_layer = 4
    
    source_hs_cache = ps.prepare_source_inputs(source_prompt, source_substring)
    source_reprs = source_hs_cache[source_layer]
    
    result = ps.patchscopes(
        source_reprs,
        target_prompt,
        target_placeholder,
        source_layer,
        target_layer,
        end_phrase="\n|\n",
        max_new_tokens=20
    )
    
    print(f"Source: '{source_substring}'")
    print(f"Result: {result}")


def example_logit_lens(ps: PatchscopesWrapper):
    """Example of logit lens analysis."""
    print("\n=== Logit Lens Example ===")
    
    prompt = "The capital of France is"
    inputs = ps.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(ps.device)
    
    with torch.no_grad():
        output = ps.model(**inputs, output_hidden_states=True)
    
    layer = 20
    token_position = inputs['input_ids'].shape[1] - 1
    
    result = ps.logit_lens(output['hidden_states'], token_position, layer, k=10)
    
    print(f"Prompt: '{prompt}'")
    print(f"Layer {layer}, position {token_position}")
    print(f"Top-5 predictions:")
    for i in range(min(5, len(result['topk_tokens']))):
        token = result['topk_tokens'][i]
        prob = result['topk_probs'][i]
        print(f"  {i+1}. '{token}' (prob: {prob:.4f})")


def example_sampling(ps: PatchscopesWrapper):
    """Example of sample-based decoding."""
    print("\n=== Sampling Example ===")
    
    source_prompt = "Diana, Princess of Wales"
    source_substring = "Wales"
    
    target_prompt = (
        "Syria: Country in the Middle East\n|\n"
        "Leonardo DiCaprio: American actor\n|\n"
        "Samsung: South Korean multinational major appliance and consumer electronics corporation\n|\n"
        "?"
    )
    target_placeholder = "?"
    
    source_layer = 2
    target_layer = 5
    
    source_hs_cache = ps.prepare_source_inputs(source_prompt, source_substring)
    source_reprs = source_hs_cache[source_layer]
    
    print(f"Source: '{source_substring}'")
    print("Generating 3 samples with temperature=0.8:")
    
    for i in range(3):
        result = ps.patchscopes(
            source_reprs,
            target_prompt,
            target_placeholder,
            source_layer,
            target_layer,
            end_phrase="\n|\n",
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8
        )
        print(f"  Sample {i+1}: {result}")


def main():
    """Run all examples."""
    print("Loading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # You may need to set your HuggingFace API key for gated models
    # import os
    # os.environ["HF_TOKEN"] = "your_token_here"
    
    model_name = "google/gemma-2-2B"
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    
    # Create wrapper
    ps = PatchscopesWrapper(model, tokenizer, device)
    
    # Run examples
    example_single_token_patching(ps)
    example_multi_token_patching(ps)
    example_logit_lens(ps)
    example_sampling(ps)
    
    print("\n=== All examples completed ===")


if __name__ == "__main__":
    main()
