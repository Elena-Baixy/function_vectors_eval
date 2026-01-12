#!/usr/bin/env python
"""
Function Vectors Replication Script

This script replicates the Function Vectors experiment from Todd et al. (ICLR 2024).
Reimplemented from understanding of plan.md and CodeWalkthrough.md without verbatim code copying.
"""

import os
os.chdir('/net/scratch2/smallyan/function_vectors_eval')

import torch
import numpy as np
import random
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import TraceDict

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Disable gradient computation for inference
torch.set_grad_enabled(False)

def set_reproducibility_seed(seed_value: int) -> None:
    """Set seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def setup_language_model(model_identifier: str, target_device: str = 'cuda'):
    """
    Load a causal language model and its tokenizer from HuggingFace.
    Returns model, tokenizer, and a configuration dictionary.
    """
    print(f"Loading model: {model_identifier}")

    tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_identifier,
        low_cpu_mem_usage=True
    ).to(target_device)

    # Build configuration based on model type
    if 'gpt-j' in model_identifier.lower():
        config = {
            'num_attention_heads': model.config.n_head,
            'num_layers': model.config.n_layer,
            'hidden_dimension': model.config.n_embd,
            'model_name': model.config.name_or_path,
            'attention_output_hooks': [f'transformer.h.{i}.attn.out_proj' for i in range(model.config.n_layer)],
            'layer_hooks': [f'transformer.h.{i}' for i in range(model.config.n_layer)],
            'adds_bos_automatically': False
        }
    else:
        raise NotImplementedError(f"Model {model_identifier} not supported in this replication")

    return model, tokenizer, config


class TaskDataset:
    """Simple dataset class for ICL input-output pairs."""

    def __init__(self, data_source):
        if isinstance(data_source, str):
            self.data = pd.read_json(data_source)
        elif isinstance(data_source, dict):
            self.data = pd.DataFrame(data_source)
        self.data = self.data[['input', 'output']]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data.iloc[idx].to_dict()
        elif isinstance(idx, (slice, list, np.ndarray)):
            return self.data.iloc[idx].to_dict(orient='list')
        elif isinstance(idx, str):
            return self.data[idx].tolist()

    def __len__(self):
        return len(self.data)


def load_task_data(task_name: str, data_root: str = 'dataset_files', split_ratio: float = 0.3, seed_val: int = 32):
    """Load a task dataset and split into train/valid/test."""

    # Search for the dataset in abstractive or extractive folders
    for folder in ['abstractive', 'extractive']:
        filepath = Path(data_root) / folder / f"{task_name}.json"
        if filepath.exists():
            break
    else:
        raise FileNotFoundError(f"Dataset {task_name} not found")

    full_dataset = TaskDataset(str(filepath))

    # Split the data
    train_data, remaining = train_test_split(full_dataset.data, test_size=split_ratio, random_state=seed_val)
    test_data, valid_data = train_test_split(remaining, test_size=split_ratio, random_state=seed_val)

    return {
        'train': TaskDataset(train_data.to_dict(orient='list')),
        'valid': TaskDataset(valid_data.to_dict(orient='list')),
        'test': TaskDataset(test_data.to_dict(orient='list'))
    }


def build_icl_prompt(
    example_pairs: dict,
    query_pair: dict = None,
    input_prefix: str = "Q:",
    output_prefix: str = "A:",
    input_separator: str = "\n",
    output_separator: str = "\n\n",
    add_bos: bool = True,
    randomize_labels: bool = False
) -> str:
    """
    Construct an in-context learning prompt from example pairs.

    Format: Q: {input}\nA: {output}\n\n for each example
    """
    prompt_parts = []

    # Add BOS token if needed
    if add_bos:
        prompt_parts.append('<|endoftext|>')

    inputs = example_pairs.get('input', [])
    outputs = example_pairs.get('output', [])

    # Optionally shuffle the output labels
    if randomize_labels and len(outputs) > 0:
        outputs = list(np.random.permutation(outputs))

    # Build examples
    for inp, out in zip(inputs, outputs):
        prompt_parts.append(f"{input_prefix} {inp}{input_separator}{output_prefix} {out}{output_separator}")

    # Add query if provided
    if query_pair is not None:
        query_input = query_pair['input']
        prompt_parts.append(f"{input_prefix} {query_input}{input_separator}{output_prefix}")

    return ''.join(prompt_parts)


def extract_attention_activations(prompt: str, model, tokenizer, model_config: dict):
    """
    Extract attention head output activations for a given prompt.
    Uses baukit's TraceDict for activation extraction.
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    hook_layers = model_config['attention_output_hooks']

    with TraceDict(model, layers=hook_layers, retain_input=True, retain_output=False) as traced:
        model(**inputs)

    return traced, inputs


def compute_mean_activations_per_head(
    dataset: dict,
    model,
    tokenizer,
    model_config: dict,
    num_icl_examples: int = 10,
    num_trials: int = 100,
    shuffle_labels: bool = False
):
    """
    Compute average activations for each attention head across multiple ICL prompts.
    This is a key step in extracting function vectors.
    """
    n_layers = model_config['num_layers']
    n_heads = model_config['num_attention_heads']
    hidden_dim = model_config['hidden_dimension']
    head_dim = hidden_dim // n_heads

    # Storage for last-token activations
    activation_sum = torch.zeros(n_layers, n_heads, head_dim)

    should_add_bos = not model_config['adds_bos_automatically']

    for trial in tqdm(range(num_trials), desc="Computing mean activations"):
        # Sample random ICL examples
        indices = np.random.choice(len(dataset['train']), num_icl_examples, replace=False)
        icl_examples = dataset['train'][indices]

        # Sample a validation example as query
        query_idx = np.random.choice(len(dataset['valid']))
        query_example = dataset['valid'][query_idx]

        # Build the prompt
        prompt = build_icl_prompt(
            icl_examples,
            query_example,
            add_bos=should_add_bos,
            randomize_labels=shuffle_labels
        )

        # Extract activations
        traced, _ = extract_attention_activations(prompt, model, tokenizer, model_config)

        # Process each layer's activations
        for layer_idx, layer_name in enumerate(model_config['attention_output_hooks']):
            # Get input to the output projection (post-attention, pre-projection)
            layer_activation = traced[layer_name].input
            if isinstance(layer_activation, tuple):
                layer_activation = layer_activation[0]

            # Shape: (batch, seq_len, hidden_dim) -> reshape to (batch, seq_len, n_heads, head_dim)
            reshaped = layer_activation.view(-1, layer_activation.size(1), n_heads, head_dim)

            # Get last token activations
            last_token_acts = reshaped[0, -1, :, :]  # (n_heads, head_dim)

            activation_sum[layer_idx] += last_token_acts.cpu()

    mean_activations = activation_sum / num_trials
    return mean_activations


# Pre-defined universal set of top causal heads for GPT-J (from the paper)
GPTJ_UNIVERSAL_HEADS = [
    (15, 5, 0.0587), (9, 14, 0.0584), (12, 10, 0.0526), (8, 1, 0.0445), (11, 0, 0.0445),
    (13, 13, 0.019), (8, 0, 0.0184), (14, 9, 0.016), (9, 2, 0.0127), (24, 6, 0.0113),
    (15, 11, 0.0092), (6, 6, 0.0069), (14, 0, 0.0068), (17, 8, 0.0068), (21, 2, 0.0067),
    (10, 11, 0.0066), (11, 2, 0.0057), (17, 0, 0.0054), (20, 11, 0.0051), (23, 0, 0.0047),
    (20, 0, 0.0046), (15, 7, 0.0045), (27, 2, 0.0045), (21, 15, 0.0044), (11, 4, 0.0044),
    (18, 6, 0.0043), (9, 6, 0.0042), (4, 12, 0.004), (11, 15, 0.004), (20, 2, 0.0036),
    (10, 0, 0.0035), (16, 9, 0.0031), (11, 14, 0.0031), (12, 4, 0.003), (9, 7, 0.003),
    (18, 3, 0.003), (19, 5, 0.003), (22, 5, 0.0027), (25, 3, 0.0026), (18, 9, 0.0025)
]


def compute_function_vector_from_universal_heads(
    mean_activations: torch.Tensor,
    model,
    model_config: dict,
    num_heads_to_use: int = 10
):
    """
    Compute a function vector by summing the outputs of the top causal attention heads.
    Uses the universal head set identified across multiple ICL tasks.
    """
    hidden_dim = model_config['hidden_dimension']
    n_heads = model_config['num_attention_heads']
    head_dim = hidden_dim // n_heads
    device = model.device

    # Use the universal head set for GPT-J
    selected_heads = GPTJ_UNIVERSAL_HEADS[:num_heads_to_use]

    # Initialize function vector
    fv = torch.zeros(1, 1, hidden_dim, device=device, dtype=model.dtype)

    for layer_idx, head_idx, _ in selected_heads:
        # Get the output projection for this layer
        out_proj = model.transformer.h[layer_idx].attn.out_proj

        # Create input vector with only this head's activation
        head_input = torch.zeros(hidden_dim, device=device)
        head_input[head_idx * head_dim:(head_idx + 1) * head_dim] = mean_activations[layer_idx, head_idx].to(device)

        # Project through the output projection
        projected = out_proj(head_input.reshape(1, 1, hidden_dim).to(model.dtype))
        fv += projected

    return fv.reshape(1, hidden_dim), selected_heads


def apply_function_vector_intervention(
    prompt: str,
    function_vector: torch.Tensor,
    intervention_layer: int,
    model,
    tokenizer,
    model_config: dict
):
    """
    Run the model with and without function vector intervention.
    The FV is added to the hidden state at the specified layer.
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    # Clean run (no intervention)
    clean_logits = model(**inputs).logits[:, -1, :]

    # Define intervention function
    def add_fv_at_layer(output, layer_name):
        current_layer = int(layer_name.split('.')[2])
        if current_layer == intervention_layer:
            if isinstance(output, tuple):
                output[0][:, -1] += function_vector.to(device).squeeze()
                return output
        return output

    # Intervention run
    with TraceDict(model, layers=model_config['layer_hooks'], edit_output=add_fv_at_layer):
        intervention_logits = model(**inputs).logits[:, -1, :]

    return clean_logits, intervention_logits


def decode_top_predictions(logits: torch.Tensor, tokenizer, k: int = 5):
    """Decode the top-k predictions from logits."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=k, dim=-1)

    results = []
    for idx, prob in zip(top_indices.squeeze(), top_probs.squeeze()):
        token = tokenizer.decode(idx.item())
        results.append((token, prob.item()))
    return results


def evaluate_accuracy(
    dataset: dict,
    function_vector: torch.Tensor,
    intervention_layer: int,
    model,
    tokenizer,
    model_config: dict,
    num_icl_shots: int = 10,
    shuffle_labels: bool = False,
    num_samples: int = 50
):
    """
    Evaluate top-1 accuracy with and without FV intervention.
    """
    should_add_bos = not model_config['adds_bos_automatically']

    clean_correct = 0
    fv_correct = 0
    total = 0

    test_indices = np.random.choice(len(dataset['test']), min(num_samples, len(dataset['test'])), replace=False)

    for idx in tqdm(test_indices, desc="Evaluating"):
        test_pair = dataset['test'][int(idx)]

        # Handle if test_pair is None or missing fields
        if test_pair is None or 'input' not in test_pair or 'output' not in test_pair:
            continue

        # Sample ICL examples
        if num_icl_shots > 0:
            icl_indices = np.random.choice(len(dataset['train']), num_icl_shots, replace=False)
            icl_examples = dataset['train'][list(icl_indices)]
        else:
            icl_examples = {'input': [], 'output': []}

        # Build prompt
        prompt = build_icl_prompt(
            icl_examples,
            test_pair,
            add_bos=should_add_bos,
            randomize_labels=shuffle_labels
        )

        # Get target token
        target = ' ' + str(test_pair['output'])  # Include space prefix
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        target_id = target_ids[0] if target_ids else None

        if target_id is None:
            continue

        # Run with and without FV
        clean_logits, fv_logits = apply_function_vector_intervention(
            prompt,
            function_vector,
            intervention_layer,
            model,
            tokenizer,
            model_config
        )

        # Check if top prediction is correct
        clean_pred = clean_logits.argmax(dim=-1).item()
        fv_pred = fv_logits.argmax(dim=-1).item()

        if clean_pred == target_id:
            clean_correct += 1
        if fv_pred == target_id:
            fv_correct += 1
        total += 1

    return {
        'clean_accuracy': clean_correct / total if total > 0 else 0,
        'fv_accuracy': fv_correct / total if total > 0 else 0,
        'total_samples': total
    }


def main():
    """Main replication function."""
    set_reproducibility_seed(42)

    # Load model (GPT-J - smallest model in the paper experiments)
    MODEL_NAME = 'EleutherAI/gpt-j-6b'
    model, tokenizer, config = setup_language_model(MODEL_NAME, device)

    print(f"\nModel configuration:")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Heads: {config['num_attention_heads']}")
    print(f"  Hidden dim: {config['hidden_dimension']}")

    # Load the antonym dataset
    set_reproducibility_seed(0)
    antonym_data = load_task_data('antonym')

    print(f"\nAntonym dataset:")
    print(f"  Train: {len(antonym_data['train'])} examples")
    print(f"  Valid: {len(antonym_data['valid'])} examples")
    print(f"  Test: {len(antonym_data['test'])} examples")

    # Compute mean activations
    set_reproducibility_seed(42)
    print("\nComputing mean head activations...")
    mean_acts = compute_mean_activations_per_head(
        antonym_data,
        model,
        tokenizer,
        config,
        num_icl_examples=10,
        num_trials=50  # Reduced for faster replication
    )
    print(f"Mean activations shape: {mean_acts.shape}")

    # Compute function vector
    function_vector, top_heads = compute_function_vector_from_universal_heads(
        mean_acts,
        model,
        config,
        num_heads_to_use=10
    )
    print(f"\nFunction vector shape: {function_vector.shape}")
    print(f"Top 10 heads used (layer, head, AIE score):")
    for head in top_heads:
        print(f"  Layer {head[0]}, Head {head[1]}: {head[2]:.4f}")

    # Intervention layer - approximately L/3 as recommended in paper
    INTERVENTION_LAYER = 9

    # Qualitative examples
    print("\n" + "=" * 50)
    print("QUALITATIVE EXAMPLES")
    print("=" * 50)

    set_reproducibility_seed(123)
    for i in range(5):
        test_pair = antonym_data['test'][i]

        # Zero-shot prompt
        zs_prompt = build_icl_prompt({'input': [], 'output': []}, test_pair, add_bos=True)

        clean_logits, fv_logits = apply_function_vector_intervention(
            zs_prompt,
            function_vector,
            INTERVENTION_LAYER,
            model,
            tokenizer,
            config
        )

        clean_pred = tokenizer.decode(clean_logits.argmax(dim=-1).item())
        fv_pred = tokenizer.decode(fv_logits.argmax(dim=-1).item())

        print(f"\nInput: '{test_pair['input']}' -> Expected: '{test_pair['output']}'")
        print(f"  Zero-shot prediction: '{clean_pred}'")
        print(f"  Zero-shot + FV:       '{fv_pred}'")

    # Quantitative evaluation
    print("\n" + "=" * 50)
    print("QUANTITATIVE EVALUATION")
    print("=" * 50)

    # Evaluate on shuffled-label ICL
    set_reproducibility_seed(42)
    print("\nEvaluating on Shuffled-Label ICL...")
    shuffled_results = evaluate_accuracy(
        antonym_data,
        function_vector,
        INTERVENTION_LAYER,
        model,
        tokenizer,
        config,
        num_icl_shots=10,
        shuffle_labels=True,
        num_samples=30
    )

    print(f"\nShuffled-Label ICL Results:")
    print(f"  Without FV: {shuffled_results['clean_accuracy']:.1%}")
    print(f"  With FV:    {shuffled_results['fv_accuracy']:.1%}")

    # Evaluate on zero-shot
    set_reproducibility_seed(42)
    print("\nEvaluating on Zero-Shot...")
    zeroshot_results = evaluate_accuracy(
        antonym_data,
        function_vector,
        INTERVENTION_LAYER,
        model,
        tokenizer,
        config,
        num_icl_shots=0,
        shuffle_labels=False,
        num_samples=30
    )

    print(f"\nZero-Shot Results:")
    print(f"  Without FV: {zeroshot_results['clean_accuracy']:.1%}")
    print(f"  With FV:    {zeroshot_results['fv_accuracy']:.1%}")

    # Results summary
    print("\n" + "=" * 60)
    print("REPLICATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Task: Antonym")
    print(f"Intervention Layer: {INTERVENTION_LAYER}")
    print(f"Number of Top Heads: 10")
    print()
    print("Results:")
    print(f"  Shuffled-Label ICL:")
    print(f"    - Baseline (without FV): {shuffled_results['clean_accuracy']:.1%}")
    print(f"    - With FV:               {shuffled_results['fv_accuracy']:.1%}")
    print(f"  Zero-Shot:")
    print(f"    - Baseline (without FV): {zeroshot_results['clean_accuracy']:.1%}")
    print(f"    - With FV:               {zeroshot_results['fv_accuracy']:.1%}")
    print()
    print("Paper Reference Values (Table 1, GPT-J):")
    print(f"  Shuffled-Label: ~39% baseline, ~91% with FV")
    print(f"  Zero-Shot: ~6% baseline, ~58% with FV")
    print()
    print("Note: Slight variations expected due to:")
    print("  - Reduced number of trials (50 vs 100)")
    print("  - Reduced evaluation samples (30 vs full test set)")
    print("  - Random sampling variability")
    print("=" * 60)

    # Save results
    results_summary = {
        'model': MODEL_NAME,
        'task': 'antonym',
        'intervention_layer': INTERVENTION_LAYER,
        'num_top_heads': 10,
        'num_trials_for_mean_activations': 50,
        'num_evaluation_samples': 30,
        'shuffled_label_icl': {
            'baseline_accuracy': shuffled_results['clean_accuracy'],
            'fv_accuracy': shuffled_results['fv_accuracy']
        },
        'zero_shot': {
            'baseline_accuracy': zeroshot_results['clean_accuracy'],
            'fv_accuracy': zeroshot_results['fv_accuracy']
        },
        'paper_reference': {
            'shuffled_label_baseline': 0.39,
            'shuffled_label_fv': 0.91,
            'zero_shot_baseline': 0.06,
            'zero_shot_fv': 0.58
        }
    }

    with open('evaluation/replications/results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\nResults saved to evaluation/replications/results.json")

    return results_summary


if __name__ == "__main__":
    results = main()
