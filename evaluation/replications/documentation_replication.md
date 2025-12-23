# Function Vectors in Large Language Models - Replication Documentation

## Goal

Replicate the core experiment from the Function Vectors paper to verify that autoregressive transformer language models develop compact vector representations of input-output functions (function vectors) within their hidden states during in-context learning.

The main claims to verify:
1. Function vectors can be extracted by summing task-conditioned mean outputs of top causal attention heads
2. Function vectors can trigger task execution in zero-shot and corrupted (shuffled-label) contexts
3. Results are reproducible with controlled random seeds

## Data

**Dataset**: Antonym task from the abstractive datasets folder
- Training set: 1,678 input-output pairs (e.g., "hardware" → "software")
- Validation set: 216 pairs
- Test set: 504 pairs

**Data Source**: `/net/scratch2/smallyan/function_vectors_eval/dataset_files/abstractive/antonym.json`

**Data Splits**: Created using sklearn's train_test_split with seed=32 and test_size=0.3

## Method

### 1. Model Setup
- **Model**: GPT-J 6B (`EleutherAI/gpt-j-6b`)
- **Device**: CUDA (GPU)
- **Configuration**: 28 layers, 16 attention heads, 4096 residual dimension

### 2. Mean Activation Computation
1. Create 100 ICL prompts with 10 randomly sampled examples each
2. For each prompt, extract attention head activations before the output projection
3. Split activations by head (256 dimensions per head)
4. Average activations across all 100 prompts at each token position

### 3. Function Vector Extraction
1. Use pre-computed universal top heads from causal mediation analysis (top 10):
   - Layer 15, Head 5 (AIE: 0.0587)
   - Layer 9, Head 14 (AIE: 0.0584)
   - Layer 12, Head 10 (AIE: 0.0526)
   - Layer 8, Head 1 (AIE: 0.0445)
   - Layer 11, Head 0 (AIE: 0.0445)
   - Layer 13, Head 13 (AIE: 0.0190)
   - Layer 8, Head 0 (AIE: 0.0184)
   - Layer 14, Head 9 (AIE: 0.0160)
   - Layer 9, Head 2 (AIE: 0.0127)
   - Layer 24, Head 6 (AIE: 0.0113)

2. For each top head, project mean activation through output projection matrix
3. Sum all projected outputs to form the function vector (shape: 1 x 4096)

### 4. Intervention Method
- Add function vector to hidden states at layer 9 (early-middle layer)
- Intervention applied to the last token position during forward pass
- Use baukit TraceDict for clean activation editing

### 5. Evaluation Protocol
- **Zero-shot**: No demonstrations, just "Q: {query}\nA:"
- **Shuffled-label**: 10 demonstrations with randomized labels
- **Metric**: Top-1 and Top-3 accuracy on 50 test samples
- **Seed**: 42 for reproducibility

## Results

### Zero-Shot Setting
| Metric | Baseline | With FV | Improvement |
|--------|----------|---------|-------------|
| Top-1 Accuracy | 0% | 32% | +32pp |
| Top-3 Accuracy | 8% | 58% | +50pp |

### Shuffled-Label Setting (10-shot)
| Metric | Baseline | With FV | Improvement |
|--------|----------|---------|-------------|
| Top-1 Accuracy | 30% | 54% | +24pp |
| Top-3 Accuracy | 50% | 68% | +18pp |

### Qualitative Examples

**Zero-shot test case**:
- Query: "static"
- Expected: "dynamic"
- Without FV: Top prediction is "static" (13.5%)
- With FV: Top prediction is "dynamic" (59.7%)

**Natural text test**:
- Input: 'The word "static" means'
- Without FV: 'The word "static" means "unchanging" or "unvarying'
- With FV: 'The word "static" means "dynamic" in the sense that it is'

## Analysis

### Key Findings

1. **Function vectors successfully trigger task execution**: Even without any demonstrations (zero-shot), adding the function vector enables the model to perform the antonym task with 32% accuracy, compared to 0% baseline.

2. **Robustness to corrupted labels**: When ICL demonstrations have shuffled labels that should confuse the model, the function vector still provides a significant boost (30% → 54%).

3. **Reproducibility confirmed**: Running the same evaluation twice with seed=42 produces identical results, confirming deterministic behavior.

### Comparison with Paper Claims

The paper reports:
- Zero-shot: 57.5% vs 5.5% baseline for GPT-J
- Shuffled-label: 90.8% vs 39.1% baseline

Our replication shows lower absolute numbers but confirms the same pattern:
- Large improvement in zero-shot setting
- Significant improvement even with corrupted labels
- The relative improvement direction matches paper claims

### Possible Discrepancies

1. **Smaller evaluation set**: We used 50 samples vs potentially more in the paper
2. **Single task**: We tested on antonym only, paper averaged over multiple tasks
3. **Head selection**: Used fixed universal heads rather than task-specific heads

## Reproducibility Notes

### Environment
- Python 3.10+
- PyTorch 1.13+
- Transformers 4.49+
- baukit (for activation editing)
- CUDA-capable GPU (tested on NVIDIA H100)

### Random Seeds
- Dataset split: seed=32
- Evaluation: seed=42
- ICL example sampling: seed=42

### Files Produced
- `replication.ipynb`: Full replication notebook
- `replication_results.json`: Numerical results
- `documentation_replication.md`: This document
- `evaluation_replication.md`: Evaluation checklist
- `self_replication_evaluation.json`: JSON summary
