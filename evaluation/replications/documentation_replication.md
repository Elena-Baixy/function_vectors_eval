# Function Vectors in Large Language Models - Replication Documentation

## Goal

This replication aims to verify the core claims of the "Function Vectors in Large Language Models" paper (Todd et al., ICLR 2024), which investigates whether autoregressive transformer language models develop compact vector representations of input-output functions within their hidden states during in-context learning.

The specific goals are:
1. Extract function vectors from ICL prompts by averaging attention head activations
2. Test function vector portability across different contexts (ICL, shuffled-label, zero-shot, natural text)
3. Verify that function vectors can trigger task execution without explicit in-context examples

## Data

The replication uses the datasets provided in the repository's `dataset_files/` directory:

- **Antonym task** (`abstractive/antonym.json`): 2398 word pairs (e.g., "hot" → "cold")
  - Train: 1678 pairs
  - Valid: 216 pairs
  - Test: 504 pairs

- **Country-Capital task** (`abstractive/country-capital.json`): 196 pairs (e.g., "France" → "Paris")
  - Train: 137 pairs
  - Valid: 17 pairs
  - Test: 42 pairs

Data is split using scikit-learn's train_test_split with 70% train, 30% temp, then 70% valid, 30% test of temp.

## Method

### 1. Function Vector Extraction

Function vectors are computed through the following process:

1. **Generate ICL prompts**: For each of N trials (N=100), sample:
   - 10 random input-output pairs from the training set as ICL examples
   - 1 query from the validation set

2. **Extract activations**: Using `baukit.TraceDict`, capture activations at the input to each attention head's output projection (`transformer.h.{layer}.attn.out_proj` for GPT-J)

3. **Compute mean activations**: Average the last-token activations across all N trials to get mean activations of shape `(n_layers, n_heads, head_dim)`

4. **Select top heads**: Use pre-computed "universal" head rankings based on Average Indirect Effect (AIE) scores from causal mediation analysis. For GPT-J, the top 10 heads are:
   - Layer 15 Head 5 (AIE: 0.0587)
   - Layer 9 Head 14 (AIE: 0.0584)
   - Layer 12 Head 10 (AIE: 0.0526)
   - Layer 8 Head 1 (AIE: 0.0445)
   - Layer 11 Head 0 (AIE: 0.0445)
   - And 5 more...

5. **Compute function vector**: For each top head, project its mean activation through the output projection matrix, then sum all projections.

### 2. Function Vector Intervention

The function vector is added to the model's hidden state at layer 9 (approximately L/3 for GPT-J's 28 layers) at the last token position during inference:

```
hidden_state[:, -1] += function_vector
```

This is implemented using `baukit.TraceDict` with an `edit_output` hook.

### 3. Evaluation Contexts

Function vectors are tested in four contexts:
1. **Clean ICL**: Standard 10-shot in-context learning (baseline)
2. **Shuffled-label ICL**: ICL with randomized output labels + FV intervention
3. **Zero-shot**: No examples, just query + FV intervention
4. **Natural text**: Free-form prompts like "The word X means" + FV intervention

### 4. Metrics

- **Top-1 Accuracy**: Percentage of test samples where the correct answer is the model's top prediction
- Computed by checking if the target token has rank 0 in the logit distribution

## Results

### Antonym Task (50 test samples)

| Context | Baseline | + Function Vector |
|---------|----------|-------------------|
| Clean ICL (10-shot) | 64.0% | 62.0% |
| Shuffled-label (10-shot) | 30.0% | 54.0% |
| Zero-shot | 0.0% | 26.0% |

### Country-Capital Task (42 test samples)

| Context | Baseline | + Function Vector |
|---------|----------|-------------------|
| Zero-shot | 7.1% | 83.3% |

### Qualitative Results

**Antonym - Natural Text:**
- Input: `The word "static" means`
- Without FV: `"unchanging" or "unvarying`
- With FV: `"dynamic" in the sense that it is`

**Country-Capital - Natural Text:**
- Input: `The capital city of Cameroon is`
- Without FV: `Yaoundé. It`
- With FV: `Yaoundé. The`

## Analysis

### Key Findings

1. **Function vectors successfully transfer task knowledge**: The dramatic improvement from 0% to 26% (antonym) and 7.1% to 83.3% (country-capital) in zero-shot settings demonstrates that FVs encode task-relevant information that can be injected without any in-context examples.

2. **FVs recover performance in corrupted contexts**: In shuffled-label settings, FVs improve accuracy from 30% to 54%, nearly recovering the clean ICL performance (64%).

3. **FVs don't harm clean ICL**: Adding FV to clean ICL maintains similar performance (64% → 62%), indicating the intervention doesn't interfere with existing task execution.

4. **Universal heads are effective**: Using pre-computed universal heads (rather than task-specific causal analysis) still produces effective function vectors, supporting the paper's claim of universal task-encoding circuits.

### Comparison with Paper Results

The paper reports for GPT-J on antonym task:
- Shuffled-label ICL+FV: ~90% (vs our 54%)
- Zero-shot+FV: ~58% (vs our 26%)

Our results show the same directional effects but with lower absolute accuracy. This discrepancy may be due to:
1. Using only 50 test samples (vs full test set)
2. Potential differences in random seed affecting example selection
3. The original implementation may use additional filtering (e.g., only samples the model gets correct on clean ICL)

### Limitations

1. Replication uses pre-computed universal head rankings rather than performing causal mediation analysis from scratch
2. Evaluation sample sizes are smaller than the full paper for computational efficiency
3. Only two tasks are replicated (antonym and country-capital) out of the 40+ in the paper

## Conclusion

This replication successfully demonstrates the core mechanism of function vectors: extracting task representations from attention head activations and using them to trigger task execution across different prompt contexts. The results confirm that function vectors are a real phenomenon that enables zero-shot task transfer, though the exact accuracy numbers differ from the original paper likely due to evaluation setup differences.
