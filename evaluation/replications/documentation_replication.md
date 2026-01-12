# Function Vectors Replication Documentation

## Goal

Replicate the core findings from "Function Vectors in Large Language Models" (Todd et al., ICLR 2024) which demonstrates that transformer language models develop compact vector representations of input-output functions (called "function vectors") within their hidden states during in-context learning.

## Data

**Dataset**: Antonym task from the repository's `dataset_files/abstractive/antonym.json`

**Splits**:
- Train: 1,678 examples
- Validation: 216 examples
- Test: 504 examples (30 samples used for evaluation)

**Format**: JSON file with input-output pairs (e.g., "swift" -> "sluggish", "parent" -> "child")

## Method

### 1. Model Loading
- **Model**: GPT-J-6B (EleutherAI/gpt-j-6b) - the smallest model used in the original paper
- **Configuration**: 28 layers, 16 attention heads, 4096 hidden dimension
- **Device**: CUDA (GPU)

### 2. Mean Activation Computation
For each of 50 trials:
1. Sample 10 random ICL examples from training set
2. Sample 1 validation example as query
3. Construct ICL prompt with format: `<BOS>Q: {input}\nA: {output}\n\n...Q: {query}\nA:`
4. Extract attention head activations using baukit's TraceDict
5. Record last-token activations for each attention head

The mean is computed across all trials to get average head activations for the antonym task.

### 3. Function Vector Extraction
Using the **universal set of top causal attention heads** identified in the paper (pre-computed via causal mediation analysis across multiple ICL tasks):

Top 10 heads for GPT-J:
| Layer | Head | AIE Score |
|-------|------|-----------|
| 15    | 5    | 0.0587    |
| 9     | 14   | 0.0584    |
| 12    | 10   | 0.0526    |
| 8     | 1    | 0.0445    |
| 11    | 0    | 0.0445    |
| 13    | 13   | 0.0190    |
| 8     | 0    | 0.0184    |
| 14    | 9    | 0.0160    |
| 9     | 2    | 0.0127    |
| 24    | 6    | 0.0113    |

For each selected head:
1. Extract the mean activation vector
2. Project through the layer's output projection matrix
3. Sum all projected vectors to form the function vector

### 4. Intervention
- **Intervention Layer**: Layer 9 (approximately L/3 = 28/3 as recommended in paper)
- **Method**: Add function vector to the hidden state at the last token position during forward pass
- **Implementation**: Using baukit's TraceDict with edit_output hook

### 5. Evaluation Contexts

**Shuffled-Label ICL**:
- 10 ICL examples with randomized output labels
- Tests if FV can recover task despite corrupted demonstrations

**Zero-Shot**:
- No ICL examples, just query
- Tests if FV can trigger task execution without any demonstrations

## Results

### Quantitative Results

| Setting | Baseline (without FV) | With FV | Paper Reference |
|---------|----------------------|---------|-----------------|
| Shuffled-Label ICL | 30.0% | 63.3% | ~39% / ~91% |
| Zero-Shot | 3.3% | 33.3% | ~6% / ~58% |

### Qualitative Examples (Zero-Shot)

| Input | Expected | Without FV | With FV |
|-------|----------|------------|---------|
| swift | sluggish | newline | swift |
| expire | renew | The | expire |
| least | most | least | least |
| parent | child | child | child |
| fragile | durable | fragile | robust |

## Analysis

### Key Findings Replicated

1. **Function vectors improve shuffled-label performance**: Adding the FV increases accuracy from 30% to 63% when ICL labels are shuffled, demonstrating the FV carries task-relevant information.

2. **Function vectors enable zero-shot task execution**: The FV improves zero-shot accuracy from 3% to 33%, showing it can trigger task execution without ICL examples.

3. **Universal heads generalize**: The pre-computed universal head set successfully extracts effective function vectors for the antonym task.

### Numerical Differences from Paper

The replicated results are lower than paper values, likely due to:

1. **Reduced trials**: 50 vs 100 trials for mean activation computation
2. **Reduced evaluation samples**: 30 vs full test set (~500 samples)
3. **Random sampling variance**: Different random seeds and example selections
4. **Potential implementation differences**: Minor differences in prompt construction or intervention mechanics

Despite lower absolute numbers, the **relative improvement pattern matches the paper** - FV intervention consistently and substantially improves performance over baselines.

### Limitations

1. Only one task (antonym) was evaluated (paper evaluates 40+ tasks)
2. Only one model (GPT-J) was used (paper includes multiple models)
3. Layer sweep was not performed (paper identifies optimal intervention layer via sweep)
4. Natural text evaluation was not performed quantitatively

## Conclusion

The replication successfully demonstrates the core finding: function vectors extracted from attention head activations can trigger in-context learning tasks in novel contexts (shuffled labels, zero-shot). The pattern of improvement matches the paper's claims even though absolute numbers differ due to reduced experimental scale.
