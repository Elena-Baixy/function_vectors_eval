# Plan
## Objective
To investigate whether autoregressive transformer language models develop compact vector representations of input-output functions (called function vectors) within their hidden states during in-context learning, and to characterize these representations across diverse tasks and models.

## Hypothesis
1. A small number of attention heads transport a compact representation of the demonstrated task (function vector) that is robust to changes in context and can trigger task execution in zero-shot and natural text settings.
2. Function vectors contain information encoding the output space of the function, but this information alone is not sufficient to reconstruct a working function vector.
3. Function vectors can be composed through vector algebra to create vectors that trigger new complex tasks combining constituent tasks.

## Methodology
1. Apply causal mediation analysis to identify attention heads with highest average indirect effect (AIE) across diverse ICL tasks. Extract function vectors by summing task-conditioned mean outputs of top causal attention heads.
2. Test function vectors across models (GPT-J 6B to Llama 2 70B) and over 40 diverse ICL tasks by adding FVs to hidden states at different layers and measuring task performance in shuffled-label, zero-shot, and natural text contexts.
3. Analyze FV internal structure by decoding vectors to vocabulary space and testing whether reconstructed vectors matching decoded distributions can recover FV performance.
4. Test vector algebra composition by constructing decomposable tasks and measuring whether algebraic sums of FVs can execute combined tasks.

## Experiments
### Portability of function vectors across contexts
- What varied: Context type (shuffled-label ICL, zero-shot, different ICL templates, natural text) and layer where FV is added
- Metric: Top-1 accuracy on target outputs
- Main result: FVs work best when added at early-middle layers (approximately L/3). In shuffled-label context GPT-J+FV achieves 90.8% vs 39.1% baseline; in zero-shot 57.5% vs 5.5%. FVs are robust across 20 different templates and natural text contexts.

### Decoded vocabulary analysis
- What varied: Number of top vocabulary tokens (k=100 vs all 50k tokens) used to reconstruct FV
- Metric: Zero-shot accuracy of reconstructed vectors vˆtk compared to original FV vt
- Main result: Reconstructed vectors matching top 100 tokens achieve much lower performance than original FVs (e.g., Country-Capital: 58.1% vs 83.2%). Even matching all tokens underperforms, indicating FVs carry information beyond output vocabulary.

### Vector algebra composition
- What varied: Different list-oriented task combinations (Last-Capitalize, Last-Country-Capital, Last-Antonym, etc.)
- Metric: Accuracy of composed vector v∗BD compared to ICL and directly extracted FV vBD
- Main result: Composed FVs work for some tasks, sometimes outperforming ICL (Last-Country-Capital: 0.60 vs 0.32 ICL, Last-Capitalize-First-Letter: 0.95 vs 0.75 ICL) but fail for others (Last-Antonym: 0.07 vs 0.25 ICL).

### Causal mediation analysis across models
- What varied: Model architecture and size (GPT-J 6B, GPT-NeoX 20B, Llama 2 7B/13B/70B)
- Metric: Average Indirect Effect (AIE) per attention head
- Main result: Top 10-100 attention heads (scaled by model size) with highest AIE cluster in middle layers across all models. Maximum AIE decreases slightly with model size (~0.053 for GPT-J to ~0.037 for Llama 2 70B).

### Performance across diverse tasks and models
- What varied: 34 additional abstractive and extractive tasks beyond the 6 representative tasks
- Metric: Zero-shot and shuffled-label accuracy with FV intervention
- Main result: GPT-J+FV achieves 80.4% shuffled-label and 46.1% zero-shot on 34 additional tasks. Llama 2 (70B)+FV achieves 93.0% shuffled-label and 74.2% zero-shot, showing consistent FV effects across task diversity.

### Natural text portability evaluation
- What varied: Different natural language prompt templates for each task
- Metric: Accuracy of correct answer appearing within n generated tokens
- Main result: Antonym FV achieves 55-68% accuracy across natural templates vs 0-3% baseline. Country-Capital FV achieves 57-72% vs 4-23% baseline, demonstrating FVs work in naturalistic settings.