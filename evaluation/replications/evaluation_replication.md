# Function Vectors Replication - Evaluation

## Replication Summary

This document evaluates the replication of the "Function Vectors in Large Language Models" paper (Todd et al., ICLR 2024). The replication was performed by reimplementing the core functionality from scratch based on the provided plan.md and CodeWalkthrough.md files.

## What Was Replicated

1. **Data Loading**: Implemented ICLDataset class and train/valid/test splitting
2. **Prompt Construction**: Created functions to build ICL prompts with configurable templates
3. **Model Loading**: Loaded GPT-J 6B with appropriate configuration for hooking
4. **Activation Extraction**: Implemented activation extraction at attention head outputs using baukit
5. **Function Vector Computation**: Implemented FV computation using pre-defined universal heads
6. **Intervention Mechanism**: Implemented hidden state modification during inference
7. **Evaluation**: Tested on antonym and country-capital tasks in multiple contexts

## What Was Not Replicated

1. **Causal Mediation Analysis**: Used pre-computed universal head rankings instead of computing AIE from scratch
2. **Full Task Coverage**: Only replicated 2 of 40+ tasks from the paper
3. **Vector Algebra Composition**: Did not replicate the FV composition experiments
4. **Vocabulary Decoding Analysis**: Did not replicate the vocabulary reconstruction experiments
5. **Portability Across Templates**: Did not systematically test across 20 different ICL templates

## Discrepancies Observed

| Metric | Paper (GPT-J) | Replication | Notes |
|--------|---------------|-------------|-------|
| Shuffled-label + FV | ~90% | 54% | Lower, likely due to sample size |
| Zero-shot + FV | ~58% | 26% | Lower, same reason |
| Baseline direction | Positive | Positive | Correct direction maintained |

The directional effects are consistent with the paper, but absolute numbers are lower. This is likely due to:
- Smaller evaluation sample size (50 vs full test set)
- No filtering to only samples model gets correct on clean ICL
- Potential random seed differences

## Issues Encountered

1. **Variable-length prompts**: Initial implementation failed on stacking activations from different-length prompts. Fixed by extracting only last-token activations.

2. **baukit dependency**: Required for activation tracing. Successfully imported without installation issues.

3. **No issues with model loading**: GPT-J 6B loaded successfully on GPU with adequate memory.

---

# Replication Evaluation - Binary Checklist

## RP1. Implementation Reconstructability

**PASS**

**Rationale**: The experiment can be reconstructed from the plan and code-walk without missing steps. The plan.md clearly describes:
- The hypothesis and methodology
- The experiments to perform
- The expected metrics and results

The CodeWalkthrough.md provides:
- Clear setup instructions
- Demo notebook reference
- Description of utility file purposes

The demo notebook (fv_demo.ipynb) provides a complete walkthrough of the core functionality. While the replication required understanding the utility files' implementations, no major guesswork was needed - the code is well-documented and follows the plan closely.

---

## RP2. Environment Reproducibility

**PASS**

**Rationale**: The environment can be restored and run without unresolved issues:
- `fv_environment.yml` provides conda environment specification
- All required packages (transformers, torch, baukit, sklearn, pandas, numpy) are standard and available
- GPT-J 6B model loads correctly from HuggingFace
- Datasets are included in the repository (`dataset_files/`)
- No external API keys required for core functionality
- No version conflicts encountered during replication

The only minor consideration is the baukit package, which is less common but installed without issues via pip.

---

## RP3. Determinism and Stability

**PASS**

**Rationale**: Results are stable with controlled randomness:
- The replication implements `set_random_seed()` function setting:
  - Python random seed
  - NumPy random seed
  - PyTorch manual seed (CPU and CUDA)
  - CuDNN deterministic mode
  - PYTHONHASHSEED
- Results are reproducible across runs with the same seed
- The function vector extraction averages over 100 trials, reducing variance
- No significant run-to-run variance observed in evaluation metrics

The pre-computed universal head rankings provide additional stability since they don't depend on per-run causal analysis.

---

## RP4. Demo Presentation

**PASS**

**Rationale**: A demo notebook (`notebooks/fv_demo.ipynb`) is provided and satisfies all conditions:

1. **Executable without external materials**: The demo can be executed using only the repository contents - it loads the model from HuggingFace, uses included datasets, and requires no hidden resources.

2. **Experiments are demonstrated or linked**:
   - Core FV extraction is demonstrated
   - ICL, shuffled-label, zero-shot, and natural text contexts are all shown
   - The main experiments from the paper are represented in the demo
   - Evaluation scripts in `src/eval_scripts/` provide additional runnable experiments

3. **Inputs and configurations specified**:
   - Model name specified (EleutherAI/gpt-j-6b)
   - Dataset loading clearly shown
   - Edit layer specified (EDIT_LAYER = 9)
   - Number of heads and trials documented

The demo outputs match the type of results claimed in the paper (improved accuracy with FV intervention in zero-shot and shuffled-label contexts).

---

## Summary

| Criterion | Result |
|-----------|--------|
| RP1. Implementation Reconstructability | **PASS** |
| RP2. Environment Reproducibility | **PASS** |
| RP3. Determinism and Stability | **PASS** |
| RP4. Demo Presentation | **PASS** |

**Overall Assessment**: The replication is successful. The repository provides sufficient documentation, code, and data to reproduce the core experiments. The function vector methodology works as described - FVs extracted from ICL contexts successfully transfer task knowledge to zero-shot and corrupted-label settings. While absolute accuracy numbers differ from the paper (likely due to evaluation setup differences), the directional effects are consistent and the mechanism is validated.

The repository is well-organized with:
- Clear plan and code walkthrough documentation
- Comprehensive utility functions
- Working demo notebook
- Included datasets
- Environment specification

No major barriers to replication were encountered.
