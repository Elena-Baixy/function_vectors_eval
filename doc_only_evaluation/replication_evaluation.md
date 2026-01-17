# Replication Evaluation — Documentation Only

## Evaluation Overview

This evaluation assesses whether the documentation provided for the "Function Vectors in Large Language Models" research contains sufficient information for an independent researcher to reproduce the experiment and its results, without access to the original code.

**Documentation Evaluated:** `/net/scratch2/smallyan/function_vectors_eval/documentation.pdf`

---

## Evaluation Notes

### RP1. Implementation Reconstructability

**Assessment: PASS**

The documentation provides extensive detail on the experimental procedure, analysis steps, and evaluation process:

1. **Method Description (Section 2):**
   - Section 2.1 describes the motivating observation with the mathematical formulation for computing average activations and performing interventions
   - Section 2.2 provides formal mathematical definitions for the autoregressive transformer model, hidden states, attention head outputs, ICL prompts, and the decoder mapping
   - Section 2.3 details the causal mediation analysis procedure with explicit equations for:
     - Task-conditioned mean activations (Equation 2)
     - Causal Indirect Effect (CIE) measurement (Equation 3)
     - Average Indirect Effect (AIE) calculation (Equation 4)
     - Function Vector construction (Equation 5)

2. **Experimental Procedure:**
   - Appendix C provides extensive experimental details including:
     - Number of prompts used (|Pt| = 100 clean prompts, |P̃t| = 25 corrupted prompts)
     - Number of attention heads (|A| = 10 for GPT-J, scaled proportionally for larger models)
     - Specific layers for FV evaluation (~|L|/3)
     - Default prompt template format (Equation 15)
     - 20 additional ICL templates tested (Table 8)

3. **Evaluation Metrics:**
   - Top-1 accuracy measurement clearly defined
   - Filtering criteria for test sets explained (only prompts where LM predicts correct output with 10-shot ICL)
   - Natural text portability evaluation with regex matching

4. **Analysis Steps:**
   - Decoding vocabulary analysis (Section 3.2) with reconstruction optimization (Equation 6)
   - Vector algebra composition experiments (Section 3.3) with explicit formulas (Equations 7-8)

The documentation is operational rather than high-level, with step-by-step procedures that could be reimplemented.

---

### RP2. Environment and Resource Specification

**Assessment: PASS**

The documentation explicitly specifies required models, datasets, and key dependencies:

1. **Models (Table 1):**
   - GPT-J: EleutherAI/gpt-j-6b, 6B parameters, 28 layers, 16 heads
   - GPT-NeoX: EleutherAI/gpt-neox-20b, 20B parameters, 44 layers, 64 heads
   - Llama 2 (7B): meta-llama/Llama-2-7b-hf
   - Llama 2 (13B): meta-llama/Llama-2-13b-hf
   - Llama 2 (70B): meta-llama/Llama-2-70b-hf
   - All accessed via HuggingFace implementations (Wolf et al., 2020)

2. **Datasets (Section 3, Appendix E, Table 9):**
   - Antonym/Synonym: Nguyen et al. (2017) - 2,398/2,881 pairs after filtering
   - Translation datasets: Conneau et al. (2017) - 4,705/5,154/5,200 pairs for French/German/Spanish
   - Sentiment: SST-2 (Socher et al., 2013) via Honovich et al. (2023) - 1,167 entries
   - CommonsenseQA: Talmor et al. (2019)
   - AG News: Zhang et al. (2015)
   - CoNLL-2003: Sang & De Meulder (2003)
   - Additional datasets constructed with ChatGPT or from Hernandez et al. (2023b)

3. **Implementation Framework:**
   - HuggingFace Transformers library explicitly cited
   - Attention head formulation following Elhage et al. (2021)

4. **Resource Availability:**
   - Open-source code and data noted as available at functions.baulab.info (footnote page 1)

---

### RP3. Determinism and Stability

**Assessment: PASS**

The documentation explicitly addresses sources of randomness and explains how variance is handled:

1. **Random Seeds:**
   - Results reported over 5 random seeds with standard deviations (e.g., Table 2, Table 3)
   - Example: "90.8 ± 0.9%" format used consistently

2. **Sampling Considerations:**
   - ICL prompt construction uses random sampling of exemplars
   - Shuffled-label prompts created by random shuffling of labels
   - Standard deviations reported across all main results

3. **Stability Analysis:**
   - Results shown across multiple model architectures (GPT-J, GPT-NeoX, Llama 2 family)
   - Results shown across multiple model sizes (7B to 70B parameters)
   - Consistent patterns observed across tasks and models (Figures 4, 26)
   - Layer-wise analysis shows reproducible patterns (peak at early-middle layers, dropoff at ~2/3 depth)

4. **Variance Handling:**
   - All main experimental results include ± standard deviation
   - Multiple tasks evaluated (40+ tasks) to demonstrate generality
   - Filtered test sets defined clearly to ensure reproducible comparisons

---

## Binary Checklist Table

| Criterion | Status | Summary |
|-----------|--------|---------|
| **RP1: Implementation Reconstructability** | **PASS** | Complete mathematical formulation provided with explicit equations, step-by-step procedures, hyperparameters, prompt templates, and evaluation metrics |
| **RP2: Environment and Resource Specification** | **PASS** | All models specified with HuggingFace IDs, dataset sources cited with sizes, implementation framework identified |
| **RP3: Determinism and Stability** | **PASS** | Results reported with standard deviations over 5 random seeds, consistent patterns across models and tasks |

---

## Summary

The documentation for "Function Vectors in Large Language Models" provides **sufficient information for independent replication**. The paper includes:

1. **Complete methodology** with formal mathematical definitions and explicit equations for all key computations
2. **Detailed experimental setup** including model specifications, dataset sources and preprocessing, hyperparameters, and prompt templates
3. **Comprehensive variance analysis** with standard deviations across multiple random seeds and consistent results across different model architectures and sizes

An independent researcher could reconstruct the experiments using the documented procedures, specified resources, and evaluation protocols. The availability of open-source code and data (as noted in the paper) would further facilitate replication, though the documentation alone provides sufficient detail for reimplementation.

**Overall Replicability Assessment: HIGH**

All three criteria (RP1, RP2, RP3) receive PASS ratings, indicating the documentation meets the standard for independent replication.
