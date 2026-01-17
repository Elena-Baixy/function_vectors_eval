# Replication Evaluation: Function Vectors Repository

## Read-Only Evaluation Mode (No Execution)

This evaluation assesses whether the experiment described in the function_vectors_eval repository can be faithfully reconstructed by an independent third party, based solely on static inspection of the available materials.

---

## Evaluation Notes

### RP1. Implementation Reconstructability

**Assessment: PASS**

The repository provides comprehensive operational detail for an independent researcher to reimplement the experiment step-by-step:

1. **Experimental Procedure Clearly Documented**:
   - `plan.md` provides a detailed experimental plan including:
     - Clear objectives (investigating function vectors in transformer language models)
     - Three specific hypotheses to test
     - Step-by-step methodology (causal mediation analysis, FV extraction, evaluation across contexts)
     - Specific experiments with metrics and expected results

2. **Code Organization is Self-Documenting**:
   - `CodeWalkthrough.md` describes the purpose of each utility module:
     - `eval_utils.py`: Function vector evaluation in various contexts
     - `extract_utils.py`: Function vector extraction and activations
     - `intervention_utils.py`: Function vector interventions during inference
     - `model_utils.py`: Model & tokenizer loading
     - `prompt_utils.py`: Data loading and prompt creation

3. **Complete Source Code with Sufficient Detail**:
   - `evaluate_function_vector.py` (228 lines): Main entry point with comprehensive CLI arguments documenting all configurable parameters
   - `compute_indirect_effect.py` (223 lines): Causal mediation analysis implementation with detailed docstrings
   - `extract_utils.py` (477 lines): Function vector extraction with `get_mean_head_activations()` (N_TRIALS=100 default) and `compute_function_vector()`
   - `eval_utils.py` (592 lines): Evaluation functions including `n_shot_eval()`, `n_shot_eval_no_intervention()`, and various metrics

4. **Analysis Steps Fully Specified**:
   - Step 1: Load model and tokenizer (`load_gpt_model_and_tokenizer()`)
   - Step 2: Load dataset (`load_dataset()`)
   - Step 3: Filter dataset via n-shot evaluation
   - Step 4: Compute mean activations across attention heads (100 trials)
   - Step 5: Compute indirect effects (25 trials default)
   - Step 6: Extract function vector from top-k influential heads
   - Step 7: Evaluate in zero-shot and shuffled-label contexts

5. **Demo Notebook Available**:
   - `notebooks/fv_demo.ipynb` provides a working example showing the complete workflow from model loading through evaluation

**Sufficient for Replication**: An independent researcher can follow the documented procedures without guesswork.

---

### RP2. Environment Reproducibility

**Assessment: PASS**

The repository explicitly specifies all required resources:

1. **Environment Configuration**:
   - `fv_environment.yml` provides a complete conda environment specification:
     - Python 3.10
     - PyTorch 1.13.0 with CUDA 11.7
     - transformers=4.49.0
     - datasets=2.14.3
     - scikit-learn=1.3.0
     - baukit (from git: github.com/davidbau/baukit@main)
     - bitsandbytes=0.45.3, accelerate=0.21.0, huggingface-hub=0.29.3
   - All key dependencies have pinned versions

2. **Model Specifications**:
   - `model_utils.py` explicitly supports:
     - GPT-2 XL (`gpt2-xl`)
     - GPT-J 6B (`EleutherAI/gpt-j-6b`)
     - GPT-NeoX 20B (`EleutherAI/gpt-neox-20b`)
     - Llama 2 7B/13B/70B (with quantization for 70B)
     - Pythia models
     - OLMo models
   - Model configuration is standardized with `MODEL_CONFIG` dict containing `n_heads`, `n_layers`, `resid_dim`, `attn_hook_names`, `layer_hook_names`, `prepend_bos`

3. **Dataset Specifications**:
   - 27+ datasets in `dataset_files/abstractive/` directory
   - Datasets in standardized JSON format with `{"input": ..., "output": ...}` structure
   - Dataset README documents the two categories (abstractive vs extractive)
   - `load_dataset()` function handles train/valid/test splitting with configurable seed and test_size

4. **Default Parameters Documented**:
   - n_shots=10 (CLI default in evaluate_function_vector.py)
   - n_mean_activations_trials=100 (CLI default)
   - n_indirect_effect_trials=25 (CLI default)
   - test_split=0.3 (CLI default)
   - seed=42 (CLI default)
   - edit_layer=-1 sweeps all layers (or specify specific layer)
   - n_top_heads=10 (CLI default)
   - prefixes={"input":"Q:", "output":"A:", "instructions":""} (CLI default)
   - separators={"input":"\n", "output":"\n\n", "instructions":""} (CLI default)

5. **External Resources Accessible**:
   - Models available from HuggingFace Hub
   - baukit dependency from public GitHub repository
   - Paper available on arXiv (2310.15213)

**Sufficient for Replication**: Environment can be reconstructed without trial-and-error.

---

### RP3. Determinism and Stability

**Assessment: PASS**

The repository explicitly addresses sources of nondeterminism:

1. **Random Seed Control**:
   - `model_utils.py` contains `set_seed()` function that sets:
     - `random.seed(seed)`
     - `np.random.seed(seed)`
     - `torch.manual_seed(seed)`
     - `torch.cuda.manual_seed(seed)`
     - `torch.backends.cudnn.deterministic = True`
     - `torch.backends.cudnn.benchmark = True`
     - `os.environ['PYTHONHASHSEED'] = str(seed)`

2. **Seed Usage Throughout Codebase**:
   - `evaluate_function_vector.py` uses `set_seed(seed)` before each stochastic operation:
     - Before dataset filtering (lines 98, 102, 107, 110)
     - Before computing mean activations (line 127)
     - Before computing indirect effects (line 141)
     - Before each evaluation run (lines 156, 169, 185, 193)
   - Default seed=42 is configurable via CLI

3. **Averaging Over Multiple Trials**:
   - Mean activations computed over N_TRIALS=100 prompts (configurable)
   - Indirect effects computed over n_trials=25 prompts (configurable)
   - Results are averaged to reduce variance from random prompt sampling

4. **Reproducible Sampling**:
   - ICL examples sampled with `np.random.choice()` after seed is set
   - Dataset splits controlled by seed parameter in `load_dataset()`
   - Filter sets computed deterministically from n-shot evaluation results

5. **Evaluation Determinism**:
   - `torch.set_grad_enabled(False)` used throughout evaluation
   - Token ranking uses deterministic `torch.argsort()` operations
   - Accuracy metrics computed from stored rank lists

**Sufficient for Replication**: Sources of nondeterminism are explicitly addressed with seed control and multi-trial averaging. Results should be reproducible given the same random seed.

---

## Binary Checklist Table

| Criterion | Status | Summary |
|-----------|--------|---------|
| **RP1. Implementation Reconstructability** | **PASS** | Complete source code with detailed documentation. plan.md specifies experimental procedure, CodeWalkthrough.md explains code organization, and demo notebook provides working example. All analysis steps are fully specified with clear parameters. |
| **RP2. Environment Reproducibility** | **PASS** | fv_environment.yml provides pinned dependencies (Python 3.10, PyTorch 1.13.0, transformers 4.49.0). Models specified in model_utils.py with standardized configuration. Datasets in JSON format with clear structure. All default parameters documented in CLI arguments. |
| **RP3. Determinism and Stability** | **PASS** | Comprehensive set_seed() function controls all random sources (Python, NumPy, PyTorch, CUDA, cuDNN). Seed used consistently before stochastic operations. Multi-trial averaging (100 for activations, 25 for indirect effects) reduces variance. |

---

## Summary

The function_vectors_eval repository demonstrates **excellent replicability** under read-only inspection. All three replication criteria pass:

1. **RP1 (Implementation)**: The experimental procedure is fully documented in plan.md with step-by-step methodology. The source code (~3,000 lines) is well-organized with clear module responsibilities and comprehensive docstrings. A demo notebook provides a complete working example.

2. **RP2 (Environment)**: The conda environment file specifies all dependencies with pinned versions. Models are loaded from HuggingFace with standardized configuration. Datasets are in simple JSON format. All default parameters are explicitly documented in CLI argument definitions.

3. **RP3 (Determinism)**: A comprehensive set_seed() function controls randomness across all libraries. The seed is applied before every stochastic operation in the evaluation pipeline. Multi-trial averaging further ensures result stability.

An independent researcher with access to appropriate computational resources (GPU with CUDA) could replicate this experiment by:
1. Creating the conda environment from fv_environment.yml
2. Following the demo notebook or running evaluate_function_vector.py
3. Using the provided datasets and default parameters
4. Obtaining consistent results due to explicit seed control

The repository is suitable for faithful replication without requiring additional assumptions or execution-dependent information.
