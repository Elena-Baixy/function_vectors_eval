# Documentation Evaluation Summary

## Replicator-Documentation Evaluation

**Original Repository:** `/net/scratch2/smallyan/function_vectors_eval`  
**Replicated Documentation:** `/net/scratch2/smallyan/function_vectors_eval/evaluation/replications/documentation_replication.md`  
**Evaluation Date:** 2026-01-16

---

## Results Comparison

The replicated documentation reports results for the Function Vectors experiment on two tasks (Antonym and Country-Capital) evaluated on GPT-J 6B.

### Antonym Task Results

| Context | Original (plan.md) | Replicated | Deviation |
|---------|-------------------|------------|-----------|
| Shuffled-label baseline | 39.1% | 30.0% | -9.1pp |
| Shuffled-label + FV | 90.8% | 54.0% | -36.8pp |
| Zero-shot baseline | 5.5% | 0.0% | -5.5pp |
| Zero-shot + FV | 57.5% | 26.0% | -31.5pp |

### Country-Capital Task Results

| Context | Original | Replicated | Note |
|---------|----------|------------|------|
| Zero-shot + FV | 57-72% (natural text) | 83.3% | Exceeds original range |

The replicated results show **consistent directional effects** (function vectors improve performance across all contexts) but exhibit **substantial numerical deviations** from the original paper's reported metrics. The replication documentation acknowledges these discrepancies and attributes them to:
1. Smaller evaluation sample size (50 samples vs. full test set)
2. No filtering to samples the model gets correct on clean ICL
3. Potential differences in random seed

---

## Conclusions Comparison

The replicated documentation presents conclusions that are **consistent** with the original findings:

| Original Claim | Replicated Conclusion | Status |
|---------------|----------------------|--------|
| FVs transport compact task representations | "Function vectors successfully transfer task knowledge" | ✓ Consistent |
| FVs are robust to context changes | "FVs recover performance in corrupted contexts" | ✓ Consistent |
| FVs work in zero-shot and natural text | "FVs enable zero-shot task transfer" | ✓ Consistent |
| Universal heads are effective | "Pre-computed universal heads still produce effective function vectors" | ✓ Consistent |

Both documents conclude that function vectors are a real phenomenon that enables task knowledge transfer across different prompting contexts. The replicated conclusions do not contradict or meaningfully differ from the original.

---

## External or Hallucinated Information

**No external or hallucinated information was detected.** The replicated documentation:

- Only reports results from experiments that were actually executed
- Correctly references the original paper (Todd et al., ICLR 2024)
- Accurately describes the methodology from the original repository
- Explicitly acknowledges limitations and what was NOT replicated
- Does not introduce unsupported claims or invented findings

---

## Evaluation Checklist

| Criterion | Result | Rationale |
|-----------|--------|-----------|
| **DE1. Result Fidelity** | **FAIL** | Results deviate significantly beyond 5% tolerance. Antonym shuffled-label +FV shows 54% vs 90.8% original (-36.8pp), and zero-shot +FV shows 26% vs 57.5% (-31.5pp). |
| **DE2. Conclusion Consistency** | **PASS** | All conclusions are consistent with the original. The replication confirms the core mechanism and directional effects without contradicting essential claims. |
| **DE3. No External Information** | **PASS** | No external references, invented findings, or hallucinated details were introduced. The documentation accurately represents the executed experiments. |

---

## Final Verdict

**REVISION REQUIRED**

The replicated documentation demonstrates the correct methodology and confirms the qualitative behavior of function vectors, but the quantitative results deviate substantially from the original metrics beyond the acceptable 5% tolerance threshold. The directional effects are preserved, but the absolute accuracy numbers differ significantly.

### Recommendations for Revision:
1. Increase evaluation sample size to match the full test set
2. Apply the same evaluation filtering as the original paper (if applicable)
3. Verify random seed and experimental configuration alignment with original setup
4. Re-run experiments with matched configurations to achieve closer numeric alignment
