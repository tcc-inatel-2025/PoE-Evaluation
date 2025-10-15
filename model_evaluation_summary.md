# Model Evaluation Summary

This document provides a concise interpretation of three key visualization analyses:  
**Delta Analysis**, **Rank Concordance**, and **Score Distributions**.

---

## 1. Delta Analysis (Overall vs Individual Metrics)

**Meaning:** Difference between `overall_score` and each metric.  
Green bars = Overall > Metric (tolerant or balanced metric).  
Red bars = Metric > Overall (strict metric).

| Metric | Interpretation |
|:--|:--|
| **Functional Correctness** | Closely tracks the overall score (mean delta ≈ +0.01). Main driver of performance. |
| **Efficiency** | Large negative deltas → overall evaluation tolerates inefficiency. |
| **Cyclomatic Complexity** | Mixed results; generally forgiven in the overall. Some models penalized more than others. |
| **Style** | Nearly neutral; style aligns closely with overall. |
| **LOC (Lines of Code)** | Negative deltas → longer code not strongly penalized. |

**Conclusion:** Overall score is dominated by correctness, slightly influenced by complexity and style, and largely indifferent to efficiency or LOC.

---

## 2. Rank Concordance (Spearman ρ)

Tests how well each metric’s *ranking* matches the overall model ranking.

| Metric | ρ (Spearman) | Direction | Meaning |
|:--|:--:|:--:|:--|
| **Functional Correctness** | 0.986 | + | Dominant factor — almost identical rank order. |
| **Style Score** | 0.943 | + | Strong alignment — good style correlates with high performance. |
| **Avg/Max Complexity** | -0.771 | - | Simpler code tends to rank higher overall. |
| **Efficiency** | 0.486 | + | Weak correlation — efficiency has little weight. |
| **LOC Score** | 0.143 | + | No meaningful correlation — code length irrelevant. |

**Takeaway:** Overall ranking ≈ correctness + style, modestly adjusted by complexity.

---

## 3. Overall Score Distributions (Box/Violin Plots)

Examines consistency and spread of `overall_score` for each model.

| Model | Median | Variance | Reliability | Verdict |
|:--|:--:|:--:|:--:|:--|
| **qwen2.5-coder_7b** | High | Moderate | Good | Best performer overall. |
| **codegemma_7b** | Mid-high | Wide | Volatile | Can be strong but inconsistent. |
| **deepsseek-coder_6.7b** | Mid | Moderate | Stable | Solid, balanced performer. |
| **codellama_7b** | Mid-low | High | Unstable | Unreliable output. |
| **starcoder2_7b** | Low | Low | Poor | Weak performance. |
| **stable-code_3b** | Low | Low | Poor | Consistently underperforming. |

**Summary Insight:**  
> Overall evaluation favors correctness and readable style above all else. Efficiency and brevity are largely ignored.  
> Qwen2.5 stands out as the only consistent top-tier model, while others oscillate between competence and collapse.
