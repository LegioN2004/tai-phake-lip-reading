# Experiment Notes: 15_ensemble_bilstm_bigru_transformer

- **Notebook**: [`notebooks/15_ensemble_bilstm_bigru_transformer.ipynb`](../../notebooks/15_ensemble_bilstm_bigru_transformer.ipynb)
- **Dataset Mode**: `80_20` (24 base train speakers / 6 base validation speakers, 220 validation utterances)
- **Goal**: Late-fusion ensembling across 3 distinct temporal architectures with identical `EfficientNetB0` visual frontends.

---

## 1. Motivation & Background
In limited-data isolated lip reading (~330 baseline samples), individual model families make complementary errors due to their inductive biases:
1. **EfficientNetB0 + BiLSTM + Attention Pooling** (~60.5% val acc): Best on subtle phoneme/viseme boundaries (e.g. $d_7$).
2. **EfficientNetB0 + BiGRU + Attention Pooling** (~60.0% val acc): Lower gating complexity, stronger on rapid articulatory shifts (e.g. $d_2$).
3. **EfficientNetB0 + 4-Block Transformer** (~52.7% val acc): Global self-attention modeling, strong on longer co-articulation patterns (e.g. $d_{10}$).

Instead of training complex new backbones, **soft probability blending** across the three models mitigates individual variance and boosts top-1 classification accuracy.

---

## 2. Models & Checkpoints Used
All models use the same Method D preprocessing (`96x96` cropped lips, uniform 30-frame sampling):
| Model | Architecture | Stage 2 Checkpoint Location |
| :--- | :--- | :--- |
| **Model 1** | EfficientNetB0 + BiLSTM (128 units/dir) + Temporal Attention Pooling | `results/efficientnetb0_bilstm_attention_pooling_80_20/efficientnetb0_bilstm_attention_pooling_stage2_best.weights.h5` |
| **Model 2** | EfficientNetB0 + BiGRU (128 units/dir) + Temporal Attention Pooling | `results/efficientnetb0_bigru_attention_pooling_80_20/efficientnetb0_bigru_attention_pooling_stage2_best.weights.h5` |
| **Model 3** | EfficientNetB0 + 4-Block Pre-LN Transformer (4 heads, 256 dim) | `results/efficientnetb0_transformer_4block_80_20/efficientnetb0_transformer_4block_stage2_best.weights.h5` |

---

## 3. Ensemble Strategies Implemented

### A. Uniform Probability Averaging (1/3 each)
$$\mathbf{P}_{\text{uniform}} = \frac{1}{3}\mathbf{P}_{\text{lstm}} + \frac{1}{3}\mathbf{P}_{\text{gru}} + \frac{1}{3}\mathbf{P}_{\text{trans}}$$
- Evaluates equal voting between recurrent champions and self-attention.
- Also tests pairwise blends (BiLSTM+BiGRU, BiLSTM+Transformer, BiGRU+Transformer).

### B. Simplex Grid Search for Optimal Ensembling Weights
$$\mathbf{P}_{\text{optimal}} = w_{\text{lstm}}\mathbf{P}_{\text{lstm}} + w_{\text{gru}}\mathbf{P}_{\text{gru}} + w_{\text{trans}}\mathbf{P}_{\text{trans}}$$
$$\text{subject to } w_1 + w_2 + w_3 = 1.0, \quad w_i \ge 0$$
- Searches simplex weights in $0.05$ increments to locate the empirical upper-bound performance achievable via late fusion.

---

## 4. Diagnostics & Artifacts Produced
- **Per-Class Comparison Table**: Inspects raw accuracy counts per digit ($d_0$ through $d_{10}$) across individual models vs ensemble to verify error correction.
- **Per-Speaker Validation Diagnostic**: Calculates accuracy for each of the 6 validation base speakers to detect if specific speakers suffer from cropping artifacts.
- **Confusion Matrix Heatmaps**: Side-by-side comparison of best individual champion vs ensemble.
- **Saved Artifacts**:
  - `results/ensemble_bilstm_bigru_transformer_80_20/per_class_ensemble_comparison.csv`
  - `results/ensemble_bilstm_bigru_transformer_80_20/per_speaker_validation_diagnostics.csv`
  - `results/ensemble_bilstm_bigru_transformer_80_20/confusion_matrix_ensemble_comparison_80_20.png`
  - `results/ensemble_bilstm_bigru_transformer_80_20/validation_ensemble_predictions_80_20.csv`
  - `results/ensemble_bilstm_bigru_transformer_80_20/metrics_ensemble_80_20.json`
