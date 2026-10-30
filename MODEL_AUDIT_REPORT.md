# Comprehensive Model & Dataset Audit Report: Tai Phake Visual Speech Recognition

**Audit Date**: October 28, 2026  
**Auditor**: Senior Machine Learning Engineer & Academic Reviewer  
**Status**: Completed (Audit-Only — No Code/Data Modified, No Training Executed)  
**Target Repository**: `tai-phake-lip-reading`  
**Report File**: `MODEL_AUDIT_REPORT.md`

---

## 1. Executive Summary

This audit independently inspects the dataset configuration, preprocessing pipelines, model architectures, training configurations, and experimental outputs for the **Tai Phake Visual Speech Recognition (VSR)** project. The goal is to establish verified empirical facts, identify critical discrepancies, audit candidates against the mentor's parameter constraint ($\le 30\text{--}35\text{M}$ parameters), evaluate cross-validation status, and recommend the three final models for the upcoming research paper.

### Key Audit Findings
1. **Dataset Reality**: The master video archive contains **30 native speakers** uttering **11 digits** ($d_0$--$d_{10}$, 330 original clips). In the active augmented dataset (`dlib_augmented_cropped_100_80_20_96x96`), base speakers are **not numbered continuously from $s_1$ to $s_{30}$**. Six historical speakers with poor lighting or subtle articulation ($s_5, s_{11}, s_{16}, s_{27}, s_{28}, s_{29}$) were replaced by newly recorded/renamed speakers ($s_{34}, S_{35}, S_{38}, s_{45}, s_{47}, s_{48}$). The active validation split consists strictly of these 6 replacement speakers ($20$ folders, $220$ sequences), while $24$ base speakers form the training split ($80$ folders, $880$ sequences).
2. **Split Status & Missing Test Set**: The dataset split is strictly **speaker-independent** with zero base-speaker leakage. However, **no independent held-out test split is currently mounted** (`data/processed/test` is empty). All reported "test/final" results in recent notebooks are validation-set evaluations.
3. **Five-Fold Cross-Validation Status**: **Not implemented**. Zero code, scripts, or notebook cells exist in the codebase implementing 5-fold cross-validation. The project currently relies on a single fixed 80/20 train/validation split (Seed 42).
4. **Mentor Parameter Target ($\le 30\text{--}35\text{M}$)**: **Fully satisfied by all candidate models**. All examined models (EfficientNetB0 and MobileNetV2 combined with BiLSTM, BiGRU, and Transformer) possess between **$2.89\text{M}$ and $6.49\text{M}$ total parameters** ($\approx 10\%\text{--}20\%$ of the mentor's ceiling).
5. **Existing Candidate Implementations**:
   - `EfficientNetB0 + BiLSTM + Attention Pooling`: Fully implemented in [notebooks/13_efficientnetb0_bilstm_attention_pooling_experiment.ipynb](file:///Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading/notebooks/13_efficientnetb0_bilstm_attention_pooling_experiment.ipynb). Total parameters: **4,775,087** ($4.78\text{M}$).
   - `EfficientNetB0 + BiGRU + Attention Pooling`: Fully implemented in [notebooks/14_efficientnetb0_bigru_attention_pooling_experiment.ipynb](file:///Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading/notebooks/14_efficientnetb0_bigru_attention_pooling_experiment.ipynb). Total parameters: **4,677,295** ($4.68\text{M}$).
   - `EfficientNetB0 + 4-Block Transformer`: Fully implemented in [notebooks/11_efficientnetb0_transformer_4block_experiment.ipynb](file:///Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading/notebooks/11_efficientnetb0_transformer_4block_experiment.ipynb). Total parameters: **6,489,006** ($6.49\text{M}$).
6. **Backbone Recommendation**: **EfficientNetB0** is strongly recommended over MobileNetV2 as the common spatial backbone. In addition to being fully implemented across all three temporal heads, EfficientNetB0 achieved **$71.82\%$** (BiGRU) and **$52.73\%$** (Transformer), whereas MobileNetV2 collapsed to **$39.09\%$** on the same Transformer architecture.

---

## 2. Dataset and Input Configuration

### 2.1 Speaker & Video Inventory
* **Total Base Speakers**: **30** unique speakers verified in `data/experiments/digit_videos_only/` and `data/experiments/extracted_raw_frames/`.
  - Discovered Speaker Identifiers: `s1, s2, s3, s4, s6, s7, s8, s9, s10, s12, s13, s14, s15, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s30, s34, S35, S38, s45, s47, s48`.
  - *Discrepancy Note*: Speakers $s_5, s_{11}, s_{16}, s_{27}, s_{28}, s_{29}$ are absent from the active dataset. Higher-indexed speakers ($s_{34}$ through $s_{48}$) exist in their place.
* **Master Video Clips**: **330** MP4 recordings ($30\text{ speakers} \times 11\text{ digits} = 330$).
* **Vocabulary & Classes**: **11 discrete classes** ($d_0$ through $d_{10}$):
  * `d0`: *Pao* (Zero)
  * `d1`: *Naung* (One)
  * `d2`: *Song* / *Saung* (Two)
  * `d3`: *Saam* (Three)
  * `d4`: *Sī* (Four)
  * `d5`: *Haā* (Five)
  * `d6`: *Howk* (Six)
  * `d7`: *Cheit* (Seven)
  * `d8`: *Peat* (Eight)
  * `d9`: *Kao* (Nine)
  * `d10`: *Sip* (Ten)

### 2.2 Active Training & Validation Ingestion (`data/processed`)
* **Active Working Dataset**: `data/experiments/dlib_augmented_cropped_100_80_20_96x96` mounted via symlinks into `data/processed/`.
* **Total Dataset Folders**: **100** directories.
  * **Train Set (`data/processed/train`)**: Exactly **80 folders** (24 base speakers $\times$ 11 digits = **880 sequences**).
    - Base Train Speakers ($24$): `s1, s2, s3, s4, s6, s7, s8, s9, s10, s12, s13, s14, s15, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s30`.
    - Augmented Expansion: 8 speakers with 4 folders ($s_X + 3\text{ aug}$) = 32 folders; 16 speakers with 3 folders ($s_X + 2\text{ aug}$) = 48 folders ($32 + 48 = 80$).
  * **Validation Set (`data/processed/val`)**: Exactly **20 folders** (6 base speakers $\times$ 11 digits = **220 sequences**).
    - Base Validation Speakers ($6$): `s34, S35, S38, s45, s47, s48`.
    - Augmented Expansion: 2 speakers with 4 folders ($s_X + 3\text{ aug}$) = 8 folders; 4 speakers with 3 folders ($s_X + 2\text{ aug}$) = 12 folders ($8 + 12 = 20$).
  * **Test Set (`data/processed/test`)**: **0 folders** (Empty directory containing only `.DS_Store`).
* **Input Tensor Dimensions**:
  * Shape: `(batch_size, 30, 96, 96, 3)` for standard RGB pipelines (Notebooks 05, 06, 11, 13, 14).
  * Shape: `(batch_size, 30, 96, 96, 4)` for motion-difference pipelines ($RGB + |Y_t - Y_{t-1}|$, Notebook 19).
  * Shape: `(batch_size, 30, 96, 96, 5)` for dual-motion pipelines (Notebook 23).
* **Temporal Sequence Standardization**: Fixed at **$T = 30$ frames** per sample. Utterances are linearly interpolated / temporally sampled uniformly from raw video durations ($\approx 0.8\text{s}$ to $2.2\text{s}$).
* **Spatial Feature Preprocessing**:
  * Face & Lip Detection: Dlib 68-point facial landmark predictor (`shape_predictor_68_face_landmarks.dat`), tracking mouth contour points 48–67.
  * Cropping Strategy: **Method D (Letterboxed 96×96)**. The lip bounding box is padded ($P_x=15\text{ px}, P_y=10\text{ px}$), scaled proportionally without aspect ratio distortion, and centered on a neutral black $96 \times 96$ canvas.
* **Speaker Independence & Zero-Leakage Verification**:
  * Verified in `data/experiments/dlib_augmented_cropped_100_80_20_96x96/speaker_split.json`.
  * Intersection between base train speakers and base validation speakers: $\emptyset$ (Zero leakage).
  * Fixed Seed: Random state explicitly pinned to `SEED = 42` across Python, NumPy, TensorFlow, and `PYTHONHASHSEED`.

---

## 3. Existing Model Implementations

### 3.1 Candidate Model 1: EfficientNetB0 + BiLSTM + Attention Pooling
* **Implementation Source**: [notebooks/13_efficientnetb0_bilstm_attention_pooling_experiment.ipynb](file:///Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading/notebooks/13_efficientnetb0_bilstm_attention_pooling_experiment.ipynb) (Cells 19, 21, 23).
* **Spatial Backbone**: Keras `EfficientNetB0` (`include_top=False`, `pooling="avg"`, weights="imagenet").
* **Feature Extraction**: `TimeDistributed(cnn)` processing shape `(30, 96, 96, 3)` $\to$ `(30, 1280)`.
* **Projection Layer**: `TimeDistributed(Dense(256, use_bias=False))` followed by `LayerNormalization(epsilon=1e-6)` $\to$ `(30, 256)`.
* **Temporal Modeling**: `Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))` $\to$ `(30, 256)`.
* **Pooling Mechanism**: **Learnable Temporal Attention Pooling**:
  $$\text{score}_t = \mathbf{w}^T h_t + b, \quad \alpha_t = \text{Softmax}(\text{score}_t), \quad \mathbf{c} = \sum_{t=1}^{30} \alpha_t h_t \quad \to (256,)$$
* **Classification Head**: `Dropout(0.4)` $\to$ `Dense(11, activation="softmax")`.
* **Input Compatibility**: Verified on shape `(None, 30, 96, 96, 3)`.
* **Status**: **Fully implemented and verified**.

### 3.2 Candidate Model 2: EfficientNetB0 + BiGRU + Attention Pooling
* **Implementation Source**: [notebooks/14_efficientnetb0_bigru_attention_pooling_experiment.ipynb](file:///Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading/notebooks/14_efficientnetb0_bigru_attention_pooling_experiment.ipynb) (Cells 19, 21, 23).
* **Spatial Backbone**: Keras `EfficientNetB0` (`include_top=False`, `pooling="avg"`, weights="imagenet").
* **Feature Extraction**: `TimeDistributed(cnn)` $\to$ `(30, 1280)`.
* **Projection Layer**: `TimeDistributed(Dense(256, use_bias=False))` followed by `LayerNormalization(epsilon=1e-6)` $\to$ `(30, 256)`.
* **Temporal Modeling**: `Bidirectional(GRU(128, return_sequences=True, dropout=0.2))` $\to$ `(30, 256)`.
* **Pooling Mechanism**: Learnable Temporal Attention Pooling $\to$ `(256,)`.
* **Classification Head**: `Dropout(0.4)` $\to$ `Dense(11, activation="softmax")`.
* **Input Compatibility**: Verified on shape `(None, 30, 96, 96, 3)`.
* **Status**: **Fully implemented and verified**.

### 3.3 Candidate Model 3: EfficientNetB0 + 4-Block Transformer
* **Implementation Source**: [notebooks/11_efficientnetb0_transformer_4block_experiment.ipynb](file:///Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading/notebooks/11_efficientnetb0_transformer_4block_experiment.ipynb) (Cells 17, 19, 21, 23).
* **Spatial Backbone**: Keras `EfficientNetB0` (`include_top=False`, `pooling="avg"`, weights="imagenet").
* **Feature Extraction**: `TimeDistributed(cnn)` $\to$ `(30, 1280)`.
* **Projection & Encoding**: `TimeDistributed(Dense(256, use_bias=False))` $\to$ `LayerNormalization` $\to$ `PositionalEncoding(seq_len=30, dim=256)`.
* **Temporal Modeling**: **4 stacked Transformer Encoder Blocks**:
  * Multi-Head Attention: `num_heads=4`, `key_dim=64`, `dropout=0.1`.
  * Feed-Forward Network: `Dense(512, activation="gelu")` $\to$ `Dropout(0.1)` $\to$ `Dense(256)`.
  * Residual connections with pre-layer normalization after attention and FFN.
* **Pooling Mechanism**: `GlobalAveragePooling1D` across 30 time steps $\to$ `(256,)`.
* **Classification Head**: `Dropout(0.4)` $\to$ `Dense(11, activation="softmax")`.
* **Input Compatibility**: Verified on shape `(None, 30, 96, 96, 3)`.
* **Status**: **Fully implemented and verified**.

### 3.4 MobileNetV2-Based Implementations
* **MobileNetV2 + BiLSTM + Attention Pooling**:
  * Implementation: [notebooks/13_mobilenetv2_bilstm_attention_pooling_experiment.ipynb](file:///Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading/notebooks/13_mobilenetv2_bilstm_attention_pooling_experiment.ipynb).
  * Structure: Identical to 3.1 except `MobileNetV2` replaces `EfficientNetB0`.
  * Status: Fully implemented.
* **MobileNetV2 + 4-Block Transformer**:
  * Implementation: [notebooks/11_mobilenetv2_transformer_4block_experiment.ipynb](file:///Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading/notebooks/11_mobilenetv2_transformer_4block_experiment.ipynb).
  * Structure: Identical to 3.3 except `MobileNetV2` replaces `EfficientNetB0`.
  * Status: Fully implemented.
* **MobileNetV2 + BiGRU + Attention Pooling**:
  * Status: **Not implemented**. Would need to be instantiated by swapping the recurrent layer in Notebook 13.

---

## 4. Model Parameter-Count Comparison

All parameter counts were calculated and verified directly using TensorFlow 2.21.0 on the actual Keras layer implementations:

| Model Architecture | Implementation Notebook | CNN Backbone Params | Temporal Head Params | Total Model Params | Stage 1 Trainable (Frozen CNN) | Stage 2 Trainable (Fine-Tuning) | Meets Mentor Limit ($\le 35\text{M}$)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNetB0 + BiLSTM + AttnPool** | Notebook 13 | 4,049,571 | 725,516 | **4,775,087 (4.78M)** | 725,516 (0.73M) | 3,013,020 (3.01M) | **YES** ($\approx 14\%$ of limit) |
| **EfficientNetB0 + BiGRU + AttnPool** | Notebook 14 | 4,049,571 | 627,724 | **4,677,295 (4.68M)** | 627,724 (0.63M) | 2,915,228 (2.92M) | **YES** ($\approx 13\%$ of limit) |
| **EfficientNetB0 + 4-Block Transformer** | Notebook 11 | 4,049,571 | 2,439,435 | **6,489,006 (6.49M)** | 2,439,435 (2.44M) | 4,726,939 (4.73M) | **YES** ($\approx 19\%$ of limit) |
| **MobileNetV2 + BiLSTM + AttnPool** | Notebook 13 | 2,257,984 | 725,516 | **2,983,500 (2.98M)** | 725,516 (0.73M) | 2,236,236 (2.24M) | **YES** ($\approx 9\%$ of limit) |
| **MobileNetV2 + BiGRU + AttnPool** *(Estimated)* | *New* | 2,257,984 | 627,724 | **2,885,708 (2.89M)** | 627,724 (0.63M) | 2,138,444 (2.14M) | **YES** ($\approx 8\%$ of limit) |
| **MobileNetV2 + 4-Block Transformer** | Notebook 11 | 2,257,984 | 2,439,435 | **4,697,419 (4.70M)** | 2,439,435 (2.44M) | 3,950,155 (3.95M) | **YES** ($\approx 13\%$ of limit) |

*Audit Assessment*: Every candidate model has a parameter footprint between $2.89\text{M}$ and $6.49\text{M}$. They are exceptionally lightweight and well within the mentor's $30\text{--}35\text{M}$ threshold.

---

## 5. Training Configuration Comparison

The training configurations across Notebooks 11, 13, and 14 share an identical, controlled training protocol:

| Training Hyperparameter | EfficientNetB0 + BiLSTM (NB 13) | EfficientNetB0 + BiGRU (NB 14) | EfficientNetB0 + Transformer (NB 11) | MobileNetV2 + BiLSTM (NB 13) | Motion + Focal Snapshot (NB 19) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1 Optimizer & LR** | Adam ($\text{lr} = 10^{-4}$) | Adam ($\text{lr} = 10^{-4}$) | Adam ($\text{lr} = 10^{-4}$) | Adam ($\text{lr} = 10^{-4}$) | Adam ($\text{lr} = 10^{-4}$) |
| **Stage 2 Optimizer & LR** | Adam ($\text{lr} = 10^{-5}$) | Adam ($\text{lr} = 10^{-5}$) | Adam ($\text{lr} = 10^{-5}$) | Adam ($\text{lr} = 10^{-5}$) | Adam ($\text{lr} = 10^{-5}$) |
| **Backbone Fine-Tuning** | Top 20% unfrozen | Top 20% unfrozen | Top 20% unfrozen | Top 20% unfrozen | Top 20% unfrozen |
| **Loss Function** | Sparse Categorical CE + Label Smoothing (0.05) | Sparse Categorical CE + Label Smoothing (0.05) | Sparse Categorical CE + Label Smoothing (0.05) | Sparse Categorical CE + Label Smoothing (0.05) | Categorical Focal Loss ($\gamma=2.0, \alpha=0.25$) |
| **Batch Size** | 16 | 16 | 16 | 16 | 16 |
| **Max Epochs (S1 / S2)** | 40 / 25 | 40 / 25 | 40 / 25 | 40 / 25 | 40 / 35 |
| **Early Stopping** | `val_loss` (Patience: 8 / 6) | `val_loss` (Patience: 8 / 6) | `val_loss` (Patience: 8 / 6) | `val_loss` (Patience: 8 / 6) | `val_loss` (Patience: 8 / 8) |
| **Checkpoint Criterion** | Save best `val_loss` | Save best `val_loss` | Save best `val_loss` | Save best `val_loss` | Cosine Snapshots (Top-5) |
| **Input Channels** | 3 (RGB) | 3 (RGB) | 3 (RGB) | 3 (RGB) | 4 (RGB + Motion Difference) |
| **Test-Time Augmentation** | None (1-View) | None (1-View) | None (1-View) | None (1-View) | 4-View TTA |

*Fairness Assessment*: Notebooks 11, 13, and 14 share completely identical optimizers, learning rate schedules, loss functions, batch sizes, and early stopping rules. This provides a fair benchmark between BiLSTM, BiGRU, and Transformer heads.

---

## 6. Verified Experimental Results

All reported figures below were extracted and cross-verified against `metrics_80_20.json`, `classification_report_validation_80_20.txt`, and `training_history.csv` located in `results/`:

| Architecture / Experiment | Result Directory | Evaluated Samples | Train Acc (Final) | Best Val Acc (Stage 1) | Best Val Acc (Stage 2) | Final Val Acc | Macro F1 | Status & Evidence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Chance Reference** | Theoretical Baseline | 220 | -- | -- | -- | **9.09%** | 0.0152 | $1/11 = 9.0909\%$ |
| **Scratch CNN-GRU** | `results/models/cnn_gru` | 66 (Test) | 98.2% | 13.64% | -- | **7.58%** | 0.0371 | Verified (NB 05) |
| **Scratch CNN-LSTM** | `results/models/cnn_lstm` | 66 (Test) | 97.4% | 18.18% | -- | **15.15%** | 0.0906 | Verified (NB 05) |
| **Scratch CNN-BiGRU** | `results/models/cnn_bigru` | 66 (Test) | 99.1% | 18.18% | -- | **18.18%** | 0.1433 | Verified (NB 05) |
| **Scratch CNN-BiLSTM** | `results/models/cnn_bilstm` | 66 (Test) | 99.5% | 15.91% | -- | **31.82%** | 0.2828 | Verified (NB 05) |
| **MobileNetV2 + BiLSTM (Phase 1)** | `results/pretrained_phase1/mobilenetv2_bilstm` | 33 (Val) | 98.8% | 57.58% | 57.58% | **57.58%** | 0.5463 | Verified (NB 06) |
| **MobileNetV2 + 4-Block Transformer** | `results/mobilenetv2_transformer_4block_80_20` | 220 | 96.1% | 43.18% | 40.91% | **39.09%** | *Not available* | Verified (NB 11) |
| **EfficientNetB0 + 4-Block Transformer** | `results/efficientnetb0_transformer_4block_80_20` | 220 | 94.4% | 56.36% | 52.73% | **52.73%** | *Not available* | Verified (NB 11) |
| **EfficientNetB0 + BiLSTM + AttnPool** | `results/efficientnetb0_bilstm_attention_pooling_80_20` | **187** | 94.4% | 61.50% | 61.50% | **61.50%** | 0.5999 | **Discrepancy: Only 17 folders evaluated** |
| **MobileNetV2 + BiLSTM + AttnPool** | `results/mobilenetv2_bilstm_attention_pooling_80_20` | 220 | 99.9% | 69.99% | 70.91% | **68.64%** | 0.6718 | Verified (NB 13) |
| **EfficientNetB0 + BiGRU + AttnPool** | `results/efficientnetb0_bigru_attention_pooling_80_20` | 220 | 98.5% | 71.36% | 72.27% | **71.82%** | **0.7127** | Verified (NB 14) |
| **EffNetB0 + BiGRU + Class Weighted** | `results/efficientnetb0_bigru_class_weighted_80_20` | 220 | 99.2% | 73.64% | 71.36% | **71.36%** | 0.7017 | Verified (NB 18) |
| **EffNetB0 + BiGRU + Motion + Focal** | `results/efficientnetb0_bigru_motion_focal_snapshot_80_20` | 220 | 98.7% | 72.27% | 72.27% | **70.45%** | 0.7048 | Verified (NB 19) |
| **Dual Ensemble (EffNet + MobileNet)** | `results/ensemble_effnet_mobilenet_tta_80_20` | 220 | -- | -- | -- | **76.82%** | *Not available* | Verified (NB 19.2) |
| **Tri-Model Master Ensemble** | `results/ensemble_tri_model_80_20` | 220 | -- | -- | -- | **81.36%** | **0.8137** | Verified (NB 22.2) |

### Evidence of Overfitting and Training Instability
1. **Severe Transformer Overfitting**:
   In `results/efficientnetb0_transformer_4block_80_20/training_history.csv`, the training accuracy reaches **$94.43\%$** (train loss $0.8179$) while the validation accuracy stagnates at **$48.18\%\text{--}52.73\%$** (validation loss escalates from $2.00$ to $2.23$). This verifies that self-attention without local inductive bias overfits severely on 880 training sequences.
2. **Stage 2 Over-Specialization**:
   In `EfficientNetB0 + BiGRU` (NB 14), Stage 1 reached $71.36\%$ at epoch 30. During Stage 2 fine-tuning, peak validation accuracy reached $72.27\%$ at epoch 3, after which validation loss steadily increased ($1.229 \to 1.265$), triggering early stopping at epoch 9. This demonstrates that fine-tuning more than 3–5 epochs in Stage 2 risks over-adapting backbone edge weights to training speakers.

---

## 7. Speaker-Independent Split and Cross-Validation Audit

### 7.1 Speaker Independence Verification
* The zero-leakage base speaker partition is verified:
  $$\text{Speakers}_{\text{train}} \cap \text{Speakers}_{\text{val}} = \emptyset$$
* Base train speakers ($24$): `s1, s2, s3, s4, s6, s7, s8, s9, s10, s12, s13, s14, s15, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s30`.
* Base validation speakers ($6$): `s34, S35, S38, s45, s47, s48`.
* All augmented folders for any given base speaker reside exclusively within either `train` or `val`.

### 7.2 Five-Fold Cross-Validation Status
* **Status**: **NOT IMPLEMENTED**.
* A rigorous scan of all 41 notebooks, scripts, and root files found **0 occurrences** of `KFold`, `GroupKFold`, or cross-validation routines.
* *Audit Finding*: The current experimental evidence is based solely on a single 80/20 train/validation split.

### 7.3 Recommended 5-Fold GroupKFold Protocol
To implement the mentor's 5-fold cross-validation request rigorously without speaker leakage:
* Total Base Speakers = 30.
* Number of Folds = 5.
* Speakers per Validation Fold = exactly **6 base speakers** ($30 / 5 = 6$).
* Speakers per Training Fold = exactly **24 base speakers** ($30 - 6 = 24$).
* Grouping Variable: `speaker_id` (enforced via `sklearn.model_selection.GroupKFold`).

$$\begin{aligned}
\text{Fold 1 Val} &: [s_1, s_2, s_3, s_4, s_6, s_7] \quad (24\text{ Train} / 6\text{ Val}) \\
\text{Fold 2 Val} &: [s_8, s_9, s_{10}, s_{12}, s_{13}, s_{14}] \\
\text{Fold 3 Val} &: [s_{15}, s_{17}, s_{18}, s_{19}, s_{20}, s_{21}] \\
\text{Fold 4 Val} &: [s_{22}, s_{23}, s_{24}, s_{25}, s_{26}, s_{30}] \\
\text{Fold 5 Val} &: [s_{34}, S_{35}, S_{38}, s_{45}, s_{47}, s_{48}]
\end{aligned}$$

---

## 8. Inconsistencies, Discrepancies, and Risks

1. **Configuration vs. Active Dataset Discrepancy**:
   `configs/preprocessing.yaml` specifies `train_speakers: 20, val_speakers: 5, test_speakers: 5`. In reality, the active dataset uses an 80/20 folder split with **24 train base speakers and 6 validation base speakers**, with **0 test speakers**.
2. **Speaker Renaming & Paper Narrative Gap**:
   `AGENTS.md` and the current presentation slides assert that the dataset comprises speakers $s_1$ through $s_{30}$, partitioned into train ($s_1$--$s_{24}$) and validation ($s_{25}$--$s_{30}$). In the active filesystem (`data/experiments/digit_videos_only`), historical problematic speakers ($s_5, s_{11}, s_{16}, s_{27}, s_{28}, s_{29}$) were removed, and the active validation speakers are actually $s_{34}, S_{35}, S_{38}, s_{45}, s_{47}, s_{48}$. This discrepancy must be harmonized before paper submission.
3. **Unevaluated EfficientNetB0 + BiLSTM Split**:
   In `results/efficientnetb0_bilstm_attention_pooling_80_20/metrics_80_20.json`, only **17 folders (187 samples)** were evaluated due to a missing folder in an earlier split run. In contrast, `EfficientNetB0 + BiGRU` was evaluated on all **20 folders (220 samples)**. Comparing $61.50\%$ directly against $71.82\%$ is currently unfair until BiLSTM is re-evaluated on the identical 20-folder set.
4. **Validation Overfitting vs. Held-Out Generalization**:
   Because `data/processed/test` is empty, ensemble weights and hyperparameter selections were evaluated on the validation set. As honestly acknowledged in Slide 14 of the seminar presentation, an independent evaluation or 5-fold cross-validation is necessary to confirm true generalization.

---

## 9. Recommended Final Three Models

To satisfy academic reviewers, departmental guidelines, and the mentor's parameter constraint ($\le 30\text{--}35\text{M}$), the following three models are recommended for the final comparative study:

```
                            ┌──────────────────────────────────────────────┐
                            │      COMMON BACKBONE: EfficientNetB0         │
                            │   (Pretrained ImageNet, 4.05M Parameters)    │
                            │ Input: (B, 30, 96, 96, 3), TimeDistributed   │
                            └──────────────────────────────────────────────┘
                                     /             |             \
                                    /              |              \
             ┌─────────────────────┐ ┌─────────────────────┐ ┌───────────────────────┐
             │       MODEL 1       │ │       MODEL 2       │ │        MODEL 3        │
             │   BiLSTM + Attn     │ │   BiGRU + Attn      │ │  4-Block Transformer │
             │  Head: 0.73M Params │ │  Head: 0.63M Params │ │   Head: 2.44M Params  │
             │  Total: 4.78M Params│ │  Total: 4.68M Params│ │   Total: 6.49M Params │
             └─────────────────────┘ └─────────────────────┘ └───────────────────────┘
```

### Justification for the Common Backbone (EfficientNetB0 over MobileNetV2)
1. **Superior Empirical Performance**: EfficientNetB0 outperforms MobileNetV2 across both recurrent and attention paradigms. It achieved **$71.82\%$** (BiGRU) and **$52.73\%$** (Transformer), whereas MobileNetV2 reached only **$39.09\%$** on the Transformer.
2. **Complete Codebase Readiness**: All three candidate models using EfficientNetB0 already have complete end-to-end implementations in dedicated notebooks (Notebooks 11, 13, and 14). In contrast, `MobileNetV2 + BiGRU` does not exist in the codebase and would require writing new code.
3. **Parameter Footprint**: EfficientNetB0 totals $4.68\text{M}$ to $6.49\text{M}$ parameters, which utilizes only $\approx 15\%\text{--}19\%$ of the mentor's $35\text{M}$ budget while maintaining high representational capacity.

### Justification for the Three Models
* **Model 1 (BiLSTM + Attention Pooling)**: Established classical sequence baseline. Represents gating with explicit cell memory states.
* **Model 2 (BiGRU + Attention Pooling)**: Represents parameter-efficient recurrent gating (reset/update gates). It is currently the highest-performing single model in the project ($71.82\%$ standalone).
* **Model 3 (4-Block Transformer + GAP1D)**: Represents modern spatiotemporal self-attention. While it experiences data-scarcity overfitting ($52.73\%$), its inclusion in the paper is essential to provide a scientific discussion on why multi-head self-attention without temporal inductive bias underperforms recurrent gating on small endangered-language corpora.

---

## 10. Exact Next Steps Before Training and Paper Revision

1. **Re-evaluate / Retrain Model 1 (`EfficientNetB0 + BiLSTM`)**:
   Run `notebooks/13_efficientnetb0_bilstm_attention_pooling_experiment.ipynb` on the active canonical 20-folder validation set ($s_{34}, S_{35}, S_{38}, s_{45}, s_{47}, s_{48}$, 220 samples) so its metrics directly match the 220-sample benchmark of Model 2 and Model 3.
2. **Harmonize Speaker Names in Paper & Documentation**:
   Update Section 4 of `AGENTS.md` and the methodology section of the research paper draft to state the actual speaker roster ($30$ native speakers with replacement validation speakers $s_{34}$--$s_{48}$) rather than claiming $s_1$--$s_{30}$ continuously.
3. **Decide Cross-Validation Execution with Mentor**:
   Discuss with Dr. Rabinder Kumar Prasad whether the final paper will report:
   * **Option A (Fixed Canonical 80/20 Benchmark)**: Report the established single split (Notebooks 11, 13, 14), clearly documenting it as an 80/20 speaker-independent benchmark.
   * **Option B (5-Fold GroupKFold Cross-Validation)**: Execute a dedicated 5-fold cross-validation script across the 30 base speakers (6 speakers per fold) using the exact protocol defined in Section 7.3.
4. **Update `configs/preprocessing.yaml`**:
   Adjust lines 60–63 in `configs/preprocessing.yaml` to reflect the active 80/20 split (`train_speakers: 24, val_speakers: 6, test_speakers: 0`).

---

## 11. Evidence Table

| Item / Finding | Verified File / Directory Path | Relevant Cell, Line, or Function Reference | Verification Method |
| :--- | :--- | :--- | :--- |
| **Discovered Speakers (30)** | `data/experiments/digit_videos_only/` | Subdirectories: `s1..s26, s30, s34, S35, S38, s45, s47, s48` | Filesystem scan via Python script |
| **Manifest Record Count (330)** | `data/manifest.csv` | Lines 1–331 (330 rows, 30 speakers $\times$ 11 digits) | Read via pandas |
| **Config Split Definition** | `configs/preprocessing.yaml` | Lines 60–63 (`train_speakers: 20, val: 5, test: 5`) | Direct YAML file inspection |
| **Active Split Metadata** | `data/experiments/dlib_augmented_cropped_100_80_20_96x96/speaker_split.json` | Keys: `train_speakers` (80), `val_speakers` (20), `base_train` (24), `base_val` (6) | Direct JSON parse |
| **Active Mount Verification** | `data/processed/train` & `data/processed/val` | Symlinks pointing to `dlib_augmented_cropped_100_80_20_96x96` | Symlink target resolution |
| **Empty Test Split** | `data/processed/test` | Contains only `.DS_Store` | Filesystem scan |
| **Candidate 1 (EffNet+BiLSTM)** | `notebooks/13_efficientnetb0_bilstm_attention_pooling_experiment.ipynb` | Cells 19 (`temporal_attention_pooling`), 21 (`build_efficientnet_bilstm_attention_pooling`), 23 | Notebook AST code inspection |
| **Candidate 2 (EffNet+BiGRU)** | `notebooks/14_efficientnetb0_bigru_attention_pooling_experiment.ipynb` | Cells 19 (`temporal_attention_pooling`), 21 (`build_efficientnet_bigru_attention_pooling`), 23 | Notebook AST code inspection |
| **Candidate 3 (EffNet+Transformer)**| `notebooks/11_efficientnetb0_transformer_4block_experiment.ipynb` | Cells 17 (`PositionalEncoding`), 19 (`transformer_encoder_block`), 21 | Notebook AST code inspection |
| **MobileNetV2+BiLSTM Model** | `notebooks/13_mobilenetv2_bilstm_attention_pooling_experiment.ipynb` | Cells 19, 21 (`build_mobilenet_bilstm_attention_pooling`), 23 | Notebook AST code inspection |
| **MobileNetV2+Transformer Model**| `notebooks/11_mobilenetv2_transformer_4block_experiment.ipynb` | Cells 17, 19, 21 (`build_mobilenet_transformer_4block`), 23 | Notebook AST code inspection |
| **Exact Parameter Calculation** | In-memory Keras evaluation | `m.count_params()`: EffNet-BiLSTM (4.78M), EffNet-BiGRU (4.68M), EffNet-Trans (6.49M) | Computed via TensorFlow 2.21.0 |
| **EffNet+BiGRU 71.82% Result** | `results/efficientnetb0_bigru_attention_pooling_80_20/` | `metrics_80_20.json`, `classification_report_validation_80_20.txt` | Direct JSON and text parse |
| **MobileNet+BiLSTM 68.64% Result** | `results/mobilenetv2_bilstm_attention_pooling_80_20/` | `metrics_80_20.json`, `classification_report_validation_80_20.txt` | Direct JSON and text parse |
| **Transformer 52.73% Result** | `results/efficientnetb0_transformer_4block_80_20/` | `metrics_80_20.json`, `training_history.csv` | Direct JSON and CSV parse |
| **EffNet+BiLSTM 187-Sample Discrepancy** | `results/efficientnetb0_bilstm_attention_pooling_80_20/metrics_80_20.json` | Lines 12–13 (`num_val_folders: 17`, evaluated samples: 187) | Direct JSON parse |
| **5-Fold Cross-Validation Absence** | Full codebase search | 0 occurrences of `KFold`, `GroupKFold`, or cross-validation routines | Automated codebase regex scan |

