# AGENTS.md — Agent & Developer Operating Manual: Tai Phake Visual Speech Recognition

This document serves as the persistent system specification, technical manual, operational protocol, and architectural guide for AI coding agents and human researchers working on the **Tai Phake Visual Speech Recognition (VSR)** repository.

---

## 1. Project Mission & Linguistic Context

* **Language**: Tai Phake (an endangered Tai-Kadai language spoken primarily in Assam and Arunachal Pradesh, India).
* **Objective**: Word-level Visual Speech Recognition (silent lip reading) of spoken numerical digits (`d0` through `d10`) from video sequences without acoustic audio.
* **Input**: Facial/mouth region-of-interest (ROI) video sequences (temporal length normalized to $T = 30$ frames).
* **Vocabulary (11 Classes)**:
  * `d0`: **Pau** (Zero)
  * `d1`: **Nung** (One)
  * `d2`: **Saung** (Two)
  * `d3`: **Sam** (Three)
  * `d4`: **Si** (Four)
  * `d5`: **Ha** (Five)
  * `d6`: **Hok** (Six)
  * `d7`: **Chit** (Seven)
  * `d8`: **Pet** (Eight)
  * `d9`: **Kao** (Nine)
  * `d10`: **Sip** (Ten)

---

## 2. Dataset Architecture & Storage Protocols

The project maintains a strict separation between raw master media and version-controlled metadata/frames:

### Two-Tier Storage Architecture
1. **Master Video Archive (Google Drive / Remote Storage)**:
   * 330 original and trimmed MP4 video clips (30 speakers `s1`–`s30` × 11 digits `d0`–`d10`).
   * **Strict Git Rule**: `*.mp4`, `*.avi`, `*.mov` files must **NEVER** be committed to GitHub (enforced by `.gitignore`).
2. **Local Frame Dataset (`data/digit/`)**:
   * Frames extracted sequentially as JPEGs (`frame_0001.jpg` … `frame_XXXX.jpg`) in-situ under `data/digit/<speaker>/<digit>/`.
   * Sequential image frames are tracked in Git as the visual ground truth.
3. **Preprocessed & Augmented Experiment Datasets (`data/experiments/`)**:
   * Isolated experiment directories (e.g. `dlib_augmented_cropped_100_80_20_96x96`, `mediapipe_lips_dataset_96x96`, `mediapipe_lips_dataset_112x112`).
   * Active training set is mounted into `data/processed/` via symlink switching scripts.

---

## 3. Preprocessing, Cropping & Face Landmark Methodologies

### 3.1 Face & Mouth Localization Libraries
* **Dlib**:
  * **Face Detector**: `dlib.get_frontal_face_detector()` (HOG + Linear SVM).
  * **Landmarks**: `models/dlib/shape_predictor_68_face_landmarks.dat` (Kazemi & Sullivan 68-point ensemble).
  * **Lip ROI**: Points 48 through 67.
* **Google MediaPipe**:
  * **Face & Mesh Detector**: `models/mediapipe/face_landmarker.task` (MediaPipe FaceLandmarker, 478 3D points).
  * **Lip ROI**: 40 canonical contour landmarks (`FACE_LANDMARKS_LIPS`).
* **OpenCV (`cv2`)**: Used strictly for affine warps, letterboxing, filtering, and colorspace conversions. **OpenCV is NOT used as a face detector** (no Haar Cascades, no YuNet, no cv2.dnn).

### 3.2 Crop Methods
* **Method A**: Adaptive tight bounding box around lips with fixed margin padding.
* **Method B**: Expanded square face-context crop.
* **Method C**: Adaptive 2:1 aspect-ratio lip crop with reflection border padding.
* **Method D (Standard 96×96 Letterbox)**:
  * Proportional aspect-preserving scaling: $\text{scale} = 96 / \max(W_{\text{roi}}, H_{\text{roi}})$.
  * Centered on a neutral black $96 \times 96$ canvas to preserve natural lip geometry without vertical distortion.

---

## 4. Speaker Partition & Augmentation Protocols

### 4.1 Strict Zero-Leakage 80/20 Partition Protocol
To ensure valid generalization, speakers are partitioned at the **base speaker** level:
* **Train Base Speakers (24 speakers, 80%)**: `s1` through `s24`.
* **Validation Base Speakers (6 speakers, 20%)**: `s25` through `s30`.
* **Zero Leakage**: No frames, augmented variants, or utterances of `s25`–`s30` are ever introduced into the training split.

### 4.2 100-Directory Expansion Scheme
To balance sample volume while maintaining strict speaker isolation:
* **Train Set (80 folders)**:
  * `s1`–`s8` (8 speakers) $\times$ 4 folders (1 original + 3 augmented: `sX`, `sX_aug1`, `sX_aug2`, `sX_aug3`) = 32 folders.
  * `s9`–`s24` (16 speakers) $\times$ 3 folders (1 original + 2 augmented: `sX`, `sX_aug1`, `sX_aug2`) = 48 folders.
  * Total Train Folders = $32 + 48 = \mathbf{80}$.
* **Validation Set (20 folders)**:
  * `s25`–`s26` (2 speakers) $\times$ 4 folders = 8 folders.
  * `s27`–`s30` (4 speakers) $\times$ 3 folders = 12 folders.
  * Total Validation Folders = $8 + 12 = \mathbf{20}$.
* **Total Dataset Folders**: Exactly 100 directories stored flat in `data/experiments/dlib_augmented_cropped_100_80_20_96x96/`.
* **Symlink Switching**: Switched via `python scripts/switch_dataset_80-20.py <dataset_name>`.

### 4.3 Conservative Lip-Reading Augmentation Limits
Geometric parameters are sampled **once per utterance variant** (using deterministic SHA-256 seed hashing) to prevent inter-frame jitter:
* **Max Translation**: $\pm 3\text{ px}$
* **Max Rotation**: $\pm 3^\circ$
* **Max Scale**: $0.97 \text{ to } 1.03$
* **Contrast ($\alpha$)**: $0.92 \text{ to } 1.08$
* **Brightness ($\beta$)**: $-10 \text{ to } +10$
* **Gaussian Noise**: $\sigma \in [1.0, 3.5]$ (50% probability)
* **Mild Blur**: Gaussian kernel $(3, 3), \sigma = 0.5$ (30% probability)
* **Border Mode**: `cv2.BORDER_REFLECT_101` (eliminates artificial seam artifacts)
* **JPEG Quality**: 95

---

## 5. Notebook Lineage & Architecture Evolution

The research and experimentation progression is organized chronologically across the `notebooks/` directory:

| Notebook | Focus & Architecture | Key Results / Insights |
| :--- | :--- | :--- |
| **`01_frame_extraction.ipynb`** | Video validation & sequential JPEG extraction | Verified 330 clips; generated master `metadata.csv` |
| **`02_dlib_experiment.ipynb`** | Dlib 68-point landmark localization & Method A–D crops | Benchmarked landmark stability on lip contours |
| **`02.5_dlib_augmented_cropped.ipynb`** | Generation of the canonical 100-folder 80/20 augmented dataset | Created `dlib_augmented_cropped_100_80_20_96x96` |
| **`03_mediapipe_experiment.ipynb`** | Google MediaPipe Face Mesh (96×96 & 112×112) | High-density 40-point lip contour extraction; $>120\text{ fps}$ |
| **`04_preprocessing_pipeline.ipynb`** | Sequence tensor packaging & normalization | Normalized sequences to shape $(N, 30, H, W, C)$ |
| **`05_model_training_baseline.ipynb`** | Custom 2D/3D CNNs + LSTM / GRU / BiLSTM / BiGRU | Established initial spatiotemporal baselines |
| **`06_pretrained_phase1_baseline.ipynb`** | Pretrained backbones (MobileNetV2, EfficientNet-B0, ResNet50) | EfficientNet-B0 and MobileNetV2 emerged as top spatial extractors |
| **`07–09_crop_method_experiments.ipynb`** | Comparative evaluation of Crop Methods B, C, and D | Method D letterboxing maximized visual phoneme preservation |
| **`10–12_transformer_experiments.ipynb`** | Spatiotemporal Multi-Block Visual Transformers | Explored self-attention temporal pooling; achieved 52.73% val acc |
| **`13–14_attention_pooling_experiments.ipynb`** | EfficientNet-B0 / MobileNetV2 + BiLSTM / BiGRU + Learned Attention Pooling | Attention-pooled recurrent heads pushed validation accuracy to 60–69% |
| **`15_ensemble_bilstm_bigru_transformer.ipynb`** | Soft-voting ensemble combining BiLSTM, BiGRU, and Transformer | Boosted multi-speaker validation accuracy to 64.09% |
| **`16_diagnostics_speaker_and_d4_d7_d10_cluster.ipynb`** | Speaker error breakdown & `{d4, d7, d10}` viseme triangle | Identified `s27` underexposure failure & dental viseme collision |
| **`17_test_time_augmentation_tta_experiment.ipynb`** | 4-View TTA (temporal shift $\pm 2$, horizontal flip) + Ensemble | **Project Peak**: **70.00% validation accuracy (0.6995 Macro F1)** |
| **`18_efficientnetb0_bigru_class_weighted_experiment.ipynb`** | Cost-sensitive loss weighting for ambiguous classes | Boosted `d7` recall to 75.0%; achieved 66.36% overall val accuracy |
| **`19_efficientnetb0_bigru_motion_focal_snapshot.ipynb`** | Single-Model Optimization: Frame Differences (4th Ch), Temporal Jitter, Focal Loss & Snapshot Ensemble | Target single-model pipeline beyond 75% without multi-model merging |

---

## 6. Key Diagnostic Insights & Error Patterns

### 6.1 Outlier Speaker Diagnosis
From Notebook 16 diagnostics across the 6 validation speakers:
* **`s27` (Severe Underperformer, 33–39% Acc)**: Suffers from extreme underexposure, harsh shadows, and low contrast around the mouth.
* **`s29` (Underperformer, 42–45% Acc)**: Characterized by minimal vertical lip displacement (subtle/muffled mouth movements).
* **`s25`, `s28`, `s30` (Strong Performers, 72–80% Acc)**: High-contrast, clear articulation, well-lit environments.

### 6.2 The `{d4, d7, d10}` Viseme Collision Triangle
The primary error cluster in the Tai Phake vocabulary involves digits with closed-teeth dental or bilabial initial/final closures:
* `d4` (*Si*) $\leftrightarrow$ `d10` (*Sip*) $\leftrightarrow$ `d7` (*Chit*)
* In Notebook 16, **18 persistent failure cases** failed across all three model architectures simultaneously (predominantly `d4` confused as `d10`, and `d7` confused as `d4` or `d10`).

---

## 7. Operational Guidelines for Agents

When interacting with this codebase, all AI agents must follow these rules:

1. **Environment**:
   * Python 3.12 managed via `uv` or `.venv`.
   * Standard execution command format: `uv run python <script.py>` or `.venv/bin/python <script.py>`.
2. **Preserve Dataset Integrity**:
   * Never modify master video files or delete base frames in `data/digit/`.
   * When creating experiment datasets, write to a new subfolder in `data/experiments/` without overwriting existing baselines.
3. **Strict Git Discipline**:
   * Do not commit video binaries (`*.mp4`), model weight files (`*.h5`, `*.keras`, `*.weights.h5`), or full tensor arrays.
   * Checkpoints and metrics belong in `results/<experiment_name>/`.
4. **Speaker Independence**:
   * Always verify that `s25`–`s30` are excluded from training when running 80/20 validation protocols.
