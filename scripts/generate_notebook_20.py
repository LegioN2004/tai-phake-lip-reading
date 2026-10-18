"""
Script: generate_notebook_20.py
Purpose: Generates notebooks/20_efficientnetb0_bigru_targeted_focal_top1.ipynb
Focus:
  1. Balanced Replaced Dataset (d1 fixed, no WeightedRandomSampler)
  2. Targeted Augmentation: Sequence-consistent random rotation (±10°) and zoom (±10%) on d1, d4, d9
  3. Updated Categorical Focal Loss with per-class weights:
     - d8: 0.5, d10: 0.5 (suppress default guessing)
     - d1: 1.5, d4: 1.5, d9: 1.5 (boost hard ambiguous classes)
     - others: 1.0
  4. Top-1 Snapshot Inference + 4-View TTA (evaluate single best checkpoint, no 3-snapshot averaging)
"""

import json
from pathlib import Path

def main():
    cells = []

    def md_cell(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source if isinstance(source, list) else [source]
        }

    def code_cell(source):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source if isinstance(source, list) else [source]
        }

    # -------------------------------------------------------------------------
    # Cell 0: Header & Comprehensive Technical Rationale
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "# 20 - Targeted Optimization: EfficientNet-B0 + BiGRU with Targeted Augmentation (d1, d4, d9), Balanced Focal Loss & Top-1 Snapshot Inference\n",
        "## Tai Phake Visual Speech Recognition (VSR) Project\n",
        "\n",
        "### Strategic Context & Motivations for Experiment 20\n",
        "In Experiment 19, our motion difference (4th channel) and snapshot ensembling established a solid spatiotemporal baseline reaching **73.18% validation accuracy** (with individual snapshot peaks at 75.00%). However, detailed error analysis and subsequent dataset verification revealed two critical bottlenecks:\n",
        "1. **Corrupted Utterance Videos in d1**: A subset of original d1 video recordings contained degraded frames/audio misalignment. The dataset has now been completely refreshed with clean, balanced recordings (`dlib_aug_cropd_98_78_20_96x96_replaced`), yielding an exact balanced distribution of 78 utterances per digit in training and 20 utterances per digit in validation. No `WeightedRandomSampler` is required.\n",
        "2. **Over-Prediction of d8 & d10 vs Severe Under-Recall of d1 & d4**: Stale model weights defaulted to predicting closed-lip/bilabial closure classes (d8, d10) while missing fast transient phonemes (d1 recall was only 15% and d4 was 5%).\n",
        "\n",
        "### Architectural & Algorithmic Modifications in Experiment 20\n",
        "\n",
        "| Innovation | Formulation / Setting | Operational Rationale |\n",
        "| :--- | :--- | :--- |\n",
        "| **1. Targeted Spatial Augmentation** | Random rotation $\\theta \\in [-10^\\circ, +10^\\circ]$, zoom $s \\in [0.90, 1.10]$ applied **only to $d_1, d_4, d_9$** | Applied consistently across all 30 frames of each utterance before motion calculation. Prevents motion channel distortion while forcing the CNN to learn angle- and scale-invariant lip phonemes for hard digits. |\n",
        "| **2. Calibrated Focal Loss Class Weights** | $w = [1.0, 1.5, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 0.5, 1.5, 0.5]$ for $d_0 \\dots d_{10}$ | $0.5\\times$ penalty on $d_8, d_{10}$ prevents false positive default-guessing; $1.5\\times$ boost on $d_1, d_4, d_9$ directly addresses the recall deficit. Easy digits remain at $1.0\\times$. |\n",
        "| **3. Temporal Cadence Jitter** | Interior frame drop (1–2 frames) on $d_1, d_4, d_7, d_9$ | Preserves temporal speed invariance for rapid bilabial closures. |\n",
        "| **4. Top-1 Snapshot Inference + 4-View TTA** | Selects single lowest-validation-loss checkpoint from Stage 2 | Evaluates the single champion model state under 4-View TTA (Baseline, $+1$ shift, $-1$ shift, Horizontal Flip) without blurring predictions across multiple checkpoints. |\n",
        "\n",
        "---"
    ]))

    # -------------------------------------------------------------------------
    # Cell 1: Environment & Setup
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 1. Setup & Reproducibility\n",
        "Configuring deterministic random seeds across Python, NumPy, and TensorFlow."
    ]))
    cells.append(code_cell([
        "# Section 1: Environment & Setup\n",
        "import os\n",
        "import sys\n",
        "import re\n",
        "import json\n",
        "import time\n",
        "import random\n",
        "import shutil\n",
        "from pathlib import Path\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from PIL import Image\n",
        "import cv2\n",
        "\n",
        "from sklearn.metrics import (\n",
        "    accuracy_score, precision_score, recall_score, f1_score,\n",
        "    confusion_matrix, classification_report\n",
        ")\n",
        "\n",
        "import tensorflow as tf\n",
        "import keras\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"SYSTEM ENVIRONMENT & VERSIONS\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Python Version     : {sys.version.split()[0]}\")\n",
        "print(f\"TensorFlow Version : {tf.__version__}\")\n",
        "print(f\"Keras Version      : {keras.__version__}\")\n",
        "print(f\"GPUs Available     : {len(tf.config.list_physical_devices('GPU'))}\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 2: Configuration & Hyperparameters
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 2. Configuration & Hyperparameters\n",
        "Configuring targeted augmentations, class-weighted Focal Loss, and Top-1 snapshot inference."
    ]))
    cells.append(code_cell([
        "# Section 2: Configuration & Hyperparameters\n",
        "SEED = 42\n",
        "random.seed(SEED)\n",
        "np.random.seed(SEED)\n",
        "tf.random.set_seed(SEED)\n",
        "os.environ[\"PYTHONHASHSEED\"] = str(SEED)\n",
        "\n",
        "DATASET_MODE = \"80_20\"\n",
        "\n",
        "# Sequence & Image Specifications (4-Channel: RGB + Motion Difference)\n",
        "NUM_CLASSES = 11\n",
        "NUM_FRAMES = 30\n",
        "FRAME_WIDTH = 96\n",
        "FRAME_HEIGHT = 96\n",
        "FRAME_CHANNELS = 4\n",
        "BATCH_SIZE = 16\n",
        "\n",
        "# Architecture Hyperparameters\n",
        "PROJECTION_DIM = 256\n",
        "GRU_UNITS = 128           # Bidirectional -> 256 recurrent output features\n",
        "GRU_DROPOUT = 0.2\n",
        "HEAD_DROPOUT = 0.4\n",
        "\n",
        "# Targeted Augmentation Hyperparameters (d1, d4, d9 only)\n",
        "TARGETED_AUG_DIGIT_IDS = [1, 4, 9]  # d1 (Nung), d4 (Si), d9 (Kao)\n",
        "TARGETED_MAX_ROTATION = 10.0        # ±10.0 degrees\n",
        "TARGETED_MAX_ZOOM = 0.10            # ±10.0% scale factor (0.90 to 1.10)\n",
        "\n",
        "# Temporal Jitter Settings\n",
        "JITTER_DIGIT_IDS = [1, 4, 7, 9]\n",
        "JITTER_PROBABILITY = 0.50\n",
        "JITTER_MAX_DROP = 2\n",
        "\n",
        "# Focal Loss with Targeted Class Weights\n",
        "# d8: 0.5, d10: 0.5 | d1: 1.5, d4: 1.5, d9: 1.5 | others: 1.0\n",
        "CLASS_WEIGHTS = [1.0, 1.5, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 0.5, 1.5, 0.5]\n",
        "FOCAL_GAMMA = 1.5\n",
        "FOCAL_ALPHA = 0.25\n",
        "LABEL_SMOOTHING = 0.05\n",
        "\n",
        "# Two-Stage Training Hyperparameters\n",
        "STAGE1_LR = 1e-4\n",
        "STAGE1_MAX_EPOCHS = 40\n",
        "STAGE1_PATIENCE = 8\n",
        "\n",
        "STAGE2_LR = 1e-5\n",
        "STAGE2_MAX_EPOCHS = 25\n",
        "STAGE2_PATIENCE = 6\n",
        "FINE_TUNE_FRACTION = 0.20\n",
        "\n",
        "DIGIT_CLASSES = [f\"d{i}\" for i in range(NUM_CLASSES)]\n",
        "CLASS_TO_ID = {d: i for i, d in enumerate(DIGIT_CLASSES)}\n",
        "ID_TO_CLASS = {i: d for i, d in enumerate(DIGIT_CLASSES)}\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"EXPERIMENT 20 CONFIGURATION: TARGETED AUG + WEIGHTED FOCAL + TOP-1 SNAPSHOT\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Targeted Augmentation Digits: {TARGETED_AUG_DIGIT_IDS} (Rot: ±{TARGETED_MAX_ROTATION}°, Zoom: ±{TARGETED_MAX_ZOOM*100:.0f}%)\")\n",
        "print(f\"Focal Loss Class Weights    : {dict(zip(DIGIT_CLASSES, CLASS_WEIGHTS))}\")\n",
        "print(f\"Evaluation Protocol         : Top-1 Stage 2 Snapshot + 4-View TTA (No multi-snapshot average)\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 3: Directory Paths
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 3. Directory Paths & Output Management\n",
        "Configuring paths to preprocessed frames in `data/processed/` and dedicated checkpoint storage in `results/efficientnetb0_bigru_targeted_focal_top1_80_20/`."
    ]))
    cells.append(code_cell([
        "# Section 3: Directory Paths Configuration\n",
        "NOTEBOOK_DIR = Path.cwd()\n",
        "PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == \"notebooks\" else NOTEBOOK_DIR\n",
        "\n",
        "DATA_DIR = PROJECT_ROOT / \"data\" / \"processed\"\n",
        "TRAIN_DIR = DATA_DIR / \"train\"\n",
        "VAL_DIR = DATA_DIR / \"val\"\n",
        "TEST_DIR = DATA_DIR / \"test\"\n",
        "\n",
        "# Dedicated Results Directory for Experiment 20\n",
        "RESULTS_DIR = PROJECT_ROOT / \"results\" / f\"efficientnetb0_bigru_targeted_focal_top1_{DATASET_MODE}\"\n",
        "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n",
        "SNAPSHOT_DIR = RESULTS_DIR / \"snapshots\"\n",
        "SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "print(f\"Project Root     : {PROJECT_ROOT}\")\n",
        "print(f\"Train Data Dir   : {TRAIN_DIR}\")\n",
        "print(f\"Val Data Dir     : {VAL_DIR}\")\n",
        "print(f\"Results Output   : {RESULTS_DIR}\")\n",
        "print(f\"Snapshots Output : {SNAPSHOT_DIR}\")\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 4: Speaker Partition Diagnostics
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 4. Speaker Partition Diagnostics\n",
        "Validating speaker isolation and asserting strict zero-leakage between training and validation sets."
    ]))
    cells.append(code_cell([
        "# Section 4: Speaker Split Verification & Diagnostics\n",
        "train_speakers = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir() and not d.name.startswith(\".\")])\n",
        "val_speakers = sorted([d.name for d in VAL_DIR.iterdir() if d.is_dir() and not d.name.startswith(\".\")])\n",
        "\n",
        "def get_base_speaker(spk_name: str) -> str:\n",
        "    return spk_name.split(\"_\")[0]\n",
        "\n",
        "base_train_speakers = sorted(\n",
        "    list({get_base_speaker(s) for s in train_speakers}),\n",
        "    key=lambda x: int(re.search(r\"\\d+\", x).group(0)) if re.search(r\"\\d+\", x) else x\n",
        ")\n",
        "base_val_speakers = sorted(\n",
        "    list({get_base_speaker(s) for s in val_speakers}),\n",
        "    key=lambda x: int(re.search(r\"\\d+\", x).group(0)) if re.search(r\"\\d+\", x) else x\n",
        ")\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(f\"SPEAKER PARTITION DIAGNOSTICS [Mode: {DATASET_MODE}]\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Training Base Speakers   ({len(base_train_speakers):2d}) : {base_train_speakers}\")\n",
        "print(f\"Training Total Folders   ({len(train_speakers):2d})\")\n",
        "print(f\"Validation Base Speakers ({len(base_val_speakers):2d}) : {base_val_speakers}\")\n",
        "print(f\"Validation Total Folders ({len(val_speakers):2d})\")\n",
        "print(\"-\" * 80)\n",
        "\n",
        "# Assert strict zero leakage\n",
        "overlap = set(base_train_speakers).intersection(set(base_val_speakers))\n",
        "assert len(overlap) == 0, f\"CRITICAL LEAKAGE DETECTED: Overlapping speakers: {overlap}\"\n",
        "print(\"[SUCCESS] STRICT ZERO-LEAKAGE VERIFIED: No base speakers overlap between Train and Val!\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 5: Utterance Manifest
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 5. Manifest Creation & Balance Verification\n",
        "Confirming that all 11 classes are perfectly balanced across Train (78 per digit) and Validation (20 per digit)."
    ]))
    cells.append(code_cell([
        "# Section 5: Utterance Manifest Indexing\n",
        "utterances = []\n",
        "\n",
        "for split_name, split_dir, spk_list in [(\"train\", TRAIN_DIR, train_speakers), (\"val\", VAL_DIR, val_speakers)]:\n",
        "    for spk in spk_list:\n",
        "        spk_path = split_dir / spk\n",
        "        for digit_dir in sorted(spk_path.iterdir()):\n",
        "            if not digit_dir.is_dir() or digit_dir.name.startswith(\".\"):\n",
        "                continue\n",
        "            digit_name = digit_dir.name\n",
        "            if digit_name not in CLASS_TO_ID:\n",
        "                continue\n",
        "            \n",
        "            frames = sorted([\n",
        "                f for f in digit_dir.iterdir()\n",
        "                if f.is_file() and f.suffix.lower() in [\".jpg\", \".jpeg\", \".png\"]\n",
        "            ])\n",
        "            if len(frames) == 0:\n",
        "                continue\n",
        "                \n",
        "            utterances.append({\n",
        "                \"speaker\": spk,\n",
        "                \"base_speaker\": get_base_speaker(spk),\n",
        "                \"digit\": digit_name,\n",
        "                \"digit_id\": CLASS_TO_ID[digit_name],\n",
        "                \"split\": split_name,\n",
        "                \"raw_frame_count\": len(frames),\n",
        "                \"frame_paths\": frames\n",
        "            })\n",
        "\n",
        "df_manifest = pd.DataFrame(utterances)\n",
        "manifest_csv_path = RESULTS_DIR / \"utterance_manifest.csv\"\n",
        "df_manifest[[\"speaker\", \"base_speaker\", \"digit\", \"digit_id\", \"split\", \"raw_frame_count\"]].to_csv(\n",
        "    manifest_csv_path, index=False\n",
        ")\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"UTTERANCE MANIFEST & BALANCE SUMMARY\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Total Utterances Indexed : {len(df_manifest):,}\")\n",
        "print(\"Train distribution per class:\")\n",
        "print(df_manifest[df_manifest['split']=='train']['digit'].value_counts().sort_index().to_dict())\n",
        "print(\"Validation distribution per class:\")\n",
        "print(df_manifest[df_manifest['split']=='val']['digit'].value_counts().sort_index().to_dict())\n",
        "print(f\"Manifest CSV saved to    : {manifest_csv_path}\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 6: Targeted Augmentation & Sequence Processing
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 6. Targeted Augmentation & Motion Difference Pipeline\n",
        "\n",
        "### Sequence-Consistent Affine Transformations\n",
        "For samples of $d_1, d_4, d_9$ during training:\n",
        "1. Sample rotation angle $\\theta \\sim \\mathcal{U}(-10^\\circ, +10^\\circ)$.\n",
        "2. Sample zoom factor $s \\sim \\mathcal{U}(0.90, 1.10)$.\n",
        "3. Compute affine matrix $M = \\text{cv2.getRotationMatrix2D}((48, 48), \\theta, s)$.\n",
        "4. Apply $M$ identically across all 30 frames with `cv2.BORDER_REFLECT_101` padding.\n",
        "5. Compute 4th channel inter-frame luminance differences $\\Delta_t = |Y_t - Y_{t-1}|$ on the augmented frames."
    ]))
    cells.append(code_cell([
        "# Section 6: Targeted Augmentation and Motion Channel Implementation\n",
        "\n",
        "def sample_30_frame_paths(frame_paths: list, target_frames: int = NUM_FRAMES) -> list:\n",
        "    n = len(frame_paths)\n",
        "    if n == target_frames:\n",
        "        indices = list(range(target_frames))\n",
        "    else:\n",
        "        indices = np.round(np.linspace(0, n - 1, target_frames)).astype(int)\n",
        "    return [str(frame_paths[i]) for i in indices]\n",
        "\n",
        "def apply_targeted_affine_aug(seq_rgb: np.ndarray, max_angle: float = TARGETED_MAX_ROTATION, max_zoom: float = TARGETED_MAX_ZOOM) -> np.ndarray:\n",
        "    \"\"\"\n",
        "    Applies random rotation (±max_angle) and random zoom (±max_zoom) consistently\n",
        "    across all frames of a sequence to preserve inter-frame physical motion continuity.\n",
        "    \"\"\"\n",
        "    angle = np.random.uniform(-max_angle, max_angle)\n",
        "    scale = np.random.uniform(1.0 - max_zoom, 1.0 + max_zoom)\n",
        "    T, H, W, C = seq_rgb.shape\n",
        "    center = (W / 2.0, H / 2.0)\n",
        "    M = cv2.getRotationMatrix2D(center, angle, scale)\n",
        "    \n",
        "    augmented_seq = np.empty_like(seq_rgb)\n",
        "    for t in range(T):\n",
        "        augmented_seq[t] = cv2.warpAffine(\n",
        "            seq_rgb[t],\n",
        "            M,\n",
        "            (W, H),\n",
        "            flags=cv2.INTER_LINEAR,\n",
        "            borderMode=cv2.BORDER_REFLECT_101\n",
        "        )\n",
        "    return augmented_seq\n",
        "\n",
        "def compute_frame_differences(seq_rgb: np.ndarray) -> np.ndarray:\n",
        "    \"\"\"\n",
        "    Calculates Frame N - Frame N-1 motion magnitude and stacks it as a 4th channel.\n",
        "    \"\"\"\n",
        "    T, H, W, _ = seq_rgb.shape\n",
        "    gray = (\n",
        "        0.2989 * seq_rgb[..., 0] +\n",
        "        0.5870 * seq_rgb[..., 1] +\n",
        "        0.1140 * seq_rgb[..., 2]\n",
        "    )[..., np.newaxis]\n",
        "    \n",
        "    diff = np.zeros((T, H, W, 1), dtype=np.float32)\n",
        "    diff[1:] = np.abs(gray[1:] - gray[:-1])\n",
        "    return np.concatenate([seq_rgb, diff], axis=-1)\n",
        "\n",
        "def apply_temporal_jitter(seq_rgb: np.ndarray, num_drop: int = 1) -> np.ndarray:\n",
        "    \"\"\"\n",
        "    Randomly drops 1 or 2 interior frames and linearly resamples back to NUM_FRAMES (30).\n",
        "    \"\"\"\n",
        "    T, H, W, C = seq_rgb.shape\n",
        "    drop_indices = np.random.choice(range(1, T - 1), size=num_drop, replace=False)\n",
        "    keep_indices = [i for i in range(T) if i not in drop_indices]\n",
        "    \n",
        "    new_t = np.linspace(0, len(keep_indices) - 1, T)\n",
        "    left_idx = np.floor(new_t).astype(int)\n",
        "    right_idx = np.ceil(new_t).astype(int)\n",
        "    alpha = (new_t - left_idx)[:, np.newaxis, np.newaxis, np.newaxis]\n",
        "    \n",
        "    kept_seq = seq_rgb[keep_indices]\n",
        "    jittered = (1.0 - alpha) * kept_seq[left_idx] + alpha * kept_seq[right_idx]\n",
        "    return jittered.astype(np.float32)\n",
        "\n",
        "df_manifest[\"sampled_frame_paths\"] = df_manifest[\"frame_paths\"].apply(sample_30_frame_paths)\n",
        "\n",
        "def load_raw_rgb_tensors():\n",
        "    num_total = len(df_manifest)\n",
        "    X_raw = np.zeros((num_total, NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)\n",
        "    y_raw = np.array(df_manifest[\"digit_id\"].values, dtype=np.int32)\n",
        "    \n",
        "    t0 = time.time()\n",
        "    for i, row in df_manifest.iterrows():\n",
        "        for t, fpath in enumerate(row[\"sampled_frame_paths\"]):\n",
        "            img = Image.open(fpath).convert(\"RGB\")\n",
        "            if img.size != (FRAME_WIDTH, FRAME_HEIGHT):\n",
        "                img = img.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.BILINEAR)\n",
        "            X_raw[i, t] = np.array(img, dtype=np.float32)\n",
        "            \n",
        "    elapsed = time.time() - t0\n",
        "    return X_raw, y_raw, elapsed\n",
        "\n",
        "print(\"Loading base RGB sequence tensors into fast contiguous RAM...\")\n",
        "X_all_rgb, y_all, load_time = load_raw_rgb_tensors()\n",
        "\n",
        "train_mask = (df_manifest[\"split\"] == \"train\").values\n",
        "val_mask = (df_manifest[\"split\"] == \"val\").values\n",
        "\n",
        "X_train_raw, y_train = X_all_rgb[train_mask], y_all[train_mask]\n",
        "X_val_raw, y_val = X_all_rgb[val_mask], y_all[val_mask]\n",
        "\n",
        "print(f\"Successfully loaded {len(X_all_rgb):,} raw video sequences in {load_time:.2f}s!\")\n",
        "print(f\"  Train RGB Set : {X_train_raw.shape}\")\n",
        "print(f\"  Val RGB Set   : {X_val_raw.shape}\")\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 7: Dataloader Generator
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 7. Dynamic Targeted Augmentation Generator (`keras.utils.PyDataset`)\n",
        "Applies targeted rotation and zoom dynamically only to $d_1, d_4, d_9$ on each training epoch."
    ]))
    cells.append(code_cell([
        "# Section 7: Dynamic Motion & Targeted Augmentation PyDataset\n",
        "\n",
        "class MotionLipDataset(keras.utils.PyDataset):\n",
        "    def __init__(\n",
        "        self,\n",
        "        X_rgb: np.ndarray,\n",
        "        y: np.ndarray,\n",
        "        batch_size: int = BATCH_SIZE,\n",
        "        shuffle: bool = True,\n",
        "        is_training: bool = False,\n",
        "        jitter_digit_ids: list = JITTER_DIGIT_IDS,\n",
        "        jitter_prob: float = JITTER_PROBABILITY,\n",
        "        targeted_aug_digit_ids: list = TARGETED_AUG_DIGIT_IDS,\n",
        "        targeted_max_angle: float = TARGETED_MAX_ROTATION,\n",
        "        targeted_max_zoom: float = TARGETED_MAX_ZOOM,\n",
        "        **kwargs\n",
        "    ):\n",
        "        super().__init__(**kwargs)\n",
        "        self.X_rgb = X_rgb\n",
        "        self.y = y\n",
        "        self.batch_size = batch_size\n",
        "        self.shuffle = shuffle\n",
        "        self.is_training = is_training\n",
        "        self.jitter_digit_ids = jitter_digit_ids\n",
        "        self.jitter_prob = jitter_prob\n",
        "        self.targeted_aug_digit_ids = targeted_aug_digit_ids\n",
        "        self.targeted_max_angle = targeted_max_angle\n",
        "        self.targeted_max_zoom = targeted_max_zoom\n",
        "        self.indices = np.arange(len(X_rgb))\n",
        "        self.on_epoch_end()\n",
        "\n",
        "    def __len__(self):\n",
        "        return int(np.ceil(len(self.X_rgb) / self.batch_size))\n",
        "\n",
        "    def on_epoch_end(self):\n",
        "        if self.shuffle:\n",
        "            np.random.shuffle(self.indices)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]\n",
        "        batch_x = []\n",
        "        batch_y = self.y[batch_idx]\n",
        "        \n",
        "        for i in batch_idx:\n",
        "            seq = self.X_rgb[i].copy()\n",
        "            digit_id = self.y[i]\n",
        "            \n",
        "            # 1. Targeted spatial affine augmentation (rotation +-10 deg, zoom +-10%) on d1, d4, d9\n",
        "            if self.is_training and (digit_id in self.targeted_aug_digit_ids):\n",
        "                seq = apply_targeted_affine_aug(\n",
        "                    seq,\n",
        "                    max_angle=self.targeted_max_angle,\n",
        "                    max_zoom=self.targeted_max_zoom\n",
        "                )\n",
        "            \n",
        "            # 2. Temporal jitter on fast transit digits\n",
        "            if self.is_training and (digit_id in self.jitter_digit_ids) and (np.random.rand() < self.jitter_prob):\n",
        "                num_drop = np.random.choice([1, 2])\n",
        "                seq = apply_temporal_jitter(seq, num_drop=num_drop)\n",
        "                \n",
        "            # 3. Calculate Frame N - Frame N-1 and stack as 4th channel\n",
        "            seq_4ch = compute_frame_differences(seq)\n",
        "            batch_x.append(seq_4ch)\n",
        "            \n",
        "        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.int32)\n",
        "\n",
        "train_generator = MotionLipDataset(\n",
        "    X_train_raw, y_train,\n",
        "    batch_size=BATCH_SIZE,\n",
        "    shuffle=True,\n",
        "    is_training=True\n",
        ")\n",
        "\n",
        "print(\"Precomputing static 4-channel validation tensor (X_val_4ch)...\")\n",
        "X_val_4ch = np.zeros(\n",
        "    (len(X_val_raw), NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),\n",
        "    dtype=np.float32\n",
        ")\n",
        "for i in range(len(X_val_raw)):\n",
        "    X_val_4ch[i] = compute_frame_differences(X_val_raw[i])\n",
        "\n",
        "print(f\"Train Generator batches per epoch : {len(train_generator)}\")\n",
        "print(f\"X_val_4ch Tensor Shape            : {X_val_4ch.shape}\")\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 8: Visual Verification of Targeted Augmentation
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 8. Visual Verification of Targeted Augmentation\n",
        "Visualizing an augmented sequence of $d_1$ (Nung) to verify continuous spatial rotation, scaling, and seamless 4th-channel motion diff calculation."
    ]))
    cells.append(code_cell([
        "# Section 8: Visualizing Targeted Augmentation on d1 sample\n",
        "d1_idx = np.where(y_train == 1)[0][0]\n",
        "sample_orig = X_train_raw[d1_idx]\n",
        "sample_aug = apply_targeted_affine_aug(sample_orig.copy(), max_angle=10.0, max_zoom=0.10)\n",
        "sample_aug_4ch = compute_frame_differences(sample_aug)\n",
        "\n",
        "frame_indices = [0, 6, 12, 18, 24]\n",
        "fig, axes = plt.subplots(3, len(frame_indices), figsize=(15, 8))\n",
        "\n",
        "for col, t in enumerate(frame_indices):\n",
        "    # Row 0: Original RGB\n",
        "    axes[0, col].imshow(sample_orig[t].astype(np.uint8))\n",
        "    axes[0, col].set_title(f\"Orig Frame {t}\", fontsize=10, fontweight=\"bold\")\n",
        "    axes[0, col].axis(\"off\")\n",
        "    \n",
        "    # Row 1: Targeted Aug (Rotation + Zoom)\n",
        "    axes[1, col].imshow(sample_aug[t].astype(np.uint8))\n",
        "    axes[1, col].set_title(f\"Aug Frame {t}\", fontsize=10, fontweight=\"bold\")\n",
        "    axes[1, col].axis(\"off\")\n",
        "    \n",
        "    # Row 2: Augmented Motion Diff\n",
        "    im = axes[2, col].imshow(sample_aug_4ch[t, ..., 3], cmap=\"magma\", vmin=0, vmax=50)\n",
        "    axes[2, col].set_title(f\"Motion Diff {t}\", fontsize=10, fontweight=\"bold\")\n",
        "    axes[2, col].axis(\"off\")\n",
        "\n",
        "plt.tight_layout()\n",
        "vis_path = RESULTS_DIR / \"targeted_aug_verification.png\"\n",
        "plt.savefig(vis_path, dpi=200)\n",
        "plt.show()\n",
        "print(f\"Targeted augmentation visualization saved to: {vis_path}\")\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 9: Categorical Focal Loss with Class Weights
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 9. Categorical Focal Loss with Per-Class Weights\n",
        "\n",
        "$$\\mathcal{L} = -\\sum_{c} y_c \\cdot w_c \\cdot \\alpha (1 - p_c)^\\gamma \\log(p_c)$$\n",
        "\n",
        "- Weights: $d_8: 0.5, d_{10}: 0.5$, $d_1: 1.5, d_4: 1.5, d_9: 1.5$, others: $1.0$.\n",
        "- Label smoothing: $\\epsilon = 0.05$."
    ]))
    cells.append(code_cell([
        "# Section 9: Categorical Focal Loss Implementation with Class Weights\n",
        "\n",
        "@keras.saving.register_keras_serializable()\n",
        "class CategoricalFocalLoss(keras.losses.Loss):\n",
        "    def __init__(\n",
        "        self,\n",
        "        gamma: float = FOCAL_GAMMA,\n",
        "        alpha: float = FOCAL_ALPHA,\n",
        "        class_weights: list = None,\n",
        "        label_smoothing: float = LABEL_SMOOTHING,\n",
        "        name: str = \"categorical_focal_loss\",\n",
        "        **kwargs\n",
        "    ):\n",
        "        super().__init__(name=name, **kwargs)\n",
        "        self.gamma = float(gamma)\n",
        "        self.alpha = float(alpha)\n",
        "        self.label_smoothing = float(label_smoothing)\n",
        "        self.class_weights = class_weights\n",
        "        if class_weights is not None:\n",
        "            self._class_weights_t = tf.constant(class_weights, dtype=tf.float32)\n",
        "        else:\n",
        "            self._class_weights_t = None\n",
        "\n",
        "    def call(self, y_true, y_pred):\n",
        "        eps = keras.backend.epsilon()\n",
        "        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)\n",
        "        \n",
        "        num_classes = tf.shape(y_pred)[-1]\n",
        "        y_true_rank = len(y_true.shape)\n",
        "        \n",
        "        if y_true_rank is not None and y_true_rank == 1:\n",
        "            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)\n",
        "        elif y_true_rank is not None and y_true_rank == 2 and y_true.shape[-1] == 1:\n",
        "            y_true = tf.one_hot(tf.cast(tf.squeeze(y_true, axis=-1), tf.int32), depth=num_classes)\n",
        "            \n",
        "        y_true = tf.cast(y_true, y_pred.dtype)\n",
        "        \n",
        "        if self.label_smoothing > 0.0:\n",
        "            c = tf.cast(num_classes, y_pred.dtype)\n",
        "            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / c)\n",
        "            \n",
        "        focal_weight = tf.pow(1.0 - y_pred, self.gamma)\n",
        "        cross_entropy = -tf.math.log(y_pred)\n",
        "        loss = self.alpha * focal_weight * cross_entropy\n",
        "        \n",
        "        if self._class_weights_t is not None:\n",
        "            loss = loss * self._class_weights_t\n",
        "            \n",
        "        per_sample_loss = tf.reduce_sum(y_true * loss, axis=-1)\n",
        "        return tf.reduce_mean(per_sample_loss)\n",
        "\n",
        "    def get_config(self):\n",
        "        config = super().get_config()\n",
        "        config.update({\n",
        "            \"gamma\": self.gamma,\n",
        "            \"alpha\": self.alpha,\n",
        "            \"class_weights\": self.class_weights,\n",
        "            \"label_smoothing\": self.label_smoothing,\n",
        "        })\n",
        "        return config\n",
        "\n",
        "focal_loss_fn = CategoricalFocalLoss(class_weights=CLASS_WEIGHTS)\n",
        "print(\"Weighted Categorical Focal Loss successfully initialized!\")\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 10: Model Architecture Assembly
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 10. Model Architecture: 4-Channel EfficientNet-B0 + BiGRU + Temporal Attention\n",
        "Preserving pretrained ImageNet features via the $1 \\times 1$ Motion Stem Adapter."
    ]))
    cells.append(code_cell([
        "# Section 10: Model Architecture Definition\n",
        "\n",
        "@keras.saving.register_keras_serializable()\n",
        "class TemporalAttentionPooling(keras.layers.Layer):\n",
        "    def __init__(self, **kwargs):\n",
        "        super().__init__(**kwargs)\n",
        "        self.score_dense = None\n",
        "\n",
        "    def build(self, input_shape):\n",
        "        self.score_dense = keras.layers.Dense(1, use_bias=False, name=\"attn_score\")\n",
        "        super().build(input_shape)\n",
        "\n",
        "    def call(self, inputs):\n",
        "        scores = self.score_dense(inputs)\n",
        "        weights = tf.nn.softmax(scores, axis=1)\n",
        "        return tf.reduce_sum(inputs * weights, axis=1)\n",
        "\n",
        "def build_4channel_efficientnet_extractor():\n",
        "    frame_in = keras.Input(shape=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), name=\"frame_4ch_input\")\n",
        "    adapter = keras.layers.Conv2D(filters=3, kernel_size=(1, 1), padding=\"same\", use_bias=False, name=\"motion_stem_adapter\")\n",
        "    adapted_rgb = adapter(frame_in)\n",
        "    \n",
        "    base_cnn = keras.applications.EfficientNetB0(weights=\"imagenet\", include_top=False, input_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3))\n",
        "    preprocessed = keras.applications.efficientnet.preprocess_input(adapted_rgb)\n",
        "    cnn_features = base_cnn(preprocessed)\n",
        "    pooled = keras.layers.GlobalAveragePooling2D(name=\"gap\")(cnn_features)\n",
        "    \n",
        "    extractor = keras.Model(inputs=frame_in, outputs=pooled, name=\"EfficientNetB0_4Ch_Extractor\")\n",
        "    \n",
        "    # Identity on RGB, 0.10 contribution on 4th motion channel\n",
        "    w = np.zeros((1, 1, 4, 3), dtype=np.float32)\n",
        "    w[0, 0, 0, 0] = 1.0\n",
        "    w[0, 0, 1, 1] = 1.0\n",
        "    w[0, 0, 2, 2] = 1.0\n",
        "    w[0, 0, 3, :] = 0.10\n",
        "    adapter.set_weights([w])\n",
        "    return extractor, base_cnn, adapter\n",
        "\n",
        "def build_full_vsr_model():\n",
        "    extractor, base_cnn, adapter = build_4channel_efficientnet_extractor()\n",
        "    seq_input = keras.Input(shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), name=\"sequence_4ch_input\")\n",
        "    \n",
        "    x = keras.layers.TimeDistributed(extractor, name=\"time_distributed_cnn\")(seq_input)\n",
        "    x = keras.layers.Dense(PROJECTION_DIM, activation=\"relu\", name=\"projection_dense\")(x)\n",
        "    x = keras.layers.Dropout(0.2, name=\"projection_dropout\")(x)\n",
        "    \n",
        "    x = keras.layers.Bidirectional(keras.layers.GRU(GRU_UNITS, return_sequences=True, dropout=GRU_DROPOUT), name=\"bidirectional_gru\")(x)\n",
        "    x = TemporalAttentionPooling(name=\"temporal_attention_pooling\")(x)\n",
        "    x = keras.layers.Dropout(HEAD_DROPOUT, name=\"head_dropout\")(x)\n",
        "    outputs = keras.layers.Dense(NUM_CLASSES, activation=\"softmax\", name=\"softmax_classification\")(x)\n",
        "    \n",
        "    model = keras.Model(inputs=seq_input, outputs=outputs, name=\"EfficientNetB0_BiGRU_Exp20\")\n",
        "    return model, base_cnn, adapter\n",
        "\n",
        "model, base_cnn, motion_adapter = build_full_vsr_model()\n",
        "print(f\"Total Model Parameters: {model.count_params():,}\")\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 11: Snapshot Tracking Callback
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 11. Top-K Snapshot Tracking Callback\n",
        "Tracks the best validation checkpoints to identify the single Top-1 champion snapshot."
    ]))
    cells.append(code_cell([
        "# Section 11: Snapshot Checkpointing Callback\n",
        "\n",
        "class TopKSnapshotCheckpoint(keras.callbacks.Callback):\n",
        "    def __init__(self, snapshot_dir: Path, k: int = 3, prefix: str = \"stage2\"):\n",
        "        super().__init__()\n",
        "        self.snapshot_dir = Path(snapshot_dir)\n",
        "        self.k = k\n",
        "        self.prefix = prefix\n",
        "        self.best_snapshots = []\n",
        "\n",
        "    def on_epoch_end(self, epoch, logs=None):\n",
        "        logs = logs or {}\n",
        "        val_loss = logs.get(\"val_loss\")\n",
        "        val_acc = logs.get(\"val_accuracy\", 0.0)\n",
        "        if val_loss is None:\n",
        "            return\n",
        "            \n",
        "        epoch_num = epoch + 1\n",
        "        temp_path = self.snapshot_dir / f\"{self.prefix}_epoch_{epoch_num:02d}_loss_{val_loss:.4f}.weights.h5\"\n",
        "        \n",
        "        if len(self.best_snapshots) < self.k or val_loss < max(s[0] for s in self.best_snapshots):\n",
        "            self.model.save_weights(temp_path)\n",
        "            self.best_snapshots.append((val_loss, epoch_num, temp_path, val_acc))\n",
        "            self.best_snapshots.sort(key=lambda x: x[0])\n",
        "            \n",
        "            if len(self.best_snapshots) > self.k:\n",
        "                removed = self.best_snapshots.pop(-1)\n",
        "                if removed[2].exists():\n",
        "                    removed[2].unlink()\n",
        "                    \n",
        "            print(f\"\\n>>> [Snapshot Tracker] Epoch {epoch_num} qualified for Top-{self.k}! (val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%)\")\n",
        "\n",
        "    def get_top1_path(self):\n",
        "        return self.best_snapshots[0][2] if self.best_snapshots else None\n",
        "\n",
        "print(\"Snapshot callback configured!\")\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 12: Stage 1 Training
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 12. Stage 1 Training: Frozen Backbone with Weighted Focal Loss\n",
        "- Backbone is completely frozen (`trainable = False`).\n",
        "- Learning Rate: $1 \\times 10^{-4}$ with Adam."
    ]))
    cells.append(code_cell([
        "# Section 12: Stage 1 Training Execution\n",
        "def freeze_backbone(backbone):\n",
        "    backbone.trainable = False\n",
        "    for layer in backbone.layers:\n",
        "        layer.trainable = False\n",
        "\n",
        "freeze_backbone(base_cnn)\n",
        "\n",
        "model.compile(\n",
        "    optimizer=keras.optimizers.Adam(learning_rate=STAGE1_LR),\n",
        "    loss=CategoricalFocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, class_weights=CLASS_WEIGHTS, label_smoothing=LABEL_SMOOTHING),\n",
        "    metrics=[\"accuracy\"]\n",
        ")\n",
        "\n",
        "stage1_weights_path = RESULTS_DIR / \"stage1_best.weights.h5\"\n",
        "checkpoint1 = keras.callbacks.ModelCheckpoint(stage1_weights_path, monitor=\"val_loss\", save_best_only=True, save_weights_only=True, verbose=1)\n",
        "early1 = keras.callbacks.EarlyStopping(monitor=\"val_loss\", patience=STAGE1_PATIENCE, restore_best_weights=True, verbose=1)\n",
        "\n",
        "print(\"Starting Stage 1 Training...\")\n",
        "t_stage1_start = time.time()\n",
        "history1 = model.fit(train_generator, validation_data=(X_val_4ch, y_val), epochs=STAGE1_MAX_EPOCHS, callbacks=[checkpoint1, early1], verbose=1)\n",
        "stage1_elapsed = time.time() - t_stage1_start\n",
        "\n",
        "if stage1_weights_path.exists():\n",
        "    model.load_weights(stage1_weights_path)\n",
        "\n",
        "stage1_best_epoch = int(np.argmin(history1.history[\"val_loss\"]) + 1)\n",
        "stage1_best_val_acc = float(max(history1.history[\"val_accuracy\"]))\n",
        "stage1_best_val_loss = float(min(history1.history[\"val_loss\"]))\n",
        "print(f\"Stage 1 Best Epoch: {stage1_best_epoch} | Loss: {stage1_best_val_loss:.4f} | Acc: {stage1_best_val_acc*100:.2f}%\")\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 13: Stage 2 Fine-Tuning
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 13. Stage 2 Fine-Tuning: Top 20% Unfrozen with Weighted Focal Loss\n",
        "- Unfreezes top 20% of layers (`BatchNormalization` layers kept in inference mode).\n",
        "- Low learning rate: $1 \\times 10^{-5}$."
    ]))
    cells.append(code_cell([
        "# Section 13: Stage 2 Fine-Tuning Execution\n",
        "def unfreeze_last_fraction(backbone, fraction=FINE_TUNE_FRACTION):\n",
        "    for layer in backbone.layers:\n",
        "        layer.trainable = False\n",
        "    total_layers = len(backbone.layers)\n",
        "    start_index = int(total_layers * (1.0 - fraction))\n",
        "    for layer in backbone.layers[start_index:]:\n",
        "        if not isinstance(layer, keras.layers.BatchNormalization):\n",
        "            layer.trainable = True\n",
        "\n",
        "unfreeze_last_fraction(base_cnn, fraction=FINE_TUNE_FRACTION)\n",
        "\n",
        "model.compile(\n",
        "    optimizer=keras.optimizers.Adam(learning_rate=STAGE2_LR),\n",
        "    loss=CategoricalFocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, class_weights=CLASS_WEIGHTS, label_smoothing=LABEL_SMOOTHING),\n",
        "    metrics=[\"accuracy\"]\n",
        ")\n",
        "\n",
        "snapshot_callback = TopKSnapshotCheckpoint(snapshot_dir=SNAPSHOT_DIR, k=3, prefix=\"stage2\")\n",
        "early2 = keras.callbacks.EarlyStopping(monitor=\"val_loss\", patience=STAGE2_PATIENCE, restore_best_weights=True, verbose=1)\n",
        "\n",
        "print(\"Starting Stage 2 Fine-Tuning...\")\n",
        "t_stage2_start = time.time()\n",
        "history2 = model.fit(train_generator, validation_data=(X_val_4ch, y_val), epochs=STAGE2_MAX_EPOCHS, callbacks=[snapshot_callback, early2], verbose=1)\n",
        "stage2_elapsed = time.time() - t_stage2_start\n",
        "\n",
        "stage2_best_epoch = int(np.argmin(history2.history[\"val_loss\"]) + 1)\n",
        "stage2_best_val_acc = float(max(history2.history[\"val_accuracy\"]))\n",
        "stage2_best_val_loss = float(min(history2.history[\"val_loss\"]))\n",
        "print(f\"Stage 2 Best Epoch: {stage2_best_epoch} | Loss: {stage2_best_val_loss:.4f} | Acc: {stage2_best_val_acc*100:.2f}%\")\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 14: Training Curves
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 14. Training & Validation History Curves\n",
        "Visualizing loss and accuracy trajectories."
    ]))
    cells.append(code_cell([
        "# Section 14: Training History Curves\n",
        "h1 = history1.history\n",
        "h2 = history2.history\n",
        "\n",
        "df_history = pd.DataFrame({\n",
        "    \"epoch\": list(range(1, len(h1[\"loss\"]) + len(h2[\"loss\"]) + 1)),\n",
        "    \"stage\": [\"Stage 1\"] * len(h1[\"loss\"]) + [\"Stage 2\"] * len(h2[\"loss\"]),\n",
        "    \"train_loss\": h1[\"loss\"] + h2[\"loss\"],\n",
        "    \"val_loss\": h1[\"val_loss\"] + h2[\"val_loss\"],\n",
        "    \"train_accuracy\": h1[\"accuracy\"] + h2[\"accuracy\"],\n",
        "    \"val_accuracy\": h1[\"val_accuracy\"] + h2[\"val_accuracy\"],\n",
        "})\n",
        "df_history.to_csv(RESULTS_DIR / \"training_history.csv\", index=False)\n",
        "\n",
        "stage1_end = len(h1[\"loss\"])\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
        "ax1.plot(df_history[\"epoch\"], df_history[\"train_loss\"], label=\"Train Loss\", color=\"#1f77b4\", lw=2)\n",
        "ax1.plot(df_history[\"epoch\"], df_history[\"val_loss\"], label=\"Val Loss\", color=\"#ff7f0e\", lw=2)\n",
        "ax1.axvline(x=stage1_end + 0.5, color=\"gray\", linestyle=\"--\", alpha=0.7, label=\"Stage Transition\")\n",
        "ax1.set_title(\"Weighted Focal Loss Trajectory\", fontsize=12, fontweight=\"bold\")\n",
        "ax1.set_xlabel(\"Epoch\"); ax1.set_ylabel(\"Loss\"); ax1.legend(); ax1.grid(True, alpha=0.3)\n",
        "\n",
        "ax2.plot(df_history[\"epoch\"], df_history[\"train_accuracy\"], label=\"Train Acc\", color=\"#2ca02c\", lw=2)\n",
        "ax2.plot(df_history[\"epoch\"], df_history[\"val_accuracy\"], label=\"Val Acc\", color=\"#d62728\", lw=2)\n",
        "ax2.axvline(x=stage1_end + 0.5, color=\"gray\", linestyle=\"--\", alpha=0.7, label=\"Stage Transition\")\n",
        "ax2.set_title(\"Validation Accuracy Trajectory\", fontsize=12, fontweight=\"bold\")\n",
        "ax2.set_xlabel(\"Epoch\"); ax2.set_ylabel(\"Accuracy\"); ax2.legend(); ax2.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "curves_path = RESULTS_DIR / \"training_history_curves.png\"\n",
        "plt.savefig(curves_path, dpi=200)\n",
        "plt.show()\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 15: Top-1 Snapshot Inference + 4-View TTA
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 15. Top-1 Snapshot Inference + 4-View TTA\n",
        "\n",
        "### Strictly Top-1 Snapshot Evaluation\n",
        "Unlike Experiment 19 which averaged predictions across 3 snapshots, we strictly load the **Top-1 best checkpoint** from Stage 2 (lowest validation loss), apply 4-View TTA, and evaluate directly."
    ]))
    cells.append(code_cell([
        "# Section 15: Top-1 Snapshot Inference + 4-View TTA Implementation\n",
        "\n",
        "def generate_4view_tta(X_seq_4ch: np.ndarray) -> list:\n",
        "    v1 = X_seq_4ch.copy()  # Baseline\n",
        "    v2 = np.empty_like(X_seq_4ch)  # Shift +1\n",
        "    v2[:-1] = X_seq_4ch[1:]; v2[-1] = X_seq_4ch[-1]\n",
        "    v3 = np.empty_like(X_seq_4ch)  # Shift -1\n",
        "    v3[0] = X_seq_4ch[0]; v3[1:] = X_seq_4ch[:-1]\n",
        "    v4 = np.flip(X_seq_4ch, axis=2).copy()  # Horizontal mirror\n",
        "    return [v1, v2, v3, v4]\n",
        "\n",
        "top1_snapshot_path = snapshot_callback.get_top1_path()\n",
        "if top1_snapshot_path is None or not top1_snapshot_path.exists():\n",
        "    all_snaps = sorted(list(SNAPSHOT_DIR.glob(\"*.weights.h5\")))\n",
        "    top1_snapshot_path = all_snaps[0] if all_snaps else RESULTS_DIR / \"stage1_best.weights.h5\"\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(f\"EVALUATING TOP-1 SNAPSHOT CHECKPOINT: {top1_snapshot_path.name}\")\n",
        "print(\"=\" * 80)\n",
        "model.load_weights(top1_snapshot_path)\n",
        "\n",
        "# Build 4-view tensors\n",
        "N_val = len(X_val_4ch)\n",
        "views_val = [np.zeros_like(X_val_4ch) for _ in range(4)]\n",
        "for i in range(N_val):\n",
        "    v_list = generate_4view_tta(X_val_4ch[i])\n",
        "    for v_idx in range(4):\n",
        "        views_val[v_idx][i] = v_list[v_idx]\n",
        "\n",
        "# Predict across 4 views\n",
        "view_probs = []\n",
        "for v_idx, v_tensor in enumerate(views_val, 1):\n",
        "    p = model.predict(v_tensor, batch_size=BATCH_SIZE, verbose=0)\n",
        "    view_probs.append(p)\n",
        "\n",
        "# Top-1 Snapshot Softmax Average across 4 views\n",
        "final_top1_probs = np.mean(view_probs, axis=0)\n",
        "final_preds = np.argmax(final_top1_probs, axis=1)\n",
        "\n",
        "final_acc = accuracy_score(y_val, final_preds)\n",
        "final_f1 = f1_score(y_val, final_preds, average=\"macro\")\n",
        "final_precision = precision_score(y_val, final_preds, average=\"macro\")\n",
        "final_recall = recall_score(y_val, final_preds, average=\"macro\")\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"TOP-1 SNAPSHOT + 4-VIEW TTA FINAL RESULTS\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Validation Accuracy : {final_acc * 100:.2f}%\")\n",
        "print(f\"Macro F1-Score      : {final_f1:.4f}\")\n",
        "print(f\"Macro Precision     : {final_precision:.4f}\")\n",
        "print(f\"Macro Recall        : {final_recall:.4f}\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 16: Classification Report & Confusion Matrix
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 16. Classification Report & Confusion Matrix\n",
        "Confirming $d_1, d_4, d_9$ recall improvements."
    ]))
    cells.append(code_cell([
        "# Section 16: Diagnostics & Detailed Report\n",
        "report_str = classification_report(y_val, final_preds, target_names=DIGIT_CLASSES, digits=4)\n",
        "print(report_str)\n",
        "\n",
        "report_dict = classification_report(y_val, final_preds, target_names=DIGIT_CLASSES, output_dict=True)\n",
        "with open(RESULTS_DIR / \"classification_report.json\", \"w\") as f:\n",
        "    json.dump(report_dict, f, indent=2)\n",
        "\n",
        "cm = confusion_matrix(y_val, final_preds)\n",
        "plt.figure(figsize=(10, 8))\n",
        "sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\", xticklabels=DIGIT_CLASSES, yticklabels=DIGIT_CLASSES, cbar_kws={\"label\": \"Utterances\"})\n",
        "plt.title(f\"Experiment 20 Confusion Matrix (Top-1 Snapshot + 4-View TTA)\\nVal Accuracy: {final_acc*100:.2f}%\")\n",
        "plt.xlabel(\"Predicted Digit\"); plt.ylabel(\"Ground Truth Digit\"); plt.tight_layout()\n",
        "cm_path = RESULTS_DIR / \"confusion_matrix.png\"\n",
        "plt.savefig(cm_path, dpi=200)\n",
        "plt.show()\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 17: Per-Speaker Breakdown
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 17. Speaker Breakdown Analysis\n",
        "Evaluating performance across the 6 validation base speakers ($s_{23}, s_{24}, s_{25}, s_{26}, s_{28}, s_{30}$)."
    ]))
    cells.append(code_cell([
        "# Section 17: Per-Speaker Accuracy Breakdown\n",
        "df_val = df_manifest[df_manifest[\"split\"] == \"val\"].copy().reset_index(drop=True)\n",
        "df_val[\"pred_id\"] = final_preds\n",
        "df_val[\"correct\"] = (df_val[\"digit_id\"] == df_val[\"pred_id\"]).astype(int)\n",
        "\n",
        "speaker_summary = []\n",
        "print(\"=\" * 80)\n",
        "print(\"PER-SPEAKER VALIDATION ACCURACY BREAKDOWN\")\n",
        "print(\"=\" * 80)\n",
        "for spk in base_val_speakers:\n",
        "    sub = df_val[df_val[\"base_speaker\"] == spk]\n",
        "    acc = sub[\"correct\"].mean()\n",
        "    total = len(sub)\n",
        "    correct_count = sub[\"correct\"].sum()\n",
        "    speaker_summary.append({\"base_speaker\": spk, \"accuracy\": float(acc), \"total\": int(total), \"correct\": int(correct_count)})\n",
        "    print(f\"  Speaker {spk:5s} : {correct_count:2d}/{total:2d} correct -> {acc * 100:6.2f}%\")\n",
        "print(\"=\" * 80)\n",
        "\n",
        "with open(RESULTS_DIR / \"per_speaker_breakdown.json\", \"w\") as f:\n",
        "    json.dump(speaker_summary, f, indent=2)\n"
    ]))

    # -------------------------------------------------------------------------
    # Cell 18: Summary Metrics Persistence
    # -------------------------------------------------------------------------
    cells.append(md_cell([
        "## 18. Final Results Persistence & Comparative Summary\n",
        "Saving final metrics JSON."
    ]))
    cells.append(code_cell([
        "# Section 18: Final Metrics Persistence\n",
        "final_metrics = {\n",
        "    \"experiment\": \"20_efficientnetb0_bigru_targeted_focal_top1\",\n",
        "    \"dataset_mode\": DATASET_MODE,\n",
        "    \"seed\": SEED,\n",
        "    \"frame_channels\": FRAME_CHANNELS,\n",
        "    \"motion_diff_applied\": True,\n",
        "    \"temporal_jitter_applied\": True,\n",
        "    \"targeted_augmentation_applied\": True,\n",
        "    \"targeted_digits\": TARGETED_AUG_DIGIT_IDS,\n",
        "    \"class_weights\": CLASS_WEIGHTS,\n",
        "    \"loss\": \"WeightedCategoricalFocalLoss\",\n",
        "    \"evaluation_mode\": \"Top-1 Snapshot + 4-View TTA\",\n",
        "    \"top1_snapshot\": str(top1_snapshot_path.name),\n",
        "    \"metrics\": {\n",
        "        \"val_accuracy\": float(final_acc),\n",
        "        \"macro_f1\": float(final_f1),\n",
        "        \"macro_precision\": float(final_precision),\n",
        "        \"macro_recall\": float(final_recall),\n",
        "    },\n",
        "    \"stages\": {\n",
        "        \"stage1_best_epoch\": stage1_best_epoch,\n",
        "        \"stage1_best_loss\": stage1_best_val_loss,\n",
        "        \"stage1_best_acc\": stage1_best_val_acc,\n",
        "        \"stage2_best_epoch\": stage2_best_epoch,\n",
        "        \"stage2_best_loss\": stage2_best_val_loss,\n",
        "        \"stage2_best_acc\": stage2_best_val_acc,\n",
        "    }\n",
        "}\n",
        "\n",
        "metrics_path = RESULTS_DIR / f\"metrics_{DATASET_MODE}.json\"\n",
        "with open(metrics_path, \"w\") as f:\n",
        "    json.dump(final_metrics, f, indent=2)\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"EXPERIMENT 20 EXECUTION COMPLETED SUCCESSFULLY\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Final Accuracy               : {final_acc * 100:.2f}%\")\n",
        "print(f\"Macro F1-Score               : {final_f1:.4f}\")\n",
        "print(f\"Baseline Milestone (Exp 19)  : 73.18%\")\n",
        "print(f\"Results Output Directory     : {RESULTS_DIR}\")\n",
        "print(\"=\" * 80)\n"
    ]))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_file = Path("notebooks/20_efficientnetb0_bigru_targeted_focal_top1.ipynb")
    with open(out_file, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully generated {out_file} with {len(cells)} cells!")

if __name__ == "__main__":
    main()
