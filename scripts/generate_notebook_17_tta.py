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

    # Cell 0: Header Markdown
    cells.append(md_cell([
        "# 17 - Test-Time Augmentation (TTA) on Champion Models\n",
        "## Tai Phake Visual Speech Recognition (VSR) Project\n",
        "\n",
        "### Strategy & Rationale (Step 3)\n",
        "Lip reading on isolated utterances exhibits natural temporal sensitivity: small shifts in phoneme onset, subtle variations in boundary frame selection, or facial asymmetry can cause a model to teeter between neighboring visemes.\n",
        "\n",
        "**Test-Time Augmentation (TTA)** boosts inference stability without retraining by evaluating each video utterance across multiple realistic transformed views and averaging their predicted softmax probability distributions:\n",
        "1. **View 1 (Baseline)**: Standard uniform temporal sampling to 30 frames ($t \\in [0, N-1]$).\n",
        "2. **View 2 (Temporal Shift +2)**: Sample biased +2 frames later (starts slightly deeper into onset).\n",
        "3. **View 3 (Temporal Shift -2)**: Sample biased -2 frames earlier (starts slightly earlier before onset).\n",
        "4. **View 4 (Horizontal Flip)**: Horizontally mirrored frames ($x \\mapsto W - 1 - x$). Bilateral human lip motion during speech is approximately symmetric, providing invariant spatial regularization.\n",
        "\n",
        "**Evaluation Scope:**\n",
        "- Tested on both co-champions: **EfficientNetB0 + BiLSTM** and **EfficientNetB0 + BiGRU**.\n",
        "- Compares single-pass accuracy vs. TTA accuracy, measuring expected gain (+1% to +3%).\n",
        "- Analyzes per-digit accuracy improvements across difficult classes ($d_1, d_2, d_4, d_7, d_9$).\n"
    ]))

    # Cell 1: Setup & Configuration
    cells.append(md_cell("## 1. Setup & Configuration"))
    cells.append(code_cell([
        "# Section 1: Setup & Configuration\n",
        "DATASET_MODE = \"80_20\"\n",
        "SEED = 42\n"
    ]))

    # Cell 2: Imports
    cells.append(md_cell("## 2. Imports"))
    cells.append(code_cell([
        "# Section 2: Imports\n",
        "import os\n",
        "import sys\n",
        "import re\n",
        "import json\n",
        "import time\n",
        "import random\n",
        "from pathlib import Path\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from PIL import Image, ImageOps\n",
        "\n",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report\n",
        "\n",
        "import tensorflow as tf\n",
        "import keras\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"ENVIRONMENT & DEPENDENCY VERSIONS\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"TensorFlow Version: {tf.__version__}\")\n",
        "print(f\"Keras Version:      {keras.__version__}\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # Cell 3: Directory Paths
    cells.append(md_cell("## 3. Directory Paths & Class Mappings"))
    cells.append(code_cell([
        "# Section 3: Paths and Constants\n",
        "NOTEBOOK_DIR = Path.cwd()\n",
        "PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == \"notebooks\" else NOTEBOOK_DIR\n",
        "\n",
        "DATA_DIR = PROJECT_ROOT / \"data\" / \"processed\"\n",
        "VAL_DIR = DATA_DIR / \"val\"\n",
        "\n",
        "RESULTS_DIR = PROJECT_ROOT / \"results\" / f\"tta_evaluation_{DATASET_MODE}\"\n",
        "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "# Champion weights paths\n",
        "WEIGHTS_LSTM = PROJECT_ROOT / \"results\" / f\"efficientnetb0_bilstm_attention_pooling_{DATASET_MODE}\" / \"efficientnetb0_bilstm_attention_pooling_stage2_best.weights.h5\"\n",
        "WEIGHTS_GRU = PROJECT_ROOT / \"results\" / f\"efficientnetb0_bigru_attention_pooling_{DATASET_MODE}\" / \"efficientnetb0_bigru_attention_pooling_stage2_best.weights.h5\"\n",
        "\n",
        "NUM_CLASSES = 11\n",
        "NUM_FRAMES = 30\n",
        "FRAME_WIDTH = 96\n",
        "FRAME_HEIGHT = 96\n",
        "FRAME_CHANNELS = 3\n",
        "BATCH_SIZE = 16\n",
        "\n",
        "DIGIT_CLASSES = [f\"d{i}\" for i in range(NUM_CLASSES)]\n",
        "CLASS_TO_ID = {d: i for i, d in enumerate(DIGIT_CLASSES)}\n",
        "ID_TO_CLASS = {i: d for i, d in enumerate(DIGIT_CLASSES)}\n",
        "\n",
        "print(f\"BiLSTM weights exist : {WEIGHTS_LSTM.exists()}\")\n",
        "print(f\"BiGRU weights exist  : {WEIGHTS_GRU.exists()}\")\n",
        "print(f\"Results Output Dir   : {RESULTS_DIR}\")\n"
    ]))

    # Cell 4: Manifest Creation
    cells.append(md_cell("## 4. Manifest Construction for Validation Set"))
    cells.append(code_cell([
        "# Section 4: Manifest Creation\n",
        "def natural_sort_key(p):\n",
        "    m = re.search(r\"(\\d+)\", Path(p).stem)\n",
        "    return int(m.group(1)) if m else str(p)\n",
        "\n",
        "def get_base_speaker(spk_name: str) -> str:\n",
        "    return spk_name.split(\"_\")[0]\n",
        "\n",
        "manifest_records = []\n",
        "for spk_dir in sorted(VAL_DIR.iterdir()):\n",
        "    if not spk_dir.is_dir() or spk_dir.name.startswith(\".\"):\n",
        "        continue\n",
        "    for digit_dir in sorted(spk_dir.iterdir()):\n",
        "        if not digit_dir.is_dir() or digit_dir.name.startswith(\".\"):\n",
        "            continue\n",
        "        digit_name = digit_dir.name\n",
        "        if digit_name not in CLASS_TO_ID:\n",
        "            continue\n",
        "        frames = sorted(\n",
        "            [f for f in digit_dir.iterdir() if f.is_file() and f.suffix.lower() in [\".jpg\", \".jpeg\", \".png\"]],\n",
        "            key=natural_sort_key\n",
        "        )\n",
        "        if len(frames) == 0:\n",
        "            continue\n",
        "        manifest_records.append({\n",
        "            \"speaker\": spk_dir.name,\n",
        "            \"base_speaker\": get_base_speaker(spk_dir.name),\n",
        "            \"digit\": digit_name,\n",
        "            \"digit_id\": CLASS_TO_ID[digit_name],\n",
        "            \"frame_count\": len(frames),\n",
        "            \"frame_paths\": [str(f) for f in frames]\n",
        "        })\n",
        "\n",
        "df_val = pd.DataFrame(manifest_records)\n",
        "print(f\"Loaded {len(df_val)} validation utterances across {df_val['base_speaker'].nunique()} base speakers: {sorted(df_val['base_speaker'].unique())}\")\n"
    ]))

    # Cell 5: TTA Sampling Functions
    cells.append(md_cell("## 5. TTA Multi-View Sampling Functions"))
    cells.append(code_cell([
        "# Section 5: Four Complementary TTA Sampling Views\n",
        "def sample_tta_indices(n_frames: int, target_frames: int = NUM_FRAMES, shift: int = 0) -> list:\n",
        "    \"\"\"\n",
        "    Generates 30 frame indices with optional temporal start/end shift.\n",
        "    shift > 0: biases towards later frames.\n",
        "    shift < 0: biases towards earlier frames.\n",
        "    \"\"\"\n",
        "    if n_frames <= target_frames:\n",
        "        return list(range(n_frames)) if n_frames == target_frames else list(np.round(np.linspace(0, n_frames - 1, target_frames)).astype(int))\n",
        "    \n",
        "    # Shiftable window within raw frames\n",
        "    start = max(0, min(shift, n_frames - target_frames))\n",
        "    end = max(target_frames - 1, min(n_frames - 1, n_frames - 1 + shift))\n",
        "    indices = np.round(np.linspace(start, end, target_frames)).astype(int)\n",
        "    return np.clip(indices, 0, n_frames - 1).tolist()\n",
        "\n",
        "# Pre-compute views for each row in manifest\n",
        "# View 1: Standard uniform\n",
        "df_val[\"paths_orig\"] = df_val[\"frame_paths\"].apply(lambda p: [p[i] for i in sample_tta_indices(len(p), 30, shift=0)])\n",
        "# View 2: Shift +2 (deeper onset)\n",
        "df_val[\"paths_shift_plus2\"] = df_val[\"frame_paths\"].apply(lambda p: [p[i] for i in sample_tta_indices(len(p), 30, shift=+2)])\n",
        "# View 3: Shift -2 (earlier onset)\n",
        "df_val[\"paths_shift_minus2\"] = df_val[\"frame_paths\"].apply(lambda p: [p[i] for i in sample_tta_indices(len(p), 30, shift=-2)])\n",
        "\n",
        "print(\"TTA temporal sampling views constructed successfully for all 220 utterances.\")\n"
    ]))

    # Cell 6: Tensor Loading for all 4 Views
    cells.append(md_cell("## 6. Tensor Loading for All 4 TTA Views"))
    cells.append(code_cell([
        "# Section 6: Loading 4 TTA Video Sequence Tensors\n",
        "N = len(df_val)\n",
        "y_val = np.array(df_val[\"digit_id\"].values, dtype=np.int32)\n",
        "\n",
        "def load_view_tensor(path_col_name, hflip=False):\n",
        "    t0 = time.time()\n",
        "    X = np.zeros((N, NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.float32)\n",
        "    for i, row in df_val.iterrows():\n",
        "        for t, fpath in enumerate(row[path_col_name]):\n",
        "            img = Image.open(fpath).convert(\"RGB\")\n",
        "            if hflip:\n",
        "                img = ImageOps.mirror(img)\n",
        "            if img.size != (FRAME_WIDTH, FRAME_HEIGHT):\n",
        "                img = img.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.BILINEAR)\n",
        "            X[i, t] = np.array(img, dtype=np.float32)\n",
        "    print(f\"Loaded {path_col_name} (hflip={hflip}) in {time.time()-t0:.2f}s\")\n",
        "    return X\n",
        "\n",
        "print(\"Loading 4 TTA views into memory:\")\n",
        "X_view1 = load_view_tensor(\"paths_orig\", hflip=False)       # 1. Original\n",
        "X_view2 = load_view_tensor(\"paths_shift_plus2\", hflip=False) # 2. Shift +2\n",
        "X_view3 = load_view_tensor(\"paths_shift_minus2\", hflip=False)# 3. Shift -2\n",
        "X_view4 = load_view_tensor(\"paths_orig\", hflip=True)        # 4. Horizontal Flip\n"
    ]))

    # Cell 7: Model Builders
    cells.append(md_cell("## 7. Model Restoration for BiLSTM and BiGRU Champions"))
    cells.append(code_cell([
        "# Section 7: Restoring BiLSTM and BiGRU Models\n",
        "def get_backbone():\n",
        "    cnn = keras.applications.EfficientNetB0(input_shape=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), include_top=False, weights=\"imagenet\", pooling=\"avg\")\n",
        "    prep = keras.applications.efficientnet.preprocess_input\n",
        "    return cnn, prep\n",
        "\n",
        "def build_bilstm():\n",
        "    cnn, prep = get_backbone()\n",
        "    inp = keras.layers.Input(shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS))\n",
        "    x = keras.layers.TimeDistributed(keras.layers.Lambda(prep))(inp)\n",
        "    x = keras.layers.TimeDistributed(cnn)(x)\n",
        "    x = keras.layers.TimeDistributed(keras.layers.Dense(256, use_bias=False))(x)\n",
        "    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)\n",
        "    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.2))(x)\n",
        "    scores = keras.layers.Dense(1, use_bias=True)(x)\n",
        "    weights = keras.layers.Softmax(axis=1)(scores)\n",
        "    x = keras.layers.Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1))([x, weights])\n",
        "    x = keras.layers.Dropout(0.4)(x)\n",
        "    out = keras.layers.Dense(NUM_CLASSES, activation=\"softmax\")(x)\n",
        "    return keras.Model(inputs=inp, outputs=out)\n",
        "\n",
        "def build_bigru():\n",
        "    cnn, prep = get_backbone()\n",
        "    inp = keras.layers.Input(shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS))\n",
        "    x = keras.layers.TimeDistributed(keras.layers.Lambda(prep))(inp)\n",
        "    x = keras.layers.TimeDistributed(cnn)(x)\n",
        "    x = keras.layers.TimeDistributed(keras.layers.Dense(256, use_bias=False))(x)\n",
        "    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)\n",
        "    x = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True, dropout=0.2))(x)\n",
        "    scores = keras.layers.Dense(1, use_bias=True)(x)\n",
        "    weights = keras.layers.Softmax(axis=1)(scores)\n",
        "    x = keras.layers.Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1))([x, weights])\n",
        "    x = keras.layers.Dropout(0.4)(x)\n",
        "    out = keras.layers.Dense(NUM_CLASSES, activation=\"softmax\")(x)\n",
        "    return keras.Model(inputs=inp, outputs=out)\n",
        "\n",
        "model_lstm = build_bilstm()\n",
        "model_lstm.load_weights(WEIGHTS_LSTM)\n",
        "print(\"[OK] BiLSTM loaded successfully.\")\n",
        "\n",
        "model_gru = build_bigru()\n",
        "model_gru.load_weights(WEIGHTS_GRU)\n",
        "print(\"[OK] BiGRU loaded successfully.\")\n"
    ]))

    # Cell 8: Evaluating TTA on BiLSTM
    cells.append(md_cell("## 8. TTA Inference & Performance on EfficientNetB0 + BiLSTM"))
    cells.append(code_cell([
        "# Section 8: Multi-View Inference on BiLSTM\n",
        "print(\"Predicting 4 TTA views on BiLSTM...\")\n",
        "p1_l = model_lstm.predict(X_view1, batch_size=BATCH_SIZE, verbose=0)\n",
        "p2_l = model_lstm.predict(X_view2, batch_size=BATCH_SIZE, verbose=0)\n",
        "p3_l = model_lstm.predict(X_view3, batch_size=BATCH_SIZE, verbose=0)\n",
        "p4_l = model_lstm.predict(X_view4, batch_size=BATCH_SIZE, verbose=0)\n",
        "\n",
        "# Soft average\n",
        "probs_tta_lstm = (p1_l + p2_l + p3_l + p4_l) / 4.0\n",
        "\n",
        "acc_orig_lstm = accuracy_score(y_val, np.argmax(p1_l, axis=1))\n",
        "acc_tta_lstm = accuracy_score(y_val, np.argmax(probs_tta_lstm, axis=1))\n",
        "f1_tta_lstm = f1_score(y_val, np.argmax(probs_tta_lstm, axis=1), average=\"macro\", zero_division=0)\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"EFFICIENTNETB0 + BiLSTM: TTA RESULTS\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Baseline Single View Accuracy : {acc_orig_lstm * 100:.2f}%  ({(np.argmax(p1_l, axis=1) == y_val).sum()}/{N})\")\n",
        "print(f\"TTA 4-View Average Accuracy   : {acc_tta_lstm * 100:.2f}%  ({(np.argmax(probs_tta_lstm, axis=1) == y_val).sum()}/{N})\")\n",
        "print(f\"Macro F1-Score (TTA)          : {f1_tta_lstm:.4f}\")\n",
        "print(f\"Net TTA Gain                  : {(acc_tta_lstm - acc_orig_lstm)*100:+.2f}%\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # Cell 9: Evaluating TTA on BiGRU
    cells.append(md_cell("## 9. TTA Inference & Performance on EfficientNetB0 + BiGRU"))
    cells.append(code_cell([
        "# Section 9: Multi-View Inference on BiGRU\n",
        "print(\"Predicting 4 TTA views on BiGRU...\")\n",
        "p1_g = model_gru.predict(X_view1, batch_size=BATCH_SIZE, verbose=0)\n",
        "p2_g = model_gru.predict(X_view2, batch_size=BATCH_SIZE, verbose=0)\n",
        "p3_g = model_gru.predict(X_view3, batch_size=BATCH_SIZE, verbose=0)\n",
        "p4_g = model_gru.predict(X_view4, batch_size=BATCH_SIZE, verbose=0)\n",
        "\n",
        "probs_tta_gru = (p1_g + p2_g + p3_g + p4_g) / 4.0\n",
        "\n",
        "acc_orig_gru = accuracy_score(y_val, np.argmax(p1_g, axis=1))\n",
        "acc_tta_gru = accuracy_score(y_val, np.argmax(probs_tta_gru, axis=1))\n",
        "f1_tta_gru = f1_score(y_val, np.argmax(probs_tta_gru, axis=1), average=\"macro\", zero_division=0)\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"EFFICIENTNETB0 + BiGRU: TTA RESULTS\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Baseline Single View Accuracy : {acc_orig_gru * 100:.2f}%  ({(np.argmax(p1_g, axis=1) == y_val).sum()}/{N})\")\n",
        "print(f\"TTA 4-View Average Accuracy   : {acc_tta_gru * 100:.2f}%  ({(np.argmax(probs_tta_gru, axis=1) == y_val).sum()}/{N})\")\n",
        "print(f\"Macro F1-Score (TTA)          : {f1_tta_gru:.4f}\")\n",
        "print(f\"Net TTA Gain                  : {(acc_tta_gru - acc_orig_gru)*100:+.2f}%\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # Cell 10: TTA Combined with Ensemble (TTA + BiLSTM + BiGRU)
    cells.append(md_cell("## 10. Synergistic Combination: TTA + Co-Champion Ensemble"))
    cells.append(code_cell([
        "# Section 10: Combining TTA Probabilities from Both Champions\n",
        "probs_tta_ensemble = (probs_tta_lstm + probs_tta_gru) / 2.0\n",
        "preds_tta_ens = np.argmax(probs_tta_ensemble, axis=1)\n",
        "acc_tta_ens = accuracy_score(y_val, preds_tta_ens)\n",
        "f1_tta_ens = f1_score(y_val, preds_tta_ens, average=\"macro\", zero_division=0)\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"SYNERGISTIC COMBINATION: TTA + (BiLSTM + BiGRU) ENSEMBLE\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Single Model Baseline (BiLSTM)      : {acc_orig_lstm * 100:.2f}%\")\n",
        "print(f\"Single Model Baseline (BiGRU)       : {acc_orig_gru * 100:.2f}%\")\n",
        "print(f\"BiLSTM + TTA                        : {acc_tta_lstm * 100:.2f}%\")\n",
        "print(f\"BiGRU + TTA                         : {acc_tta_gru * 100:.2f}%\")\n",
        "print(f\"BiLSTM + BiGRU Ensemble + TTA       : {acc_tta_ens * 100:.2f}%  ({(preds_tta_ens == y_val).sum()}/{N}) | Macro F1: {f1_tta_ens:.4f}\")\n",
        "print(f\"Total Gain over Raw BiLSTM Baseline : {(acc_tta_ens - acc_orig_lstm)*100:+.2f}%\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # Cell 11: Per-Class Comparison Table
    cells.append(md_cell("## 11. Per-Class Accuracy Improvements with TTA"))
    cells.append(code_cell([
        "# Section 11: Per-Class Breakdown Table\n",
        "rows = []\n",
        "for d_i, d_name in enumerate(DIGIT_CLASSES):\n",
        "    mask = (y_val == d_i)\n",
        "    tot = mask.sum()\n",
        "    c_orig_l = (np.argmax(p1_l, axis=1)[mask] == d_i).sum()\n",
        "    c_tta_l = (np.argmax(probs_tta_lstm, axis=1)[mask] == d_i).sum()\n",
        "    c_orig_g = (np.argmax(p1_g, axis=1)[mask] == d_i).sum()\n",
        "    c_tta_g = (np.argmax(probs_tta_gru, axis=1)[mask] == d_i).sum()\n",
        "    c_tta_ens = (preds_tta_ens[mask] == d_i).sum()\n",
        "    rows.append({\n",
        "        \"Digit\": d_name,\n",
        "        \"Samples\": tot,\n",
        "        \"BiLSTM (Raw)\": f\"{c_orig_l}/{tot} ({c_orig_l/tot*100:.1f}%)\",\n",
        "        \"BiLSTM (TTA)\": f\"{c_tta_l}/{tot} ({c_tta_l/tot*100:.1f}%)\",\n",
        "        \"BiGRU (Raw)\": f\"{c_orig_g}/{tot} ({c_orig_g/tot*100:.1f}%)\",\n",
        "        \"BiGRU (TTA)\": f\"{c_tta_g}/{tot} ({c_tta_g/tot*100:.1f}%)\",\n",
        "        \"Ensemble+TTA\": f\"{c_tta_ens}/{tot} ({c_tta_ens/tot*100:.1f}%)\",\n",
        "    })\n",
        "\n",
        "df_class_tta = pd.DataFrame(rows)\n",
        "print(df_class_tta.to_string(index=False))\n",
        "df_class_tta.to_csv(RESULTS_DIR / \"tta_per_class_comparison.csv\", index=False)\n"
    ]))

    # Cell 12: Confusion Matrix Heatmap
    cells.append(md_cell("## 12. Confusion Matrix Heatmaps"))
    cells.append(code_cell([
        "# Section 12: Confusion Matrix Heatmaps\n",
        "fig, axes = plt.subplots(1, 2, figsize=(18, 7))\n",
        "\n",
        "# 1. Raw BiLSTM\n",
        "cm_raw = confusion_matrix(y_val, np.argmax(p1_l, axis=1))\n",
        "sns.heatmap(cm_raw, annot=True, fmt=\"d\", cmap=\"Blues\", xticklabels=DIGIT_CLASSES, yticklabels=DIGIT_CLASSES, ax=axes[0], cbar=True)\n",
        "axes[0].set_title(f\"Raw BiLSTM (Single View): {acc_orig_lstm*100:.2f}%\", fontsize=12, fontweight=\"bold\")\n",
        "axes[0].set_xlabel(\"Predicted\")\n",
        "axes[0].set_ylabel(\"True\")\n",
        "\n",
        "# 2. TTA Ensemble\n",
        "cm_tta = confusion_matrix(y_val, preds_tta_ens)\n",
        "sns.heatmap(cm_tta, annot=True, fmt=\"d\", cmap=\"Purples\", xticklabels=DIGIT_CLASSES, yticklabels=DIGIT_CLASSES, ax=axes[1], cbar=True)\n",
        "axes[1].set_title(f\"BiLSTM + BiGRU + TTA: {acc_tta_ens*100:.2f}%\", fontsize=12, fontweight=\"bold\")\n",
        "axes[1].set_xlabel(\"Predicted\")\n",
        "axes[1].set_ylabel(\"True\")\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(RESULTS_DIR / \"confusion_matrix_tta_comparison.png\", dpi=200)\n",
        "plt.show()\n"
    ]))

    # Cell 13: Save Metrics & Summary
    cells.append(md_cell("## 13. Save Metrics Artifacts"))
    cells.append(code_cell([
        "# Section 13: Save Metrics Artifacts\n",
        "tta_metrics = {\n",
        "    \"experiment\": \"17_tta_evaluation\",\n",
        "    \"dataset_mode\": DATASET_MODE,\n",
        "    \"views\": [\"original\", \"shift_+2\", \"shift_-2\", \"horizontal_flip\"],\n",
        "    \"bilstm\": {\n",
        "        \"raw_accuracy\": float(acc_orig_lstm),\n",
        "        \"tta_accuracy\": float(acc_tta_lstm),\n",
        "        \"tta_macro_f1\": float(f1_tta_lstm),\n",
        "        \"gain\": float(acc_tta_lstm - acc_orig_lstm)\n",
        "    },\n",
        "    \"bigru\": {\n",
        "        \"raw_accuracy\": float(acc_orig_gru),\n",
        "        \"tta_accuracy\": float(acc_tta_gru),\n",
        "        \"tta_macro_f1\": float(f1_tta_gru),\n",
        "        \"gain\": float(acc_tta_gru - acc_orig_gru)\n",
        "    },\n",
        "    \"tta_ensemble\": {\n",
        "        \"accuracy\": float(acc_tta_ens),\n",
        "        \"macro_f1\": float(f1_tta_ens),\n",
        "        \"gain_over_raw_bilstm\": float(acc_tta_ens - acc_orig_lstm)\n",
        "    }\n",
        "}\n",
        "\n",
        "with open(RESULTS_DIR / \"metrics_tta.json\", \"w\") as f:\n",
        "    json.dump(tta_metrics, f, indent=2)\n",
        "print(f\"Saved metrics JSON to: {RESULTS_DIR / 'metrics_tta.json'}\")\n"
    ]))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": ".venv",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_file = Path("notebooks/17_test_time_augmentation_tta_experiment.ipynb")
    with open(out_file, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully generated {out_file} with {len(cells)} cells!")

if __name__ == "__main__":
    main()
