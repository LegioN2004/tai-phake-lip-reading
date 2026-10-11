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
        "# 16 - Validation Speaker Diagnostics & d4/d7/d10 Error Cluster Analysis\n",
        "## Tai Phake Visual Speech Recognition (VSR) Project\n",
        "\n",
        "### Investigation Goal & Rationale (Step 2)\n",
        "In our 80/20 speaker-independent experiments (24 base train / 6 base val speakers), both the **BiLSTM** and **BiGRU** models achieve ~60% accuracy, but error matrices reveal persistent systemic confusions:\n",
        "1. **The {d4, d7, d10} Confusion Triangle**: Visual similarity between digits 4, 7, and 10 causes the models to repeatedly confuse them with each other.\n",
        "2. **Speaker-Specific Degradation**: If certain validation speakers yield abnormally low accuracy (e.g. <30%) while others reach >70%, the failure is caused by **data/landmark quality** (e.g. crop alignment, head pose, lighting) rather than model capacity.\n",
        "\n",
        "This notebook performs an exhaustive failure analysis:\n",
        "- Evaluates per-speaker accuracy across all 6 validation base speakers for all 3 champion models.\n",
        "- Isolates every single misclassified video utterance within the {d4, d7, d10} triangle.\n",
        "- Quantifies which speaker dominates the confusions.\n",
        "- Visualizes sample frame sequences of persistent error cases side-by-side to verify lip crop fidelity.\n"
    ]))

    # Cell 1: Config
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
        "from PIL import Image\n",
        "\n",
        "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
        "\n",
        "import tensorflow as tf\n",
        "import keras\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"ENVIRONMENT & DEPENDENCY VERSIONS\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"TensorFlow: {tf.__version__} | Keras: {keras.__version__}\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # Cell 3: Constants & Directories
    cells.append(md_cell("## 3. Directory Paths & Class Mappings"))
    cells.append(code_cell([
        "# Section 3: Paths and Constants\n",
        "NOTEBOOK_DIR = Path.cwd()\n",
        "PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == \"notebooks\" else NOTEBOOK_DIR\n",
        "\n",
        "DATA_DIR = PROJECT_ROOT / \"data\" / \"processed\"\n",
        "VAL_DIR = DATA_DIR / \"val\"\n",
        "\n",
        "RESULTS_DIR = PROJECT_ROOT / \"results\" / f\"diagnostics_val_speakers_{DATASET_MODE}\"\n",
        "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "WEIGHTS_LSTM = PROJECT_ROOT / \"results\" / f\"efficientnetb0_bilstm_attention_pooling_{DATASET_MODE}\" / \"efficientnetb0_bilstm_attention_pooling_stage2_best.weights.h5\"\n",
        "WEIGHTS_GRU = PROJECT_ROOT / \"results\" / f\"efficientnetb0_bigru_attention_pooling_{DATASET_MODE}\" / \"efficientnetb0_bigru_attention_pooling_stage2_best.weights.h5\"\n",
        "WEIGHTS_TRANS = PROJECT_ROOT / \"results\" / f\"efficientnetb0_transformer_4block_{DATASET_MODE}\" / \"efficientnetb0_transformer_4block_stage2_best.weights.h5\"\n",
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
        "print(f\"Results Output Directory : {RESULTS_DIR}\")\n",
        "print(f\"BiLSTM Checkpoint Exists : {WEIGHTS_LSTM.exists()}\")\n",
        "print(f\"BiGRU Checkpoint Exists  : {WEIGHTS_GRU.exists()}\")\n",
        "print(f\"Trans Checkpoint Exists  : {WEIGHTS_TRANS.exists()}\")\n"
    ]))

    # Cell 4: Manifest & Data Loading
    cells.append(md_cell("## 4. Manifest Construction & Validation Tensor Loading"))
    cells.append(code_cell([
        "# Section 4: Manifest Construction\n",
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
        "base_val_speakers = sorted(df_val[\"base_speaker\"].unique().tolist())\n",
        "print(f\"Loaded {len(df_val)} validation utterances across {len(base_val_speakers)} base speakers: {base_val_speakers}\")\n",
        "\n",
        "def sample_30_frames(frame_paths: list, target_frames: int = NUM_FRAMES) -> list:\n",
        "    n = len(frame_paths)\n",
        "    if n == target_frames:\n",
        "        indices = list(range(target_frames))\n",
        "    else:\n",
        "        indices = np.round(np.linspace(0, n - 1, target_frames)).astype(int)\n",
        "    return [str(frame_paths[i]) for i in indices]\n",
        "\n",
        "df_val[\"sampled_frame_paths\"] = df_val[\"frame_paths\"].apply(sample_30_frames)\n",
        "\n",
        "X_val = np.zeros((len(df_val), NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.float32)\n",
        "y_val = np.array(df_val[\"digit_id\"].values, dtype=np.int32)\n",
        "\n",
        "t0 = time.time()\n",
        "for i, row in df_val.iterrows():\n",
        "    for t, fpath in enumerate(row[\"sampled_frame_paths\"]):\n",
        "        img = Image.open(fpath).convert(\"RGB\")\n",
        "        if img.size != (FRAME_WIDTH, FRAME_HEIGHT):\n",
        "            img = img.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.BILINEAR)\n",
        "        X_val[i, t] = np.array(img, dtype=np.float32)\n",
        "\n",
        "print(f\"Loaded validation tensor {X_val.shape} in {time.time()-t0:.2f}s.\")\n"
    ]))

    # Cell 5: Model Builders
    cells.append(md_cell("## 5. Model Instantiation & Weights Restoration"))
    cells.append(code_cell([
        "# Section 5: Model Definitions\n",
        "def get_backbone():\n",
        "    cnn = keras.applications.EfficientNetB0(input_shape=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), include_top=False, weights=\"imagenet\", pooling=\"avg\")\n",
        "    prep = keras.applications.efficientnet.preprocess_input\n",
        "    return cnn, prep\n",
        "\n",
        "# 1. BiLSTM\n",
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
        "# 2. BiGRU\n",
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
        "# 3. Transformer\n",
        "class PositionalEncoding(keras.layers.Layer):\n",
        "    def __init__(self, seq_len=NUM_FRAMES, dim=256, **kwargs):\n",
        "        super().__init__(**kwargs)\n",
        "        pos = np.arange(seq_len)[:, np.newaxis]\n",
        "        i = np.arange(dim)[np.newaxis, :]\n",
        "        angles = pos / np.power(10000.0, (2 * (i // 2)) / np.float32(dim))\n",
        "        pe = np.zeros((1, seq_len, dim), dtype=np.float32)\n",
        "        pe[0, :, 0::2] = np.sin(angles[:, 0::2])\n",
        "        pe[0, :, 1::2] = np.cos(angles[:, 1::2])\n",
        "        self.pe = tf.constant(pe, dtype=tf.float32)\n",
        "    def call(self, inputs):\n",
        "        return inputs + self.pe\n",
        "\n",
        "def build_transformer():\n",
        "    cnn, prep = get_backbone()\n",
        "    inp = keras.layers.Input(shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS))\n",
        "    x = keras.layers.TimeDistributed(keras.layers.Lambda(prep))(inp)\n",
        "    x = keras.layers.TimeDistributed(cnn)(x)\n",
        "    x = keras.layers.TimeDistributed(keras.layers.Dense(256, use_bias=False))(x)\n",
        "    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)\n",
        "    x = PositionalEncoding()(x)\n",
        "    for b_id in range(4):\n",
        "        n1 = keras.layers.LayerNormalization(epsilon=1e-6)(x)\n",
        "        attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(n1, n1)\n",
        "        x = keras.layers.Add()([x, keras.layers.Dropout(0.2)(attn)])\n",
        "        n2 = keras.layers.LayerNormalization(epsilon=1e-6)(x)\n",
        "        ffn = keras.layers.Dense(512, activation=\"gelu\")(n2)\n",
        "        ffn = keras.layers.Dense(256)(keras.layers.Dropout(0.2)(ffn))\n",
        "        x = keras.layers.Add()([x, keras.layers.Dropout(0.2)(ffn)])\n",
        "    x = keras.layers.GlobalAveragePooling1D()(x)\n",
        "    out = keras.layers.Dense(NUM_CLASSES, activation=\"softmax\")(keras.layers.Dropout(0.4)(x))\n",
        "    return keras.Model(inputs=inp, outputs=out)\n",
        "\n",
        "model_lstm = build_bilstm()\n",
        "model_lstm.load_weights(WEIGHTS_LSTM)\n",
        "model_gru = build_bigru()\n",
        "model_gru.load_weights(WEIGHTS_GRU)\n",
        "model_trans = build_transformer()\n",
        "model_trans.load_weights(WEIGHTS_TRANS)\n",
        "\n",
        "print(\"[OK] Restored all 3 models from checkpoint weights.\")\n"
    ]))

    # Cell 6: Generate Predictions
    cells.append(md_cell("## 6. Validation Predictions & Overall Performance"))
    cells.append(code_cell([
        "# Section 6: Compute Softmax Probability Matrices\n",
        "probs_lstm = model_lstm.predict(X_val, batch_size=BATCH_SIZE, verbose=0)\n",
        "probs_gru = model_gru.predict(X_val, batch_size=BATCH_SIZE, verbose=0)\n",
        "probs_trans = model_trans.predict(X_val, batch_size=BATCH_SIZE, verbose=0)\n",
        "probs_ens = (probs_lstm + probs_gru + probs_trans) / 3.0\n",
        "\n",
        "df_val[\"pred_lstm\"] = np.argmax(probs_lstm, axis=1)\n",
        "df_val[\"pred_gru\"] = np.argmax(probs_gru, axis=1)\n",
        "df_val[\"pred_trans\"] = np.argmax(probs_trans, axis=1)\n",
        "df_val[\"pred_ens\"] = np.argmax(probs_ens, axis=1)\n",
        "\n",
        "df_val[\"correct_lstm\"] = (df_val[\"digit_id\"] == df_val[\"pred_lstm\"])\n",
        "df_val[\"correct_gru\"] = (df_val[\"digit_id\"] == df_val[\"pred_gru\"])\n",
        "df_val[\"correct_trans\"] = (df_val[\"digit_id\"] == df_val[\"pred_trans\"])\n",
        "df_val[\"correct_ens\"] = (df_val[\"digit_id\"] == df_val[\"pred_ens\"])\n",
        "\n",
        "print(\"Overall Validation Accuracies:\")\n",
        "print(f\"  BiLSTM      : {df_val['correct_lstm'].mean()*100:.2f}%\")\n",
        "print(f\"  BiGRU       : {df_val['correct_gru'].mean()*100:.2f}%\")\n",
        "print(f\"  Transformer : {df_val['correct_trans'].mean()*100:.2f}%\")\n",
        "print(f\"  Ensemble    : {df_val['correct_ens'].mean()*100:.2f}%\")\n"
    ]))

    # Cell 7: Part 1 - Per-Speaker Diagnostic
    cells.append(md_cell("## 7. Diagnostic 1: Per-Speaker Accuracy Breakdown"))
    cells.append(code_cell([
        "# Section 7: Per-Speaker Accuracy Analysis\n",
        "speaker_summary = []\n",
        "for spk in base_val_speakers:\n",
        "    mask = (df_val[\"base_speaker\"] == spk)\n",
        "    n_spk = mask.sum()\n",
        "    speaker_summary.append({\n",
        "        \"Base Speaker\": spk,\n",
        "        \"Utterances\": n_spk,\n",
        "        \"BiLSTM Acc (%)\": round(df_val.loc[mask, \"correct_lstm\"].mean() * 100, 2),\n",
        "        \"BiGRU Acc (%)\": round(df_val.loc[mask, \"correct_gru\"].mean() * 100, 2),\n",
        "        \"Transformer Acc (%)\": round(df_val.loc[mask, \"correct_trans\"].mean() * 100, 2),\n",
        "        \"Ensemble Acc (%)\": round(df_val.loc[mask, \"correct_ens\"].mean() * 100, 2),\n",
        "    })\n",
        "\n",
        "df_speaker_summary = pd.DataFrame(speaker_summary)\n",
        "print(df_speaker_summary.to_string(index=False))\n",
        "\n",
        "spk_csv = RESULTS_DIR / \"per_speaker_accuracy_breakdown.csv\"\n",
        "df_speaker_summary.to_csv(spk_csv, index=False)\n",
        "print(f\"\\nSaved speaker summary table to: {spk_csv}\")\n",
        "\n",
        "# Plot per-speaker comparison barplot\n",
        "plt.figure(figsize=(10, 5))\n",
        "bar_df = df_speaker_summary.melt(id_vars=[\"Base Speaker\", \"Utterances\"], value_vars=[\"BiLSTM Acc (%)\", \"BiGRU Acc (%)\", \"Transformer Acc (%)\", \"Ensemble Acc (%)\"], var_name=\"Model\", value_name=\"Accuracy\")\n",
        "sns.barplot(data=bar_df, x=\"Base Speaker\", y=\"Accuracy\", hue=\"Model\", palette=\"Set2\")\n",
        "plt.title(\"Validation Accuracy by Speaker: Identifying Anomaly / Outlier Speakers\", fontsize=13, fontweight=\"bold\")\n",
        "plt.ylabel(\"Accuracy (%)\")\n",
        "plt.grid(axis=\"y\", linestyle=\"--\", alpha=0.5)\n",
        "plt.tight_layout()\n",
        "plt.savefig(RESULTS_DIR / \"speaker_accuracy_barplot.png\", dpi=200)\n",
        "plt.show()\n"
    ]))

    # Cell 8: Part 2 - Analysis of the d4/d7/d10 Cluster
    cells.append(md_cell("## 8. Diagnostic 2: The {d4, d7, d10} Confusion Triangle Breakdown"))
    cells.append(code_cell([
        "# Section 8: The {d4, d7, d10} Error Cluster Analysis\n",
        "target_triangle = [4, 7, 10]\n",
        "target_names = [\"d4\", \"d7\", \"d10\"]\n",
        "\n",
        "# Filter manifest for true labels in {d4, d7, d10}\n",
        "df_triangle = df_val[df_val[\"digit_id\"].isin(target_triangle)].copy()\n",
        "print(f\"Total utterances belonging to {{d4, d7, d10}}: {len(df_triangle)} (Expected ~60 samples)\")\n",
        "\n",
        "# Error in triangle: True in triangle, but predicted incorrectly\n",
        "df_triangle[\"err_lstm\"] = (~df_triangle[\"correct_lstm\"])\n",
        "df_triangle[\"err_gru\"] = (~df_triangle[\"correct_gru\"])\n",
        "df_triangle[\"err_trans\"] = (~df_triangle[\"correct_trans\"])\n",
        "df_triangle[\"err_all_three\"] = (df_triangle[\"err_lstm\"] & df_triangle[\"err_gru\"] & df_triangle[\"err_trans\"])\n",
        "\n",
        "# Count errors per speaker inside the triangle\n",
        "spk_triangle_errors = []\n",
        "for spk in base_val_speakers:\n",
        "    m = (df_triangle[\"base_speaker\"] == spk)\n",
        "    n_total = m.sum()\n",
        "    err_l = df_triangle.loc[m, \"err_lstm\"].sum()\n",
        "    err_g = df_triangle.loc[m, \"err_gru\"].sum()\n",
        "    err_t = df_triangle.loc[m, \"err_trans\"].sum()\n",
        "    err_all = df_triangle.loc[m, \"err_all_three\"].sum()\n",
        "    spk_triangle_errors.append({\n",
        "        \"Speaker\": spk,\n",
        "        \"Samples in {d4,d7,d10}\": n_total,\n",
        "        \"BiLSTM Errors\": f\"{err_l}/{n_total} ({err_l/n_total*100:.1f}%)\",\n",
        "        \"BiGRU Errors\": f\"{err_g}/{n_total} ({err_g/n_total*100:.1f}%)\",\n",
        "        \"Transformer Errors\": f\"{err_t}/{n_total} ({err_t/n_total*100:.1f}%)\",\n",
        "        \"All 3 Failed\": f\"{err_all}/{n_total} ({err_all/n_total*100:.1f}%)\",\n",
        "    })\n",
        "\n",
        "df_spk_err = pd.DataFrame(spk_triangle_errors)\n",
        "print(\"\\n--- Errors in {d4, d7, d10} Grouped by Speaker ---\")\n",
        "print(df_spk_err.to_string(index=False))\n",
        "df_spk_err.to_csv(RESULTS_DIR / \"d4_d7_d10_errors_by_speaker.csv\", index=False)\n"
    ]))

    # Cell 9: Detailed confusion submatrix for d4, d7, d10
    cells.append(md_cell("## 9. Sub-Confusion Matrix for {d4, d7, d10} Across Models"))
    cells.append(code_cell([
        "# Section 9: Isolated Confusion Sub-Matrices\n",
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))\n",
        "\n",
        "for idx, (m_name, p_col) in enumerate([(\"BiLSTM\", \"pred_lstm\"), (\"BiGRU\", \"pred_gru\"), (\"Transformer\", \"pred_trans\")]):\n",
        "    # Sub-confusion matrix strictly between target classes\n",
        "    sub_mask = df_val[\"digit_id\"].isin(target_triangle)\n",
        "    y_true_sub = df_val.loc[sub_mask, \"digit_id\"]\n",
        "    y_pred_sub = df_val.loc[sub_mask, p_col]\n",
        "    \n",
        "    cm_sub = confusion_matrix(y_true_sub, y_pred_sub, labels=target_triangle)\n",
        "    sns.heatmap(cm_sub, annot=True, fmt=\"d\", cmap=\"YlOrRd\", xticklabels=target_names, yticklabels=target_names, ax=axes[idx], cbar=False)\n",
        "    axes[idx].set_title(f\"{m_name} Triangle Confusions\", fontweight=\"bold\")\n",
        "    axes[idx].set_xlabel(\"Predicted\")\n",
        "    axes[idx].set_ylabel(\"True\")\n",
        "\n",
        "plt.suptitle(\"Confusion Transitions Among {d4, d7, d10}\", fontsize=14, fontweight=\"bold\", y=1.03)\n",
        "plt.tight_layout()\n",
        "plt.savefig(RESULTS_DIR / \"d4_d7_d10_sub_confusion_matrix.png\", dpi=200)\n",
        "plt.show()\n"
    ]))

    # Cell 10: Persistent Error Inspection
    cells.append(md_cell("## 10. Listing All Persistent Failure Utterances Across Champions"))
    cells.append(code_cell([
        "# Section 10: Specific Misclassified Utterances Where All 3 Models Failed\n",
        "persistent_fails = df_val[df_val[\"err_all_three\" if \"err_all_three\" in df_val else \"digit_id\" != \"pred_lstm\"]].copy()\n",
        "persistent_fails = df_val[\n",
        "    (df_val[\"digit_id\"].isin(target_triangle)) &\n",
        "    (~df_val[\"correct_lstm\"]) &\n",
        "    (~df_val[\"correct_gru\"]) &\n",
        "    (~df_val[\"correct_trans\"])\n",
        "].copy()\n",
        "\n",
        "persistent_fails[\"True_Class\"] = persistent_fails[\"digit_id\"].map(ID_TO_CLASS)\n",
        "persistent_fails[\"BiLSTM_Pred\"] = persistent_fails[\"pred_lstm\"].map(ID_TO_CLASS)\n",
        "persistent_fails[\"BiGRU_Pred\"] = persistent_fails[\"pred_gru\"].map(ID_TO_CLASS)\n",
        "persistent_fails[\"Trans_Pred\"] = persistent_fails[\"pred_trans\"].map(ID_TO_CLASS)\n",
        "\n",
        "summary_cols = [\"speaker\", \"base_speaker\", \"True_Class\", \"BiLSTM_Pred\", \"BiGRU_Pred\", \"Trans_Pred\"]\n",
        "print(f\"\\nTotal Persistent Failures in Triangle (Failed by all 3 models): {len(persistent_fails)}\")\n",
        "print(persistent_fails[summary_cols].to_string(index=False))\n",
        "\n",
        "persistent_csv = RESULTS_DIR / \"persistent_triangle_failures.csv\"\n",
        "persistent_fails[summary_cols].to_csv(persistent_csv, index=False)\n",
        "print(f\"\\nSaved persistent failure table to: {persistent_csv}\")\n"
    ]))

    # Cell 11: Frame Strip Visualizer for Misclassified Samples
    cells.append(md_cell("## 11. Visual Inspection: Lip Crop Strip Visualizer for Triangle Errors"))
    cells.append(code_cell([
        "# Section 11: Visualizing Lip Sequence Strips for Persistent Failures\n",
        "def plot_utterance_strip(row, max_frames=8):\n",
        "    paths = row[\"sampled_frame_paths\"]\n",
        "    indices = np.linspace(0, len(paths)-1, max_frames, dtype=int)\n",
        "    fig, axes = plt.subplots(1, max_frames, figsize=(16, 2.5))\n",
        "    for idx, f_i in enumerate(indices):\n",
        "        img = Image.open(paths[f_i])\n",
        "        axes[idx].imshow(img)\n",
        "        axes[idx].axis(\"off\")\n",
        "        axes[idx].set_title(f\"t={f_i}\", fontsize=9)\n",
        "    title = f\"Spk: {row['speaker']} | True: {row['True_Class']} | Preds: [LSTM: {row['BiLSTM_Pred']}, GRU: {row['BiGRU_Pred']}, Trans: {row['Trans_Pred']}]\"\n",
        "    plt.suptitle(title, fontsize=11, fontweight=\"bold\")\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "print(\"Visualizing up to 4 persistent misclassified samples to verify bounding box/cropping quality:\")\n",
        "for i, (_, row) in enumerate(persistent_fails.head(4).iterrows()):\n",
        "    plot_utterance_strip(row)\n"
    ]))

    # Cell 12: Summary Report
    cells.append(md_cell("## 12. Diagnostic Conclusions & Actionable Insights"))
    cells.append(code_cell([
        "# Section 12: Summary Diagnostics Report\n",
        "report_text = f\"\"\"================================================================================\n",
        "VALIDATION DIAGNOSTICS & {target_names} TRIANGLE SUMMARY\n",
        "================================================================================\n",
        "1. Outlier Speaker Diagnostics:\n",
        "{df_speaker_summary.to_string(index=False)}\n",
        "\n",
        "2. {target_names} Cluster Errors by Speaker:\n",
        "{df_spk_err.to_string(index=False)}\n",
        "\n",
        "3. Total Persistent Tri-Model Failures in Triangle: {len(persistent_fails)}\n",
        "Saved all diagnostic CSVs and figures to:\n",
        "{RESULTS_DIR}\n",
        "================================================================================\n",
        "\"\"\"\n",
        "print(report_text)\n",
        "with open(RESULTS_DIR / \"diagnostic_summary_report.txt\", \"w\") as f:\n",
        "    f.write(report_text)\n"
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

    out_file = Path("notebooks/16_diagnostics_speaker_and_d4_d7_d10_cluster.ipynb")
    with open(out_file, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully generated {out_file} with {len(cells)} cells!")

if __name__ == "__main__":
    main()
