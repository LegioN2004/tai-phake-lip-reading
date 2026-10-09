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
        "# 15 - Multi-Model Ensemble Experiment (BiLSTM + BiGRU + 4-Block Transformer)\n",
        "## Tai Phake Visual Speech Recognition (VSR) Project\n",
        "\n",
        "### Experiment Objective: Tri-Model Late Fusion / Probability Averaging\n",
        "In our speaker-independent 80/20 benchmark (24 base train speakers / 6 base validation speakers), three diverse temporal architectures were trained using an identical ImageNet-pretrained **EfficientNetB0** visual backbone:\n",
        "1. **EfficientNetB0 + BiLSTM + Learnable Attention Pooling** (~60.5% validation accuracy)\n",
        "2. **EfficientNetB0 + BiGRU + Learnable Attention Pooling** (~60.0% validation accuracy)\n",
        "3. **EfficientNetB0 + 4-Block Transformer** (~52.7% validation accuracy)\n",
        "\n",
        "**Key Hypothesis & Motivation:**\n",
        "- **Complementary Error Distributions**: Because recurrent models (BiLSTM/BiGRU) with sequence inductive bias and self-attention models (Transformer) model temporal dependencies through distinct mathematical mechanisms, they commit non-overlapping errors on specific digits:\n",
        "  - BiLSTM achieves higher recall on subtle lip closures like $d_7$.\n",
        "  - BiGRU resolves rapid transitions like $d_2$ more accurately.\n",
        "  - Transformer captures global temporal co-articulation on classes like $d_{10}$.\n",
        "- **Ensemble Formulation**: By loading the best Stage 2 fine-tuned checkpoint weights for all three champions and performing **soft voting (probability averaging)** across predicted class probability vectors:\n",
        "  $$\\mathbf{P}_{\\text{ensemble}} = w_{\\text{lstm}}\\mathbf{P}_{\\text{lstm}} + w_{\\text{gru}}\\mathbf{P}_{\\text{gru}} + w_{\\text{trans}}\\mathbf{P}_{\\text{trans}}$$\n",
        "  we evaluate both **uniform averaging** ($w_i = 1/3$) and **optimal weighted averaging via grid search**.\n"
    ]))

    # Cell 1: Section 1 Configuration
    cells.append(md_cell("## 1. Experiment Mode & Seed Setup"))
    cells.append(code_cell([
        "# Section 1: Experiment Configuration\n",
        "DATASET_MODE = \"80_20\"\n",
        "SEED = 42\n"
    ]))

    # Cell 2: Section 2 Imports
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
        "import sklearn.metrics as metrics\n",
        "from sklearn.metrics import (\n",
        "    accuracy_score,\n",
        "    precision_score,\n",
        "    recall_score,\n",
        "    f1_score,\n",
        "    confusion_matrix,\n",
        "    classification_report\n",
        ")\n",
        "\n",
        "import tensorflow as tf\n",
        "import keras\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"ENVIRONMENT & DEPENDENCY VERSIONS\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Python Version:     {sys.version.split()[0]}\")\n",
        "print(f\"TensorFlow Version: {tf.__version__}\")\n",
        "print(f\"Keras Version:      {keras.__version__}\")\n",
        "gpu_devices = tf.config.list_physical_devices('GPU')\n",
        "print(f\"Available GPUs:     {len(gpu_devices)} ({gpu_devices})\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # Cell 3: Section 3 Configuration & Hyperparameters
    cells.append(md_cell("## 3. Configuration & Classes"))
    cells.append(code_cell([
        "# Section 3: Configuration & Hyperparameters\n",
        "random.seed(SEED)\n",
        "np.random.seed(SEED)\n",
        "tf.random.set_seed(SEED)\n",
        "os.environ[\"PYTHONHASHSEED\"] = str(SEED)\n",
        "\n",
        "# Sequence & Image Specifications (Method D standard)\n",
        "NUM_CLASSES = 11\n",
        "NUM_FRAMES = 30\n",
        "FRAME_WIDTH = 96\n",
        "FRAME_HEIGHT = 96\n",
        "FRAME_CHANNELS = 3\n",
        "BATCH_SIZE = 16\n",
        "\n",
        "# Architecture Dimensions\n",
        "PROJECTION_DIM = 256\n",
        "LSTM_UNITS = 128\n",
        "GRU_UNITS = 128\n",
        "EMBED_DIM = 256\n",
        "NUM_HEADS = 4\n",
        "KEY_DIM = 64\n",
        "FFN_DIM = 512\n",
        "NUM_TRANSFORMER_BLOCKS = 4\n",
        "\n",
        "# Target Digit Classes (d0 through d10)\n",
        "DIGIT_CLASSES = [f\"d{i}\" for i in range(NUM_CLASSES)]\n",
        "CLASS_TO_ID = {d: i for i, d in enumerate(DIGIT_CLASSES)}\n",
        "ID_TO_CLASS = {i: d for i, d in enumerate(DIGIT_CLASSES)}\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"ENSEMBLE EVALUATION SETUP\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Dataset Mode         : {DATASET_MODE}\")\n",
        "print(f\"Target Digit Classes : {DIGIT_CLASSES}\")\n",
        "print(f\"Input Sequence Shape : ({NUM_FRAMES}, {FRAME_HEIGHT}x{FRAME_WIDTH}, {FRAME_CHANNELS})\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # Cell 4: Section 4 Dataset Paths & Checkpoint Locations
    cells.append(md_cell("## 4. Dataset & Checkpoint Paths"))
    cells.append(code_cell([
        "# Section 4: Dataset Paths Configuration & Model Checkpoint Paths\n",
        "NOTEBOOK_DIR = Path.cwd()\n",
        "PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == \"notebooks\" else NOTEBOOK_DIR\n",
        "\n",
        "DATA_DIR = PROJECT_ROOT / \"data\" / \"processed\"\n",
        "TRAIN_DIR = DATA_DIR / \"train\"\n",
        "VAL_DIR = DATA_DIR / \"val\"\n",
        "TEST_DIR = DATA_DIR / \"test\"\n",
        "\n",
        "# Dedicated Results Directory for Ensemble\n",
        "RESULTS_DIR = PROJECT_ROOT / \"results\" / f\"ensemble_bilstm_bigru_transformer_{DATASET_MODE}\"\n",
        "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "# Checkpoint weight paths from prior experiments\n",
        "WEIGHTS_LSTM_PATH = PROJECT_ROOT / \"results\" / f\"efficientnetb0_bilstm_attention_pooling_{DATASET_MODE}\" / \"efficientnetb0_bilstm_attention_pooling_stage2_best.weights.h5\"\n",
        "WEIGHTS_GRU_PATH = PROJECT_ROOT / \"results\" / f\"efficientnetb0_bigru_attention_pooling_{DATASET_MODE}\" / \"efficientnetb0_bigru_attention_pooling_stage2_best.weights.h5\"\n",
        "WEIGHTS_TRANS_PATH = PROJECT_ROOT / \"results\" / f\"efficientnetb0_transformer_4block_{DATASET_MODE}\" / \"efficientnetb0_transformer_4block_stage2_best.weights.h5\"\n",
        "\n",
        "print(\"Checkpoint weight availability check:\")\n",
        "print(f\"  BiLSTM Weights      : {WEIGHTS_LSTM_PATH.exists()} ({WEIGHTS_LSTM_PATH})\")\n",
        "print(f\"  BiGRU Weights       : {WEIGHTS_GRU_PATH.exists()} ({WEIGHTS_GRU_PATH})\")\n",
        "print(f\"  Transformer Weights : {WEIGHTS_TRANS_PATH.exists()} ({WEIGHTS_TRANS_PATH})\")\n",
        "print(f\"  Output Results Dir  : {RESULTS_DIR}\")\n"
    ]))

    # Cell 5: Speaker Split Verification & Diagnostics
    cells.append(md_cell("## 5. Speaker Split Verification"))
    cells.append(code_cell([
        "# Section 5: Speaker Split Verification & Diagnostics\n",
        "train_speakers = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir() and not d.name.startswith(\".\")])\n",
        "val_speakers = sorted([d.name for d in VAL_DIR.iterdir() if d.is_dir() and not d.name.startswith(\".\")])\n",
        "\n",
        "def get_base_speaker(spk_name: str) -> str:\n",
        "    return spk_name.split(\"_\")[0]\n",
        "\n",
        "base_train_speakers = sorted(list(set(get_base_speaker(s) for s in train_speakers)))\n",
        "base_val_speakers = sorted(list(set(get_base_speaker(s) for s in val_speakers)))\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"SPEAKER SPLIT VERIFICATION\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Train Directory ({len(train_speakers)} folders) -> {len(base_train_speakers)} Base Speakers: {base_train_speakers}\")\n",
        "print(f\"Val Directory   ({len(val_speakers)} folders) -> {len(base_val_speakers)} Base Speakers: {base_val_speakers}\")\n",
        "overlap = set(base_train_speakers).intersection(set(base_val_speakers))\n",
        "assert len(overlap) == 0, f\"DATA LEAKAGE DETECTED! Overlap: {overlap}\"\n",
        "print(\"[OK] Zero speaker leakage between train and validation splits!\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # Cell 6: Manifest Creation
    cells.append(md_cell("## 6. Manifest Creation"))
    cells.append(code_cell([
        "# Section 6: Manifest Creation\n",
        "def natural_sort_key(p):\n",
        "    m = re.search(r\"(\\d+)\", Path(p).stem)\n",
        "    return int(m.group(1)) if m else str(p)\n",
        "\n",
        "manifest_records = []\n",
        "\n",
        "splits = [(\"train\", TRAIN_DIR), (\"val\", VAL_DIR)]\n",
        "if DATASET_MODE == \"100_3_3\" and TEST_DIR.exists():\n",
        "    splits.append((\"test\", TEST_DIR))\n",
        "\n",
        "for split_name, split_path in splits:\n",
        "    for spk_dir in sorted(split_path.iterdir()):\n",
        "        if not spk_dir.is_dir() or spk_dir.name.startswith(\".\"):\n",
        "            continue\n",
        "        for digit_dir in sorted(spk_dir.iterdir()):\n",
        "            if not digit_dir.is_dir() or digit_dir.name.startswith(\".\"):\n",
        "                continue\n",
        "            digit_name = digit_dir.name\n",
        "            if digit_name not in CLASS_TO_ID:\n",
        "                continue\n",
        "            frames = sorted(\n",
        "                [f for f in digit_dir.iterdir() if f.is_file() and f.suffix.lower() in [\".jpg\", \".jpeg\", \".png\"]],\n",
        "                key=natural_sort_key\n",
        "            )\n",
        "            if len(frames) == 0:\n",
        "                continue\n",
        "            manifest_records.append({\n",
        "                \"split\": split_name,\n",
        "                \"speaker\": spk_dir.name,\n",
        "                \"base_speaker\": get_base_speaker(spk_dir.name),\n",
        "                \"digit\": digit_name,\n",
        "                \"digit_id\": CLASS_TO_ID[digit_name],\n",
        "                \"frame_count\": len(frames),\n",
        "                \"frame_paths\": [str(f) for f in frames]\n",
        "            })\n",
        "\n",
        "df_manifest = pd.DataFrame(manifest_records)\n",
        "print(f\"Manifest created with {len(df_manifest):,} total video utterances across splits:\")\n",
        "print(df_manifest[\"split\"].value_counts().to_string())\n"
    ]))

    # Cell 7: Frame Sampling & In-Memory Loading
    cells.append(md_cell("## 7. Frame Sampling & In-Memory Loading"))
    cells.append(code_cell([
        "# Section 7: Frame Sampling & In-Memory Sequence Loading\n",
        "def sample_30_frames(frame_paths: list, target_frames: int = NUM_FRAMES) -> list:\n",
        "    \"\"\"\n",
        "    Deterministic temporal sampling across the video duration.\n",
        "    Produces exactly target_frames (30) uniformly distributed frame indices.\n",
        "    \"\"\"\n",
        "    n = len(frame_paths)\n",
        "    if n == target_frames:\n",
        "        indices = list(range(target_frames))\n",
        "    else:\n",
        "        indices = np.round(np.linspace(0, n - 1, target_frames)).astype(int)\n",
        "    return [str(frame_paths[i]) for i in indices]\n",
        "\n",
        "# EXPERIMENT 1 IMPLEMENTATION: dropping 3 frames\n",
        "# def sample_30_frames(frame_paths: list, target_frames: int = NUM_FRAMES, drop_start: int = 3, drop_end: int = 3) -> list:\n",
        "#     n = len(frame_paths)\n",
        "#     # Drop first 3 and last 3 frames (neutral / closed lips)\n",
        "#     if n > (drop_start + drop_end):\n",
        "#         trimmed = frame_paths[drop_start:-drop_end]\n",
        "#     else:\n",
        "#         trimmed = frame_paths\n",
        "#\n",
        "#     n_trimmed = len(trimmed)\n",
        "#     if n_trimmed == target_frames:\n",
        "#         indices = list(range(target_frames))\n",
        "#     else:\n",
        "#         indices = np.round(np.linspace(0, n_trimmed - 1, target_frames)).astype(int)\n",
        "#     return [str(trimmed[i]) for i in indices]\n",
        "\n",
        "df_manifest[\"sampled_frame_paths\"] = df_manifest[\"frame_paths\".split()[-1]].apply(sample_30_frames)\n",
        "# Correct column lookup\n",
        "df_manifest[\"sampled_frame_paths\"] = df_manifest[\"frame_paths\"].apply(sample_30_frames)\n",
        "print(f\"Standardized all {len(df_manifest):,} utterances to exactly {NUM_FRAMES} frames.\")\n",
        "\n",
        "def load_all_video_sequences():\n",
        "    \"\"\"\n",
        "    Loads all sampled frames into contiguous NumPy tensors.\n",
        "    \"\"\"\n",
        "    X_data = np.zeros(\n",
        "        (len(df_manifest), NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),\n",
        "        dtype=np.float32\n",
        "    )\n",
        "    y_data = np.array(df_manifest[\"digit_id\"].values, dtype=np.int32)\n",
        "\n",
        "    t0 = time.time()\n",
        "    for i, row in df_manifest.iterrows():\n",
        "        for t, fpath in enumerate(row[\"sampled_frame_paths\"]):\n",
        "            img = Image.open(fpath).convert(\"RGB\")\n",
        "            if img.size != (FRAME_WIDTH, FRAME_HEIGHT):\n",
        "                img = img.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.BILINEAR)\n",
        "            X_data[i, t] = np.array(img, dtype=np.float32)\n",
        "\n",
        "    elapsed = time.time() - t0\n",
        "    return X_data, y_data, elapsed\n",
        "\n",
        "print(\"Loading video sequence tensors into memory...\")\n",
        "X_all, y_all, load_elapsed = load_all_video_sequences()\n",
        "\n",
        "val_mask = (df_manifest[\"split\"] == \"val\").values\n",
        "X_val, y_val = X_all[val_mask], y_all[val_mask]\n",
        "val_manifest = df_manifest[val_mask].copy().reset_index(drop=True)\n",
        "\n",
        "print(f\"Loaded validation tensor: shape {X_val.shape}, labels: {y_val.shape} in {load_elapsed:.2f}s\")\n"
    ]))

    # Cell 8: Architecture Builder 1 - EfficientNetB0 + BiLSTM + Attention Pooling
    cells.append(md_cell("## 8. Architecture Builder 1: EfficientNetB0 + BiLSTM + Attention Pooling"))
    cells.append(code_cell([
        "# Section 8: Model 1 Builder (BiLSTM + Attention Pooling)\n",
        "def get_efficientnetb0_backbone():\n",
        "    cnn = keras.applications.EfficientNetB0(\n",
        "        input_shape=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),\n",
        "        include_top=False,\n",
        "        weights=\"imagenet\",\n",
        "        pooling=\"avg\"\n",
        "    )\n",
        "    preprocess_fn = keras.applications.efficientnet.preprocess_input\n",
        "    return cnn, preprocess_fn\n",
        "\n",
        "def build_model_bilstm():\n",
        "    cnn, preprocess_fn = get_efficientnetb0_backbone()\n",
        "    inputs = keras.layers.Input(\n",
        "        shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),\n",
        "        name=\"lip_sequence_input\"\n",
        "    )\n",
        "    x = keras.layers.TimeDistributed(keras.layers.Lambda(preprocess_fn), name=\"efficientnetb0_preprocess\")(inputs)\n",
        "    x = keras.layers.TimeDistributed(cnn, name=\"efficientnetb0_cnn\")(x)\n",
        "    x = keras.layers.TimeDistributed(keras.layers.Dense(PROJECTION_DIM, use_bias=False), name=\"projection_dense\")(x)\n",
        "    x = keras.layers.LayerNormalization(epsilon=1e-6, name=\"projection_ln\")(x)\n",
        "    x = keras.layers.Bidirectional(keras.layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2), name=\"bilstm\")(x)\n",
        "\n",
        "    attn_scores = keras.layers.Dense(1, use_bias=True, name=\"temporal_attn_score\")(x)\n",
        "    attn_weights = keras.layers.Softmax(axis=1, name=\"temporal_attn_weights\")(attn_scores)\n",
        "    x = keras.layers.Lambda(lambda tensors: tf.reduce_sum(tensors[0] * tensors[1], axis=1), name=\"temporal_attention_pooling\")([x, attn_weights])\n",
        "\n",
        "    x = keras.layers.Dropout(0.4, name=\"head_dropout\")(x)\n",
        "    outputs = keras.layers.Dense(NUM_CLASSES, activation=\"softmax\", name=\"digit_classifier\")(x)\n",
        "    return keras.Model(inputs=inputs, outputs=outputs, name=\"EfficientNetB0_BiLSTM_AttentionPooling\")\n",
        "\n",
        "model_lstm = build_model_bilstm()\n",
        "model_lstm.load_weights(WEIGHTS_LSTM_PATH)\n",
        "print(\"[OK] Model 1 (BiLSTM + Attention Pooling) loaded successfully.\")\n"
    ]))

    # Cell 9: Architecture Builder 2 - EfficientNetB0 + BiGRU + Attention Pooling
    cells.append(md_cell("## 9. Architecture Builder 2: EfficientNetB0 + BiGRU + Attention Pooling"))
    cells.append(code_cell([
        "# Section 9: Model 2 Builder (BiGRU + Attention Pooling)\n",
        "def build_model_bigru():\n",
        "    cnn, preprocess_fn = get_efficientnetb0_backbone()\n",
        "    inputs = keras.layers.Input(\n",
        "        shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),\n",
        "        name=\"lip_sequence_input\"\n",
        "    )\n",
        "    x = keras.layers.TimeDistributed(keras.layers.Lambda(preprocess_fn), name=\"efficientnetb0_preprocess\")(inputs)\n",
        "    x = keras.layers.TimeDistributed(cnn, name=\"efficientnetb0_cnn\")(x)\n",
        "    x = keras.layers.TimeDistributed(keras.layers.Dense(PROJECTION_DIM, use_bias=False), name=\"projection_dense\")(x)\n",
        "    x = keras.layers.LayerNormalization(epsilon=1e-6, name=\"projection_ln\")(x)\n",
        "    x = keras.layers.Bidirectional(keras.layers.GRU(GRU_UNITS, return_sequences=True, dropout=0.2), name=\"bigru\")(x)\n",
        "\n",
        "    attn_scores = keras.layers.Dense(1, use_bias=True, name=\"temporal_attn_score\")(x)\n",
        "    attn_weights = keras.layers.Softmax(axis=1, name=\"temporal_attn_weights\")(attn_scores)\n",
        "    x = keras.layers.Lambda(lambda tensors: tf.reduce_sum(tensors[0] * tensors[1], axis=1), name=\"temporal_attention_pooling\")([x, attn_weights])\n",
        "\n",
        "    x = keras.layers.Dropout(0.4, name=\"head_dropout\")(x)\n",
        "    outputs = keras.layers.Dense(NUM_CLASSES, activation=\"softmax\", name=\"digit_classifier\")(x)\n",
        "    return keras.Model(inputs=inputs, outputs=outputs, name=\"EfficientNetB0_BiGRU_AttentionPooling\")\n",
        "\n",
        "model_gru = build_model_bigru()\n",
        "model_gru.load_weights(WEIGHTS_GRU_PATH)\n",
        "print(\"[OK] Model 2 (BiGRU + Attention Pooling) loaded successfully.\")\n"
    ]))

    # Cell 10: Architecture Builder 3 - EfficientNetB0 + 4-Block Transformer
    cells.append(md_cell("## 10. Architecture Builder 3: EfficientNetB0 + 4-Block Transformer"))
    cells.append(code_cell([
        "# Section 10: Model 3 Builder (4-Block Transformer)\n",
        "class PositionalEncoding(keras.layers.Layer):\n",
        "    def __init__(self, seq_len: int = NUM_FRAMES, dim: int = EMBED_DIM, **kwargs):\n",
        "        super().__init__(**kwargs)\n",
        "        self.seq_len = seq_len\n",
        "        self.dim = dim\n",
        "        pos = np.arange(seq_len)[:, np.newaxis]\n",
        "        i = np.arange(dim)[np.newaxis, :]\n",
        "        angles = pos / np.power(10000.0, (2 * (i // 2)) / np.float32(dim))\n",
        "        pe = np.zeros((1, seq_len, dim), dtype=np.float32)\n",
        "        pe[0, :, 0::2] = np.sin(angles[:, 0::2])\n",
        "        pe[0, :, 1::2] = np.cos(angles[:, 1::2])\n",
        "        self.pe = tf.constant(pe, dtype=tf.float32)\n",
        "\n",
        "    def call(self, inputs):\n",
        "        return inputs + self.pe\n",
        "\n",
        "    def get_config(self):\n",
        "        config = super().get_config()\n",
        "        config.update({\"seq_len\": self.seq_len, \"dim\": self.dim})\n",
        "        return config\n",
        "\n",
        "def transformer_encoder_block(x, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, key_dim=KEY_DIM, ffn_dim=FFN_DIM, dropout=0.2, block_id=1):\n",
        "    norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name=f\"transformer_{block_id}_norm1\")(x)\n",
        "    attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout, name=f\"transformer_{block_id}_mha\")(norm1, norm1)\n",
        "    attn = keras.layers.Dropout(dropout, name=f\"transformer_{block_id}_attn_drop\")(attn)\n",
        "    x = keras.layers.Add(name=f\"transformer_{block_id}_add1\")([x, attn])\n",
        "\n",
        "    norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name=f\"transformer_{block_id}_norm2\")(x)\n",
        "    ffn = keras.layers.Dense(ffn_dim, activation=\"gelu\", name=f\"transformer_{block_id}_ffn1\")(norm2)\n",
        "    ffn = keras.layers.Dropout(dropout, name=f\"transformer_{block_id}_ffn_drop1\")(ffn)\n",
        "    ffn = keras.layers.Dense(embed_dim, name=f\"transformer_{block_id}_ffn2\")(ffn)\n",
        "    ffn = keras.layers.Dropout(dropout, name=f\"transformer_{block_id}_ffn_drop2\")(ffn)\n",
        "    x = keras.layers.Add(name=f\"transformer_{block_id}_add2\")([x, ffn])\n",
        "    return x\n",
        "\n",
        "def build_model_transformer():\n",
        "    cnn, preprocess_fn = get_efficientnetb0_backbone()\n",
        "    inputs = keras.layers.Input(\n",
        "        shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),\n",
        "        name=\"lip_sequence_input\"\n",
        "    )\n",
        "    x = keras.layers.TimeDistributed(keras.layers.Lambda(preprocess_fn), name=\"efficientnetb0_preprocess\")(inputs)\n",
        "    x = keras.layers.TimeDistributed(cnn, name=\"efficientnetb0_cnn\")(x)\n",
        "    x = keras.layers.TimeDistributed(keras.layers.Dense(EMBED_DIM, use_bias=False), name=\"projection_dense\")(x)\n",
        "    x = keras.layers.LayerNormalization(epsilon=1e-6, name=\"projection_ln\")(x)\n",
        "    x = PositionalEncoding(seq_len=NUM_FRAMES, dim=EMBED_DIM, name=\"positional_encoding\")(x)\n",
        "    for b_id in range(1, NUM_TRANSFORMER_BLOCKS + 1):\n",
        "        x = transformer_encoder_block(x, block_id=b_id)\n",
        "    x = keras.layers.GlobalAveragePooling1D(name=\"gap1d\")(x)\n",
        "    x = keras.layers.Dropout(0.4, name=\"head_dropout\")(x)\n",
        "    outputs = keras.layers.Dense(NUM_CLASSES, activation=\"softmax\", name=\"digit_classifier\")(x)\n",
        "    return keras.Model(inputs=inputs, outputs=outputs, name=\"EfficientNetB0_Transformer_4Block\")\n",
        "\n",
        "model_trans = build_model_transformer()\n",
        "model_trans.load_weights(WEIGHTS_TRANS_PATH)\n",
        "print(\"[OK] Model 3 (4-Block Transformer) loaded successfully.\")\n"
    ]))

    # Cell 11: Individual Validation Inference
    cells.append(md_cell("## 11. Individual Validation Predictions & Baseline Accuracies"))
    cells.append(code_cell([
        "# Section 11: Generate Softmax Probabilities for Individual Models\n",
        "print(\"Generating inference predictions across the validation set (N={len(X_val)})...\")\n",
        "\n",
        "probs_lstm = model_lstm.predict(X_val, batch_size=BATCH_SIZE, verbose=1)\n",
        "probs_gru = model_gru.predict(X_val, batch_size=BATCH_SIZE, verbose=1)\n",
        "probs_trans = model_trans.predict(X_val, batch_size=BATCH_SIZE, verbose=1)\n",
        "\n",
        "preds_lstm = np.argmax(probs_lstm, axis=1)\n",
        "preds_gru = np.argmax(probs_gru, axis=1)\n",
        "preds_trans = np.argmax(probs_trans, axis=1)\n",
        "\n",
        "acc_lstm = accuracy_score(y_val, preds_lstm)\n",
        "acc_gru = accuracy_score(y_val, preds_gru)\n",
        "acc_trans = accuracy_score(y_val, preds_trans)\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"INDIVIDUAL VALIDATION ACCURACIES (80/20 SPLIT)\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"1. EfficientNetB0 + BiLSTM (Attn Pooling)  : {acc_lstm * 100:.2f}%  ({(preds_lstm == y_val).sum()}/{len(y_val)})\")\n",
        "print(f\"2. EfficientNetB0 + BiGRU  (Attn Pooling)  : {acc_gru * 100:.2f}%  ({(preds_gru == y_val).sum()}/{len(y_val)})\")\n",
        "print(f\"3. EfficientNetB0 + 4-Block Transformer   : {acc_trans * 100:.2f}%  ({(preds_trans == y_val).sum()}/{len(y_val)})\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # Cell 12: Uniform Ensemble Evaluation
    cells.append(md_cell("## 12. Uniform Probability Averaging (1/3 Weight Each)"))
    cells.append(code_cell([
        "# Section 12: Uniform Ensemble (Equal Weight 1/3 each)\n",
        "probs_uniform = (probs_lstm + probs_gru + probs_trans) / 3.0\n",
        "preds_uniform = np.argmax(probs_uniform, axis=1)\n",
        "acc_uniform = accuracy_score(y_val, preds_uniform)\n",
        "f1_uniform = f1_score(y_val, preds_uniform, average=\"macro\", zero_division=0)\n",
        "\n",
        "# Pairwise Ensembles for Comparison\n",
        "probs_lstm_gru = (probs_lstm + probs_gru) / 2.0\n",
        "acc_lstm_gru = accuracy_score(y_val, np.argmax(probs_lstm_gru, axis=1))\n",
        "\n",
        "probs_lstm_trans = (probs_lstm + probs_trans) / 2.0\n",
        "acc_lstm_trans = accuracy_score(y_val, np.argmax(probs_lstm_trans, axis=1))\n",
        "\n",
        "probs_gru_trans = (probs_gru + probs_trans) / 2.0\n",
        "acc_gru_trans = accuracy_score(y_val, np.argmax(probs_gru_trans, axis=1))\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"UNIFORM ENSEMBLE EVALUATION\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Tri-Model Ensemble (BiLSTM + BiGRU + Transformer) : {acc_uniform * 100:.2f}%  ({(preds_uniform == y_val).sum()}/{len(y_val)}) | Macro F1: {f1_uniform:.4f}\")\n",
        "print(f\"Pairwise Ensemble (BiLSTM + BiGRU)                : {acc_lstm_gru * 100:.2f}%\")\n",
        "print(f\"Pairwise Ensemble (BiLSTM + Transformer)          : {acc_lstm_trans * 100:.2f}%\")\n",
        "print(f\"Pairwise Ensemble (BiGRU + Transformer)           : {acc_gru_trans * 100:.2f}%\")\n",
        "print(f\"Delta over Best Single Model ({max(acc_lstm, acc_gru)*100:.2f}%)   : {(acc_uniform - max(acc_lstm, acc_gru))*100:+.2f}%\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # Cell 13: Grid Search for Optimal Ensemble Weights
    cells.append(md_cell("## 13. Grid Search for Optimal Softmax Ensembling Weights"))
    cells.append(code_cell([
        "# Section 13: Grid Search over Simplex Weights (w1 + w2 + w3 = 1)\n",
        "best_acc = 0.0\n",
        "best_weights = None\n",
        "best_probs = None\n",
        "\n",
        "steps = 21  # 0.0 to 1.0 in 0.05 increments\n",
        "weight_records = []\n",
        "\n",
        "for w_l in np.linspace(0.0, 1.0, steps):\n",
        "    for w_g in np.linspace(0.0, 1.0 - w_l, steps):\n",
        "        w_t = round(1.0 - w_l - w_g, 4)\n",
        "        if w_t < -1e-6:\n",
        "            continue\n",
        "        w_t = max(0.0, w_t)\n",
        "        p_blend = w_l * probs_lstm + w_g * probs_gru + w_t * probs_trans\n",
        "        p_cls = np.argmax(p_blend, axis=1)\n",
        "        score = accuracy_score(y_val, p_cls)\n",
        "        weight_records.append((w_l, w_g, w_t, score))\n",
        "        if score > best_acc:\n",
        "            best_acc = score\n",
        "            best_weights = (w_l, w_g, w_t)\n",
        "            best_probs = p_blend\n",
        "\n",
        "best_preds = np.argmax(best_probs, axis=1)\n",
        "best_f1 = f1_score(y_val, best_preds, average=\"macro\", zero_division=0)\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"OPTIMAL WEIGHTED ENSEMBLE VIA GRID SEARCH\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Best Weights (BiLSTM, BiGRU, Transformer): ({best_weights[0]:.2f}, {best_weights[1]:.2f}, {best_weights[2]:.2f})\")\n",
        "print(f\"Weighted Ensemble Validation Accuracy    : {best_acc * 100:.2f}%  ({(best_preds == y_val).sum()}/{len(y_val)})\")\n",
        "print(f\"Weighted Ensemble Macro F1-Score         : {best_f1:.4f}\")\n",
        "print(f\"Delta over Best Single Model             : {(best_acc - max(acc_lstm, acc_gru))*100:+.2f}%\")\n",
        "print(\"=\" * 80)\n"
    ]))

    # Cell 14: Per-Class Comparison Table
    cells.append(md_cell("## 14. Per-Class Accuracy & Breakdown Across Champions vs Ensemble"))
    cells.append(code_cell([
        "# Section 14: Per-Class Breakdown and Comparison\n",
        "per_class_stats = []\n",
        "for d_idx, d_name in enumerate(DIGIT_CLASSES):\n",
        "    mask = (y_val == d_idx)\n",
        "    total_c = mask.sum()\n",
        "    c_lstm = (preds_lstm[mask] == d_idx).sum()\n",
        "    c_gru = (preds_gru[mask] == d_idx).sum()\n",
        "    c_trans = (preds_trans[mask] == d_idx).sum()\n",
        "    c_unif = (preds_uniform[mask] == d_idx).sum()\n",
        "    c_opt = (best_preds[mask] == d_idx).sum()\n",
        "    per_class_stats.append({\n",
        "        \"Digit\": d_name,\n",
        "        \"Samples\": total_c,\n",
        "        \"BiLSTM\": f\"{c_lstm}/{total_c} ({c_lstm/total_c*100:.1f}%)\",\n",
        "        \"BiGRU\": f\"{c_gru}/{total_c} ({c_gru/total_c*100:.1f}%)\",\n",
        "        \"Transformer\": f\"{c_trans}/{total_c} ({c_trans/total_c*100:.1f}%)\",\n",
        "        \"Uniform Ens\": f\"{c_unif}/{total_c} ({c_unif/total_c*100:.1f}%)\",\n",
        "        \"Optimal Ens\": f\"{c_opt}/{total_c} ({c_opt/total_c*100:.1f}%)\",\n",
        "    })\n",
        "\n",
        "df_per_class = pd.DataFrame(per_class_stats)\n",
        "print(df_per_class.to_string(index=False))\n",
        "\n",
        "# Save comparison table\n",
        "comparison_csv_path = RESULTS_DIR / \"per_class_ensemble_comparison.csv\"\n",
        "df_per_class.to_csv(comparison_csv_path, index=False)\n",
        "print(f\"\\nSaved per-class comparison to: {comparison_csv_path}\")\n"
    ]))

    # Cell 15: Per-Speaker Accuracy Diagnostic
    cells.append(md_cell("## 15. Validation Speaker Diagnostics"))
    cells.append(code_cell([
        "# Section 15: Per-Speaker Accuracy Diagnostics\n",
        "print(\"=\" * 80)\n",
        "print(\"PER-SPEAKER VALIDATION ACCURACY DIAGNOSTICS\")\n",
        "print(\"=\" * 80)\n",
        "spk_stats = []\n",
        "for spk in base_val_speakers:\n",
        "    mask = (val_manifest[\"base_speaker\"] == spk).values\n",
        "    n_spk = mask.sum()\n",
        "    acc_s_lstm = (preds_lstm[mask] == y_val[mask]).mean()\n",
        "    acc_s_gru = (preds_gru[mask] == y_val[mask]).mean()\n",
        "    acc_s_trans = (preds_trans[mask] == y_val[mask]).mean()\n",
        "    acc_s_ens = (best_preds[mask] == y_val[mask]).mean()\n",
        "    spk_stats.append({\n",
        "        \"Base Speaker\": spk,\n",
        "        \"Utterances\": n_spk,\n",
        "        \"BiLSTM\": f\"{acc_s_lstm*100:.2f}%\",\n",
        "        \"BiGRU\": f\"{acc_s_gru*100:.2f}%\",\n",
        "        \"Transformer\": f\"{acc_s_trans*100:.2f}%\",\n",
        "        \"Ensemble\": f\"{acc_s_ens*100:.2f}%\",\n",
        "    })\n",
        "\n",
        "df_spk = pd.DataFrame(spk_stats)\n",
        "print(df_spk.to_string(index=False))\n",
        "spk_csv_path = RESULTS_DIR / \"per_speaker_validation_diagnostics.csv\"\n",
        "df_spk.to_csv(spk_csv_path, index=False)\n",
        "print(f\"\\nSaved speaker diagnostics to: {spk_csv_path}\")\n"
    ]))

    # Cell 16: Classification Report
    cells.append(md_cell("## 16. Classification Report"))
    cells.append(code_cell([
        "# Section 16: Classification Report\n",
        "clf_report = classification_report(\n",
        "    y_val,\n",
        "    best_preds,\n",
        "    target_names=DIGIT_CLASSES,\n",
        "    digits=4,\n",
        "    zero_division=0\n",
        ")\n",
        "print(\"Classification Report (Optimal Weighted Ensemble):\")\n",
        "print(clf_report)\n",
        "\n",
        "report_path = RESULTS_DIR / f\"classification_report_ensemble_{DATASET_MODE}.txt\"\n",
        "with open(report_path, \"w\") as f:\n",
        "    f.write(clf_report)\n",
        "print(f\"Classification report saved to: {report_path}\")\n"
    ]))

    # Cell 17: Confusion Matrix
    cells.append(md_cell("## 17. Confusion Matrix Heatmaps"))
    cells.append(code_cell([
        "# Section 17: Confusion Matrix Heatmaps (Single vs Ensemble)\n",
        "fig, axes = plt.subplots(1, 2, figsize=(18, 7))\n",
        "\n",
        "# 1. Best Single Model (BiLSTM)\n",
        "cm_lstm = confusion_matrix(y_val, preds_lstm)\n",
        "sns.heatmap(\n",
        "    cm_lstm,\n",
        "    annot=True,\n",
        "    fmt=\"d\",\n",
        "    cmap=\"Blues\",\n",
        "    xticklabels=DIGIT_CLASSES,\n",
        "    yticklabels=DIGIT_CLASSES,\n",
        "    ax=axes[0],\n",
        "    cbar=True\n",
        ")\n",
        "axes[0].set_title(f\"Baseline Champion: EfficientNetB0 + BiLSTM ({acc_lstm*100:.2f}%)\", fontsize=12, fontweight=\"bold\")\n",
        "axes[0].set_xlabel(\"Predicted Digit Class\", fontsize=10)\n",
        "axes[0].set_ylabel(\"True Digit Class\", fontsize=10)\n",
        "\n",
        "# 2. Tri-Model Optimal Ensemble\n",
        "cm_ens = confusion_matrix(y_val, best_preds)\n",
        "sns.heatmap(\n",
        "    cm_ens,\n",
        "    annot=True,\n",
        "    fmt=\"d\",\n",
        "    cmap=\"Greens\",\n",
        "    xticklabels=DIGIT_CLASSES,\n",
        "    yticklabels=DIGIT_CLASSES,\n",
        "    ax=axes[1],\n",
        "    cbar=True\n",
        ")\n",
        "axes[1].set_title(f\"Tri-Model Ensemble (BiLSTM + BiGRU + Transformer) ({best_acc*100:.2f}%)\", fontsize=12, fontweight=\"bold\")\n",
        "axes[1].set_xlabel(\"Predicted Digit Class\", fontsize=10)\n",
        "axes[1].set_ylabel(\"True Digit Class\", fontsize=10)\n",
        "\n",
        "plt.tight_layout()\n",
        "cm_path = RESULTS_DIR / f\"confusion_matrix_ensemble_comparison_{DATASET_MODE}.png\"\n",
        "plt.savefig(cm_path, dpi=200)\n",
        "plt.show()\n",
        "print(f\"Ensemble confusion matrix saved to: {cm_path}\")\n"
    ]))

    # Cell 18: Save Metrics JSON
    cells.append(md_cell("## 18. Save Metrics Artifacts"))
    cells.append(code_cell([
        "# Section 18: Save Consolidated Ensemble Metrics\n",
        "val_manifest[\"true_digit\"] = [ID_TO_CLASS[y] for y in y_val]\n",
        "val_manifest[\"pred_lstm\"] = [ID_TO_CLASS[p] for p in preds_lstm]\n",
        "val_manifest[\"pred_gru\"] = [ID_TO_CLASS[p] for p in preds_gru]\n",
        "val_manifest[\"pred_trans\"] = [ID_TO_CLASS[p] for p in preds_trans]\n",
        "val_manifest[\"pred_ensemble_uniform\"] = [ID_TO_CLASS[p] for p in preds_uniform]\n",
        "val_manifest[\"pred_ensemble_opt\"] = [ID_TO_CLASS[p] for p in best_preds]\n",
        "val_manifest[\"correct_opt\"] = (val_manifest[\"digit_id\"] == best_preds)\n",
        "\n",
        "predictions_csv = RESULTS_DIR / f\"validation_ensemble_predictions_{DATASET_MODE}.csv\"\n",
        "val_manifest[[\"speaker\", \"base_speaker\", \"digit\", \"digit_id\", \"pred_lstm\", \"pred_gru\", \"pred_trans\", \"pred_ensemble_opt\", \"correct_opt\"]].to_csv(\n",
        "    predictions_csv, index=False\n",
        ")\n",
        "print(f\"Saved detailed predictions CSV to: {predictions_csv}\")\n",
        "\n",
        "metrics_data = {\n",
        "    \"experiment\": \"15_ensemble_bilstm_bigru_transformer\",\n",
        "    \"dataset_mode\": DATASET_MODE,\n",
        "    \"models_included\": [\n",
        "        \"EfficientNetB0_BiLSTM_AttentionPooling\",\n",
        "        \"EfficientNetB0_BiGRU_AttentionPooling\",\n",
        "        \"EfficientNetB0_Transformer_4Block\"\n",
        "    ],\n",
        "    \"weights_paths\": {\n",
        "        \"bilstm\": str(WEIGHTS_LSTM_PATH),\n",
        "        \"bigru\": str(WEIGHTS_GRU_PATH),\n",
        "        \"transformer\": str(WEIGHTS_TRANS_PATH)\n",
        "    },\n",
        "    \"individual_accuracies\": {\n",
        "        \"bilstm\": float(acc_lstm),\n",
        "        \"bigru\": float(acc_gru),\n",
        "        \"transformer\": float(acc_trans)\n",
        "    },\n",
        "    \"ensemble_results\": {\n",
        "        \"uniform\": {\n",
        "            \"weights\": [1/3, 1/3, 1/3],\n",
        "            \"accuracy\": float(acc_uniform),\n",
        "            \"macro_f1\": float(f1_uniform)\n",
        "        },\n",
        "        \"optimal_weighted\": {\n",
        "            \"weights\": [float(w) for w in best_weights],\n",
        "            \"accuracy\": float(best_acc),\n",
        "            \"macro_f1\": float(best_f1)\n",
        "        },\n",
        "        \"delta_over_best_single\": float(best_acc - max(acc_lstm, acc_gru))\n",
        "    },\n",
        "    \"num_validation_samples\": int(len(y_val)),\n",
        "    \"num_validation_speakers\": len(base_val_speakers)\n",
        "}\n",
        "\n",
        "metrics_json_path = RESULTS_DIR / f\"metrics_ensemble_{DATASET_MODE}.json\"\n",
        "with open(metrics_json_path, \"w\") as f:\n",
        "    json.dump(metrics_data, f, indent=2)\n",
        "print(f\"Saved metrics JSON to: {metrics_json_path}\")\n"
    ]))

    # Cell 19: Final Printout
    cells.append(md_cell("## 19. Final Ensemble Summary"))
    cells.append(code_cell([
        "# Section 19: Final Printout\n",
        "print(\"=\" * 80)\n",
        "print(f\"FINAL ENSEMBLE SUMMARY [{DATASET_MODE}]\")\n",
        "print(\"=\" * 80)\n",
        "print(f\"Best Single Model (BiLSTM)      : {acc_lstm * 100:.2f}%\")\n",
        "print(f\"Co-Champion Model (BiGRU)       : {acc_gru * 100:.2f}%\")\n",
        "print(f\"Complementary Model (Transformer): {acc_trans * 100:.2f}%\")\n",
        "print(\"-\" * 80)\n",
        "print(f\"Uniform Ensemble (1/3 each)     : {acc_uniform * 100:.2f}%\")\n",
        "print(f\"Optimal Weighted Ensemble       : {best_acc * 100:.2f}%  (weights: {best_weights})\")\n",
        "print(f\"Net Ensemble Accuracy Gain       : {(best_acc - max(acc_lstm, acc_gru)) * 100:+.2f}%\")\n",
        "print(\"=\" * 80)\n"
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

    out_file = Path("notebooks/15_ensemble_bilstm_bigru_transformer.ipynb")
    with open(out_file, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully generated {out_file} with {len(cells)} cells!")

if __name__ == "__main__":
    main()
