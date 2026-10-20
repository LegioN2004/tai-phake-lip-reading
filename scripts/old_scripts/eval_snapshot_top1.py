import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf
import keras

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "efficientnetb0_bigru_motion_focal_snapshot_80_20"
SNAPSHOT_PATH = RESULTS_DIR / "snapshots" / "stage2_epoch_15_loss_0.1993.weights.h5"

print("Snapshot Path:", SNAPSHOT_PATH, "Exists:", SNAPSHOT_PATH.exists())

# Manifest
df_manifest = pd.read_csv(RESULTS_DIR / "utterance_manifest.csv")
val_mask = (df_manifest["split"] == "val").values
df_val = df_manifest[val_mask].reset_index(drop=True)
print(f"Validation utterances: {len(df_val)}")

NUM_FRAMES = 30
FRAME_HEIGHT = 96
FRAME_WIDTH = 96
FRAME_CHANNELS = 4
PROJECTION_DIM = 256
GRU_UNITS = 128
GRU_DROPOUT = 0.2
HEAD_DROPOUT = 0.3
NUM_CLASSES = 11

def compute_frame_differences(seq_rgb: np.ndarray) -> np.ndarray:
    T, H, W, _ = seq_rgb.shape
    gray = (
        0.2989 * seq_rgb[..., 0] +
        0.5870 * seq_rgb[..., 1] +
        0.1140 * seq_rgb[..., 2]
    )[..., np.newaxis]
    diff = np.zeros((T, H, W, 1), dtype=np.float32)
    diff[1:] = np.abs(gray[1:] - gray[:-1])
    return np.concatenate([seq_rgb, diff], axis=-1)

# Load validation raw RGB into RAM
VAL_DIR = PROJECT_ROOT / "data" / "processed" / "val"
X_val_rgb = np.zeros((len(df_val), NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)
y_val = df_val["digit_id"].values

for i, row in df_val.iterrows():
    folder = row["speaker"]
    digit = row["digit"]
    dp = VAL_DIR / folder / digit
    fpaths = sorted(dp.glob("*.jpg"))
    # Uniform sample to 30 frames
    indices = np.round(np.linspace(0, len(fpaths) - 1, NUM_FRAMES)).astype(int)
    for t, idx in enumerate(indices):
        img = Image.open(fpaths[idx]).convert("RGB")
        X_val_rgb[i, t] = np.array(img, dtype=np.float32)

X_val_4ch = np.zeros((len(df_val), NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=np.float32)
for i in range(len(df_val)):
    X_val_4ch[i] = compute_frame_differences(X_val_rgb[i])

print("Precomputed X_val_4ch:", X_val_4ch.shape)

# Build Model exactly matching Notebook 19
@keras.saving.register_keras_serializable()
class TemporalAttentionPooling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = None

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.score_dense = keras.layers.Dense(1, use_bias=False, name="attn_score")
        super().build(input_shape)

    def call(self, inputs):
        scores = self.score_dense(inputs)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)

def build_4channel_efficientnet_extractor():
    frame_in = keras.Input(shape=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), name="frame_4ch_input")
    adapter = keras.layers.Conv2D(
        filters=3,
        kernel_size=(1, 1),
        padding="same",
        use_bias=False,
        name="motion_stem_adapter"
    )
    adapted_rgb = adapter(frame_in)
    base_cnn = keras.applications.EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3)
    )
    preprocessed = keras.applications.efficientnet.preprocess_input(adapted_rgb)
    cnn_features = base_cnn(preprocessed)
    pooled = keras.layers.GlobalAveragePooling2D(name="gap")(cnn_features)
    extractor = keras.Model(inputs=frame_in, outputs=pooled, name="EfficientNetB0_4Ch_Extractor")
    return extractor, base_cnn, adapter

def build_full_vsr_model():
    extractor, base_cnn, adapter = build_4channel_efficientnet_extractor()
    seq_input = keras.Input(shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), name="sequence_4ch_input")
    x = keras.layers.TimeDistributed(extractor, name="time_distributed_cnn")(seq_input)
    x = keras.layers.Dense(PROJECTION_DIM, activation="relu", name="projection_dense")(x)
    x = keras.layers.Dropout(0.2, name="projection_dropout")(x)
    x = keras.layers.Bidirectional(
        keras.layers.GRU(GRU_UNITS, return_sequences=True, dropout=GRU_DROPOUT),
        name="bidirectional_gru"
    )(x)
    x = TemporalAttentionPooling(name="temporal_attention_pooling")(x)
    x = keras.layers.Dropout(HEAD_DROPOUT, name="head_dropout")(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="softmax_classification")(x)
    model = keras.Model(inputs=seq_input, outputs=outputs, name="EfficientNetB0_BiGRU_Motion4Ch")
    return model

model = build_full_vsr_model()
model.load_weights(SNAPSHOT_PATH)
print("Loaded Snapshot 1 weights successfully!")

# 4-View TTA
def generate_4view_tta(X_seq_4ch):
    v1 = X_seq_4ch.copy()
    v2 = np.empty_like(X_seq_4ch)
    v2[:-1] = X_seq_4ch[1:]
    v2[-1] = X_seq_4ch[-1]
    v3 = np.empty_like(X_seq_4ch)
    v3[0] = X_seq_4ch[0]
    v3[1:] = X_seq_4ch[:-1]
    v4 = np.flip(X_seq_4ch, axis=2).copy()
    return [v1, v2, v3, v4]

N_val = len(X_val_4ch)
views_val = [np.zeros_like(X_val_4ch) for _ in range(4)]
for i in range(N_val):
    v_list = generate_4view_tta(X_val_4ch[i])
    for v_idx in range(4):
        views_val[v_idx][i] = v_list[v_idx]

tta_probs = np.zeros((N_val, NUM_CLASSES), dtype=np.float32)
for v_idx in range(4):
    p = model.predict(views_val[v_idx], batch_size=16, verbose=0)
    tta_probs += p
tta_probs /= 4.0

y_pred = np.argmax(tta_probs, axis=-1)
acc = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(y_val, y_pred, average="macro")
macro_prec = precision_score(y_val, y_pred, average="macro")
macro_rec = recall_score(y_val, y_pred, average="macro")

print("\n" + "=" * 80)
print(f"TOP-1 SNAPSHOT (Epoch 15) + 4-VIEW TTA EVALUATION RESULTS")
print("=" * 80)
print(f"Validation Accuracy : {acc * 100:.2f}% ({np.sum(y_pred == y_val)}/{len(y_val)})")
print(f"Macro F1-Score      : {macro_f1:.4f}")
print(f"Macro Precision     : {macro_prec:.4f}")
print(f"Macro Recall        : {macro_rec:.4f}")
print("-" * 80)
digits = [f"d{i}" for i in range(11)]
print(classification_report(y_val, y_pred, target_names=digits, digits=4))

# Per-speaker breakdown
df_val["correct"] = (y_pred == y_val)
spk_group = df_val.groupby("base_speaker")["correct"].agg(["sum", "count"])
spk_group["acc"] = spk_group["sum"] / spk_group["count"] * 100
print("Per-Speaker Breakdown:")
print(spk_group)
