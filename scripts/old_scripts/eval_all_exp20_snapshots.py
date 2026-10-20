"""
Script: eval_all_exp20_snapshots.py
Evaluates all Experiment 20 snapshots + ensemble and prints the full confusion matrix.
"""

import ast
import json
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

NUM_CLASSES = 11
NUM_FRAMES = 30
FRAME_WIDTH = 96
FRAME_HEIGHT = 96
FRAME_CHANNELS = 4
BATCH_SIZE = 16

DIGIT_CLASSES = [f"d{i}" for i in range(NUM_CLASSES)]

PROJECT_ROOT = Path("/Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading")
RESULTS_DIR = PROJECT_ROOT / "results" / "efficientnetb0_bigru_targeted_focal_top1_80_20"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"

df = pd.read_csv(RESULTS_DIR / "utterance_manifest.csv")
val_df = df[df["split"] == "val"].reset_index(drop=True)

def sample_30(frames):
    n = len(frames)
    indices = np.round(np.linspace(0, n - 1, 30)).astype(int) if n != 30 else list(range(30))
    return [str(frames[i]) for i in indices]

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

def generate_4view_tta(X_seq_4ch: np.ndarray) -> list:
    v1 = X_seq_4ch.copy()
    v2 = np.empty_like(X_seq_4ch)
    v2[:-1] = X_seq_4ch[1:]; v2[-1] = X_seq_4ch[-1]
    v3 = np.empty_like(X_seq_4ch)
    v3[0] = X_seq_4ch[0]; v3[1:] = X_seq_4ch[:-1]
    v4 = np.flip(X_seq_4ch, axis=2).copy()
    return [v1, v2, v3, v4]

@keras.saving.register_keras_serializable()
class TemporalAttentionPooling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = None

    def build(self, input_shape):
        self.score_dense = keras.layers.Dense(1, use_bias=False, name="attn_score")
        super().build(input_shape)

    def call(self, inputs):
        scores = self.score_dense(inputs)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)

def build_model():
    frame_in = keras.Input(shape=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), name="frame_4ch_input")
    adapter = keras.layers.Conv2D(filters=3, kernel_size=(1, 1), padding="same", use_bias=False, name="motion_stem_adapter")
    adapted_rgb = adapter(frame_in)
    base_cnn = keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3))
    preprocessed = keras.applications.efficientnet.preprocess_input(adapted_rgb)
    cnn_features = base_cnn(preprocessed)
    pooled = keras.layers.GlobalAveragePooling2D(name="gap")(cnn_features)
    extractor = keras.Model(inputs=frame_in, outputs=pooled, name="EfficientNetB0_4Ch_Extractor")

    seq_input = keras.Input(shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), name="sequence_4ch_input")
    x = keras.layers.TimeDistributed(extractor, name="time_distributed_cnn")(seq_input)
    x = keras.layers.Dense(256, activation="relu", name="projection_dense")(x)
    x = keras.layers.Dropout(0.2, name="projection_dropout")(x)
    x = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True, dropout=0.2), name="bidirectional_gru")(x)
    x = TemporalAttentionPooling(name="temporal_attention_pooling")(x)
    x = keras.layers.Dropout(0.4, name="head_dropout")(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="softmax_classification")(x)
    return keras.Model(inputs=seq_input, outputs=outputs, name="EfficientNetB0_BiGRU_Exp20")

print("Loading validation sequences...")
VAL_DIR = PROJECT_ROOT / "data" / "processed" / "val"
X_val_raw = np.zeros((len(val_df), NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)
y_val = val_df["digit_id"].values

for i, row in val_df.iterrows():
    folder = row["speaker"]
    digit = row["digit"]
    dp = VAL_DIR / folder / digit
    fpaths = sorted([f for f in dp.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    indices = np.round(np.linspace(0, len(fpaths) - 1, NUM_FRAMES)).astype(int) if len(fpaths) != NUM_FRAMES else list(range(NUM_FRAMES))
    for t, idx in enumerate(indices):
        img = Image.open(fpaths[idx]).convert("RGB")
        if img.size != (FRAME_WIDTH, FRAME_HEIGHT):
            img = img.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.BILINEAR)
        X_val_raw[i, t] = np.array(img, dtype=np.float32)

X_val_4ch = np.zeros((len(val_df), NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=np.float32)
for i in range(len(val_df)):
    X_val_4ch[i] = compute_frame_differences(X_val_raw[i])

views_val = [np.zeros_like(X_val_4ch) for _ in range(4)]
for i in range(len(X_val_4ch)):
    v_list = generate_4view_tta(X_val_4ch[i])
    for v_idx in range(4):
        views_val[v_idx][i] = v_list[v_idx]

model = build_model()

snapshots = sorted(list(SNAPSHOT_DIR.glob("*.weights.h5")))
print(f"Found {len(snapshots)} snapshots in {SNAPSHOT_DIR}:")
for s in snapshots:
    print(f"  - {s.name}")

all_snapshot_probs = []

for idx, snap_path in enumerate(snapshots, 1):
    print("\n" + "=" * 80)
    print(f"EVALUATING SNAPSHOT {idx}: {snap_path.name}")
    print("=" * 80)
    model.load_weights(snap_path)
    
    view_probs = []
    for v_idx, v_tensor in enumerate(views_val, 1):
        p = model.predict(v_tensor, batch_size=BATCH_SIZE, verbose=0)
        view_probs.append(p)
        
    avg_probs = np.mean(view_probs, axis=0)
    all_snapshot_probs.append(avg_probs)
    preds = np.argmax(avg_probs, axis=1)
    
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    print(f"Accuracy: {acc * 100:.2f}% | Macro F1: {f1:.4f}")
    
    cm = confusion_matrix(y_val, preds)
    print("Confusion Matrix:")
    print("     " + " ".join([f"{d:>4}" for d in DIGIT_CLASSES]))
    for r_idx, row in enumerate(cm):
        print(f"{DIGIT_CLASSES[r_idx]:<4} " + " ".join([f"{val:>4}" for val in row]))

if len(all_snapshot_probs) > 1:
    print("\n" + "=" * 80)
    print("SNAPSHOT ENSEMBLE (Average of all 3 snapshots + 4-View TTA)")
    print("=" * 80)
    ens_probs = np.mean(all_snapshot_probs, axis=0)
    ens_preds = np.argmax(ens_probs, axis=1)
    ens_acc = accuracy_score(y_val, ens_preds)
    ens_f1 = f1_score(y_val, ens_preds, average="macro")
    print(f"Ensemble Accuracy: {ens_acc * 100:.2f}% | Macro F1: {ens_f1:.4f}")
    cm_ens = confusion_matrix(y_val, ens_preds)
    print("Confusion Matrix:")
    print("     " + " ".join([f"{d:>4}" for d in DIGIT_CLASSES]))
    for r_idx, row in enumerate(cm_ens):
        print(f"{DIGIT_CLASSES[r_idx]:<4} " + " ".join([f"{val:>4}" for val in row]))
