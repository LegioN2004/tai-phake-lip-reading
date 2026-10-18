"""
Script: train_experiment_21.py
Purpose: Runs full end-to-end training and Top-1 Snapshot + 4-View TTA evaluation for Experiment 21.
Specifications:
  1. Balanced clean dataset (dlib_aug_cropd_98_78_20_96x96_replaced).
  2. Remove Targeted Affine Augmentation: No random rotation or zoom.
  3. Calibrated Focal Loss Weights: [1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 0.1]
     (Heavily suppresses d8 and d10 to 0.1 to stop default guessing; d2 given 1.2; others 1.0).
  4. Increase Focal Loss Gamma: gamma = 2.0 (focus harder on hard misclassifications).
  5. Temporal Frame Dropping: Randomly drop 2-3 frames and interpolate back to 30 frames for all classes.
  6. Extended Fine-Tuning: Stage 2 early stopping patience increased to 10 epochs (max 35 epochs).
  7. Top-1 Snapshot Inference + 4-View TTA (evaluate single best checkpoint).
"""

import os
import sys
import re
import json
import time
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import tensorflow as tf
import keras

def main():
    print("=" * 80)
    print("STARTING EXPERIMENT 21: FOCAL SUPPRESSION (d8/d10=0.1) + GAMMA 2.0 + TEMPORAL DROP")
    print("=" * 80)

    # 1. Reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    DATASET_MODE = "80_20"
    NUM_CLASSES = 11
    NUM_FRAMES = 30
    FRAME_WIDTH = 96
    FRAME_HEIGHT = 96
    FRAME_CHANNELS = 4
    BATCH_SIZE = 16

    PROJECTION_DIM = 256
    GRU_UNITS = 128
    GRU_DROPOUT = 0.2
    HEAD_DROPOUT = 0.4

    # Focal Loss with Class Weights:
    # d8: 0.1, d10: 0.1 (heavily suppressed to stop default-guessing)
    # d2: 1.2, others: 1.0
    CLASS_WEIGHTS = [1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 0.1]
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.25
    LABEL_SMOOTHING = 0.05

    STAGE1_LR = 1e-4
    STAGE1_MAX_EPOCHS = 40
    STAGE1_PATIENCE = 8

    STAGE2_LR = 1e-5
    STAGE2_MAX_EPOCHS = 35
    STAGE2_PATIENCE = 10
    FINE_TUNE_FRACTION = 0.20

    DIGIT_CLASSES = [f"d{i}" for i in range(NUM_CLASSES)]
    CLASS_TO_ID = {d: i for i, d in enumerate(DIGIT_CLASSES)}
    ID_TO_CLASS = {i: d for i, d in enumerate(DIGIT_CLASSES)}

    PROJECT_ROOT = Path("/Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading")
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"

    RESULTS_DIR = PROJECT_ROOT / "results" / f"efficientnetb0_bigru_suppressed_focal_temporal_drop_{DATASET_MODE}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean any stale snapshot files
    for old_file in SNAPSHOT_DIR.glob("*.weights.h5"):
        old_file.unlink()

    print(f"Results Directory   : {RESULTS_DIR}")
    print(f"Snapshots Directory : {SNAPSHOT_DIR}")
    print(f"Class Weights       : {CLASS_WEIGHTS}")
    print(f"Focal Loss Gamma    : {FOCAL_GAMMA}")
    print(f"Stage 2 Max Epochs  : {STAGE2_MAX_EPOCHS} (Patience: {STAGE2_PATIENCE})")

    # 2. Manifest and Speaker Verification
    train_speakers = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
    val_speakers = sorted([d.name for d in VAL_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])

    def get_base_speaker(spk_name: str) -> str:
        return spk_name.split("_")[0]

    base_train_speakers = sorted(list({get_base_speaker(s) for s in train_speakers}))
    base_val_speakers = sorted(list({get_base_speaker(s) for s in val_speakers}))

    overlap = set(base_train_speakers).intersection(set(base_val_speakers))
    assert len(overlap) == 0, f"LEAKAGE DETECTED: Overlapping speakers: {overlap}"
    print(f"Zero leakage verified! Base Train: {len(base_train_speakers)}, Base Val: {len(base_val_speakers)}")

    utterances = []
    for split_name, split_dir, spk_list in [("train", TRAIN_DIR, train_speakers), ("val", VAL_DIR, val_speakers)]:
        for spk in spk_list:
            spk_path = split_dir / spk
            for digit_dir in sorted(spk_path.iterdir()):
                if not digit_dir.is_dir() or digit_dir.name.startswith("."):
                    continue
                digit_name = digit_dir.name
                if digit_name not in CLASS_TO_ID:
                    continue
                frames = sorted([
                    f for f in digit_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ])
                if len(frames) == 0:
                    continue
                utterances.append({
                    "speaker": spk,
                    "base_speaker": get_base_speaker(spk),
                    "digit": digit_name,
                    "digit_id": CLASS_TO_ID[digit_name],
                    "split": split_name,
                    "raw_frame_count": len(frames),
                    "frame_paths": frames
                })

    df_manifest = pd.DataFrame(utterances)
    df_manifest.to_csv(RESULTS_DIR / "utterance_manifest.csv", index=False)
    print(f"Total Utterances: {len(df_manifest)} (Train: {(df_manifest['split']=='train').sum()}, Val: {(df_manifest['split']=='val').sum()})")

    def sample_30_frame_paths(frame_paths: list, target_frames: int = NUM_FRAMES) -> list:
        n = len(frame_paths)
        if n == target_frames:
            indices = list(range(target_frames))
        else:
            indices = np.round(np.linspace(0, n - 1, target_frames)).astype(int)
        return [str(frame_paths[i]) for i in indices]

    df_manifest["sampled_frame_paths"] = df_manifest["frame_paths"].apply(sample_30_frame_paths)

    # 3. Temporal Frame Dropping & Motion Channel Functions
    def apply_temporal_frame_drop(seq_rgb: np.ndarray, num_drop: int = None) -> np.ndarray:
        """
        Randomly drops 2-3 interior frames from the 30-frame sequence for all classes
        and linearly interpolates back to exactly 30 frames.
        Forces the model to learn the visual motion dynamics rather than memorizing static frames.
        """
        T, H, W, C = seq_rgb.shape
        if num_drop is None:
            num_drop = int(np.random.choice([2, 3]))
        drop_indices = np.random.choice(range(1, T - 1), size=num_drop, replace=False)
        keep_indices = [i for i in range(T) if i not in drop_indices]
        new_t = np.linspace(0, len(keep_indices) - 1, T)
        left_idx = np.floor(new_t).astype(int)
        right_idx = np.ceil(new_t).astype(int)
        alpha = (new_t - left_idx)[:, np.newaxis, np.newaxis, np.newaxis]
        kept_seq = seq_rgb[keep_indices]
        jittered = (1.0 - alpha) * kept_seq[left_idx] + alpha * kept_seq[right_idx]
        return jittered.astype(np.float32)

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

    # 4. Load Raw RGB Sequences
    print("Loading base RGB sequences into memory...")
    t0 = time.time()
    num_total = len(df_manifest)
    X_all_rgb = np.zeros((num_total, NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)
    y_all = np.array(df_manifest["digit_id"].values, dtype=np.int32)

    for i, row in df_manifest.iterrows():
        for t, fpath in enumerate(row["sampled_frame_paths"]):
            img = Image.open(fpath).convert("RGB")
            if img.size != (FRAME_WIDTH, FRAME_HEIGHT):
                img = img.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.BILINEAR)
            X_all_rgb[i, t] = np.array(img, dtype=np.float32)

    load_time = time.time() - t0
    train_mask = (df_manifest["split"] == "train").values
    val_mask = (df_manifest["split"] == "val").values

    X_train_raw, y_train = X_all_rgb[train_mask], y_all[train_mask]
    X_val_raw, y_val = X_all_rgb[val_mask], y_all[val_mask]
    print(f"Loaded {num_total} sequences in {load_time:.2f}s! Train: {X_train_raw.shape}, Val: {X_val_raw.shape}")

    # 5. Dataloader PyDataset
    class MotionLipDataset(keras.utils.PyDataset):
        def __init__(
            self,
            X_rgb: np.ndarray,
            y: np.ndarray,
            batch_size: int = BATCH_SIZE,
            shuffle: bool = True,
            is_training: bool = False,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.X_rgb = X_rgb
            self.y = y
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.is_training = is_training
            self.indices = np.arange(len(X_rgb))
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.X_rgb) / self.batch_size))

        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indices)

        def __getitem__(self, idx):
            batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch_x = []
            batch_y = self.y[batch_idx]

            for i in batch_idx:
                seq = self.X_rgb[i].copy()

                # 1. Temporal frame dropping (2-3 frames dropped and interpolated) for all classes in training
                if self.is_training:
                    seq = apply_temporal_frame_drop(seq)

                # 2. Calculate Frame N - Frame N-1 and stack as 4th channel
                seq_4ch = compute_frame_differences(seq)
                batch_x.append(seq_4ch)

            return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.int32)

    train_generator = MotionLipDataset(
        X_train_raw, y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        is_training=True
    )

    print("Precomputing static 4-channel validation tensor (X_val_4ch)...")
    X_val_4ch = np.zeros(
        (len(X_val_raw), NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),
        dtype=np.float32
    )
    for i in range(len(X_val_raw)):
        X_val_4ch[i] = compute_frame_differences(X_val_raw[i])

    # 6. Focal Loss Implementation
    @keras.saving.register_keras_serializable()
    class CategoricalFocalLoss(keras.losses.Loss):
        def __init__(
            self,
            gamma: float = FOCAL_GAMMA,
            alpha: float = FOCAL_ALPHA,
            class_weights: list = None,
            label_smoothing: float = LABEL_SMOOTHING,
            name: str = "categorical_focal_loss",
            **kwargs
        ):
            super().__init__(name=name, **kwargs)
            self.gamma = float(gamma)
            self.alpha = float(alpha)
            self.label_smoothing = float(label_smoothing)
            self.class_weights = class_weights
            if class_weights is not None:
                self._class_weights_t = tf.constant(class_weights, dtype=tf.float32)
            else:
                self._class_weights_t = None

        def call(self, y_true, y_pred):
            eps = keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

            num_classes = tf.shape(y_pred)[-1]
            y_true_rank = len(y_true.shape)

            if y_true_rank is not None and y_true_rank == 1:
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
            elif y_true_rank is not None and y_true_rank == 2 and y_true.shape[-1] == 1:
                y_true = tf.one_hot(tf.cast(tf.squeeze(y_true, axis=-1), tf.int32), depth=num_classes)

            y_true = tf.cast(y_true, y_pred.dtype)

            if self.label_smoothing > 0.0:
                c = tf.cast(num_classes, y_pred.dtype)
                y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / c)

            focal_weight = tf.pow(1.0 - y_pred, self.gamma)
            cross_entropy = -tf.math.log(y_pred)
            loss = self.alpha * focal_weight * cross_entropy

            if self._class_weights_t is not None:
                loss = loss * self._class_weights_t

            per_sample_loss = tf.reduce_sum(y_true * loss, axis=-1)
            return tf.reduce_mean(per_sample_loss)

        def get_config(self):
            config = super().get_config()
            config.update({
                "gamma": self.gamma,
                "alpha": self.alpha,
                "class_weights": self.class_weights,
                "label_smoothing": self.label_smoothing,
            })
            return config

    # 7. Model Architecture
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

    def build_4channel_efficientnet_extractor():
        frame_in = keras.Input(shape=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), name="frame_4ch_input")
        adapter = keras.layers.Conv2D(filters=3, kernel_size=(1, 1), padding="same", use_bias=False, name="motion_stem_adapter")
        adapted_rgb = adapter(frame_in)
        base_cnn = keras.applications.EfficientNetB0(weights="imagenet", include_top=False, input_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3))
        preprocessed = keras.applications.efficientnet.preprocess_input(adapted_rgb)
        cnn_features = base_cnn(preprocessed)
        pooled = keras.layers.GlobalAveragePooling2D(name="gap")(cnn_features)
        extractor = keras.Model(inputs=frame_in, outputs=pooled, name="EfficientNetB0_4Ch_Extractor")

        w = np.zeros((1, 1, 4, 3), dtype=np.float32)
        w[0, 0, 0, 0] = 1.0
        w[0, 0, 1, 1] = 1.0
        w[0, 0, 2, 2] = 1.0
        w[0, 0, 3, :] = 0.10
        adapter.set_weights([w])
        return extractor, base_cnn, adapter

    def build_full_vsr_model():
        extractor, base_cnn, adapter = build_4channel_efficientnet_extractor()
        seq_input = keras.Input(shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), name="sequence_4ch_input")
        x = keras.layers.TimeDistributed(extractor, name="time_distributed_cnn")(seq_input)
        x = keras.layers.Dense(PROJECTION_DIM, activation="relu", name="projection_dense")(x)
        x = keras.layers.Dropout(0.2, name="projection_dropout")(x)
        x = keras.layers.Bidirectional(keras.layers.GRU(GRU_UNITS, return_sequences=True, dropout=GRU_DROPOUT), name="bidirectional_gru")(x)
        x = TemporalAttentionPooling(name="temporal_attention_pooling")(x)
        x = keras.layers.Dropout(HEAD_DROPOUT, name="head_dropout")(x)
        outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="softmax_classification")(x)
        model = keras.Model(inputs=seq_input, outputs=outputs, name="EfficientNetB0_BiGRU_Exp21")
        return model, base_cnn, adapter

    model, base_cnn, motion_adapter = build_full_vsr_model()
    print(f"Model assembled! Total Parameters: {model.count_params():,}")

    # 8. Snapshot Ensembling Callback
    class TopKSnapshotCheckpoint(keras.callbacks.Callback):
        def __init__(self, snapshot_dir: Path, k: int = 3, prefix: str = "stage2"):
            super().__init__()
            self.snapshot_dir = Path(snapshot_dir)
            self.k = k
            self.prefix = prefix
            self.best_snapshots = []

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_loss = logs.get("val_loss")
            val_acc = logs.get("val_accuracy", 0.0)
            if val_loss is None:
                return

            epoch_num = epoch + 1
            temp_path = self.snapshot_dir / f"{self.prefix}_epoch_{epoch_num:02d}_loss_{val_loss:.4f}.weights.h5"

            if len(self.best_snapshots) < self.k or val_loss < max(s[0] for s in self.best_snapshots):
                self.model.save_weights(temp_path)
                self.best_snapshots.append((val_loss, epoch_num, temp_path, val_acc))
                self.best_snapshots.sort(key=lambda x: x[0])

                if len(self.best_snapshots) > self.k:
                    removed = self.best_snapshots.pop(-1)
                    if removed[2].exists():
                        removed[2].unlink()

                print(f"\n>>> [Snapshot Tracker] Epoch {epoch_num:02d} qualified! (val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%)")
                for rank, s in enumerate(self.best_snapshots, 1):
                    print(f"      Rank {rank}: Epoch {s[1]:02d} | Val Loss: {s[0]:.4f} | Val Acc: {s[3]*100:.2f}% ({s[2].name})")

        def get_top1_path(self):
            return self.best_snapshots[0][2] if self.best_snapshots else None

    # 9. Stage 1 Training Execution
    print("\n" + "=" * 80)
    print("STAGE 1: FROZEN BACKBONE TRAINING (Weighted Focal Loss, Gamma=2.0)")
    print("=" * 80)
    base_cnn.trainable = False
    for l in base_cnn.layers:
        l.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE1_LR),
        loss=CategoricalFocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, class_weights=CLASS_WEIGHTS, label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"]
    )

    stage1_weights_path = RESULTS_DIR / "stage1_best.weights.h5"
    checkpoint1 = keras.callbacks.ModelCheckpoint(
        stage1_weights_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    early1 = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=STAGE1_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    t_s1 = time.time()
    history1 = model.fit(
        train_generator,
        validation_data=(X_val_4ch, y_val),
        epochs=STAGE1_MAX_EPOCHS,
        callbacks=[checkpoint1, early1],
        verbose=1
    )
    s1_elapsed = time.time() - t_s1

    if stage1_weights_path.exists():
        model.load_weights(stage1_weights_path)

    stage1_best_epoch = int(np.argmin(history1.history["val_loss"]) + 1)
    stage1_best_val_acc = float(max(history1.history["val_accuracy"]))
    stage1_best_val_loss = float(min(history1.history["val_loss"]))
    print(f"Stage 1 Best Epoch: {stage1_best_epoch} | Loss: {stage1_best_val_loss:.4f} | Acc: {stage1_best_val_acc*100:.2f}% | Elapsed: {s1_elapsed:.2f}s")

    # 10. Stage 2 Fine-Tuning Execution
    print("\n" + "=" * 80)
    print("STAGE 2: TOP 20% FINE-TUNING (Snapshot Tracking, Patience=10)")
    print("=" * 80)
    def unfreeze_last_fraction(backbone, fraction=FINE_TUNE_FRACTION):
        for layer in backbone.layers:
            layer.trainable = False
        total_layers = len(backbone.layers)
        start_index = int(total_layers * (1.0 - fraction))
        for layer in backbone.layers[start_index:]:
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True

    unfreeze_last_fraction(base_cnn, fraction=FINE_TUNE_FRACTION)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE2_LR),
        loss=CategoricalFocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, class_weights=CLASS_WEIGHTS, label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"]
    )

    snapshot_callback = TopKSnapshotCheckpoint(snapshot_dir=SNAPSHOT_DIR, k=3, prefix="stage2")
    early2 = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=STAGE2_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    t_s2 = time.time()
    history2 = model.fit(
        train_generator,
        validation_data=(X_val_4ch, y_val),
        epochs=STAGE2_MAX_EPOCHS,
        callbacks=[snapshot_callback, early2],
        verbose=1
    )
    s2_elapsed = time.time() - t_s2

    stage2_best_epoch = int(np.argmin(history2.history["val_loss"]) + 1)
    stage2_best_val_acc = float(max(history2.history["val_accuracy"]))
    stage2_best_val_loss = float(min(history2.history["val_loss"]))
    print(f"Stage 2 Best Epoch: {stage2_best_epoch} | Loss: {stage2_best_val_loss:.4f} | Acc: {stage2_best_val_acc*100:.2f}% | Elapsed: {s2_elapsed:.2f}s")

    # 11. Save Training Curves
    h1 = history1.history
    h2 = history2.history
    df_history = pd.DataFrame({
        "epoch": list(range(1, len(h1["loss"]) + len(h2["loss"]) + 1)),
        "stage": ["Stage 1"] * len(h1["loss"]) + ["Stage 2"] * len(h2["loss"]),
        "train_loss": h1["loss"] + h2["loss"],
        "val_loss": h1["val_loss"] + h2["val_loss"],
        "train_accuracy": h1["accuracy"] + h2["accuracy"],
        "val_accuracy": h1["val_accuracy"] + h2["val_accuracy"],
    })
    df_history.to_csv(RESULTS_DIR / "training_history.csv", index=False)

    stage1_end = len(h1["loss"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(df_history["epoch"], df_history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(df_history["epoch"], df_history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    ax1.axvline(x=stage1_end + 0.5, color="gray", linestyle="--", alpha=0.7, label="Stage Transition")
    ax1.set_title("Focal Loss Trajectory (Gamma=2.0, d8/d10=0.1)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(df_history["epoch"], df_history["train_accuracy"], label="Train Acc", color="#2ca02c", lw=2)
    ax2.plot(df_history["epoch"], df_history["val_accuracy"], label="Val Acc", color="#d62728", lw=2)
    ax2.axvline(x=stage1_end + 0.5, color="gray", linestyle="--", alpha=0.7, label="Stage Transition")
    ax2.set_title("Validation Accuracy Trajectory", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_history_curves.png", dpi=200)
    plt.close()

    # 12. Top-1 Snapshot Inference + 4-View TTA
    print("\n" + "=" * 80)
    print("TOP-1 SNAPSHOT INFERENCE + 4-VIEW TTA")
    print("=" * 80)
    top1_snapshot_path = snapshot_callback.get_top1_path()
    if top1_snapshot_path is None or not top1_snapshot_path.exists():
        all_snaps = sorted(list(SNAPSHOT_DIR.glob("*.weights.h5")))
        top1_snapshot_path = all_snaps[0] if all_snaps else RESULTS_DIR / "stage1_best.weights.h5"

    print(f"Loading Top-1 Snapshot Weights: {top1_snapshot_path.name}")
    model.load_weights(top1_snapshot_path)

    def generate_4view_tta(X_seq_4ch: np.ndarray) -> list:
        v1 = X_seq_4ch.copy()
        v2 = np.empty_like(X_seq_4ch)
        v2[:-1] = X_seq_4ch[1:]; v2[-1] = X_seq_4ch[-1]
        v3 = np.empty_like(X_seq_4ch)
        v3[0] = X_seq_4ch[0]; v3[1:] = X_seq_4ch[:-1]
        v4 = np.flip(X_seq_4ch, axis=2).copy()
        return [v1, v2, v3, v4]

    N_val = len(X_val_4ch)
    views_val = [np.zeros_like(X_val_4ch) for _ in range(4)]
    for i in range(N_val):
        v_list = generate_4view_tta(X_val_4ch[i])
        for v_idx in range(4):
            views_val[v_idx][i] = v_list[v_idx]

    view_probs = []
    for v_idx, v_tensor in enumerate(views_val, 1):
        print(f"Computing forward pass for TTA View {v_idx}/4...")
        p = model.predict(v_tensor, batch_size=BATCH_SIZE, verbose=0)
        view_probs.append(p)

    final_top1_probs = np.mean(view_probs, axis=0)
    final_preds = np.argmax(final_top1_probs, axis=1)

    final_acc = accuracy_score(y_val, final_preds)
    final_f1 = f1_score(y_val, final_preds, average="macro")
    final_precision = precision_score(y_val, final_preds, average="macro")
    final_recall = recall_score(y_val, final_preds, average="macro")

    print("\n" + "=" * 80)
    print("FINAL EVALUATION METRICS (TOP-1 SNAPSHOT + 4-VIEW TTA)")
    print("=" * 80)
    print(f"Top-1 Snapshot Checkpoint : {top1_snapshot_path.name}")
    print(f"Validation Accuracy       : {final_acc * 100:.2f}% ({int(np.sum(final_preds == y_val))}/{len(y_val)})")
    print(f"Macro F1-Score            : {final_f1:.4f}")
    print(f"Macro Precision           : {final_precision:.4f}")
    print(f"Macro Recall              : {final_recall:.4f}")
    print("-" * 80)

    report_str = classification_report(y_val, final_preds, target_names=DIGIT_CLASSES, digits=4)
    print(report_str)

    report_dict = classification_report(y_val, final_preds, target_names=DIGIT_CLASSES, output_dict=True)
    with open(RESULTS_DIR / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    cm = confusion_matrix(y_val, final_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=DIGIT_CLASSES, yticklabels=DIGIT_CLASSES, cbar_kws={"label": "Utterances"})
    plt.title(f"Experiment 21 Confusion Matrix (Top-1 Snapshot + 4-View TTA)\nVal Accuracy: {final_acc*100:.2f}%", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Digit"); plt.ylabel("Ground Truth Digit"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # Per-Speaker Breakdown
    df_val = df_manifest[df_manifest["split"] == "val"].copy().reset_index(drop=True)
    df_val["pred_id"] = final_preds
    df_val["correct"] = (df_val["digit_id"] == df_val["pred_id"]).astype(int)

    speaker_summary = []
    print("=" * 80)
    print("PER-SPEAKER ACCURACY BREAKDOWN")
    print("=" * 80)
    for spk in base_val_speakers:
        sub = df_val[df_val["base_speaker"] == spk]
        acc = sub["correct"].mean()
        total = len(sub)
        correct_count = sub["correct"].sum()
        speaker_summary.append({"base_speaker": spk, "accuracy": float(acc), "total": int(total), "correct": int(correct_count)})
        print(f"  Speaker {spk:5s} : {correct_count:2d}/{total:2d} correct -> {acc * 100:6.2f}%")
    print("=" * 80)

    with open(RESULTS_DIR / "per_speaker_breakdown.json", "w") as f:
        json.dump(speaker_summary, f, indent=2)

    final_metrics = {
        "experiment": "21_efficientnetb0_bigru_suppressed_focal_temporal_drop",
        "dataset_mode": DATASET_MODE,
        "seed": SEED,
        "frame_channels": FRAME_CHANNELS,
        "motion_diff_applied": True,
        "temporal_frame_dropping_applied": True,
        "temporal_frame_dropping_num": "2-3 frames interpolated for all classes",
        "targeted_affine_augmentation_applied": False,
        "class_weights": CLASS_WEIGHTS,
        "focal_gamma": FOCAL_GAMMA,
        "focal_alpha": FOCAL_ALPHA,
        "stage2_patience": STAGE2_PATIENCE,
        "stage2_max_epochs": STAGE2_MAX_EPOCHS,
        "loss": "SuppressedCategoricalFocalLoss",
        "evaluation_mode": "Top-1 Snapshot + 4-View TTA",
        "top1_snapshot": str(top1_snapshot_path.name),
        "metrics": {
            "val_accuracy": float(final_acc),
            "macro_f1": float(final_f1),
            "macro_precision": float(final_precision),
            "macro_recall": float(final_recall),
        },
        "stages": {
            "stage1_best_epoch": stage1_best_epoch,
            "stage1_best_loss": stage1_best_val_loss,
            "stage1_best_acc": stage1_best_val_acc,
            "stage1_elapsed": s1_elapsed,
            "stage2_best_epoch": stage2_best_epoch,
            "stage2_best_loss": stage2_best_val_loss,
            "stage2_best_acc": stage2_best_val_acc,
            "stage2_elapsed": s2_elapsed,
        }
    }

    with open(RESULTS_DIR / f"metrics_{DATASET_MODE}.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("Experiment 21 execution finished successfully!")

if __name__ == "__main__":
    main()
