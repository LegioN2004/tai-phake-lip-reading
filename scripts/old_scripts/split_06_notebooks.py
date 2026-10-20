import json
from pathlib import Path
import copy

def split_notebook_06():
    with open("notebooks/06_pretrained_phase1_baseline.ipynb") as f:
        nb6 = json.load(f)

    models_config = [
        {
            "filename": "06_MobileNet_pretrained_phase1.ipynb",
            "model_name": "MobileNetV2",
            "slug": "mobilenetv2_bilstm",
            "cell_idx": 12,
        },
        {
            "filename": "06_EfficientNet_pretrained_phase1.ipynb",
            "model_name": "EfficientNetB0",
            "slug": "efficientnetb0_bilstm",
            "cell_idx": 14,
        },
        {
            "filename": "06_ResNet_pretrained_phase1.ipynb",
            "model_name": "ResNet50",
            "slug": "resnet50_bilstm",
            "cell_idx": 16,
        },
    ]

    for cfg in models_config:
        m_name = cfg["model_name"]
        slug = cfg["slug"]
        c_idx = cfg["cell_idx"]

        new_nb = {
            "cells": [],
            "metadata": copy.deepcopy(nb6["metadata"]),
            "nbformat": nb6["nbformat"],
            "nbformat_minor": nb6["nbformat_minor"],
        }

        # Cell 0: Header markdown
        c0 = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# 06 - Pretrained {m_name} + BiLSTM Baseline\n",
                "## Tai Phake Visual Speech Recognition (VSR) Project\n",
                "\n",
                f"**Target Architecture**: Pretrained **{m_name}** backbone with BiLSTM sequence classifier.\n",
                "\n",
                "---\n",
                "### Experimental Controls & Configuration:\n",
                "- **Dataset**: Speaker-independent splits (`data/processed/train`, `val`, `test`).\n",
                f"- **Backbone**: `{m_name}` (ImageNet weights, feature extraction per frame).\n",
                "- **Sequence Head**: Bidirectional LSTM (64 units) + Dropout(0.3) + Dense(11, softmax).\n",
                "- **Temporal Standardization**: Exactly 30 frames per utterance.\n",
                "- **Optimization**: Adam (lr=1e-4), max 80 epochs, Early Stopping (patience=15).\n",
                "- **Consolidated Output**: Complete evaluation metrics (accuracy in %, loss, F1), classification report, confusion matrix, and training curves are organized neatly at the end of the notebook."
            ]
        }
        new_nb["cells"].append(c0)

        # Cells 1 to 8: Environment, Splits, Manifest, Data Tensors
        for i in range(1, 9):
            new_nb["cells"].append(copy.deepcopy(nb6["cells"][i]))

        # Cell 9: Markdown for Architecture Builder
        new_nb["cells"].append(copy.deepcopy(nb6["cells"][9]))

        # Cell 10: Code for Architecture Builder
        new_nb["cells"].append(copy.deepcopy(nb6["cells"][10]))

        # Cell 11: Section Markdown for the specific model
        m_md = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"## 6. Model Training & Evaluation: {m_name} + BiLSTM\n",
                f"Executes feature extraction using `{m_name}`, trains the BiLSTM temporal classifier with EarlyStopping, and produces consolidated evaluation metrics."
            ]
        }
        new_nb["cells"].append(m_md)

        # Cell 12: Code for model training & evaluation
        # Let's craft the clean self-contained script based on cell 12 / 14 / 16
        code_src = f"""# Section 6: {m_name} + BiLSTM Training and Consolidated Evaluation
m_name = "{m_name}"
slug = "{slug}"
model_dir = RESULTS_DIR / slug
model_dir.mkdir(parents=True, exist_ok=True)

print(f"=== Initializing {{m_name}} + BiLSTM ===")
cnn, feat_dim = get_backbone(m_name)

# 1. Feature Extraction
print(f"Extracting {{m_name}} features (dim={{feat_dim}}) across all {{len(X_all)}} sequence frames...")
t_feat_start = time.time()
all_frames_flat = X_all.reshape(-1, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
feat_flat = cnn.predict(all_frames_flat, batch_size=64, verbose=0)
feat_all = feat_flat.reshape(len(X_all), NUM_FRAMES, feat_dim)
print(f"Features extracted in {{time.time() - t_feat_start:.2f}}s! Representation shape: {{feat_all.shape}}")

F_train, F_val, F_test = feat_all[train_mask], feat_all[val_mask], feat_all[test_mask]

# 2. Build Head & Architecture Summary
head = build_bilstm_head(feat_dim)
total_params = head.count_params() + cnn.count_params()
trainable_params = sum(w.numpy().size for w in head.trainable_weights)
print(f"Architecture Parameters:")
print(f"  Total Parameters:         {{total_params:,}}")
print(f"  Trainable Parameters:     {{trainable_params:,}}")
print(f"  Non-Trainable Parameters: {{cnn.count_params():,}}")

# 3. Training with EarlyStopping
print(f"\\nTraining on {{len(train_speakers)}} train speakers with EarlyStopping (val_loss monitoring on {{len(val_speakers)}} val speakers)...")
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

t_train_start = time.time()
history = head.fit(
    F_train, y_train,
    validation_data=(F_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    shuffle=True,
    verbose=1
)
train_time = time.time() - t_train_start
print(f"Training completed in {{train_time:.2f}}s")

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
best_val_loss = float(np.min(history.history["val_loss"]))
best_val_acc = float(history.history["val_accuracy"][best_epoch - 1])

# Save training history
df_hist = pd.DataFrame(history.history)
df_hist["epoch"] = range(1, len(df_hist) + 1)
df_hist.to_csv(model_dir / "training_history.csv", index=False)

# Save End-to-End unified model
end_to_end_model = build_end_to_end_model(cnn, head, slug)
end_to_end_model.save(model_dir / "best_model.keras")
print(f"Saved end-to-end model checkpoint to: {{model_dir / 'best_model.keras'}}")

# 4. Evaluation on Unseen Test Speakers
print(f"\\nEvaluating {{m_name}} + BiLSTM on {{len(test_speakers)}} Unseen Test Speakers...")
test_probs = head.predict(F_test, verbose=0)
test_preds = np.argmax(test_probs, axis=1)

test_acc = accuracy_score(y_test, test_preds)
test_prec = precision_score(y_test, test_preds, average="macro", zero_division=0)
test_rec = recall_score(y_test, test_preds, average="macro", zero_division=0)
test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
cm = confusion_matrix(y_test, test_preds, labels=list(range(NUM_CLASSES)))

correct_count = int(np.sum(y_test == test_preds))
incorrect_count = int(np.sum(y_test != test_preds))

# Save classification report
report_dict = classification_report(y_test, test_preds, target_names=DIGIT_CLASSES, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report_dict).transpose()
df_report.to_csv(model_dir / "per_class_metrics.csv")

# Save metrics JSON & CSV
metrics_summary = {{
    "model_name": m_name + " + BiLSTM",
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "non_trainable_parameters": cnn.count_params(),
    "training_time_seconds": round(train_time, 2),
    "best_epoch": best_epoch,
    "best_val_loss": round(best_val_loss, 4),
    "best_val_accuracy": round(best_val_acc, 4),
    "test_accuracy": round(test_acc, 4),
    "test_accuracy_pct": round(test_acc * 100, 2),
    "test_macro_precision": round(test_prec, 4),
    "test_macro_recall": round(test_rec, 4),
    "test_macro_f1": round(test_f1, 4),
    "correct_test_predictions": correct_count,
    "incorrect_test_predictions": incorrect_count
}}
with open(model_dir / "metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)
pd.DataFrame([metrics_summary]).to_csv(model_dir / "metrics.csv", index=False)

# Save Predictions for every test utterance
df_test_meta = df_manifest[df_manifest["split"] == "test"].copy().reset_index(drop=True)
df_test_preds = pd.DataFrame({{
    "speaker": df_test_meta["speaker"],
    "true_digit": df_test_meta["digit"],
    "true_digit_id": y_test,
    "pred_digit_id": test_preds,
    "pred_digit": [ID_TO_CLASS[p] for p in test_preds],
    "is_correct": (y_test == test_preds),
    "confidence": np.max(test_probs, axis=1)
}})
for c_idx, d_name in enumerate(DIGIT_CLASSES):
    df_test_preds[f"prob_{{d_name}}"] = test_probs[:, c_idx]
df_test_preds.to_csv(model_dir / "test_predictions.csv", index=False)

# Save Confusion Matrix CSV
df_cm = pd.DataFrame(cm, index=DIGIT_CLASSES, columns=DIGIT_CLASSES)
df_cm.to_csv(model_dir / "confusion_matrix.csv")

# ==============================================================================
# 5. CONSOLIDATED ALL-IN-ONE OUTPUT RESULTS (EASY TO READ AT A GLANCE)
# ==============================================================================
print("\\n" + "=" * 80)
print(f"                 FINAL EVALUATION SUMMARY: {{m_name.upper()}} + BiLSTM")
print("=" * 80)
print(f"  Test Accuracy:           {{test_acc * 100:.2f}}%  ({{correct_count}}/{{len(y_test)}} correct)")
print(f"  Validation Accuracy:     {{best_val_acc * 100:.2f}}%  (Best Epoch: {{best_epoch}})")
print(f"  Validation Loss:         {{best_val_loss:.4f}}")
print(f"  Macro Precision:         {{test_prec:.4f}}")
print(f"  Macro Recall:            {{test_rec:.4f}}")
print(f"  Macro F1-Score:          {{test_f1:.4f}}")
print(f"  Total / Trainable Params:{{total_params:,}} / {{trainable_params:,}}")
print(f"  Training Duration:       {{train_time:.2f}}s")
print("-" * 80)
print("Per-Class Metrics (Unseen Test Speakers):")
display(df_report.loc[DIGIT_CLASSES, ["precision", "recall", "f1-score", "support"]])
print("=" * 80)

# Training Curves Plot
plt.figure(figsize=(13, 4.5))
plt.subplot(1, 2, 1)
plt.plot(df_hist["epoch"], df_hist["loss"], label="Train Loss", color="royalblue", lw=2)
plt.plot(df_hist["epoch"], df_hist["val_loss"], label="Val Loss", color="crimson", lw=2)
plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({{best_epoch}})")
plt.title(f"{{m_name}} + BiLSTM: Loss Trajectory", fontsize=12, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss (Sparse Categorical Crossentropy)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(df_hist["epoch"], df_hist["accuracy"] * 100, label="Train Accuracy (%)", color="royalblue", lw=2)
plt.plot(df_hist["epoch"], df_hist["val_accuracy"] * 100, label="Val Accuracy (%)", color="crimson", lw=2)
plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({{best_epoch}})")
plt.title(f"{{m_name}} + BiLSTM: Accuracy Trajectory (%)", fontsize=12, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_dir / "training_curves.png", dpi=300)
plt.show()

# Confusion Matrix Heatmap
plt.figure(figsize=(8.5, 6.5))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True, square=True)
plt.title(f"Confusion Matrix: {{m_name}} + BiLSTM (Test Accuracy: {{test_acc*100:.2f}}%)", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Class", fontsize=11)
plt.ylabel("True Class", fontsize=11)
plt.tight_layout()
plt.savefig(model_dir / "confusion_matrix.png", dpi=300)
plt.show()

print(f"\\nAll model artifacts saved to: {{model_dir.resolve()}}")
"""

        c12 = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code_src.splitlines()]
        }
        new_nb["cells"].append(c12)

        out_file = Path("notebooks") / cfg["filename"]
        with open(out_file, "w") as f:
            json.dump(new_nb, f, indent=1)
        print(f"Created {out_file} successfully.")

if __name__ == "__main__":
    split_notebook_06()
