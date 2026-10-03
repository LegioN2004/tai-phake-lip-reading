import json
import copy
from pathlib import Path

# Load reference baseline notebook
with open('notebooks/06_pretrained_phase1_baseline.ipynb', 'r') as f:
    nb_base = json.load(f)

configs = [
    {
        'path': 'notebooks/06_MobileNet_pretrained_phase1.ipynb',
        'model_name': 'MobileNetV2',
        'slug': 'mobilenetv2_bilstm',
        'display_name': 'MobileNetV2',
        'desc': 'Pretrained **MobileNetV2** backbone with BiLSTM sequence classifier using two-stage training (frozen backbone warm-up followed by fine-tuning of the top 20% CNN layers).'
    },
    {
        'path': 'notebooks/06_EfficientNet_pretrained_phase1.ipynb',
        'model_name': 'EfficientNetB0',
        'slug': 'efficientnetb0_bilstm',
        'display_name': 'EfficientNetB0',
        'desc': 'Pretrained **EfficientNetB0** backbone with BiLSTM sequence classifier using two-stage training (frozen backbone warm-up followed by fine-tuning of the top 20% CNN layers).'
    },
    {
        'path': 'notebooks/06_ResNet_pretrained_phase1.ipynb',
        'model_name': 'ResNet50',
        'slug': 'resnet50_bilstm',
        'display_name': 'ResNet50',
        'desc': 'Pretrained **ResNet50** backbone with BiLSTM sequence classifier using two-stage training (frozen backbone warm-up followed by fine-tuning of the top 20% CNN layers).'
    }
]

for cfg in configs:
    m_name = cfg['model_name']
    slug = cfg['slug']
    display_name = cfg['display_name']

    # Header markdown
    cell0_md = (
        f"# 06 - Pretrained {display_name} + BiLSTM Baseline\n"
        f"## Tai Phake Visual Speech Recognition (VSR) Project\n\n"
        f"**Target Architecture**: {cfg['desc']}\n\n"
        "---\n"
        "### Experimental Controls & Configuration:\n"
        "- **Dataset**: Speaker-independent splits (`data/processed/train`, `val`, `test`).\n"
        f"- **Backbone**: `{display_name}` (ImageNet weights, native frame preprocessing, two-stage transfer learning).\n"
        "- **Sequence Head**: Bidirectional LSTM (128 units) + Dropout(0.3) + Dense(11, softmax).\n"
        "- **Temporal Standardization**: Exactly 30 frames per utterance.\n"
        "- **Stage 1 (Frozen Backbone)**: Adam (lr=1e-4), max 40 epochs, Early Stopping (patience=8).\n"
        "- **Stage 2 (Fine-tuning Top 20%)**: Adam (lr=1e-5), max 25 epochs, Early Stopping (patience=6, BatchNorm frozen).\n"
        f"- **Consolidated Output**: Full evaluation metrics (accuracy, macro precision/recall/F1), per-class report, confusion matrix, training curves, and saved artifacts in `results/pretrained_phase1/{slug}/`.\n"
    )

    # Section 5 markdown
    cell9_md = (
        "## 5. Architectural Builder & Two-Stage Training Strategy\n"
        "The architecture consumes raw video frames $(30, 96, 96, 3)$ end-to-end:\n"
        f"$$\\text{{Frame Sequence }} (30\\times 96\\times 96\\times 3) \\xrightarrow{{\\text{{Preprocessing}}}} \\text{{TimeDistributed}}({display_name}\\text{{ Backbone}}) \\xrightarrow{{}} \\text{{BiLSTM}}(128) \\xrightarrow{{\\text{{Dropout}}(0.3)}} \\xrightarrow{{\\text{{Dense}}(11, \\text{{softmax}})}} \\hat{{y}}$$\n\n"
        "### Two-Stage Optimization:\n"
        "1. **Stage 1 (Feature Alignment / Frozen Backbone)**: CNN weights are completely frozen (`trainable = False`), training only the BiLSTM and classification head with learning rate $1\\times 10^{-4}$.\n"
        "2. **Stage 2 (Fine-Tuning Top 20%)**: The top 20% of CNN layers are unfrozen (with `BatchNormalization` layers preserved frozen for stability) and fine-tuned with a lower learning rate $1\\times 10^{-5}$.\n"
    )

    # Training cell code
    train_code = (
        f"# Section 6: {display_name} + BiLSTM Two-Stage Training\n"
        f"model, history1, history2 = train_two_stage(\n"
        f"    \"{m_name}\",\n"
        f"    X_train,\n"
        f"    y_train,\n"
        f"    X_val,\n"
        f"    y_val\n"
        f")\n"
    )

    # Evaluation & Artifacts cell code
    eval_code = f'''# Section 7: Evaluation on Unseen Test Speakers & Artifact Generation
m_name = "{m_name}"
slug = "{slug}"
model_dir = RESULTS_DIR / slug
model_dir.mkdir(parents=True, exist_ok=True)

# 1. Predictions on Unseen Test Speakers
print(f"\\nEvaluating {{m_name}} + BiLSTM on {{len(test_speakers)}} Unseen Test Speakers...")
test_probs = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)
test_preds = np.argmax(test_probs, axis=1)

test_acc = accuracy_score(y_test, test_preds)
test_prec = precision_score(y_test, test_preds, average="macro", zero_division=0)
test_rec = recall_score(y_test, test_preds, average="macro", zero_division=0)
test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
cm = confusion_matrix(y_test, test_preds, labels=list(range(NUM_CLASSES)))

correct_count = int(np.sum(y_test == test_preds))
incorrect_count = int(np.sum(y_test != test_preds))

# 2. Combine Two-Stage Histories for Metrics and Plots
combined_loss = history1.history["loss"] + history2.history["loss"]
combined_val_loss = history1.history["val_loss"] + history2.history["val_loss"]
combined_acc = history1.history["accuracy"] + history2.history["accuracy"]
combined_val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]

stage1_len = len(history1.history["loss"])
stage2_len = len(history2.history["loss"])
total_epochs = stage1_len + stage2_len

df_hist = pd.DataFrame({{
    "epoch": range(1, total_epochs + 1),
    "stage": [1] * stage1_len + [2] * stage2_len,
    "loss": combined_loss,
    "val_loss": combined_val_loss,
    "accuracy": combined_acc,
    "val_accuracy": combined_val_acc
}})
df_hist.to_csv(model_dir / "training_history.csv", index=False)

best_epoch = int(np.argmin(df_hist["val_loss"]) + 1)
best_val_loss = float(np.min(df_hist["val_loss"]))
best_val_acc = float(df_hist.loc[best_epoch - 1, "val_accuracy"])

# Save End-to-End model checkpoint
model.save(model_dir / "best_model.keras")
print(f"Saved end-to-end model checkpoint to: {{model_dir / 'best_model.keras'}}")

# Save classification report
report_dict = classification_report(y_test, test_preds, target_names=DIGIT_CLASSES, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report_dict).transpose()
df_report.to_csv(model_dir / "per_class_metrics.csv")

# Model parameters
total_params = int(model.count_params())
trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
non_trainable_params = total_params - trainable_params

# Save metrics JSON & CSV
metrics_summary = {{
    "model_name": m_name + " + BiLSTM",
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "non_trainable_parameters": non_trainable_params,
    "stage1_epochs": stage1_len,
    "stage2_epochs": stage2_len,
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
# CONSOLIDATED ALL-IN-ONE OUTPUT RESULTS (EASY TO READ AT A GLANCE)
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
print("-" * 80)
print("Per-Class Metrics (Unseen Test Speakers):")
display(df_report.loc[DIGIT_CLASSES, ["precision", "recall", "f1-score", "support"]])
print("=" * 80)

# Training Curves Plot
plt.figure(figsize=(13, 4.5))
plt.subplot(1, 2, 1)
plt.plot(df_hist["epoch"], df_hist["loss"], label="Train Loss", color="royalblue", lw=2)
plt.plot(df_hist["epoch"], df_hist["val_loss"], label="Val Loss", color="crimson", lw=2)
if stage2_len > 0:
    plt.axvline(stage1_len + 0.5, color="orange", linestyle=":", label="Stage 2 Start")
plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({{best_epoch}})")
plt.title(f"{{m_name}} + BiLSTM: Loss Trajectory", fontsize=12, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss (Sparse Categorical Crossentropy)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(df_hist["epoch"], df_hist["accuracy"] * 100, label="Train Accuracy (%)", color="royalblue", lw=2)
plt.plot(df_hist["epoch"], df_hist["val_accuracy"] * 100, label="Val Accuracy (%)", color="crimson", lw=2)
if stage2_len > 0:
    plt.axvline(stage1_len + 0.5, color="orange", linestyle=":", label="Stage 2 Start")
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
'''

    new_cells = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [l + '\n' for l in cell0_md.splitlines()]},
        copy.deepcopy(nb_base['cells'][1]),
        copy.deepcopy(nb_base['cells'][2]),
        copy.deepcopy(nb_base['cells'][3]),
        copy.deepcopy(nb_base['cells'][4]),
        copy.deepcopy(nb_base['cells'][5]),
        copy.deepcopy(nb_base['cells'][6]),
        copy.deepcopy(nb_base['cells'][7]),
        copy.deepcopy(nb_base['cells'][8]),
        {'cell_type': 'markdown', 'metadata': {}, 'source': [l + '\n' for l in cell9_md.splitlines()]},
        copy.deepcopy(nb_base['cells'][10]),
        copy.deepcopy(nb_base['cells'][11]),
        copy.deepcopy(nb_base['cells'][12]),
        {'cell_type': 'markdown', 'metadata': {}, 'source': [f"## 6. Two-Stage Training: {display_name} + BiLSTM\n"]},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [l + '\n' for l in train_code.splitlines()]},
        {'cell_type': 'markdown', 'metadata': {}, 'source': [f"## 7. Model Evaluation & Artifacts: {display_name} + BiLSTM\n"]},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [l + '\n' for l in eval_code.splitlines()]}
    ]

    nb_new = {
        'cells': new_cells,
        'metadata': copy.deepcopy(nb_base.get('metadata', {})),
        'nbformat': nb_base.get('nbformat', 4),
        'nbformat_minor': nb_base.get('nbformat_minor', 2)
    }

    with open(cfg['path'], 'w') as f:
        json.dump(nb_new, f, indent=1)

    print(f"Successfully wrote {cfg['path']} ({len(new_cells)} cells)")
