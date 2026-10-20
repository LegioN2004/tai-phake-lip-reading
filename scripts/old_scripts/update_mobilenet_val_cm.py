import json
import ast
from pathlib import Path

nb_path = Path("notebooks/06_MobileNet_pretrained_phase1.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_code = '''# Section 7: Evaluation on Unseen Test Speakers & Artifact Generation
m_name = "MobileNetV2"
slug = "mobilenetv2_bilstm"
model_dir = RESULTS_DIR / slug
model_dir.mkdir(parents=True, exist_ok=True)

# 1. Predictions on Unseen Test Speakers
print(f"\\nEvaluating {m_name} + BiLSTM on {len(test_speakers)} Unseen Test Speakers...")
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

# 1b. Predictions on Validation Speakers & Validation Confusion Matrix
print(f"\\nEvaluating {m_name} + BiLSTM on {len(val_speakers)} Validation Speakers...")
val_probs = model.predict(
    X_val,
    batch_size=BATCH_SIZE,
    verbose=1
)
val_preds = np.argmax(val_probs, axis=1)

val_acc = accuracy_score(y_val, val_preds)
val_prec = precision_score(y_val, val_preds, average="macro", zero_division=0)
val_rec = recall_score(y_val, val_preds, average="macro", zero_division=0)
val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
cm_val = confusion_matrix(y_val, val_preds, labels=list(range(NUM_CLASSES)))

val_correct_count = int(np.sum(y_val == val_preds))
val_incorrect_count = int(np.sum(y_val != val_preds))

# 2. Combine Two-Stage Histories for Metrics and Plots
combined_loss = history1.history["loss"] + history2.history["loss"]
combined_val_loss = history1.history["val_loss"] + history2.history["val_loss"]
combined_acc = history1.history["accuracy"] + history2.history["accuracy"]
combined_val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]

stage1_len = len(history1.history["loss"])
stage2_len = len(history2.history["loss"])
total_epochs = stage1_len + stage2_len

df_hist = pd.DataFrame({
    "epoch": range(1, total_epochs + 1),
    "stage": [1] * stage1_len + [2] * stage2_len,
    "loss": combined_loss,
    "val_loss": combined_val_loss,
    "accuracy": combined_acc,
    "val_accuracy": combined_val_acc
})
df_hist.to_csv(model_dir / "training_history.csv", index=False)

best_epoch = int(np.argmin(df_hist["val_loss"]) + 1)
best_val_loss = float(np.min(df_hist["val_loss"]))
best_val_acc = float(df_hist.loc[best_epoch - 1, "val_accuracy"])

# Save End-to-End model checkpoint
model.save(model_dir / "best_model.keras")
print(f"Saved end-to-end model checkpoint to: {model_dir / 'best_model.keras'}")

# Save classification report (Test)
report_dict = classification_report(y_test, test_preds, target_names=DIGIT_CLASSES, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report_dict).transpose()
df_report.to_csv(model_dir / "per_class_metrics.csv")

# Save classification report (Validation)
report_dict_val = classification_report(y_val, val_preds, target_names=DIGIT_CLASSES, output_dict=True, zero_division=0)
df_report_val = pd.DataFrame(report_dict_val).transpose()
df_report_val.to_csv(model_dir / "per_class_metrics_val.csv")

# Model parameters
total_params = int(model.count_params())
trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
non_trainable_params = total_params - trainable_params

# Save metrics JSON & CSV
metrics_summary = {
    "model_name": m_name + " + BiLSTM",
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "non_trainable_parameters": non_trainable_params,
    "stage1_epochs": stage1_len,
    "stage2_epochs": stage2_len,
    "best_epoch": best_epoch,
    "best_val_loss": round(best_val_loss, 4),
    "best_val_accuracy": round(best_val_acc, 4),
    "val_eval_accuracy": round(val_acc, 4),
    "val_eval_accuracy_pct": round(val_acc * 100, 2),
    "val_macro_precision": round(val_prec, 4),
    "val_macro_recall": round(val_rec, 4),
    "val_macro_f1": round(val_f1, 4),
    "test_accuracy": round(test_acc, 4),
    "test_accuracy_pct": round(test_acc * 100, 2),
    "test_macro_precision": round(test_prec, 4),
    "test_macro_recall": round(test_rec, 4),
    "test_macro_f1": round(test_f1, 4),
    "correct_test_predictions": correct_count,
    "incorrect_test_predictions": incorrect_count,
    "correct_val_predictions": val_correct_count,
    "incorrect_val_predictions": val_incorrect_count
}
with open(model_dir / "metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)
pd.DataFrame([metrics_summary]).to_csv(model_dir / "metrics.csv", index=False)

# Save Predictions for every test utterance
df_test_meta = df_manifest[df_manifest["split"] == "test"].copy().reset_index(drop=True)
df_test_preds = pd.DataFrame({
    "speaker": df_test_meta["speaker"],
    "true_digit": df_test_meta["digit"],
    "true_digit_id": y_test,
    "pred_digit_id": test_preds,
    "pred_digit": [ID_TO_CLASS[p] for p in test_preds],
    "is_correct": (y_test == test_preds),
    "confidence": np.max(test_probs, axis=1)
})
for c_idx, d_name in enumerate(DIGIT_CLASSES):
    df_test_preds[f"prob_{d_name}"] = test_probs[:, c_idx]
df_test_preds.to_csv(model_dir / "test_predictions.csv", index=False)

# Save Predictions for every validation utterance
df_val_meta = df_manifest[df_manifest["split"] == "val"].copy().reset_index(drop=True)
df_val_preds = pd.DataFrame({
    "speaker": df_val_meta["speaker"],
    "true_digit": df_val_meta["digit"],
    "true_digit_id": y_val,
    "pred_digit_id": val_preds,
    "pred_digit": [ID_TO_CLASS[p] for p in val_preds],
    "is_correct": (y_val == val_preds),
    "confidence": np.max(val_probs, axis=1)
})
for c_idx, d_name in enumerate(DIGIT_CLASSES):
    df_val_preds[f"prob_{d_name}"] = val_probs[:, c_idx]
df_val_preds.to_csv(model_dir / "val_predictions.csv", index=False)

# Save Confusion Matrix CSVs
df_cm = pd.DataFrame(cm, index=DIGIT_CLASSES, columns=DIGIT_CLASSES)
df_cm.to_csv(model_dir / "confusion_matrix.csv")
df_cm.to_csv(model_dir / "confusion_matrix_test.csv")

df_cm_val = pd.DataFrame(cm_val, index=DIGIT_CLASSES, columns=DIGIT_CLASSES)
df_cm_val.to_csv(model_dir / "confusion_matrix_val.csv")

# ==============================================================================
# CONSOLIDATED ALL-IN-ONE OUTPUT RESULTS (EASY TO READ AT A GLANCE)
# ==============================================================================
print("\\n" + "=" * 80)
print(f"                 FINAL EVALUATION SUMMARY: {m_name.upper()} + BiLSTM")
print("=" * 80)
print(f"  Test Accuracy:           {test_acc * 100:.2f}%  ({correct_count}/{len(y_test)} correct)")
print(f"  Validation Accuracy:     {val_acc * 100:.2f}%  ({val_correct_count}/{len(y_val)} correct)")
print(f"  Best Epoch Val Accuracy: {best_val_acc * 100:.2f}%  (Best Epoch: {best_epoch})")
print(f"  Best Epoch Val Loss:     {best_val_loss:.4f}")
print(f"  Test Macro Precision:    {test_prec:.4f}")
print(f"  Test Macro Recall:       {test_rec:.4f}")
print(f"  Test Macro F1-Score:     {test_f1:.4f}")
print(f"  Total / Trainable Params:{total_params:,} / {trainable_params:,}")
print("-" * 80)
print("Per-Class Metrics (Unseen Test Speakers):")
display(df_report.loc[DIGIT_CLASSES, ["precision", "recall", "f1-score", "support"]])
print("-" * 80)
print("Per-Class Metrics (Validation Speakers):")
display(df_report_val.loc[DIGIT_CLASSES, ["precision", "recall", "f1-score", "support"]])
print("=" * 80)

# Training Curves Plot
plt.figure(figsize=(13, 4.5))
plt.subplot(1, 2, 1)
plt.plot(df_hist["epoch"], df_hist["loss"], label="Train Loss", color="royalblue", lw=2)
plt.plot(df_hist["epoch"], df_hist["val_loss"], label="Val Loss", color="crimson", lw=2)
if stage2_len > 0:
    plt.axvline(stage1_len + 0.5, color="orange", linestyle=":", label="Stage 2 Start")
plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({best_epoch})")
plt.title(f"{m_name} + BiLSTM: Loss Trajectory", fontsize=12, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss (Sparse Categorical Crossentropy)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(df_hist["epoch"], df_hist["accuracy"] * 100, label="Train Accuracy (%)", color="royalblue", lw=2)
plt.plot(df_hist["epoch"], df_hist["val_accuracy"] * 100, label="Val Accuracy (%)", color="crimson", lw=2)
if stage2_len > 0:
    plt.axvline(stage1_len + 0.5, color="orange", linestyle=":", label="Stage 2 Start")
plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({best_epoch})")
plt.title(f"{m_name} + BiLSTM: Accuracy Trajectory (%)", fontsize=12, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_dir / "training_curves.png", dpi=300)
plt.show()

# Validation Confusion Matrix Heatmap
plt.figure(figsize=(8.5, 6.5))
sns.heatmap(df_cm_val, annot=True, fmt="d", cmap="Blues", cbar=True, square=True)
plt.title(f"Confusion Matrix (Validation): {m_name} + BiLSTM (Val Accuracy: {val_acc*100:.2f}%)", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Class", fontsize=11)
plt.ylabel("True Class", fontsize=11)
plt.tight_layout()
plt.savefig(model_dir / "confusion_matrix_val.png", dpi=300)
plt.show()

# Test Confusion Matrix Heatmap
plt.figure(figsize=(8.5, 6.5))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True, square=True)
plt.title(f"Confusion Matrix (Test): {m_name} + BiLSTM (Test Accuracy: {test_acc*100:.2f}%)", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Class", fontsize=11)
plt.ylabel("True Class", fontsize=11)
plt.tight_layout()
plt.savefig(model_dir / "confusion_matrix.png", dpi=300)
plt.savefig(model_dir / "confusion_matrix_test.png", dpi=300)
plt.show()

print(f"\\nAll model artifacts saved to: {model_dir.resolve()}")
'''

# Verify AST syntax
ast.parse(new_code)
print("AST parse validated successfully.")

# Update notebook cell 16
nb["cells"][16]["source"] = [line + "\n" for line in new_code.strip().split("\n")]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook 06_MobileNet_pretrained_phase1.ipynb successfully updated!")
