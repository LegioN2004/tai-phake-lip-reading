import json
from pathlib import Path
import copy

def split_notebook_05():
    with open("notebooks/05_model_training_baseline.ipynb") as f:
        nb5 = json.load(f)

    # 4 models in notebook 05
    models_config = [
        {
            "filename": "05_CnnLstm_model_training_baseline.ipynb",
            "model_type": "cnn_lstm",
            "display_name": "CNN + LSTM",
            "title": "# 05 - Controlled Baseline: CNN + LSTM Model Training & Benchmark"
        },
        {
            "filename": "05_CnnBiLstm_model_training_baseline.ipynb",
            "model_type": "cnn_bilstm",
            "display_name": "CNN + BiLSTM",
            "title": "# 05 - Controlled Baseline: CNN + BiLSTM Model Training & Benchmark"
        },
        {
            "filename": "05_CnnGru_model_training_baseline.ipynb",
            "model_type": "cnn_gru",
            "display_name": "CNN + GRU",
            "title": "# 05 - Controlled Baseline: CNN + GRU Model Training & Benchmark"
        },
        {
            "filename": "05_CnnBiGru_model_training_baseline.ipynb",
            "model_type": "cnn_bigru",
            "display_name": "CNN + BiGRU",
            "title": "# 05 - Controlled Baseline: CNN + BiGRU Model Training & Benchmark"
        },
    ]

    for cfg in models_config:
        m_name = cfg["model_type"]
        disp_name = cfg["display_name"]

        new_nb = {
            "cells": [],
            "metadata": copy.deepcopy(nb5["metadata"]),
            "nbformat": nb5["nbformat"],
            "nbformat_minor": nb5["nbformat_minor"],
        }

        # Cell 0: Header markdown
        c0 = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"{cfg['title']}\n",
                "## Tai Phake Visual Speech Recognition (VSR) Project\n",
                "\n",
                f"**Target Architecture**: Controlled **{disp_name}** (`{m_name}`) baseline.\n",
                "\n",
                "---\n",
                "### Experimental Controls & Configuration:\n",
                "- **Dataset**: Canonical speaker-independent partitions (`data/processed/train`, `val`, `test`).\n",
                "- **Spatio-Temporal Pipeline**: Lightweight 2D CNN backbone + recurrent aggregator.\n",
                "- **Temporal Standardization**: Exactly 30 frames per utterance.\n",
                "- **Optimization**: Adam (lr=1e-4), max 100 epochs, Early Stopping (patience=15, monitor='val_loss').\n",
                "- **Consolidated Output**: Complete evaluation metrics (accuracy in %, loss, macro-averages), classification report, confusion matrix, and training curves are organized neatly at the end of the notebook."
            ]
        }
        new_nb["cells"].append(c0)

        # Cell 1: MD Imports
        new_nb["cells"].append(copy.deepcopy(nb5["cells"][1]))

        # Cell 2: Code Imports (add seaborn for heatmap plot if needed)
        c2 = copy.deepcopy(nb5["cells"][2])
        c2_src = "".join(c2["source"])
        if "seaborn" not in c2_src:
            c2_src = c2_src.replace(
                "import matplotlib.pyplot as plt",
                "import matplotlib.pyplot as plt\nimport seaborn as sns"
            )
            c2["source"] = [line + "\n" for line in c2_src.splitlines()]
        new_nb["cells"].append(c2)

        # Cell 3: MD Config & DataLoader
        new_nb["cells"].append(copy.deepcopy(nb5["cells"][3]))

        # Cell 4: Code Config & DataLoader
        new_nb["cells"].append(copy.deepcopy(nb5["cells"][4]))

        # Cell 5: MD Architecture Builder
        new_nb["cells"].append(copy.deepcopy(nb5["cells"][5]))

        # Cell 6: Code Architecture Builder
        new_nb["cells"].append(copy.deepcopy(nb5["cells"][6]))

        # Cell 7: MD Section for single model training
        m_md = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"## 4. Controlled Model Execution: {disp_name}\n",
                f"Trains the `{m_name}` model with EarlyStopping, saves checkpoints, and outputs consolidated evaluation metrics on the unseen test set."
            ]
        }
        new_nb["cells"].append(m_md)

        # Cell 8: Code training and consolidated outputs
        code_src = f"""# Section 4: {disp_name} ({m_name}) Training and Consolidated Evaluation
m_name = "{m_name}"
disp_name = "{disp_name}"

print("=" * 75)
print(f"          BASELINE EXPERIMENT: {{disp_name.upper()}} ({{m_name}})")
print("=" * 75)

model_save_dir = MODELS_DIR / m_name
model_save_dir.mkdir(parents=True, exist_ok=True)
best_model_path = model_save_dir / "best_model.keras"
hist_csv_path = model_save_dir / "training_history.csv"
metrics_json_path = model_save_dir / "metrics.json"

if not best_model_path.exists() or RE_TRAIN:
    print(f"Training {{m_name}} from scratch (max {{MAX_EPOCHS}} epochs, patience {{PATIENCE}})...")
    tf.keras.backend.clear_session()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model = build_controlled_model(m_name)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(best_model_path), monitor="val_loss", save_best_only=True, verbose=0)
    ]
    t0 = time.time()
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=MAX_EPOCHS, callbacks=callbacks, verbose=1)
    train_time = time.time() - t0
    print(f"Training completed in {{train_time:.2f}}s")
    
    hist_df = pd.DataFrame(history.history)
    hist_df.index.name = "epoch"
    hist_df.to_csv(hist_csv_path)
else:
    print(f"Loading existing best checkpoint for {{m_name}} from: {{best_model_path.resolve()}}")
    hist_df = pd.read_csv(hist_csv_path, index_col="epoch")
    train_time = 0.0

best_epoch = int(np.argmin(hist_df["val_loss"]) + 1)
best_val_loss = float(np.min(hist_df["val_loss"]))
best_val_acc = float(hist_df["val_accuracy"].iloc[best_epoch - 1])

# Evaluate on TEST set only with selected checkpoint
best_model = tf.keras.models.load_model(str(best_model_path))
total_params = int(best_model.count_params())

y_true_list, y_pred_list, y_prob_list = [], [], []
for x_b, y_b in test_dataset:
    preds = best_model.predict(x_b, verbose=0)
    y_prob_list.extend(preds)
    y_pred_list.extend(np.argmax(preds, axis=1))
    y_true_list.extend(y_b.numpy())

y_true = np.array(y_true_list)
y_pred = np.array(y_pred_list)
y_prob = np.array(y_prob_list)

test_acc = float(accuracy_score(y_true, y_pred))
test_macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
test_macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
test_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

correct_count = int(np.sum(y_true == y_pred))
incorrect_count = int(np.sum(y_true != y_pred))

# Save predictions.csv
pred_df = test_df.copy()
pred_df["true_label"] = y_true
pred_df["predicted_label"] = y_pred
pred_df["predicted_digit"] = [ID_TO_CLASS[idx] for idx in y_pred]
pred_df["correct"] = (y_true == y_pred)
for cls_idx in range(NUM_CLASSES):
    pred_df[f"prob_d{{cls_idx}}"] = y_prob[:, cls_idx]
pred_df.to_csv(model_save_dir / "predictions.csv", index=False)

# Save metrics.json & CSV
metrics_data = {{
    "model": m_name,
    "display_name": disp_name,
    "parameters_count": total_params,
    "epochs_trained": len(hist_df),
    "best_epoch": best_epoch,
    "best_val_loss": round(best_val_loss, 4),
    "best_val_accuracy": round(best_val_acc, 4),
    "test_accuracy": round(test_acc, 4),
    "test_accuracy_pct": round(test_acc * 100, 2),
    "test_macro_precision": round(test_macro_prec, 4),
    "test_macro_recall": round(test_macro_rec, 4),
    "test_macro_f1": round(test_macro_f1, 4),
    "correct_test_predictions": correct_count,
    "incorrect_test_predictions": incorrect_count,
    "per_class": {{f"d{{i}}": {{"precision": float(per_class_prec[i]), "recall": float(per_class_rec[i]), "f1": float(per_class_f1[i]), "support": int(np.sum(y_true == i))}} for i in range(NUM_CLASSES)}},
    "confusion_matrix": cm.tolist()
}}
with open(metrics_json_path, "w") as f:
    json.dump(metrics_data, f, indent=2)

# Save config.json
config_data = {{
    "model_type": m_name,
    "display_name": disp_name,
    "input_shape": [NUM_FRAMES, TARGET_HEIGHT, TARGET_WIDTH, CHANNELS],
    "cnn_feature_dim": CNN_FEATURE_DIM,
    "rnn_units": RNN_HIDDEN_UNITS,
    "rnn_layers": RNN_LAYERS,
    "dropout": DROPOUT_RATE,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": PATIENCE,
    "optimizer": "Adam",
    "loss": "SparseCategoricalCrossentropy",
    "parameters_count": total_params
}}
with open(model_save_dir / "config.json", "w") as f:
    json.dump(config_data, f, indent=2)

# Per-class summary DataFrame
df_per_class = pd.DataFrame({{
    "Digit": EXPECTED_DIGITS,
    "Precision": [round(p, 4) for p in per_class_prec],
    "Recall": [round(r, 4) for r in per_class_rec],
    "F1-Score": [round(f, 4) for f in per_class_f1],
    "Support": [int(np.sum(y_true == i)) for i in range(NUM_CLASSES)]
}}).set_index("Digit")
df_per_class.to_csv(model_save_dir / "per_class_metrics.csv")

# ==============================================================================
# 5. CONSOLIDATED ALL-IN-ONE OUTPUT RESULTS (EASY TO READ AT A GLANCE)
# ==============================================================================
print("\\n" + "=" * 80)
print(f"                 FINAL EVALUATION SUMMARY: {{disp_name.upper()}} ({{m_name}})")
print("=" * 80)
print(f"  Test Accuracy:           {{test_acc * 100:.2f}}%  ({{correct_count}}/{{len(y_true)}} correct)")
print(f"  Validation Accuracy:     {{best_val_acc * 100:.2f}}%  (Best Epoch: {{best_epoch}})")
print(f"  Validation Loss:         {{best_val_loss:.4f}}")
print(f"  Macro Precision:         {{test_macro_prec:.4f}}")
print(f"  Macro Recall:            {{test_macro_rec:.4f}}")
print(f"  Macro F1-Score:          {{test_macro_f1:.4f}}")
print(f"  Total Parameters:        {{total_params:,}}")
if train_time > 0:
    print(f"  Training Duration:       {{train_time:.2f}}s")
print("-" * 80)
print("Per-Class Metrics (Unseen Test Set):")
display(df_per_class)
print("=" * 80)

# 1. Training & Validation History Trajectories Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
epochs_range = range(1, len(hist_df) + 1)

axes[0].plot(epochs_range, hist_df["loss"], label="Train Loss", color="royalblue", lw=2)
axes[0].plot(epochs_range, hist_df["val_loss"], label="Val Loss", color="crimson", lw=2)
axes[0].axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({{best_epoch}})")
axes[0].set_title(f"{{disp_name}}: Loss Trajectory", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend()

axes[1].plot(epochs_range, hist_df["accuracy"] * 100, label="Train Accuracy (%)", color="royalblue", lw=2)
axes[1].plot(epochs_range, hist_df["val_accuracy"] * 100, label="Val Accuracy (%)", color="crimson", lw=2)
axes[1].axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({{best_epoch}})")
axes[1].set_title(f"{{disp_name}}: Accuracy Trajectory (%)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend()

plt.tight_layout()
plt.savefig(model_save_dir / "training_curves.png", dpi=150)
plt.show()

# 2. Confusion Matrix Heatmap
df_cm = pd.DataFrame(cm, index=EXPECTED_DIGITS, columns=EXPECTED_DIGITS)
df_cm.to_csv(model_save_dir / "confusion_matrix.csv")

plt.figure(figsize=(8.5, 6.5))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True, square=True)
plt.title(f"Confusion Matrix: {{disp_name}} (Test Accuracy: {{test_acc * 100:.2f}}%)", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Class", fontsize=11)
plt.ylabel("True Class", fontsize=11)
plt.tight_layout()
plt.savefig(model_save_dir / "confusion_matrix.png", dpi=150)
plt.show()

print(f"\\nAll model artifacts saved to: {{model_save_dir.resolve()}}")
"""

        c8 = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code_src.splitlines()]
        }
        new_nb["cells"].append(c8)

        out_file = Path("notebooks") / cfg["filename"]
        with open(out_file, "w") as f:
            json.dump(new_nb, f, indent=1)
        print(f"Created {out_file} successfully.")

if __name__ == "__main__":
    split_notebook_05()
