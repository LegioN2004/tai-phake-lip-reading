import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from pathlib import Path

# Set random seed for exact reproducibility
np.random.seed(42)

PROJECT_ROOT = Path("/Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading")
GHAPLA_DIR = PROJECT_ROOT / "results" / "ghapla"
GHAPLA_DIR.mkdir(parents=True, exist_ok=True)

# 11 Tai Phake digits with phonetic descriptions
DIGIT_DETAILS = {
    0: {"label": "d0", "word": "Pao", "meaning": "Zero", "type": "Clean Bilabial Plosive /p/"},
    1: {"label": "d1", "word": "Naung", "meaning": "One", "type": "Problematic Motionless Nasal /n/"},
    2: {"label": "d2", "word": "Saung", "meaning": "Two", "type": "Sibilant + Rounded Vowel"},
    3: {"label": "d3", "word": "Saam", "meaning": "Three", "type": "Clean Bilabial Nasal /m/ Coda"},
    4: {"label": "d4", "word": "Si", "meaning": "Four", "type": "Problematic Dental Sibilant /s/ (s25 collision)"},
    5: {"label": "d5", "word": "Ha", "meaning": "Five", "type": "Open Glottal Aperture /h/"},
    6: {"label": "d6", "word": "Hok", "meaning": "Six", "type": "Velar Coda Closure /k/"},
    7: {"label": "d7", "word": "Chit", "meaning": "Seven", "type": "Fatal Viseme Collision (Affricate /tʃ/)"},
    8: {"label": "d8", "word": "Pet", "meaning": "Eight", "type": "Clean High-Energy Bilabial /p/"},
    9: {"label": "d9", "word": "Kao", "meaning": "Nine", "type": "Invisible Soft-Palate Velar /k/"},
    10: {"label": "d10", "word": "Sip", "meaning": "Ten", "type": "Dental /s/ + Bilabial Closure /p/"}
}

CLASS_LABELS = [f"d{i}" for i in range(11)]
cm_df = pd.read_csv(GHAPLA_DIR / "simulated_90_4pct_confusion_matrix.csv", index_col=0)

y_true = []
y_pred = []
y_probs = []

# Class-specific confidence parameters reflecting realistic model calibration
class_base_conf = {
    0: 4.4, 1: 2.3, 2: 3.3, 3: 4.6, 4: 2.4,
    5: 3.5, 6: 3.4, 7: 2.1, 8: 4.6, 9: 2.3, 10: 2.2
}

confusion_affinities = {
    4: [10, 7],
    10: [4, 7, 2],
    7: [4, 1, 10],
    9: [2, 0],
    2: [9, 6],
    1: [7, 4],
    5: [1, 3],
    6: [2]
}

for true_idx, true_cls in enumerate(CLASS_LABELS):
    row = cm_df.loc[true_cls]
    for pred_idx, pred_cls in enumerate(CLASS_LABELS):
        count = int(row[pred_cls])
        for _ in range(count):
            y_true.append(true_idx)
            y_pred.append(pred_idx)
            
            # Base logits with slight random jitter
            logits = np.random.normal(-1.2, 0.35, size=11)
            
            # Phonetically related competitors get slightly elevated baseline logits
            if true_idx in confusion_affinities:
                for rel_cls in confusion_affinities[true_idx]:
                    if rel_cls != pred_idx and rel_cls != true_idx:
                        logits[rel_cls] = np.random.normal(0.2, 0.3)
            
            if true_idx == pred_idx:
                # Correct prediction
                mean_conf = class_base_conf[true_idx]
                sample_conf = np.random.normal(mean_conf, 0.45)
                logits[true_idx] = sample_conf
                
                # Runner-up logit for difficult classes
                if true_idx in confusion_affinities:
                    primary_comp = confusion_affinities[true_idx][0]
                    logits[primary_comp] = sample_conf - np.random.uniform(0.9, 1.7)
            else:
                # Misclassification
                pred_logit = np.random.normal(2.3, 0.25)
                margin = np.random.uniform(0.18, 0.50)
                true_logit = pred_logit - margin
                
                logits[pred_idx] = pred_logit
                logits[true_idx] = true_logit
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            prob = exp_logits / np.sum(exp_logits)
            
            # Verify strict argmax
            if np.argmax(prob) != pred_idx:
                logits[pred_idx] = np.max(logits) + 0.4
                exp_logits = np.exp(logits - np.max(logits))
                prob = exp_logits / np.sum(exp_logits)
            
            y_probs.append(prob)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

# Confirm accuracy is exactly 90.45% (199 / 220)
accuracy = np.mean(y_true == y_pred)
assert np.isclose(accuracy, 199/220, atol=1e-4), f"Accuracy mismatch: {accuracy}"

y_true_bin = label_binarize(y_true, classes=list(range(11)))

# Calculate One-vs-Rest ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(11):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Micro-average ROC
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Macro-average ROC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(11)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(11):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= 11

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Colors palette for 11 classes + averages
colors = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# Create Publication-Ready 2-Panel ROC Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

# ==================== Panel 1: Full Range [0, 1] x [0, 1] ====================
ax1.plot(fpr["micro"], tpr["micro"],
         label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})',
         color='deeppink', linestyle=':', linewidth=3)

ax1.plot(fpr["macro"], tpr["macro"],
         label=f'Macro-average (AUC = {roc_auc["macro"]:.4f})',
         color='navy', linestyle='--', linewidth=3)

for i in range(11):
    label_text = f"{DIGIT_DETAILS[i]['label']}: {DIGIT_DETAILS[i]['word']} (AUC = {roc_auc[i]:.4f})"
    ax1.plot(fpr[i], tpr[i], color=colors[i], lw=1.8, label=label_text)

ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Random Chance (AUC = 0.5000)')
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.05])
ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold')
ax1.set_title('One-vs-Rest ROC Curves (Full Range [0, 1])\nHypothetical 90.45% Validation Scenario (220 Samples)', fontsize=13, fontweight='bold', pad=12)
ax1.legend(loc='lower right', fontsize=8.5, frameon=True, facecolor='white', framealpha=0.95)
ax1.grid(True, linestyle='--', alpha=0.5)

# ==================== Panel 2: Zoomed View (Top-Left Corner) ====================
# Highlights the clear divergence of problematic classes vs. clean classes
ax2.plot(fpr["micro"], tpr["micro"],
         label=f'Micro-avg (AUC = {roc_auc["micro"]:.4f})',
         color='deeppink', linestyle=':', linewidth=3)

ax2.plot(fpr["macro"], tpr["macro"],
         label=f'Macro-avg (AUC = {roc_auc["macro"]:.4f})',
         color='navy', linestyle='--', linewidth=3)

for i in range(11):
    label_text = f"{DIGIT_DETAILS[i]['label']}: {DIGIT_DETAILS[i]['word']} (AUC = {roc_auc[i]:.4f})"
    ax2.plot(fpr[i], tpr[i], color=colors[i], lw=2.2, label=label_text)

ax2.set_xlim([-0.005, 0.12])
ax2.set_ylim([0.70, 1.01])
ax2.set_xlabel('False Positive Rate (Zoomed: 0.00 to 0.12)', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Positive Rate (Zoomed: 0.70 to 1.00)', fontsize=12, fontweight='bold')
ax2.set_title('Zoomed Articulatory Divergence Region\nShowing Visible Degradation in Ambiguous Visemes (d7, d10, d4, d9, d1)', fontsize=13, fontweight='bold', pad=12)
ax2.legend(loc='lower right', fontsize=8.5, frameon=True, facecolor='white', framealpha=0.95)
ax2.grid(True, linestyle='--', alpha=0.6)

# Add prominent watermark banner
plt.suptitle('THEORETICAL ROC/AUC AUDIT: SIMULATED 90.45% VALIDATION ACCURACY\n[NOT FROM REAL PHYSICAL TRAINING — FOR FEASIBILITY AUDIT ONLY]',
             fontsize=14, fontweight='heavy', color='#8B0000', y=1.02)

plt.tight_layout()
plot_path = GHAPLA_DIR / "simulated_90_4pct_roc_curve.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved ROC Curve figure to: {plot_path}")

# ==================== Save Metrics JSON ====================
roc_metrics = {
    "audit_type": "HYPOTHETICAL_FEASIBILITY_AUDIT_NOT_REAL_TRAINING",
    "simulated_validation_accuracy": 0.9045,
    "accuracy_pct": "90.45%",
    "macro_average_auc": round(float(roc_auc["macro"]), 4),
    "micro_average_auc": round(float(roc_auc["micro"]), 4),
    "per_class_auc": {
        DIGIT_DETAILS[i]["label"]: {
            "digit_name": DIGIT_DETAILS[i]["word"],
            "phonetic_type": DIGIT_DETAILS[i]["type"],
            "auc": round(float(roc_auc[i]), 4),
            "simulated_recall_at_top1": float(cm_df.iloc[i, i]) / 20.0
        }
        for i in range(11)
    },
    "key_articulatory_findings": [
        "Clean bilabial plosives and nasals (d0, d3, d8) achieve near-perfect AUC (>=0.999) due to distinct mouth closure.",
        "Problematic viseme cluster (d7 Chit, d10 Sip, d4 Si) displays depressed AUCs (0.994 - 0.997) due to false-positive cross-firing across dental thresholds.",
        "Multiclass 1-vs-Rest AUC inflation: In an 11-class problem, each class faces a 1:10 positive-to-negative ratio (20 pos vs 200 neg). Because 195+ negative samples have near-zero scores, OvR AUC is mathematically propelled above 0.99 despite noticeable errors."
    ]
}

with open(GHAPLA_DIR / "simulated_90_4pct_roc_auc.json", "w") as f:
    json.dump(roc_metrics, f, indent=2)
print("Saved ROC AUC metrics JSON.")

# ==================== Save CSV Summary Table ====================
summary_rows = []
for i in range(11):
    summary_rows.append({
        "Digit": DIGIT_DETAILS[i]["label"],
        "Word": DIGIT_DETAILS[i]["word"],
        "Phonetic Description": DIGIT_DETAILS[i]["type"],
        "Simulated Recall (Top-1)": f"{int(cm_df.iloc[i, i])}/20 ({cm_df.iloc[i, i]/20.0*100:.1f}%)",
        "OvR ROC AUC": f"{roc_auc[i]:.4f}"
    })
summary_rows.append({
    "Digit": "OVERALL",
    "Word": "Macro-Average",
    "Phonetic Description": "Unweighted mean across all 11 digit classes",
    "Simulated Recall (Top-1)": "199/220 (90.45%)",
    "OvR ROC AUC": f"{roc_auc['macro']:.4f}"
})
summary_rows.append({
    "Digit": "OVERALL",
    "Word": "Micro-Average",
    "Phonetic Description": "Aggregated across all 2,420 binary class decisions",
    "Simulated Recall (Top-1)": "199/220 (90.45%)",
    "OvR ROC AUC": f"{roc_auc['micro']:.4f}"
})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(GHAPLA_DIR / "simulated_90_4pct_roc_summary.csv", index=False)
print("Saved ROC summary table CSV.")
