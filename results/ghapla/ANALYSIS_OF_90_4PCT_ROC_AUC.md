# Theoretical ROC & AUC Analysis: Deconstructing the 90.45% Validation Scenario

**Audit Directory**: `results/ghapla/`  
**Target Scenario**: Simulated 90.45% Validation Accuracy ($199 / 220$ Correct Predictions)  
**Status**: Theoretical Plausibility Audit (Simulated Benchmark — Not From Real Physical Training)  
**Associated Deliverables**:
* Plot: `results/ghapla/simulated_90_4pct_roc_curve.png`
* Metrics JSON: `results/ghapla/simulated_90_4pct_roc_auc.json`
* Summary CSV: `results/ghapla/simulated_90_4pct_roc_summary.csv`
* Confusion Matrix: `results/ghapla/simulated_90_4pct_confusion_matrix.csv`

---

## 1. Executive Summary & The "AUC Paradox"

When evaluating the hypothetical **90.45% validation accuracy** model across the 11 balanced Tai Phake digit classes (`d0`–`d10`, 20 samples per class, $N = 220$):

* **Macro-Average ROC AUC**: $\mathbf{0.9987}$ ($99.87\%$)
* **Micro-Average ROC AUC**: $\mathbf{0.9988}$ ($99.88\%$)
* **Per-Class AUC Range**: $\mathbf{0.9942}$ (hardest: $d_7$ *Chit*) to $\mathbf{1.0000}$ (clean bilabials: $d_0, d_3, d_8$)

```
+-------------------------------------------------------------------------------------------------+
|                                     THE 90.4% AUC PARADOX                                       |
|                                                                                                 |
|   Validation Accuracy = 90.45%   ═══════════════════════►   One-vs-Rest ROC AUC = 0.9987       |
|   (21 errors out of 220 samples)                            (Near-perfect rank discrimination) |
+-------------------------------------------------------------------------------------------------+
```

### Why is ROC AUC $\approx 0.999$ when Accuracy is "Only" $90.4\%$?
In multi-class classification ($K = 11$), One-vs-Rest (OvR) ROC curves exhibit **severe class imbalance**:
* For each individual class, there are **20 positive instances** ($y = c$) and **200 negative instances** ($y \neq c$)—a **1:10 positive-to-negative ratio** ($90.9\%$ negatives).
* ROC AUC mathematically measures the Wilcoxon-Mann-Whitney probability:
  $$\text{AUC} = P(\text{Score}_{\text{positive}} > \text{Score}_{\text{negative}})$$
* Even if class $d_7$ suffers 5 errors and gets confused with dental twins $d_4$ and $d_{10}$, the model assigns near-zero probabilities ($P < 0.02$) to the remaining **$190+$ negative samples** belonging to unrelated phonemes ($d_0$ *Pao*, $d_3$ *Saam*, $d_5$ *Ha*, $d_8$ *Pet*, etc.).
* Consequently, out of the $20 \times 200 = 4,000$ possible (positive, negative) sample pairs, over **$3,975+$ pairs are ranked in the correct order**, resulting in an astronomical AUC $> 0.99$.

---

## 2. Per-Class ROC AUC Breakdown Under the 90.45% Distribution

The table below details how the 90.45% simulation translates into One-vs-Rest ROC AUC across the 11 classes:

| Class | Word | Meaning | Phonetic / Articulatory Profile | Top-1 Recall | OvR ROC AUC | Curvature Profile |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **$d_0$** | *Pao* | Zero | Clean bilabial plosive onset /p/ | $20/20$ ($100.0\%$) | **$1.0000$** | Perfect right-angle curve at $(0, 1)$ |
| **$d_1$** | *Naung* | One | Problematic motionless alveolar nasal /n/ | $17/20$ ($85.0\%$) | **$0.9980$** | Slight bend near $FPR \approx 0.01$ |
| **$d_2$** | *Saung* | Two | Alveolar sibilant + rounded diphthong | $19/20$ ($95.0\%$) | **$0.9985$** | Minor dip from $d_9$ confusion |
| **$d_3$** | *Saam* | Three | Distinct bilabial nasal coda /m/ | $20/20$ ($100.0\%$) | **$1.0000$** | Perfect right-angle curve at $(0, 1)$ |
| **$d_4$** | *Si* | Four | Dental sibilant /s/ ($s_{25}$ closure collision) | $18/20$ ($90.0\%$) | **$0.9975$** | Visible threshold depression |
| **$d_5$** | *Ha* | Five | Broad vertical glottal aperture /h/ | $19/20$ ($95.0\%$) | **$1.0000$** | Near-instant rise to ceiling |
| **$d_6$** | *Hok* | Six | Rounded open vowel + velar stop /k/ | $19/20$ ($95.0\%$) | **$1.0000$** | Near-instant rise to ceiling |
| **$d_7$** | *Chit* | Seven | **Hardest Viseme**: Postalveolar affricate /tʃ/ | $15/20$ ($75.0\%$) | **$0.9942$** | **Lowest AUC**: Significant sag below $0.80$ TPR |
| **$d_8$** | *Pet* | Eight | High-energy bilabial plosive /p/ | $20/20$ ($100.0\%$) | **$1.0000$** | Perfect right-angle curve at $(0, 1)$ |
| **$d_9$** | *Kao* | Nine | Invisible soft-palate velar plosive /k/ | $16/20$ ($80.0\%$) | **$0.9980$** | Sagging elbow between $TPR \in [0.80, 0.95]$ |
| **$d_{10}$** | *Sip* | Ten | Dental onset /s/ + bilabial coda closure /p/ | $16/20$ ($80.0\%$) | **$0.9960$** | Cross-firing with $d_4$ and $d_7$ |
| **OVERALL** | **Macro-Avg** | — | **Unweighted arithmetic mean of 11 classes** | **$199/220$ ($90.45\%$)** | **$0.9987$** | Smooth, high-convexity average |
| **OVERALL** | **Micro-Avg** | — | **Aggregated over 2,420 binary decisions** | **$199/220$ ($90.45\%$)** | **$0.9988$** | Global true positive dominance |

---

## 3. Visual Anatomy of the ROC Curve (`simulated_90_4pct_roc_curve.png`)

The generated figure in `results/ghapla/simulated_90_4pct_roc_curve.png` contains **two synchronized panels**:

```
+-------------------------------------------------------+-------------------------------------------------------+
| PANEL 1: FULL ROC SPACE [0, 1] x [0, 1]               | PANEL 2: ZOOMED ARTICULATORY DIVERGENCE REGION        |
|                                                       |          (FPR: 0.00 to 0.12, TPR: 0.70 to 1.00)       |
|                                                       |                                                       |
| 1.00 ┌──────────────────────────────────────────────┐ | 1.00 ┌───────────────────═══════════════════════════┐ |
|      │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                      │ |      │    ▲▲▲ d0, d3, d8 (Bilabials reach 1.000)    │ |
|      │█                                             │ | 0.95 │───- - - - - - - - - - - - - - - - - - - - - -│ |
|      │█                                             │ | 0.90 │              ▲ d4 (Si) reaches 0.90           │ |
| 0.50 │█                                             │ | 0.85 │         ▲ d1 (Naung) reaches 0.85             │ |
|      │█            Random Chance (AUC=0.50)         │ | 0.80 │    ▲ d9 (Kao) & d10 (Sip) plateau at 0.80     │ |
|      │█           ╱                                 │ | 0.75 │ ▲ d7 (Chit) drops down to 0.75 TPR            │ |
| 0.00 └───────────╱──────────────────────────────────┘ | 0.70 └──────────────────────────────────────────────┘ |
|     0.00                    0.50               1.00   |     0.00          0.04          0.08          0.12    |
|                False Positive Rate                    |                  False Positive Rate                  |
+-------------------------------------------------------+-------------------------------------------------------+
```

### Why the Zoomed Inset is Essential
1. **The Optical Illusion of Panel 1**:
   * On the standard $[0, 1] \times [0, 1]$ plot, every single curve hugs the top-left boundary. A superficial glance makes it look as though all 11 digits are performing identically at near-perfection.
2. **The Truth in Panel 2 (Zoomed Region)**:
   * When magnifying the region where $FPR \in [0.00, 0.12]$ and $TPR \in [0.70, 1.00]$, the curves **severely diverge**:
     * Clean bilabials ($d_0, d_3, d_8$) shoot straight to $1.00$ TPR with $0.00$ FPR.
     * The problematic visemes ($d_7$ *Chit*, $d_9$ *Kao*, $d_{10}$ *Sip*) lag significantly: at zero false positives, their True Positive Rate is only $0.75\text{--}0.80$. To capture the remaining 4–5 utterances, the classification threshold must be lowered so far that false alarms enter from the competing dental classes.

---

## 4. The Academic Trap: Why Reviewers Scrutinize 0.99+ AUCs

If this ROC curve is included in an academic research paper alongside a claim of $90.4\%$ accuracy:

1. **Reviewer Suspicion of Test Leakage**:
   * On silent video without acoustic waveforms, achieving an OvR AUC of **$0.9987$** on an unconstrained speaker-independent split is virtually unprecedented in the lip-reading literature.
   * Reviewers will immediately ask: *"Did the authors compute ROC on the training split, or did identity leakage allow the network to recognize the speaker's face rather than visual speech?"*
2. **The Metric of Choice: Precision-Recall (PR) Curves**:
   * Because of the 1:10 class imbalance in OvR evaluation, knowledgeable reviewers frequently demand **Precision-Recall (PR) curves** or **Average Precision (AP)** rather than ROC AUC.
   * On a PR curve, where false positives are penalized directly against the small pool of 20 positives:
     * $d_7$ PR-AUC would drop from $0.9942$ down to $\approx \mathbf{0.835}$.
     * $d_9$ PR-AUC would drop from $0.9980$ down to $\approx \mathbf{0.852}$.
     * Macro Average Precision would be $\approx \mathbf{0.910}$, providing a much more realistic picture of the model's actual performance.

---

## 5. Summary Table for Paper / Presentation

| Metric | Empirical 81.36% Ensemble (Notebook 22.2) | Hypothetical 90.45% Simulation (`ghapla`) | Delta ($\Delta$) | Research Plausibility Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Accuracy** | **81.36%** ($179/220$) | **90.45%** ($199/220$) | $+9.09\%$ | Fictitious (+20 artificial correct samples) |
| **Total Validation Errors** | **41 errors** | **21 errors** | $-20$ errors | Defies physical video anomalies |
| **Macro ROC AUC** | $\approx \mathbf{0.965}$ | $\mathbf{0.9987}$ | $+0.0337$ | Pushed into near-perfect territory |
| **Micro ROC AUC** | $\approx \mathbf{0.978}$ | $\mathbf{0.9988}$ | $+0.0208$ | Dominated by 1:10 negative class volume |
| **Worst-Class AUC ($d_7$)** | $\approx \mathbf{0.892}$ | $\mathbf{0.9942}$ | $+0.1022$ | Artificially masks dental affricate collision |
| **Scientific Integrity** | **100% Genuine & Defensible** | **Fabricated Simulation** | — | **Real 81.36% benchmark is publication-grade** |
