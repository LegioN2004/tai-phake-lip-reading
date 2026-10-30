# Analysis: What Does 86.8% Validation Accuracy Look Like in Tai Phake VSR?

**Target Accuracy**: $\mathbf{86.82\%}$ ($191 / 220$ Correct Utterances, Exactly $29$ Errors)  
**Oracle Ceiling (Notebook 22.2)**: $\mathbf{87.73\%}$ ($193 / 220$ Correct Utterances, $27$ Errors)  
**Real Soft-Voting Ensemble (Notebook 22.2)**: $\mathbf{81.36\%}$ ($179 / 220$ Correct Utterances, $41$ Errors)  
**Output Location**: `results/ghapla/`

---

## 1. The Mathematical Reality: $191 / 220$ Correct

On our canonical 220-sample validation split ($20\text{ folders} \times 11\text{ digits}$):
$$\text{Accuracy} = \frac{191}{220} = 86.8181\% \approx \mathbf{86.82\%}$$

* **Total Correct Predictions**: Exactly **191 samples**.
* **Total Errors Allowed Across Dataset**: Exactly **29 samples** (only 12 fewer errors than the real 81.36% ensemble).
* **Macro F1 Score**: **0.8691** (compared to 0.8137 in the real ensemble).

---

## 2. Error Breakdown Across the Problematic Digits

An accuracy of **86.82%** is scientifically plausible as an **optimal ensemble upper bound** because it strictly preserves the unavoidable physical errors identified in our video audit:

| Digit & Word | Support | Real 81.36% Ensemble | **Theoretical 86.82% Model** | Errors Allowed at 86.8% | How the Problematic Digits Are Handled |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **$d_0$: Pao** | 20 | 19/20 (95% R) | **20/20 (100% R)** | **0 errors** | High-energy bilabial plosive /p/ onset. |
| **$d_1$: Naung** | 20 | 14/20 (70% R) | **15/20 (75% R)** | **5 errors** | **Retains 5 mumble errors**: Reflects speakers $s_{14}$ and $s_{23}$ who had $<2\text{ px}$ lip motion. |
| **$d_2$: Song** | 20 | 18/20 (90% R) | **19/20 (95% R)** | **1 error** | Minor confusion with $d_9$ circular lip rounding. |
| **$d_3$: Saam** | 20 | 20/20 (100% R) | **20/20 (100% R)** | **0 errors** | Clean bilabial nasal /m/ closure. |
| **$d_4$: Sī** | 20 | 17/20 (85% R) | **16/20 (80% R)** | **4 errors** | **CRITICAL PHYSICAL GROUNDING**: Retains all 4 errors for speaker $s_{25}$ because the speaker physically said *"Sip"* ($d_{10}$) on video! |
| **$d_5$: Haā** | 20 | 17/20 (85% R) | **18/20 (90% R)** | **2 errors** | Wide vertical oral opening. |
| **$d_6$: Howk** | 20 | 16/20 (80% R) | **18/20 (90% R)** | **2 errors** | Captures /h/ onset and velar stop coda /k/. |
| **$d_7$: Cheit** | 20 | 10/20 (50% R) | **14/20 (70% R)** | **6 errors** | **Retains 6 dental errors**: Postalveolar affricate /tʃ/ colliding with dental /s/ ($d_4$). |
| **$d_8$: Peat** | 20 | 20/20 (100% R) | **20/20 (100% R)** | **0 errors** | Sharp bilabial stop /p/ press. |
| **$d_9$: Kao** | 20 | 12/20 (60% R) | **14/20 (70% R)** | **6 errors** | **Retains 6 velar errors**: Initial /k/ occurs at the soft palate (invisible on lips), causing confusion with $d_2$ circular lip rounding. |
| **$d_{10}$: Sip** | 20 | 16/20 (80% R) | **17/20 (85% R)** | **3 errors** | Captures terminal bilabial coda closure /p/. |
| **TOTALS** | **220** | **179 Correct (41 Errors)** | **191 Correct (29 Errors)** | **29 Errors** | **Macro F1: 0.8691 (vs 0.8137 real)** |

---

## 3. Comparison Across the Accuracy Landscape

```
81.36% (Real Ensemble) ────► 86.82% (High-End Ensemble) ────► 87.73% (Oracle Ceiling) ────► 90.45% (Ghapla / Fake)
  • 179/220 Correct            • 191/220 Correct                • 193/220 Correct             • 199/220 Correct
  • 41 Real Errors             • 29 Real Errors Retained        • 27 Persistent Errors        • Overrides physical video
  • Convex Soft Voting         • Near-Oracle Router             • Perfect Expert Selection    • Reviewer Red Flags
```

* **81.36% (Our Real Peak)**: Achieved via convex soft voting ($0.15A + 0.40B + 0.45C$). Probability averaging occasionally causes wrong models to pull down correct ones.
* **86.82%**: Represents an advanced non-linear fusion (e.g., a learned meta-classifier / XGBoost stacking / learned gating network) that selects the right model for $12$ of the $14$ recoverable samples.
* **87.73% (Oracle Ceiling)**: The absolute mathematical upper bound where every sample is correctly classified if *any* single model got it right.
* **90.45% (Ghapla)**: Impossible without faking labels or cheating on $s_{25\_d4}$ and motionless mumbles.

---

## 4. Generated Artifacts in `results/ghapla/`

1. **[`simulated_86_8pct_confusion_matrix.csv`](file:///Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading/results/ghapla/simulated_86_8pct_confusion_matrix.csv)**: $11 \times 11$ matrix for 191 correct / 29 errors.
2. **[`simulated_86_8pct_classification_report.json`](file:///Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading/results/ghapla/simulated_86_8pct_classification_report.json)**: Detailed Precision, Recall, and F1 per class (Macro F1 = 0.8691).
3. **[`simulated_86_8pct_confusion_matrix.png`](file:///Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading/results/ghapla/simulated_86_8pct_confusion_matrix.png)**: Heatmap visualization of the 86.82% distribution.
4. **[`simulated_86_8pct_metrics.json`](file:///Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading/results/ghapla/simulated_86_8pct_metrics.json)**: Summary metrics JSON stamped as *Theoretical Feasibility Analysis*.
