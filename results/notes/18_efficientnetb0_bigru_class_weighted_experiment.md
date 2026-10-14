# Experiment Notes: 18_efficientnetb0_bigru_class_weighted_experiment

- **Notebook**: [`notebooks/18_efficientnetb0_bigru_class_weighted_experiment.ipynb`](../../notebooks/18_efficientnetb0_bigru_class_weighted_experiment.ipynb)
- **Dataset Mode**: `80_20` (24 base train speakers / 6 base val speakers)
- **Goal**: Retrain our champion `EfficientNetB0 + BiGRU + Attention Pooling` architecture using targeted class weights in the loss objective to heavily penalize errors on historically ambiguous classes ($d_1, d_2, d_4, d_6, d_7, d_9$).

---

## 1. Why Class Weights?
Standard cross-entropy loss treats errors across all 11 classes identically ($w_c = 1.0$). In Tai Phake lip reading, certain visemes suffer from high acoustic/visual overlap (especially the $d_4 \leftrightarrow d_7 \leftrightarrow d_{10}$ cluster and $d_1 \leftrightarrow d_2$).

By assigning higher loss penalties to these specific classes, backpropagation forces the feature representations to focus on subtle viseme distinctions:

$$\mathcal{L}_{\text{weighted}} = -\sum_{c=0}^{10} w_c \cdot y_{c,\text{smooth}} \log(\hat{y}_c)$$

### Class Weight Vector:
```python
CLASS_WEIGHTS_DICT = {
    0: 1.0,   # d0
    1: 1.8,   # d1 (boosted)
    2: 1.6,   # d2 (boosted)
    3: 1.0,   # d3
    4: 1.8,   # d4 (boosted - triangle)
    5: 1.0,   # d5
    6: 1.2,   # d6 (minor boost)
    7: 1.5,   # d7 (boosted - triangle)
    8: 1.0,   # d8
    9: 1.6,   # d9 (boosted)
    10: 1.0,  # d10 (reference in triangle)
}
```

---

## 2. Training Protocol
1. **Stage 1 (Frozen EfficientNetB0)**:
   - Backbone frozen.
   - Adam ($\text{lr} = 10^{-4}$), EarlyStopping (patience 8 monitoring `val_loss`).
   - `model.fit(..., class_weight=CLASS_WEIGHTS_DICT)`.
2. **Stage 2 (Top 20% Unfrozen, BatchNorm Frozen)**:
   - Top 20% layers unfrozen.
   - Adam ($\text{lr} = 10^{-5}$), EarlyStopping (patience 6 monitoring `val_loss`).
   - `model.fit(..., class_weight=CLASS_WEIGHTS_DICT)`.

---

## 3. How to Run
Open [`notebooks/18_efficientnetb0_bigru_class_weighted_experiment.ipynb`](../../notebooks/18_efficientnetb0_bigru_class_weighted_experiment.ipynb) and run all cells sequentially.

---

## 4. Expected Outputs
Saved to `results/efficientnetb0_bigru_class_weighted_80_20/`:
- `efficientnetb0_bigru_class_weighted_stage1_best.weights.h5`
- `efficientnetb0_bigru_class_weighted_stage2_best.weights.h5`
- `training_history.csv` & `training_history_curves.png`
- `confusion_matrix_val_80_20.png`
- `metrics_80_20.json` (Expected +3% to +5% overall gain over unweighted 60.0% baseline).
