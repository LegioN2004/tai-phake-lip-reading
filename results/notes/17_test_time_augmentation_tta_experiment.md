# Experiment Notes: 17_test_time_augmentation_tta_experiment

- **Notebook**: [`notebooks/17_test_time_augmentation_tta_experiment.ipynb`](../../notebooks/17_test_time_augmentation_tta_experiment.ipynb)
- **Dataset Mode**: `80_20` (6 validation base speakers, 220 utterances)
- **Goal**: Evaluate inference stabilization through Test-Time Augmentation (TTA) on our co-champion recurrent models (BiLSTM and BiGRU).

---

## 1. What is Test-Time Augmentation (TTA)?
In video lip reading, isolated utterances often contain boundary uncertainty (whether the mouth opens at frame 2 or frame 4) and subtle facial landmark jitter.
Instead of predicting on only 1 view, TTA evaluates **4 complementary views per utterance** and averages their predicted softmax probabilities:
$$\mathbf{P}_{\text{TTA}} = \frac{1}{4} \left( \mathbf{P}_{\text{orig}} + \mathbf{P}_{\text{shift}+2} + \mathbf{P}_{\text{shift}-2} + \mathbf{P}_{\text{hflip}} \right)$$

1. **View 1 (Original)**: Standard uniform interpolation across all raw frames.
2. **View 2 (Temporal Shift +2)**: Starts 2 frames later into utterance (deeper onset trajectory).
3. **View 3 (Temporal Shift -2)**: Starts 2 frames earlier (pre-onset resting viseme).
4. **View 4 (Horizontal Flip)**: Horizontally mirrored mouth crop (human lip movements during vowels/consonants are bilaterally symmetric, providing spatial invariance).

---

## 2. Models Evaluated
- **Model 1**: `EfficientNetB0 + BiLSTM + Attention Pooling` (`results/efficientnetb0_bilstm_attention_pooling_80_20/`)
- **Model 2**: `EfficientNetB0 + BiGRU + Attention Pooling` (`results/efficientnetb0_bigru_attention_pooling_80_20/`)
- **Combined Ensemble + TTA**: Averaging TTA predictions from both co-champions simultaneously.

---

## 3. How to Run
Open [`notebooks/17_test_time_augmentation_tta_experiment.ipynb`](../../notebooks/17_test_time_augmentation_tta_experiment.ipynb) and run all cells sequentially. No training is performed; it loads existing Stage 2 checkpoint weights and computes inference.

---

## 4. Expected Outputs
Saved to `results/tta_evaluation_80_20/`:
- `tta_per_class_comparison.csv`: Comparison of raw accuracy vs TTA accuracy for each digit ($d_0$ through $d_{10}$).
- `confusion_matrix_tta_comparison.png`: Side-by-side confusion matrix of raw model vs TTA ensemble.
- `metrics_tta.json`: Consolidated metrics detailing individual view performance and net accuracy gain (+1% to +3%).
