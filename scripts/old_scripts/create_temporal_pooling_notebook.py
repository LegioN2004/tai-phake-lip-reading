import json
from pathlib import Path

source_path = Path("notebooks/11_Stage2Rem_mobilenetv2_transformer_4block_experiment.ipynb")
dest_path = Path("notebooks/12_temporal_pooling_mobilenetv2_transformer_4block_experiment.ipynb")

with open(source_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Clear outputs and execution_counts for clean notebook state
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

# 1. Update Cell 0: Header & Experiment Context
cell_0_text = """# 12 - MobileNetV2 + 4-Block Transformer with Learnable Temporal Attention Pooling
## Tai Phake Visual Speech Recognition (VSR) Project

### Controlled Experiment: Learnable Temporal Attention Pooling vs. GlobalAveragePooling1D

**Purpose:**
Determine whether replacing fixed `GlobalAveragePooling1D` temporal aggregation with a learnable temporal attention/pooling mechanism improves word-level sequence classification accuracy over the 42.42% baseline.

**Baseline:**
- 96×96 Method D mouth crops, 30 frames
- Augmented training (original + aug1..aug4 = 120 dirs) + original-only validation (6 dirs) in `80_20` mode
- MobileNetV2 ImageNet pretrained, 100% frozen
- 4-block Transformer encoder (4 heads, key_dim=64, embed_dim=256, FFN=512, dropout=0.2)
- Stage 2 disabled (single `model.fit()`)
- **GlobalAveragePooling1D (fixed uniform weighting)**
- **Baseline Validation Accuracy: 42.42%**

**Single Controlled Change:**
- Replace `keras.layers.GlobalAveragePooling1D` with a **Learnable Temporal Attention Pooling mechanism**:
  $$e_t = \\mathbf{w}^\\top \\mathbf{h}_t + b, \\quad \\alpha_t = \\frac{\\exp(e_t)}{\\sum_{\\tau=1}^{T} \\exp(e_\\tau)}, \\quad \\mathbf{c} = \\sum_{t=1}^{T} \\alpha_t \\mathbf{h}_t$$
- Everything else (data, split, 100% frozen MobileNetV2, Transformer depth/heads/dims, Stage 1 hyperparams, single fit) remains **strictly identical**.
"""
nb["cells"][0]["source"] = [line + "\n" for line in cell_0_text.strip().split("\n")]

# 2. Update Cell 5: Configuration comment / printouts if needed
src_5 = "".join(nb["cells"][5]["source"])
src_5 = src_5.replace("EXPERIMENT CONFIGURATION & HYPERPARAMETERS [Stage 2 Removed]",
                      "EXPERIMENT CONFIGURATION & HYPERPARAMETERS [Temporal Attention Pooling]")
nb["cells"][5]["source"] = [line + "\n" for line in src_5.strip().split("\n")]

# 3. Update Cell 6: Markdown for Section 4
cell_6_text = """## 4. Dataset Paths
Configuring paths for the active Method D processed dataset (`data/processed`) and creating a dedicated output directory for this temporal attention pooling experiment (`results/mobilenetv2_transformer_4block_temporal_pooling_96x96_{DATASET_MODE}`)."""
nb["cells"][6]["source"] = [line + "\n" for line in cell_6_text.strip().split("\n")]

# 4. Update Cell 7: Dedicated Results Directory
src_7 = "".join(nb["cells"][7]["source"])
src_7 = src_7.replace("mobilenetv2_transformer_4block_stage2_removed_96x96_{DATASET_MODE}",
                      "mobilenetv2_transformer_4block_temporal_pooling_96x96_{DATASET_MODE}")
src_7 = src_7.replace("# Dedicated Results Directory per Dataset Mode (Stage 2 Removed)",
                      "# Dedicated Results Directory per Dataset Mode (Temporal Attention Pooling)")
nb["cells"][7]["source"] = [line + "\n" for line in src_7.strip().split("\n")]

# 5. Update Cell 9: split_metadata experiment tag
src_9 = "".join(nb["cells"][9]["source"])
src_9 = src_9.replace('"experiment": "mobilenetv2_transformer_4block_stage2_removed"',
                      '"experiment": "mobilenetv2_transformer_4block_temporal_pooling"')
nb["cells"][9]["source"] = [line + "\n" for line in src_9.strip().split("\n")]

# 6. Update Cell 20: Markdown for Model Architecture
cell_20_text = """## 11. MobileNetV2 + 4-Block Transformer Model with Learnable Temporal Attention Pooling
Building the end-to-end architecture:
```
Input: (30, 96, 96, 3)
   ↓
TimeDistributed MobileNetV2 (preprocess_input + ImageNet backbone)
   ↓
1280-dimensional feature vector per frame
   ↓
Dense(256)
   ↓
LayerNormalization
   ↓
Sinusoidal Positional Encoding
   ↓
4 × Transformer Encoder Blocks (Pre-LN, 4 heads, key_dim=64, FFN=512)
   ↓
Learnable Temporal Attention Pooling:
   • Dense(1, name='temporal_attn_score') -> (batch, 30, 1)
   • Softmax(axis=1, name='temporal_attn_weights') -> (batch, 30, 1)
   • Sum(weights * frames, axis=1, name='temporal_attention_pooling') -> (batch, 256)
   ↓
Dropout(0.4)
   ↓
Dense(11, activation="softmax")
```"""
nb["cells"][20]["source"] = [line + "\n" for line in cell_20_text.strip().split("\n")]

# 7. Update Cell 21: Model Assembly with Learnable Temporal Attention Pooling
src_21 = "".join(nb["cells"][21]["source"])
old_pool = """    # 6. Temporal Sequence Pooling
    x = keras.layers.GlobalAveragePooling1D(name="gap1d")(x)  # (batch, 256)"""

new_pool = """    # 6. Learnable Temporal Attention Pooling
    attn_scores = keras.layers.Dense(1, use_bias=True, name="temporal_attn_score")(x)  # (batch, 30, 1)
    attn_weights = keras.layers.Softmax(axis=1, name="temporal_attn_weights")(attn_scores)  # (batch, 30, 1)
    x = keras.layers.Lambda(
        lambda tensors: tf.reduce_sum(tensors[0] * tensors[1], axis=1),
        name="temporal_attention_pooling"
    )([x, attn_weights])  # (batch, 256)"""

assert old_pool in src_21, "old_pool string not found in Cell 21!"
src_21 = src_21.replace(old_pool, new_pool)
src_21 = src_21.replace('name="MobileNetV2_Transformer_4Block"',
                        'name="MobileNetV2_Transformer_4Block_TemporalPooling"')
nb["cells"][21]["source"] = [line + "\n" for line in src_21.strip().split("\n")]

# 8. Update Cell 23: Model Summary printout title
src_23 = "".join(nb["cells"][23]["source"])
src_23 = src_23.replace("             MOBILENETV2 + 4-BLOCK TRANSFORMER ARCHITECTURE\n                     [STAGE 2 FINE-TUNING REMOVED]",
                        "  MOBILENETV2 + 4-BLOCK TRANSFORMER + LEARNABLE TEMPORAL ATTENTION POOLING\n                     [STAGE 2 FINE-TUNING REMOVED]")
nb["cells"][23]["source"] = [line + "\n" for line in src_23.strip().split("\n")]

# 9. Update Cell 36: Markdown Save Results
cell_36_text = """## 19. Save Results
Exporting test predictions CSV and the consolidated experiment metrics JSON file to `results/mobilenetv2_transformer_4block_temporal_pooling_96x96_{DATASET_MODE}/`."""
nb["cells"][36]["source"] = [line + "\n" for line in cell_36_text.strip().split("\n")]

# 10. Update Cell 37: Final metrics JSON metadata
src_37 = "".join(nb["cells"][37]["source"])
src_37 = src_37.replace('"model_name": "MobileNetV2_Transformer_4Block"',
                        '"model_name": "MobileNetV2_Transformer_4Block_TemporalPooling"')
src_37 = src_37.replace('"temporal_head": "Transformer_4x_4heads_256dim"',
                        '"temporal_head": "Transformer_4x_4heads_256dim_temporal_attention_pooling"')
nb["cells"][37]["source"] = [line + "\n" for line in src_37.strip().split("\n")]

# 11. Update Cell 38: Final Experiment Summary Table
cell_38_text = """## 20. Final Experiment Summary
Architectural comparison table across the core temporal modeling experiments:

| Architectural Component | MobileNetV2 + 4-Block Transformer (GAP Baseline) | MobileNetV2 + 4-Block Transformer (Learnable Temporal Pooling - This Experiment) |
| :--- | :--- | :--- |
| **Visual Frontend** | MobileNetV2 (ImageNet weights, 100% frozen) | MobileNetV2 (ImageNet weights, 100% frozen) |
| **Resolution** | 96 × 96 RGB (30 frames) | 96 × 96 RGB (30 frames) |
| **Temporal Head** | 4× Transformer Encoder Blocks | 4× Transformer Encoder Blocks |
| **Attention Mechanism** | Multi-Head Self-Attention (4 heads) | Multi-Head Self-Attention (4 heads) |
| **Attention Key Dim** | 64 | 64 |
| **FFN Dimension** | 512 (GELU) | 512 (GELU) |
| **Sequence Pooling** | **GlobalAveragePooling1D (fixed uniform)** | **Learnable Temporal Attention Pooling (Dense + Softmax + Weighted Sum)** |
| **Stage 1 Protocol** | Frozen CNN, Adam (lr=1e-4), EarlyStopping | Frozen CNN, Adam (lr=1e-4), EarlyStopping |
| **Stage 2 Protocol** | **DISABLED** | **DISABLED** |
| **Loss Function** | Label Smoothing ($0.1$) | Label Smoothing ($0.1$) |
| **Validation Accuracy** | **42.42% (Baseline)** | *(Evaluated upon run)* |"""
nb["cells"][38]["source"] = [line + "\n" for line in cell_38_text.strip().split("\n")]

# 12. Update Cell 39: Printout banner
src_39 = "".join(nb["cells"][39]["source"])
src_39 = src_39.replace("FINAL EVALUATION SUMMARY: MobileNetV2 + 4-Block Transformer",
                        "FINAL EVALUATION SUMMARY: MobileNetV2 + 4-Block Transformer [Temporal Attention Pooling]")
nb["cells"][39]["source"] = [line + "\n" for line in src_39.strip().split("\n")]

# Write new notebook
with open(dest_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully wrote new notebook: {dest_path}")
