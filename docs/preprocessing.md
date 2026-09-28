# Preprocessing Pipeline Documentation

## Pipeline Overview

The preprocessing pipeline prepares raw visual speech frames for spatial-temporal deep learning models (CNN + BiLSTM).

```
Raw Frames (from data/frames/)
       │
       ▼
Face Landmark Detection (Dlib / MediaPipe)
       │
       ▼
Mouth / Lip Boundary Calculation
       │
       ▼
ROI Cropping & Resizing (96x96)
       │
       ▼
Normalization & Grayscale Conversion
       │
       ▼
Temporal Sequence Padding / Truncation (T=30)
       │
       ▼
Speaker-Independent Tensor Generation (data/processed/)
```

---

## Technical Specifications
- **Mouth ROI Size**: $96 \times 96$ pixels
- **Padding Factor**: 0.2 proportional margin around lip contours
- **Color Format**: Grayscale ($1$ channel) or RGB ($3$ channels)
- **Sequence Length**: Fixed $T = 30$ frames (with edge padding for shorter utterances)
- **Splits**: 20 speakers train (approx. 200 videos), 5 speakers validation (approx. 50 videos), 5 speakers test (approx. 50 videos).
