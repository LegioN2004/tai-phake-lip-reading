# Dlib Facial Landmark & Mouth Extraction (Person 2)

## Responsibility & Goals
Person 2 is responsible for evaluating the classical **Dlib 68-point facial landmark detector** on the Tai Phake visual speech dataset.

---

## Landmark Model Details
- **Pre-trained Model**: `shape_predictor_68_face_landmarks.dat`
- **Mouth Region Landmarks**:
  - Outer Lip: Landmarks 48 through 59
  - Inner Lip: Landmarks 60 through 67
- **Evaluation Criteria**:
  - Face detection success rate across 30 speakers
  - Sensitivity to head pose, tilt, and rapid articulatory motion
  - Bounding box stability across video duration
  - Inference latency per frame (CPU vs GPU)
  - Failure cases and visual documentation in `results/dlib/`
