# MediaPipe Face Mesh & Mouth Extraction (Person 3)

## Responsibility & Goals
Person 3 is responsible for evaluating **Google MediaPipe Face Mesh** for lip landmark tracking on the Tai Phake visual speech dataset.

---

## Face Mesh Details
- **Architecture**: Lightweight attention-based neural network providing 468/478 3D facial landmarks in real time.
- **Lip Contour Indices**:
  - Upper lip: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
  - Lower lip: 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
- **Evaluation Criteria**:
  - Landmark tracking robustness under variable lighting and motion blur
  - Jitter / temporal stability of mouth bounding box
  - Failure cases under occlusion or fast mouth openings
  - Quantitative and qualitative comparison with Dlib (logged in `results/mediapipe/`)
