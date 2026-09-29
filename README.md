# Tai Phake Visual Speech Recognition (VSR)

**Project Name**: `tai-phake-visual-speech`

An AI research system designed to recognize spoken digits in the **Tai Phake** language from visual mouth movements (silent lip reading) in video.

---

## 1. Dataset Architecture & Storage Rules

The dataset is managed in a **two-tier architecture**:

### 1. Original Recordings (Google Drive Master Archive)
- True raw, uncut recordings of speakers.
- Preserved permanently in **Google Drive / institutional storage**.
- **Not downloaded locally** to preserve laptop disk storage.
- Must remain untouched.

### 2. Trimmed Digit Videos (Local Working Dataset in `data/digit/`)
- Approximately 330 segmented/trimmed MP4 video files (30 speakers × 11 digits).
- Each video contains **one speaker saying one Tai Phake digit** (`d0` to `d10`).
- Downloaded/copied locally into `data/digit/` for dataset validation and frame extraction.
- **Never committed to GitHub** (strictly excluded by `.gitignore`).

### Tai Phake Vocabulary (11 Digits)
- `d0`: **Pau** (Zero)
- `d1`: **Nung** (One)
- `d2`: **Saung** (Two)
- `d3`: **Sam** (Three)
- `d4`: **Si** (Four)
- `d5`: **Ha** (Five)
- `d6`: **Hok** (Six)
- `d7`: **Chit** (Seven)
- `d8`: **Pet** (Eight)
- `d9`: **Kao** (Nine)
- `d10`: **Sip** (Ten)

---

## 2. Strict Git Governance Rules

> **CRITICAL GITHUB RULES**:
> - **The original and trimmed MP4 video files must NEVER be placed in the GitHub repository** (strictly ignored by `.gitignore`).
> - **Extracted sequential image frames (`*.jpg`) in `data/digit/` ARE tracked and version-controlled by Git.**
> - The directory `data/processed/` (preprocessed tensors) is local-only and ignored by `.gitignore`.
> - Model checkpoints and weights (`*.h5`, `*.keras`, `*.pth`, `*.pt`, `*.ckpt`) are excluded from Git.
> - GitHub tracks source code, notebooks, configurations, documentation, metadata files (`data/metadata.csv`), and extracted frame images.

---

## 3. Directory Structure

```
tai-phake-visual-speech/
├── README.md
├── data/
│   ├── digit/                         # 30 speakers (s1..s30) × 11 digits (d0..d10) [Frames tracked in Git; MP4 videos local-only]
│   │   ├── s1/
│   │   │   ├── d0/                    # Contains d0.mp4 (local) and in-situ extracted frames (tracked)
│   │   │   └── ...
│   │   └── ...
│   ├── processed/                     # [Local only] Preprocessed mouth ROI tensors
│   ├── metadata.csv                   # Master dataset metadata (tracked in Git)
│   └── README.md                      # Data guidelines and schema description
├── notebooks/
│   ├── 01_dataset_analysis.ipynb      # Person 1: Dataset exploration & validation
│   ├── 02_dlib_experiment.ipynb       # Person 2: Dlib 68-landmark experiments
│   ├── 03_mediapipe_experiment.ipynb  # Person 3: MediaPipe Face Mesh experiments
│   └── 04_preprocessing_pipeline.ipynb# Person 4: Integration, cropping & normalization
├── extract_frames.py                  # Person 1: In-situ sequential frame extraction
├── validate_videos.py                 # Person 1: Video decodability & metadata generator
├── configs/
│   └── preprocessing.yaml             # Central pipeline and model configuration
├── results/
│   ├── dataset/                       # Dataset distribution plots & stats
│   ├── dlib/                          # Dlib landmark evaluation logs & images
│   ├── mediapipe/                     # MediaPipe landmark evaluation logs & images
│   └── comparison/                    # Comparative benchmark analysis
├── docs/
│   ├── dataset.md                     # Dataset documentation & phonetics
│   ├── preprocessing.md               # Preprocessing specifications & tensor formats
│   ├── dlib.md                        # Person 2 Dlib technical guide
│   └── mediapipe.md                   # Person 3 MediaPipe technical guide
├── models/                            # [Local only] Model weights & checkpoints
├── requirements.txt                   # Project dependencies
├── .python-version                    # Python 3.12 (managed via uv)
├── .gitignore                         # Strict exclusion rules for media and models
└── LICENSE                            # MIT License
```

---

## 4. Python Environment Setup with UV (Python 3.12)

This project uses [uv](https://github.com/astral-sh/uv) to manage Python and virtual environments for speed and reliability. Python **3.12** is used to ensure maximum library compatibility.

### 1. Create Virtual Environment with Python 3.12
```bash
# uv will automatically use Python 3.12 from .python-version
uv venv .venv --python 3.12

# Activate environment:
# Windows (PowerShell):
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install Project Dependencies
```bash
uv pip install -r requirements.txt
```

### 3. Video Validation & Metadata Generation
Validates integrity and decodability of all videos and updates `data/metadata.csv`:
```bash
uv run python validate_videos.py
```

### 4. Frame Extraction (Person 1 Pipeline)
Extracted frames are saved in-situ alongside the source video (`data/digit/<speaker>/<digit>/frame_XXXX.jpg`).

```bash
# 1. Dry run (verifies decodability without writing files to disk)
uv run python extract_frames.py --dry-run

# 2. Extract single speaker/digit for testing
uv run python extract_frames.py --speaker s1 --digit d0

# 3. Full extraction across all 30 speakers and 11 digits
uv run python extract_frames.py

# 4. Configurable extraction: extract every 2nd frame (reduce dataset size)
uv run python extract_frames.py --step 2

# 5. Configurable extraction: sample at fixed target FPS (e.g. 15 FPS)
uv run python extract_frames.py --target-fps 15.0

# 6. Re-extract and overwrite existing frames
uv run python extract_frames.py --overwrite
```
