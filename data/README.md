# Dataset Directory Guide & Storage Rules

> **IMPORTANT**:
> - The **original full raw recordings** remain safely stored **ONLY in Google Drive / institutional cloud storage** to preserve laptop disk space. They are NOT downloaded locally.
> - The **~330 trimmed digit videos** stored in `data/digit/` are our local working copy.
> - **Extracted frames are generated directly inside each digit folder** (e.g. `data/digit/s1/d0/frame_0001.png`), eliminating the need for a separate frames directory.
> - **NEVER commit video files or frame images to GitHub.**
> - The directories `data/digit/`, `data/trimmed/`, and `data/processed/` are local-only and strictly ignored by `.gitignore`.

---

## 1. Two-Tier Dataset Architecture

### Tier 1: Original Raw Recordings (Cloud Only)
- Uncut master recordings of speakers.
- Preserved in **Google Drive / Institutional Storage** only.
- Read-only, master archive. **No local copy is created or required**.

### Tier 2: Trimmed Digit Videos & In-Situ Frames (Local in `data/digit/`)
- Approximately 330 segmented/trimmed MP4 clips (30 speakers × 11 digits `d0`–`d10`).
- Each video contains **one speaker uttering one Tai Phake digit**.
- Extracted directly from the Google Drive zip into `data/digit/`.
- Frames are extracted directly into each respective `d` subfolder alongside the video.

---

## 2. Directory Structure

```
data/
├── digit/               # [Local working copy] 30 speakers (s1..s30) × 11 digits (d0..d10)
│   ├── s1/
│   │   ├── d0/
│   │   │   ├── d0.mp4           # Trimmed video clip
│   │   │   ├── frame_0001.png   # In-situ extracted frames
│   │   │   ├── frame_0002.png
│   │   │   └── ...
│   │   ├── d1/
│   │   │   ├── d1.mp4
│   │   │   └── ...
│   │   └── ...d10/
│   ├── s2/
│   └── ...
│   └── s30/
├── processed/           # Preprocessed lip crops, normalized arrays & tensors (local only)
├── metadata.csv         # Small dataset metadata table (Tracked in Git)
└── README.md            # Dataset documentation (Tracked in Git)
```

---

## 3. Storage Protocol & Data Integrity Rules

1. **Trimmed Video Immutability**:
   - The trimmed videos in `data/digit/` must **never be modified, renamed, moved, compressed, or overwritten**.
   - All validation and frame extraction scripts access these videos in **strictly read-only mode**.

2. **Git Version Control**:
   - Only `metadata.csv` and `README.md` are tracked by Git.
   - `data/digit/*`, `data/trimmed/*`, and `data/processed/*` are ignored by `.gitignore`.

3. **`metadata.csv` Schema**:
   - `video_id`: Unique utterance identifier (e.g., `s01_d0`)
   - `speaker_id`: Speaker identifier (`s1` / `s01` to `s30`)
   - `digit`: Digit code (`d0` to `d10`)
   - `digit_label`: Integer index (0 to 10)
   - `tai_phake_word`: Spoken phonetic word (e.g. `Sun`, `Nung`, `Song`, ..., `Sip`)
   - `file_name`: Original video filename
   - `relative_path`: Path relative to `data/digit/`
   - `duration_sec`: Video duration in seconds
   - `fps`: Video frame rate
   - `frame_count`: Number of visual frames
   - `width`, `height`: Video resolution
   - `is_valid`: Boolean integrity status
   - `notes`: Validation flags or anomalies
