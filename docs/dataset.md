# Tai Phake VSR - Dataset Documentation

## Overview
This document outlines the visual speech dataset collected for recognizing spoken Tai Phake digits.

- **Language**: Tai Phake (endangered Tai-Phake language spoken in Assam and Arunachal Pradesh, Northeast India)
- **Vocabulary**: 11 digits (`d0` to `d10`):
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
- **Speakers**: 30 distinct native/heritage speakers (`s1` to `s30`)
- **Total Utterances**: ~330 videos (30 speakers × 11 digits)
- **Modality**: Video + Audio (primary recognition relies on visual lip movements)

---

## Two-Tier Dataset Architecture

### Tier 1: Original Raw Recordings (Cloud Archive)
- The true original uncut recordings are archived safely in **Google Drive / Institutional Storage**.
- To prevent laptop storage exhaustion, these original raw recordings are **NOT downloaded or stored locally**.
- They remain permanently untouched.

### Tier 2: Trimmed Digit Videos & In-Situ Frames (Local in `data/digit/`)
- Approximately 330 trimmed/segmented MP4 video files across 30 speakers × 11 digits (`d0`–`d10`).
- Each video contains one speaker uttering one digit.
- Extracted frames are stored directly inside each respective digit subfolder (`data/digit/<speaker>/<digit>/frame_XXXX.png`).
- Organized as:
  ```
  data/digit/
  ├── s1/
  │   ├── d0/
  │   │   ├── d0.mp4
  │   │   ├── frame_0001.png
  │   │   └── ...
  │   ├── d1/
  │   └── ...d10/
  ├── s2/
  └── ...
  └── s30/
  ```

---

## Data Management & Git Governance

1. **Local Working Directories (Ignored by Git)**:
   - `data/digit/`: Working copy of ~330 trimmed videos and in-situ extracted frames.
   - `data/processed/`: Cropped mouth sequences and model tensors.
2. **Tracked Files (Committed to Git)**:
   - `data/metadata.csv`: Master metadata table capturing video IDs, speakers, digits, duration, frame counts, and validation status.
   - `data/README.md`: Directory rules and schema.
3. **Data Immutability Guarantee**:
   - Trimmed videos in `data/trimmed/` are treated as read-only.
   - Pipeline scripts never modify, rename, compress, or overwrite video files.
