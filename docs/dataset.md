# Tai Phake VSR - Dataset Documentation

## Overview
This document outlines the visual speech dataset collected for recognizing spoken Tai Phake digits.

- **Language**: Tai Phake (endangered Tai-Kadai language spoken in Assam and Arunachal Pradesh, Northeast India)
- **Vocabulary**: 11 digits (`d0` to `d10`):
  - `d0`: **Sun** (Zero)
  - `d1`: **Nung** (One)
  - `d2`: **Song** (Two)
  - `d3`: **Sam** (Three)
  - `d4`: **Si** (Four)
  - `d5`: **Ha** (Five)
  - `d6`: **Hok** (Six)
  - `d7`: **Jet** (Seven)
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

### Tier 2: Trimmed Digit Videos (Local Working Copy in `data/trimmed/`)
- Approximately 300 trimmed/segmented MP4 video files.
- Each video contains one speaker uttering one digit.
- Organized approximately as:
  ```
  data/trimmed/
  ├── s1/
  │   ├── d0/d0.mp4
  │   ├── d1/d1.mp4
  │   └── ...
  ├── s2/
  └── ...
  └── s30/
  ```
- Downloaded/copied locally for Person 1's validation and frame extraction work.

---

## Data Management & Git Governance

1. **Local Working Directories (Ignored by Git)**:
   - `data/trimmed/`: Working copy of ~300 trimmed videos.
   - `data/frames/`: Extracted visual frames.
   - `data/processed/`: Cropped mouth sequences and model tensors.
2. **Tracked Files (Committed to Git)**:
   - `data/metadata.csv`: Master metadata table capturing video IDs, speakers, digits, duration, frame counts, and validation status.
   - `data/README.md`: Directory rules and schema.
3. **Data Immutability Guarantee**:
   - Trimmed videos in `data/trimmed/` are treated as read-only.
   - Pipeline scripts never modify, rename, compress, or overwrite video files.
