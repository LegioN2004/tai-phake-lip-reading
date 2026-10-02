# Tai Phake Lip-Reading Dataset Sanity Check Report

> **Final Verdict**: **`PASS WITH WARNINGS`**
>
> - 91 sample(s) flagged with non-critical warnings (e.g. resolution variations or frame count discrepancies)
> - 88 sample(s) have frame resolutions other than standard 720x1280

---

## 1. Dataset Overview
- **Target Directory**: `/Users/milinda/ghq/github.com/legion2004/tai-phake-lip-reading/data/digit`
- **Expected Structure**: 30 speakers (`s1`..`s30`) × 11 digits (`d0`..`d10`) = **330 samples**
- **Total Samples Inspected**: 330
- **Valid Samples with Extracted Frames**: 330
- **Total Extracted Frames**: 13,610
- **Corrupted / Invalid Files**: 0

## 2. Frame Count Statistics
| Metric | Frames per Utterance |
| :--- | :--- |
| Minimum | 15.0 frames |
| Maximum | 92.0 frames |
| Mean | 41.24 frames |
| Median | 40.0 frames |
| Std Dev | 11.97 frames |

## 3. Frame Resolutions Across Dataset
| Resolution (Width × Height) | Aspect Ratio | Samples | Percentage |
| :--- | :--- | :--- | :--- |
| `720x1280` | Portrait (Standard) | 242 | 73.3% |
| `1280x720` | Landscape | 76 | 23.0% |
| `720x1080` | Portrait (Alternative) | 10 | 3.0% |
| `1080x1920` | Portrait (Alternative) | 1 | 0.3% |
| `960x1280` | Portrait (Alternative) | 1 | 0.3% |

## 4. Dataset Balance Matrix
All 30 speakers have complete sequences for all 11 digit classes (`d0` to `d10`):

| Speaker | Valid Classes (of 11) | Min Frames | Max Frames | Mean Frames |
| :--- | :--- | :--- | :--- | :--- |
| `s1` | 11 / 11 | 30 | 55 | 40.2 |
| `s2` | 11 / 11 | 23 | 46 | 35.4 |
| `s3` | 11 / 11 | 22 | 66 | 34.3 |
| `s4` | 11 / 11 | 31 | 56 | 39.7 |
| `s5` | 11 / 11 | 22 | 47 | 33.3 |
| `s6` | 11 / 11 | 37 | 47 | 41.2 |
| `s7` | 11 / 11 | 26 | 52 | 35.8 |
| `s8` | 11 / 11 | 28 | 56 | 37.7 |
| `s9` | 11 / 11 | 44 | 66 | 54.3 |
| `s10` | 11 / 11 | 41 | 68 | 50.4 |
| `s11` | 11 / 11 | 34 | 77 | 47.0 |
| `s12` | 11 / 11 | 28 | 55 | 39.4 |
| `s13` | 11 / 11 | 33 | 60 | 47.7 |
| `s14` | 11 / 11 | 30 | 68 | 44.6 |
| `s15` | 11 / 11 | 38 | 92 | 58.4 |
| `s16` | 11 / 11 | 40 | 59 | 50.2 |
| `s17` | 11 / 11 | 25 | 53 | 36.6 |
| `s18` | 11 / 11 | 29 | 49 | 41.6 |
| `s19` | 11 / 11 | 40 | 64 | 46.9 |
| `s20` | 11 / 11 | 35 | 60 | 43.5 |
| `s21` | 11 / 11 | 49 | 90 | 70.3 |
| `s22` | 11 / 11 | 25 | 55 | 37.2 |
| `s23` | 11 / 11 | 30 | 48 | 38.9 |
| `s24` | 11 / 11 | 28 | 39 | 32.3 |
| `s25` | 11 / 11 | 33 | 44 | 39.6 |
| `s26` | 11 / 11 | 15 | 43 | 28.7 |
| `s27` | 11 / 11 | 25 | 45 | 33.0 |
| `s28` | 11 / 11 | 21 | 55 | 34.7 |
| `s29` | 11 / 11 | 27 | 37 | 33.1 |
| `s30` | 11 / 11 | 18 | 54 | 31.3 |

## 5. Discrepancies & Warnings
Total samples with notices/warnings: 91

### Notable Sample Observations:
| Sample | Observation / Warning |
| :--- | :--- |
| `s3_d0` | Video resolution (720x1080) differs from expected (720x1280) |
| `s3_d1` | Video resolution (720x1080) differs from expected (720x1280) |
| `s3_d2` | Video resolution (720x1080) differs from expected (720x1280) |
| `s3_d3` | Video resolution (720x1080) differs from expected (720x1280) |
| `s3_d4` | Video resolution (720x1080) differs from expected (720x1280) |
| `s3_d5` | Video resolution (720x1080) differs from expected (720x1280) |
| `s3_d7` | Video resolution (720x1080) differs from expected (720x1280) |
| `s3_d8` | Video resolution (720x1080) differs from expected (720x1280) |
| `s3_d9` | Video resolution (720x1080) differs from expected (720x1280) |
| `s3_d10` | Video resolution (720x1080) differs from expected (720x1280) |
| `s4_d9` | Decoded frame count (54) differs from container header (56) |
| `s6_d5` | Video resolution (1280x720) differs from expected (720x1280) |
| `s8_d0` | Video resolution (1280x720) differs from expected (720x1280) |
| `s8_d1` | Video resolution (1280x720) differs from expected (720x1280) |
| `s8_d2` | Video resolution (1280x720) differs from expected (720x1280) |
| `s8_d3` | Video resolution (1280x720) differs from expected (720x1280) |
| `s8_d4` | Video resolution (1280x720) differs from expected (720x1280) |
| `s8_d5` | Video resolution (1280x720) differs from expected (720x1280) |
| `s8_d6` | Video resolution (1280x720) differs from expected (720x1280) |
| `s8_d7` | Video resolution (1280x720) differs from expected (720x1280) |
| `s8_d8` | Video resolution (1280x720) differs from expected (720x1280) |
| `s8_d9` | Decoded frame count (28) differs from container header (35) |
| `s8_d9` | Video resolution (1280x720) differs from expected (720x1280) |
| `s8_d10` | Video resolution (1280x720) differs from expected (720x1280) |
| `s9_d0` | Video resolution (1280x720) differs from expected (720x1280) |
| `s9_d1` | Video resolution (1280x720) differs from expected (720x1280) |
| `s9_d2` | Video resolution (1280x720) differs from expected (720x1280) |
| `s9_d3` | Video resolution (1280x720) differs from expected (720x1280) |
| `s9_d4` | Video resolution (1280x720) differs from expected (720x1280) |
| `s9_d5` | Video resolution (1280x720) differs from expected (720x1280) |
| ... | *and 64 more entries (see sample_inventory.csv)* |

## 6. Duplicate Detection & Data Integrity
- **Duplicate MP4 Hash Groups**: 0
- **Unreadable / Corrupt Image Files**: 0
- **Frame Sequence Gaps**: None (all 330 samples have consecutive 1-indexed frames: `frame_0001.jpg` onward)

## 7. Generated Reports
- Detailed sample manifest: [`sample_inventory.csv`](sample_inventory.csv)
- Structural issue log: [`structure_errors.csv`](structure_errors.csv)
- Invalid files log: [`invalid_files.csv`](invalid_files.csv)
- Complete JSON dump: [`dataset_summary.json`](dataset_summary.json)
