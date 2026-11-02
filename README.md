# Data Heist: CTF + ML Challenge 🥈

**2nd place** in *Data Heist: Crack. Discover. Predict.*, a 4-hour team competition that combined **Capture the Flag (CTF)** security challenges with a **Kaggle regression task**.

The task: crack five CTF challenges to unlock five datasets, work out which data is real signal and which is a deliberate trap, then predict Formula-1-style **lap times** (`lap_time_sec`, scored by RMSE) for 9,544 laps from 8 races that never appear in training.

| | |
|---|---|
| **Final standing** | 2nd place |
| **Best public LB RMSE (this repo)** | 0.907, down from a 1.144 first baseline |
| **Stack** | Python, pandas, LightGBM, PyTorch (GPU), Optuna, scikit-learn |

---

## Phase 1: Cracking the CTF

Each challenge unlocked one dataset ("channel"):

| # | Challenge | Technique | Dataset unlocked |
|---|---|---|---|
| 1 | *False Format* | A "CSV" that was really **Base64 → gzip**; decoded to recover the payload | ch1: car setup |
| 2 | *Cookie Crumbs* | Web dashboard; **cookie tampering / privilege escalation** to reveal hidden content | ch2: fuel & stint |
| 3 | *Undertow* | Repaired a **corrupted WAV header** and read a message hidden in its **spectrogram** | ch3: car telemetry |
| 4 | *Network Ghost* | Reassembled **hex fragments from a packet capture** (pcapng) | ch4: weather |
| 5 | *Lost in the Binary* | Reverse-engineered a stripped x86-64 ELF; recovered an **XOR-encrypted** access string | ch5: race context |

Challenge files belong to the organisers and are not redistributed here.

---

## Phase 2: Investigating the data

The brief warned that *"some datasets may contain misleading information… or deliberately poor-quality data."* Most of the work, and most of the score, came from finding those traps.

### 1. The test races are unseen, so validation must hold out whole races
Train covers 40 races (season × circuit); test is 8 *different* races. Random K-fold reported ~0.85 RMSE. **Race-grouped CV** gave ~1.5 on the same model, which matched the leaderboard. From then on, every decision was made with race-grouped CV.

### 2. Effects are multiplicative, so predict a ratio
Safety car ≈ ×1.32 lap time, VSC ≈ ×1.23, engine modes and rain scale lap time too. Predicting `log(lap_time / circuit_baseline)` instead of raw seconds **halved the error** (CV 1.48 → 0.72).

### 3. Planted leaks, and how they were exposed
Several columns tracked lap time closely in **train** but were pure noise in **test**. Leaky columns score well in CV, so CV alone can't catch them. Three detection methods:

- **Train-vs-test relationship check.** Train a model *without* feature X, then compare how X relates to that prediction in train vs test. A real feature behaves the same in both; a leak's relationship collapses in test.
- **Feature-correlation shift.** e.g. `ride_height ~ brake_bias`: r = −0.57 in train, +0.03 in test.
- **Cross-channel agreement.** Several channels contain the same column. In test they always agree; in train the faked copies disagree.

| Column | Train | Test | Verdict |
|---|---|---|---|
| `ride_height_mm`, `brake_bias_pct` (ch1) | r ≈ ±0.6 with expected lap time | r ≈ 0 | ❌ leak |
| `engine_mode` (ch1) | agrees with ch3's copy only 54% of the time; modes 4–5 sit on "fast" laps | identical to ch3; no context link | ❌ leak (hidden behind mode 1 = safety car) |
| `engine_mode` (ch3) | same per-mode pattern as test | same | ✅ honest and predictive |
| `fuel_used_kg` (ch3) | matches ch2's copy only 9% of the time | 100% match | ❌ leak; used ch2's copy |
| `tyre_temp_c` (ch3) | r = 0.57 with context | r = 0.28 | ❌ excluded |
| ch4 weather | test laps only, no labels | | ❌ decoy |

Removing `ride_height` / `brake_bias` moved the public LB from **1.144 to 0.978**. Removing ch1's `engine_mode` moved it from **0.952 to 0.907**.

### 4. Weather needs its own model
Rainy and wet-tyre laps had 3–4 s errors with a systematic bias, and test contained tyre/rain combinations never seen in training. Each tyre has a "sweet spot" (e.g. intermediates are only ~0.5% slow in light rain while slicks are 4–5% slow). The fix was a **two-stage model**:

1. **Dry-pace model**, trained on dry laps only.
2. **Weather factor**: a small, heavily regularised GBM on `log(actual / dry prediction)` using compound, current and lagged rain, laps since rain stopped, track status and tyre age.

Wet-lap RMSE dropped from 2.61 to 1.83, and the systematic bias went to ~0.

### 5. Ideas tested and rejected (by CV, not by leaderboard)
Lagged-rain features in a single GBM, log-linear models (1.06 vs 0.63), CatBoost, XGBoost, linear-leaf GBMs for temperature extrapolation (temperature turned out to have ~0 effect), deeper hyperparameter tuning (it fitted the CV splits rather than generalising), and reconstructing relative pace from `gap_ahead_sec` (pure noise).

---

## Phase 3: Final model

```
                    ┌─ LightGBM (default params, 3 seeds) ─┐
dry laps ──────────►├─ LightGBM (Optuna-tuned, 3 seeds)    ├─ equal-weight mean ─┐
                    └─ MLP w/ categorical embeddings (GPU) ┘                     │
                                                                                 ├─► log-ratio ─► × circuit baseline ─► lap time
wet / rainy laps ──► weather-factor GBM (5 seeds) ───────────────────────────────┘
```

- **Target:** `log(lap_time) − log(circuit_baseline)`.
- **Ensemble weights:** equal, never fitted to the leaderboard. Optimised weights gained only 0.002 in CV, not worth the overfitting risk.
- **Robustness check:** on a fresh fold split that played no part in any design choice, the ensemble beat the single model in 33/40 held-out races, and won in 96% of 5,000 bootstrapped 8-race test sets.

### Leaderboard progression (public LB, ~29% of test)

| Version | Change | Public RMSE |
|---|---|---|
| v1 | LightGBM, log-ratio target | 1.144 |
| v2 | + rain features, tuning (still leaky) | 1.197 |
| v3b | **removed ride_height / brake_bias leak** | 0.978 |
| v4 | two-stage dry + weather model | 0.968 |
| v5 | dropped weak setup columns | 0.964 |
| v6 | 3-model ensemble (LGBM ×2 + NN) | 0.952 |
| v7 | **removed ch1 engine_mode leak** | **0.907** |
| v8 | + 8.4k extra laps from ch2/ch3, honest engine mode | 0.931 |

The final selection was v8 plus a teammate's model, which together placed **2nd** on the private leaderboard.

---

## Repository layout

```
src/
├── prep.py, prep2.py, prep3.py  # loading, 4-channel merge, imputation of race context
├── fe.py, rainfe.py             # feature engineering (tyre/rain interactions, rain lags)
├── frame.py, cv.py, drycv.py    # race-grouped CV harnesses
├── twostage.py                  # dry-pace + weather-factor model
├── nn.py                        # PyTorch MLP with categorical embeddings
├── final.py, final2.py          # final ensembles (v6/v7, v8)
├── submit.py                    # strict submission writer (schema / row-count checks)
├── sub_*.py                     # scripts that produced each submission
├── leak*.py, corrshift.py, emcheck.py, adv.py, overlap.py   # leak detection
├── robust*.py, disagree.py, tempx.py                         # overfitting / extrapolation checks
└── eda*.py, resid*.py, wet.py, rain.py, exp*.py, ...         # exploration & experiments
```

## Reproducing

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate on Linux/macOS
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU build
```

Place the competition data (not included) at:

```
data/B_merged_ch5_ch1/{train,test}.csv
train_test_splits/{ch1,ch2,ch3,ch5}/{train,test}.csv
```

Then run the scripts from `src/`:

```bash
cd src
python prep3.py      # builds the merged 4-channel feature table (cached to ../oof/)
python sub_v7.py     # best public-LB model
python final2.py     # v8 (4-channel model)
```

Submissions are written to `submissions/`.

## Takeaways

- **Validate the way you'll be tested.** Grouping CV folds by race was the single most important decision.
- **Leaks don't show up in CV.** A train-vs-test consistency check, plus cross-checking the same column across sources, found traps that CV rewarded.
- **Model the structure.** A multiplicative target and a dedicated weather model beat brute-force tuning.
- **With 71% of the test hidden, prefer stable choices:** equal-weight ensembles, seed averaging, and rejecting gains that don't hold on a fresh split.
