# ANN System Identification — Unbalanced Disc

ANN part of the design assignment: identify the dynamics of the Unbalanced Disc
(input `u` = motor voltage in [-3, +3] V, output `θ` = disk angle in rad),
**without using the angular velocity `ω`**.

Two deliverable models:
- **Simple approach** — NARX-MLP, `(n_a=8, n_b=8)`, MLP 64-64-tanh.
- **Advanced approach** — LSTM (hidden=128, 2 layers), trained in simulation mode (BPTT).

All models use the `(sin θ, cos θ)` angle encoding and `u/3` input normalization.

---

## Directory layout

```
ML4S&C-assignment/
├── README.md
├── final/                  ← the clean deliverable pipeline (run these)
│   ├── 1_data_prep.ipynb           data loading, sin/cos encoding, 80/20 split
│   ├── 2_narx_simple.ipynb         trains NARX (8,8) end-to-end + writes narx-mlp-* submissions
│   └── 3_lstm_advanced.ipynb       trains LSTM h=128 L=2 + writes lstm-v3-* submissions
├── exploration/            ← how we got there (kept for the report, not needed for execution)
│   ├── 1_narx_grid_ranked-on-prediction-only.ipynb        initial NARX grid -> (4,6)
│   ├── 2_narx_grid_ranked-on-prediction+simulation.ipynb  re-ranked grid -> motivates (8,8)
│   ├── 3_advanced_NOE-vs-LSTM32_T-sweep.ipynb             NOE vs LSTM(h=32), T sweep
│   └── 4_advanced_NOE-stabilize-and-LSTM64.ipynb          NOE-v2 (failed) + LSTM(h=64)
├── artifacts/              ← shared data + trained weights (see below)
├── submissions/            ← the 4 files to hand in + archive/ (old deliverables used in exploration)
└── ml-control-venv/        ← Python venv (PyTorch + MPS, Apple Silicon)
```

Notebooks use **absolute** paths (`WORK_DIR`, `REPO_DIR`, `ARTIFACTS`), so moving them
between folders does not break anything.

---

## Execution order

`final/2_*` and `final/3_*` both consume `artifacts/step2_processed.npz`, which is
produced by `final/1_data_prep.ipynb`. Beyond that there are no hidden dependencies:
each notebook in `final/` is self-contained (it trains the model, saves the weights,
and writes its own submission files).

**Simple model:**
```
final/1_data_prep.ipynb         # -> artifacts/step2_processed.npz
final/2_narx_simple.ipynb       # trains NARX (8,8) -> best_narx.pt + narx-mlp-* submissions
```

**Advanced model** (needs `1_data_prep` first):
```
final/3_lstm_advanced.ipynb     # trains LSTM h=128 L=2 -> best_advanced.pt + lstm-v3-* submissions
```

**Optional — the exploration notebooks** (only to reproduce the analysis/figures for the report):
```
exploration/1_..._prediction-only.ipynb        (the original PRED-only grid)
exploration/2_..._prediction+simulation.ipynb  (the dual-metric analysis that motivates (8,8))
exploration/3_advanced_NOE-vs-LSTM32...ipynb   (first NOE vs LSTM comparison)
exploration/4_advanced_NOE-stabilize...ipynb   (NOE stabilization + LSTM h=64)
```

Run everything with the venv kernel **`ml-control-venv`**:
```bash
source ml-control-venv/bin/activate
jupyter lab
```

---

## Single source of truth for submissions

- `narx-mlp-*` are written **only by `final/2_narx_simple.ipynb`**.
- `lstm-v3-*` are written **only by `final/3_lstm_advanced.ipynb`**.

To regenerate the simple model from scratch: open `final/2_narx_simple.ipynb` and run all cells.

---

## Why NARX (8, 8) and not (4, 6)?

`exploration/1_...prediction-only.ipynb` ranked the 16-model grid on **prediction** RMSE
only, picking `(4, 6)` (best one-step). `exploration/2_...prediction+simulation.ipynb`
re-ranked the same grid also on **simulation** (free-run) RMSE and revealed a Pareto
trade-off: `(8, 8)` gives ~22% lower SIM error while keeping PRED on par with the repo's
linear baseline. We therefore select `(8, 8)` on a balanced PRED+SIM criterion.

---

## Final results (validation set, last 20% of training data)

| Model | val PRED [rad / °] | val SIM [rad / °] |
|---|---|---|
| **NARX-MLP (8,8)** — simple    | 0.0068 / 0.39° | 0.1221 / 7.00° |
| **LSTM h=128, L=2** — advanced | 0.0266 / 1.52° | 0.0385 / 2.20° |
| Linear ARX (repo baseline)     | 0.0067 / 0.38° | 0.255  / 14.6° |
| Good NN (repo baseline)        | 0.0038 / 0.22° | 0.0271 / 1.55° |
| Lower bound (repo)             | —              | 0.0195 / 1.12° |

PRED = one-step-ahead prediction. SIM = free-run (recursive). Repo baselines may use `ω`; ours do not.

---

## Files to submit to the lecturers

```
submissions/narx-mlp-prediction-submission-file.npz   # simple, prediction task
submissions/narx-mlp-simulation-submission-file.npz   # simple, simulation task
submissions/lstm-v3-prediction-submission-file.npz    # advanced, prediction task
submissions/lstm-v3-simulation-submission-file.npz    # advanced, simulation task
```

`submissions/archive/` keeps the intermediate LSTM versions (h=32, h=64) for traceability.

---

## artifacts/

```
step2_processed.npz        normalized signals + 80/20 temporal split
step3_grid_results.csv     NARX grid ranked on PRED          (from exploration/1)
step3b_grid_with_sim.csv   NARX grid ranked on PRED + SIM     (from exploration/2)
best_narx.pt               final simple model (NARX 8,8)
best_advanced.pt           final advanced model (LSTM h=128, L=2)
```
