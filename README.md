# per-strategy-gate-trainer

Trains and calibrates per-strategy gate models (LogReg / RF / GBM / SVM) for Polymarket strategies.

## Why it exists

The main 5-minute crypto up/down trading engine fires dozens of named strategies, each with a different base win rate. A single global model under- or over-gates most of them. This repo trains a **dedicated calibrated ensemble per `(strategy, side)`** and finds a per-strategy probability threshold that lifts each viable strategy to a statistically defensible 70–80% win rate. The output is a `deploy_map.json` the live bot consults at fire time: look up the strategy, apply its specific model + threshold, otherwise fall through to the global gate or SHADOW.

## How it works

| File | Role |
|------|------|
| `per_strategy_train.py` | The trainer / entry point. Loads historical ticks + fire recaps, builds features, and for every `(strategy, side)` with ≥50 free (un-gated) fires: chronological 70/30 walk-forward split → trains a 5-model soft-vote ensemble (LogReg L2, LogReg L1, RandomForest, GradientBoosting, calibrated RBF-SVM) → scans thresholds for the one maximizing Wilson-lower-bound WR → validates with Bonferroni-corrected binomial p and a 1000-iter permutation test → emits the artifacts. |
| `flip_features.py` | **Vendored from the engine** — runtime feature extractor that mirrors `vps_feature_extraction.py` exactly (same `WINDOWS`, `crowd_num`, `safe_*` helpers, snapshot-at-cd logic) so training-time and live features match bit-for-bit. Excludes look-ahead/leaky fields. Edit upstream, not here. |

Gating tiers in the output: `TARGET` (Wilson-lo ≥ 0.70, all rigor tests pass) and `VIABLE` (Wilson-lo ≥ 0.55, relaxed). Anything else is skipped.

## Requirements

- Python 3.9+
- `numpy`, `scikit-learn` (the only third-party deps; everything else is stdlib)
- Read access to the historical data files (see **Data**)

```bash
pip install numpy scikit-learn
```

No wallet, key, or network access is needed — this is fully offline training and handles no funds.

## Usage

```bash
python3 per_strategy_train.py
```

The script `chdir`s into `/home/polybot/polymarket-bot/data` and inserts `/home/polybot/polymarket-bot` on the path (adjust these two constants near the top if your checkout differs). It prints a per-strategy results table and a TARGET/VIABLE summary, then writes:

- `per_strategy_models.pkl` — fitted model packs (models + scaler + feature list), the deploy map, and per-strategy diagnostics.
- `deploy_map.json` — the runtime lookup table (`"strategy|side" → {threshold, wr, wilson_lo, tier, ...}`) consumed by the live bot's gate.

## Data

This repo ships **code only**. It reads two history files from the private `polymarket-data` repo (expected under the engine's `data/` dir):

- `market_history.jsonl` — per-market tick replays (keyed by slug).
- `market_recap_history.jsonl` — per-market fire recaps with strategy, side, entry, and hypothetical PnL.

Point the path constants at your local `polymarket-data` checkout before running. The `.jsonl`, `.pkl`, and `.json` artifacts are git-ignored and never committed.

> Private research software. No warranty; trades/handles real funds at your own risk.
