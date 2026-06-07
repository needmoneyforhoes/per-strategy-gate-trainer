# per-strategy-gate-trainer

Trains a calibrated gate model per `(strategy, side)` for the 5-minute crypto up/down engine and emits a `deploy_map.json` the live bot consults at fire time.

A single global gate under- or over-gates strategies that have different base win rates. This trains a dedicated ensemble per strategy and picks a per-strategy probability threshold. At fire time the bot looks up `strategy|side` in the map and applies that model plus threshold; misses fall through to the global gate or SHADOW.

## Contents

| File | What it does |
|------|--------------|
| `per_strategy_train.py` | Trainer and entry point. Loads ticks and fire recaps, builds features, and for each `(strategy, side)` with at least 50 free fires: chronological 70/30 split, trains a 5-model soft-vote ensemble (LogReg L2, LogReg L1, RandomForest, GradientBoosting, calibrated RBF-SVM), scans thresholds for the max Wilson lower-bound WR, then validates with Bonferroni-corrected binomial p and a 1000-iter permutation test. Writes the artifacts. |
| `flip_features.py` | Runtime feature extractor vendored from the engine. Mirrors `vps_feature_extraction.py` (same `WINDOWS`, `crowd_num`, `safe_*` helpers, snapshot-at-cd logic) so training and live features match. Drops look-ahead fields. Edit upstream, not here. |

Output tiers: `TARGET` is Wilson-lo >= 0.70 with all rigor tests passing, `VIABLE` is Wilson-lo >= 0.55 relaxed. Anything else is skipped.

## Requirements

Python 3.9+, `numpy`, `scikit-learn`. Everything else is stdlib.

```bash
pip install numpy scikit-learn
```

## Usage

```bash
python3 per_strategy_train.py
```

Prints a per-strategy results table and a TARGET/VIABLE summary, then writes:

- `per_strategy_models.pkl`: fitted model packs (models, scaler, feature list), the deploy map, and per-strategy diagnostics.
- `deploy_map.json`: runtime lookup table, `"strategy|side"` to `{threshold, wr, wilson_lo, tier, ...}`.

## Data

Code only. Reads two history files from the private `polymarket-data` repo, expected under the engine's `$DATA_DIR`:

- `market_history.jsonl`: per-market tick replays, keyed by slug.
- `market_recap_history.jsonl`: per-market fire recaps (strategy, side, entry, hypothetical PnL).

The script `chdir`s into `$DATA_DIR` and inserts the engine root on `sys.path`; adjust the two path constants near the top of `per_strategy_train.py` if your checkout differs. Generated `.jsonl`, `.pkl`, and `.json` artifacts are git-ignored.

Offline training, no credentials required.
