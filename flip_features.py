"""
flip_features.py — runtime feature extractor for flip_gate.

Mirrors vps_feature_extraction.py EXACTLY so runtime values match training:
  - Same WINDOWS dict (cd boundaries, inclusive both ends)
  - Same crowd_num conversion ('DN' → 1, else → 0)
  - Same safe_avg / safe_std / safe_slope / safe_max / safe_min logic
  - Same pct_above / consec_above logic

Also adds context features (hour_utc, day_of_week) computed from current wall-clock
since training pulled them from the post-market record.

Excludes leaky features (sc_200_winner_ask, sc_200_loser_bid) which can never be
computed at runtime — they require post-market winner knowledge.
"""
import math
import statistics
import datetime

# Tick column indices — matches entry_model_gate.py and feature_matrix.json
COL_CD          = 0
COL_UP_BID      = 1
COL_UP_ASK      = 2
COL_DN_BID      = 3
COL_DN_ASK      = 4
COL_UP_DEPTH    = 5
COL_DN_DEPTH    = 6
COL_UP_ZC       = 7
COL_DN_ZC       = 8
COL_UP_SR       = 9
COL_DN_SR       = 10
COL_BN_PRICE    = 11
COL_BN_DELTA    = 12
COL_UP_EMA      = 13
COL_DN_EMA      = 14
COL_UP_D3       = 15
COL_DN_D3       = 16
COL_UP_D10      = 17
COL_DN_D10      = 18
COL_CROWD_SIDE  = 19
COL_CROWD_CONV  = 20
COL_ASK_SUM     = 21
COL_DEP_VEL_UP  = 22
COL_DEP_VEL_DN  = 23
COL_CL_DELTA    = 24
COL_CL_AGE      = 25
COL_BN_SPREAD   = 26

# IDENTICAL to vps_feature_extraction.py
WINDOWS = {
    'entry':     (240, 310),
    'early':     (200, 240),
    'mid':       (160, 200),
    'late':      (120, 160),
    'very_late': ( 90, 120),
}

SC_SNAPSHOTS = {200: 'sc_200', 150: 'sc_150', 120: 'sc_120', 90: 'sc_90'}


# ── helpers — IDENTICAL to vps_feature_extraction.py ───────────────────────
def safe_avg(lst):
    lst = [x for x in lst if x is not None]
    return sum(lst) / len(lst) if lst else None

def safe_std(lst):
    lst = [x for x in lst if x is not None]
    return statistics.stdev(lst) if len(lst) >= 2 else None

def safe_min(lst):
    lst = [x for x in lst if x is not None]
    return min(lst) if lst else None

def safe_max(lst):
    lst = [x for x in lst if x is not None]
    return max(lst) if lst else None

def safe_slope(ys):
    ys = [y for y in ys if y is not None]
    n = len(ys)
    if n < 3: return None
    xs = list(range(n))
    xm = sum(xs) / n; ym = sum(ys) / n
    num = sum((xs[i] - xm) * (ys[i] - ym) for i in range(n))
    den = sum((xs[i] - xm) ** 2 for i in range(n))
    return num / den if den else None

def pct_above(lst, thr):
    lst = [x for x in lst if x is not None]
    return sum(1 for x in lst if x > thr) / len(lst) if lst else None

def pct_below(lst, thr):
    lst = [x for x in lst if x is not None]
    return sum(1 for x in lst if x < thr) / len(lst) if lst else None

def consec_above(lst, thr):
    lst = [x for x in lst if x is not None]
    best = cur = 0
    for x in lst:
        cur = cur + 1 if x > thr else 0
        best = max(best, cur)
    return best

def crowd_num(x):
    """IDENTICAL to vps_feature_extraction.py.
    String 'DN' → 1, anything else → 0. None → None."""
    if x is None: return None
    if isinstance(x, str): return 1 if x == 'DN' else 0
    return x


# ── per-window aggregator ──────────────────────────────────────────────────
def _window_features(window_ticks, prefix):
    """Compute all window-aggregate features for given prefix.
    Returns dict with keys '{prefix}_*'.
    Output is identical to vps_feature_extraction.py per-window section.
    """
    feat = {}
    if not window_ticks:
        # Return Nones for all expected keys (matches training behavior)
        for k in ('n', 'up_ask_avg', 'up_ask_std', 'up_ask_slope', 'up_ask_min', 'up_ask_max',
                  'dn_ask_avg', 'dn_ask_std', 'dn_ask_slope', 'dn_ask_min', 'dn_ask_max',
                  'spread_avg', 'spread_std', 'spread_slope', 'spread_pct_dn',
                  'up_ema_avg', 'dn_ema_avg', 'ema_spread_avg',
                  'bn_avg', 'bn_std', 'bn_slope', 'bn_neg_pct', 'bn_pos_pct',
                  'crowd_dn_pct', 'conv_avg', 'conv_std',
                  'up_depth_avg', 'dn_depth_avg', 'depth_ratio_avg',
                  'up_sr_avg', 'dn_sr_avg', 'sr_ratio_avg',
                  'up_zc_avg', 'dn_zc_avg', 'ask_sum_avg',
                  'up_d3_avg', 'dn_d3_avg', 'up_d10_avg', 'dn_d10_avg',
                  'cl_delta_avg', 'cl_age_avg',
                  'up_pct_above50', 'dn_pct_above50', 'up_consec_above50'):
            feat[f'{prefix}_{k}'] = None
        return feat
    
    # Pull each column from the tick rows. Column-by-column, raw values.
    def col(idx):
        return [t[idx] if len(t) > idx else None for t in window_ticks]
    
    up_ask  = col(COL_UP_ASK)
    dn_ask  = col(COL_DN_ASK)
    up_ema  = col(COL_UP_EMA)
    dn_ema  = col(COL_DN_EMA)
    bn      = col(COL_BN_DELTA)
    crowd_s = col(COL_CROWD_SIDE)
    conv    = col(COL_CROWD_CONV)
    up_dep  = col(COL_UP_DEPTH)
    dn_dep  = col(COL_DN_DEPTH)
    up_sr   = col(COL_UP_SR)
    dn_sr   = col(COL_DN_SR)
    up_zc   = col(COL_UP_ZC)
    dn_zc   = col(COL_DN_ZC)
    ask_sum = col(COL_ASK_SUM)
    up_d3   = col(COL_UP_D3)
    dn_d3   = col(COL_DN_D3)
    up_d10  = col(COL_UP_D10)
    dn_d10  = col(COL_DN_D10)
    cl_d    = col(COL_CL_DELTA)
    cl_a    = col(COL_CL_AGE)
    
    # Derived series — IDENTICAL to training
    spread = [dn - up if dn is not None and up is not None else None
              for dn, up in zip(dn_ask, up_ask)]
    ema_sp = [dn - up if dn is not None and up is not None else None
              for dn, up in zip(dn_ema, up_ema)]
    dep_rat = [dn / up if dn and up and up > 0 else None
               for dn, up in zip(dn_dep, up_dep)]
    sr_rat = [dn / up if dn and up and up > 0 else None
              for dn, up in zip(dn_sr, up_sr)]
    crowd_n = [crowd_num(x) for x in crowd_s]   # ← THIS IS THE KEY FIX
    
    feat[f'{prefix}_n']               = len(window_ticks)
    feat[f'{prefix}_up_ask_avg']      = safe_avg(up_ask)
    feat[f'{prefix}_up_ask_std']      = safe_std(up_ask)
    feat[f'{prefix}_up_ask_slope']    = safe_slope(up_ask)
    feat[f'{prefix}_up_ask_min']      = safe_min(up_ask)
    feat[f'{prefix}_up_ask_max']      = safe_max(up_ask)
    feat[f'{prefix}_dn_ask_avg']      = safe_avg(dn_ask)
    feat[f'{prefix}_dn_ask_std']      = safe_std(dn_ask)
    feat[f'{prefix}_dn_ask_slope']    = safe_slope(dn_ask)
    feat[f'{prefix}_dn_ask_min']      = safe_min(dn_ask)
    feat[f'{prefix}_dn_ask_max']      = safe_max(dn_ask)
    feat[f'{prefix}_spread_avg']      = safe_avg(spread)
    feat[f'{prefix}_spread_std']      = safe_std(spread)
    feat[f'{prefix}_spread_slope']    = safe_slope(spread)
    feat[f'{prefix}_spread_pct_dn']   = pct_above(spread, 0.0)
    feat[f'{prefix}_up_ema_avg']      = safe_avg(up_ema)
    feat[f'{prefix}_dn_ema_avg']      = safe_avg(dn_ema)
    feat[f'{prefix}_ema_spread_avg']  = safe_avg(ema_sp)
    feat[f'{prefix}_bn_avg']          = safe_avg(bn)
    feat[f'{prefix}_bn_std']          = safe_std(bn)
    feat[f'{prefix}_bn_slope']        = safe_slope(bn)
    feat[f'{prefix}_bn_neg_pct']      = pct_below(bn, 0.0)
    feat[f'{prefix}_bn_pos_pct']      = pct_above(bn, 0.0)
    feat[f'{prefix}_crowd_dn_pct']    = safe_avg(crowd_n)   # avg of 0/1 ints
    feat[f'{prefix}_conv_avg']        = safe_avg(conv)
    feat[f'{prefix}_conv_std']        = safe_std(conv)
    feat[f'{prefix}_up_depth_avg']    = safe_avg(up_dep)
    feat[f'{prefix}_dn_depth_avg']    = safe_avg(dn_dep)
    feat[f'{prefix}_depth_ratio_avg'] = safe_avg(dep_rat)
    feat[f'{prefix}_up_sr_avg']       = safe_avg(up_sr)
    feat[f'{prefix}_dn_sr_avg']       = safe_avg(dn_sr)
    feat[f'{prefix}_sr_ratio_avg']    = safe_avg(sr_rat)
    feat[f'{prefix}_up_zc_avg']       = safe_avg(up_zc)
    feat[f'{prefix}_dn_zc_avg']       = safe_avg(dn_zc)
    feat[f'{prefix}_ask_sum_avg']     = safe_avg(ask_sum)
    feat[f'{prefix}_up_d3_avg']       = safe_avg(up_d3)
    feat[f'{prefix}_dn_d3_avg']       = safe_avg(dn_d3)
    feat[f'{prefix}_up_d10_avg']      = safe_avg(up_d10)
    feat[f'{prefix}_dn_d10_avg']      = safe_avg(dn_d10)
    feat[f'{prefix}_cl_delta_avg']    = safe_avg(cl_d)
    feat[f'{prefix}_cl_age_avg']      = safe_avg(cl_a)
    feat[f'{prefix}_up_pct_above50']  = pct_above(up_ask, 0.5)
    feat[f'{prefix}_dn_pct_above50']  = pct_above(dn_ask, 0.5)
    feat[f'{prefix}_up_consec_above50'] = consec_above(up_ask, 0.5)
    
    return feat


def _snapshot_at_cd(replay_ticks, target_cd, prefix):
    """Find tick closest to target_cd (must be at or below the cd) and extract snapshot."""
    feat = {}
    candidates = [t for t in replay_ticks if len(t) > COL_CD and t[COL_CD] is not None and t[COL_CD] <= target_cd]
    if not candidates:
        feat[f'{prefix}_crowd_dn']    = None
        feat[f'{prefix}_crowd_conv']  = None
        feat[f'{prefix}_bn_delta']    = None
        return feat
    # Closest tick at or below target_cd (highest cd that's <= target)
    tick = max(candidates, key=lambda t: t[COL_CD])
    crowd_s = tick[COL_CROWD_SIDE] if len(tick) > COL_CROWD_SIDE else None
    feat[f'{prefix}_crowd_dn']   = crowd_num(crowd_s)
    feat[f'{prefix}_crowd_conv'] = tick[COL_CROWD_CONV] if len(tick) > COL_CROWD_CONV else None
    feat[f'{prefix}_bn_delta']   = tick[COL_BN_DELTA]   if len(tick) > COL_BN_DELTA   else None
    return feat


def build_features(replay_ticks, current_time=None):
    """Build the full feature dict matching feature_matrix.json column names.
    
    Args:
        replay_ticks: list of tick rows (each row is a list)
        current_time: datetime for hour_utc/day_of_week. Defaults to now.
    
    Returns dict with all feature keys produced by vps_feature_extraction.py
    (minus the post-market mw_*, gc_*, and side-relative sc_*_winner/loser fields).
    """
    feat = {}
    
    # Per-window features
    for win_name, (cd_lo, cd_hi) in WINDOWS.items():
        # IDENTICAL to training: cd_lo <= t.cd <= cd_hi (inclusive both ends)
        wt = [t for t in replay_ticks
              if len(t) > COL_CD and t[COL_CD] is not None
              and cd_lo <= t[COL_CD] <= cd_hi]
        feat.update(_window_features(wt, win_name))
    
    # Snapshot-at-cd features
    for cd_target, prefix in SC_SNAPSHOTS.items():
        feat.update(_snapshot_at_cd(replay_ticks, cd_target, prefix))
    
    # Context features — pulled from current wall-clock time
    if current_time is None:
        current_time = datetime.datetime.utcnow()
    feat['hour_utc']    = current_time.hour
    feat['day_of_week'] = current_time.weekday()
    
    return feat
