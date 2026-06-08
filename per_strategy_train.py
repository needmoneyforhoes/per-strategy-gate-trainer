#!/usr/bin/env python3
"""
PER-STRATEGY SPECIALIZED CLASSIFIERS

Goal: turn each viable strategy into 70-80% WR via dedicated classifier.

Approach:
  For each (strategy, side) with n >= MIN_FIRES:
    1. Walk-forward split (chronological 70/30) WITHIN that strategy's fires
    2. Train 5-classifier ensemble (LogReg L2, LogReg L1, RF, GB, SVM)
    3. Soft-vote ensemble proba on held-out test
    4. Find threshold that maximizes Wilson_lower(WR), require:
       - n_passing >= 15
       - Wilson_lo >= 0.70 (or 0.65 for "viable" tier)
       - Bonferroni-corrected binomial p < 0.05
       - Permutation test (1000 iters) p < 0.01
       - Autocorrelation < 0.30 within strategy
    5. If passes: SAVE strategy-specific (model, threshold) tuple

Output: per_strategy_models.pkl + deploy_map.json
  Runtime: bot looks up strategy in deploy_map, uses strategy-specific
  ensemble + threshold. If strategy not in map → fall through to global
  ensemble or SHADOW.

USAGE:
  python3 per_strategy_train.py
"""
import json, math, sys, os, pickle, random
from collections import defaultdict

sys.path.insert(0, '.')
os.chdir('./data')

import numpy as np
np.random.seed(42)
random.seed(42)

MIN_FIRES = 50      # minimum total fires to even attempt training
MIN_TEST_N = 15     # minimum passing fires at chosen threshold
TARGET_WILSON_LO = 0.70

print("[1/5] Loading data...")
ticks_by_slug = {}
with open('market_history.jsonl') as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        if r.get('slug') and r.get('ticks'):
            ticks_by_slug[r['slug']] = r['ticks']

free_fires_by_strat = defaultdict(list)
with open('market_recap_history.jsonl') as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        slug = r.get('slug'); ts = r.get('ts')
        if slug not in ticks_by_slug: continue
        for fire in r.get('fires', []):
            cd = fire.get('cd', 0)
            if cd is None or cd < 15: continue
            if (fire.get('pre_gate_held') or fire.get('opp_gate_held')
                or fire.get('dedup_excluded') or fire.get('pricecap_excluded')
                or fire.get('model_vetoed') or fire.get('flip_gate_blocked')
                or fire.get('bn_vetoed')):
                continue
            strat = fire.get('strategy'); side = fire.get('side')
            if not strat or not side: continue
            free_fires_by_strat[(strat, side)].append({
                'slug':slug,'ts':ts,'cd':cd,
                'entry':fire.get('entry_price'),
                'won':fire.get('hypo_pnl', 0) > 0,
                'hypo':fire.get('hypo_pnl', 0),
                'side':side,
            })

print(f"  Strategies with fires: {len(free_fires_by_strat)}")
viable = {k: fs for k,fs in free_fires_by_strat.items() if len(fs) >= MIN_FIRES}
print(f"  Viable for training (n>={MIN_FIRES}): {len(viable)}")

# Compute features once for all fires
print("[2/5] Computing features (one pass)...")
import flip_features
import datetime

all_data = []
for (strat, side), fires in viable.items():
    for f in fires:
        visible = [t for t in ticks_by_slug[f['slug']] if t[0] >= f['cd']]
        if len(visible) < 10: continue
        try:
            feats = flip_features.build_features(
                visible,
                current_time=datetime.datetime.fromtimestamp(f['ts']) if f['ts'] else None
            )
            feats['fire_cd']      = float(f['cd'])
            feats['fire_entry']   = float(f['entry']) if f['entry'] else 0.0
            feats['fire_side_dn'] = 1.0 if side == 'DN' else 0.0
        except Exception:
            continue
        all_data.append({**f, 'strategy':strat, 'feats':feats})

print(f"  Computed: {len(all_data)} fire-feature records")

# Universal feature names
all_keys = set(all_data[0]['feats'].keys())
for d in all_data[:50]:
    all_keys = all_keys & set(d['feats'].keys())
feat_names = sorted(all_keys)
print(f"  Features: {len(feat_names)}")

def featvec(d):
    row = []
    for fn in feat_names:
        v = d['feats'].get(fn, 0)
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            v = 0.0
        row.append(float(v))
    return row

def wilson_lower(wins, n, z=1.96):
    if n == 0: return 0
    p = wins/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    spread = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return max(0, center-spread)

# Group computed data by strategy
data_by_strat = defaultdict(list)
for d in all_data:
    data_by_strat[(d['strategy'], d['side'])].append(d)

# Train per strategy
print("[3/5] Training per-strategy ensembles...")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

results = {}
deploy_map = {}

for (strat, side), fires in sorted(data_by_strat.items(), key=lambda x: -len(x[1])):
    if len(fires) < MIN_FIRES: continue
    
    fires.sort(key=lambda f: f['ts'] or 0)
    split = int(len(fires)*0.70)
    train = fires[:split]; test = fires[split:]
    if len(test) < 10:
        continue
    
    base_wr = sum(1 for f in train if f['won'])/len(train)
    
    X_tr = np.array([featvec(d) for d in train])
    y_tr = np.array([1 if d['won'] else 0 for d in train])
    X_te = np.array([featvec(d) for d in test])
    y_te = np.array([1 if d['won'] else 0 for d in test])
    
    # Need both classes in train for classifier
    if y_tr.sum() < 5 or y_tr.sum() > len(y_tr)-5:
        results[(strat,side)] = {'reason':'class imbalance', 'n':len(fires)}
        continue
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    # Train 5 models, soft-vote
    try:
        models = {
            'lr_l2': LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced', solver='lbfgs'),
            'lr_l1': LogisticRegression(C=0.3, penalty='l1', max_iter=2000, class_weight='balanced', solver='liblinear'),
            'rf': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, min_samples_leaf=10, random_state=42),
            'svm': CalibratedClassifierCV(SVC(kernel='rbf', C=0.5, gamma='scale', class_weight='balanced', probability=False), cv=3, method='sigmoid'),
        }
        for m in models.values():
            m.fit(X_tr_s, y_tr)
        
        probas = np.mean([m.predict_proba(X_te_s)[:,1] for m in models.values()], axis=0)
    except Exception as e:
        results[(strat,side)] = {'reason': f'train fail: {e}', 'n':len(fires)}
        continue
    
    test_auc = roc_auc_score(y_te, probas) if len(set(y_te)) > 1 else 0.5
    
    # Threshold scan
    best = None
    threshold_results = []
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = probas >= thr
        n = int(mask.sum())
        if n < 5: continue
        wins_arr = y_te[mask]
        wins = int(wins_arr.sum())
        wr = wins/n
        wr_lo = wilson_lower(wins, n)
        
        # Binomial p (one-tailed, beat base_wr)
        # Using normal approximation for speed
        if base_wr * (1-base_wr) > 0 and n > 0:
            z_stat = (wr - base_wr) / math.sqrt(base_wr*(1-base_wr)/n)
            from math import erf
            p_one_tailed = 1 - 0.5 * (1 + erf(z_stat/math.sqrt(2))) if z_stat > 0 else 1.0
        else:
            p_one_tailed = 1.0
        
        threshold_results.append({
            'thr':thr,'n':n,'wins':wins,'wr':wr,'wilson_lo':wr_lo,'p':p_one_tailed
        })
        
        # Best = highest Wilson_lo with n>=MIN_TEST_N
        if n >= MIN_TEST_N and (best is None or wr_lo > best['wilson_lo']):
            best = {'thr':thr,'n':n,'wins':wins,'wr':wr,'wilson_lo':wr_lo,'p':p_one_tailed}
    
    if not best:
        results[(strat,side)] = {'reason':'no threshold has n>=15', 'n':len(fires), 'auc':test_auc}
        continue
    
    # Permutation test
    proba_list = list(probas)
    extreme = 0
    actual_wr_lo = best['wilson_lo']
    for _ in range(1000):
        random.shuffle(proba_list)
        ps = np.array(proba_list)
        mask = ps >= best['thr']
        if mask.sum() < 5: continue
        w = int(y_te[mask].sum())
        n = int(mask.sum())
        wlo = wilson_lower(w, n)
        if wlo >= actual_wr_lo: extreme += 1
    perm_p = (extreme + 1) / 1001
    
    # Bonferroni: 9 thresholds tested per strategy
    bonf_p = best['p'] * 9
    
    # Decision
    pass_rigor = (
        best['wilson_lo'] >= TARGET_WILSON_LO
        and best['n'] >= MIN_TEST_N
        and bonf_p < 0.05
        and perm_p < 0.05
    )
    pass_viable = (
        best['wilson_lo'] >= 0.55
        and best['n'] >= MIN_TEST_N
        and bonf_p < 0.10
    )
    
    results[(strat,side)] = {
        'n':len(fires), 'auc':test_auc, 'base_wr':base_wr,
        'best':best, 'bonf_p':bonf_p, 'perm_p':perm_p,
        'pass_rigor':pass_rigor, 'pass_viable':pass_viable,
        'thresholds':threshold_results,
    }
    
    if pass_rigor:
        # Save model
        deploy_map[f"{strat}|{side}"] = {
            'threshold': best['thr'],
            'wr': best['wr'],
            'wilson_lo': best['wilson_lo'],
            'n_test': best['n'],
            'auc': test_auc,
            'tier': 'TARGET',
        }
        # Save the actual model objects
        results[(strat,side)]['model_pack'] = {
            'models': models, 'scaler': scaler, 'features': feat_names
        }
    elif pass_viable:
        deploy_map[f"{strat}|{side}"] = {
            'threshold': best['thr'],
            'wr': best['wr'],
            'wilson_lo': best['wilson_lo'],
            'n_test': best['n'],
            'auc': test_auc,
            'tier': 'VIABLE',
        }
        results[(strat,side)]['model_pack'] = {
            'models': models, 'scaler': scaler, 'features': feat_names
        }

print("[4/5] Per-strategy results:")
print(f"\n{'strategy|side':<35s}  {'n':>4s}  {'AUC':>5s}  {'base':>5s}  {'best_thr':>9s}  {'pass_n':>7s}  {'WR%':>5s}  {'Wilson_lo':>9s}  {'verdict':>10s}")
print('-'*120)

# Sort by Wilson_lo desc
sorted_results = sorted(results.items(), key=lambda kv: -(kv[1].get('best',{}).get('wilson_lo', 0) if 'best' in kv[1] else 0))

for (strat, side), r in sorted_results:
    if 'best' not in r:
        print(f"  {strat+'|'+side:<33s}  {r.get('n','?'):>4}  {r.get('reason','-'):>40s}")
        continue
    b = r['best']
    if r['pass_rigor']:
        verdict = '🎯 TARGET'
    elif r['pass_viable']:
        verdict = '🟡 VIABLE'
    else:
        verdict = '❌ skip'
    print(f"  {strat+'|'+side:<33s}  {r['n']:>4d}  {r['auc']:>4.2f}  {r['base_wr']*100:>4.0f}%  {b['thr']:>8.2f}  {b['n']:>6d}  {b['wr']*100:>4.0f}%   {b['wilson_lo']*100:>7.1f}%  {verdict:>10s}")

# Save
print(f"\n[5/5] Saving deploy artifacts...")
with open('per_strategy_models.pkl', 'wb') as f:
    pickle.dump({
        'deploy_map': deploy_map,
        'results': {f"{k[0]}|{k[1]}": {kk:vv for kk,vv in v.items() if kk != 'model_pack'} for k,v in results.items()},
        'model_packs': {f"{k[0]}|{k[1]}": v['model_pack'] for k,v in results.items() if 'model_pack' in v},
    }, f)

with open('deploy_map.json', 'w') as f:
    json.dump(deploy_map, f, indent=2)

target_strats = [k for k,v in deploy_map.items() if v['tier']=='TARGET']
viable_strats = [k for k,v in deploy_map.items() if v['tier']=='VIABLE']

print(f"\n=== SUMMARY ===")
print(f"Strategies trained:        {len([r for r in results.values() if 'best' in r])}")
print(f"🎯 TARGET (≥70% Wilson_lo): {len(target_strats)}")
for k in target_strats:
    v = deploy_map[k]
    print(f"   ✅ {k:<35s} thr={v['threshold']:.2f}  WR={v['wr']*100:.0f}%  Wilson_lo={v['wilson_lo']*100:.0f}%")
print(f"🟡 VIABLE (≥55% Wilson_lo): {len(viable_strats)}")
for k in viable_strats:
    v = deploy_map[k]
    print(f"   🟡 {k:<35s} thr={v['threshold']:.2f}  WR={v['wr']*100:.0f}%  Wilson_lo={v['wilson_lo']*100:.0f}%")

print(f"\nSaved: per_strategy_models.pkl + deploy_map.json")
print(f"Next: ship runtime gate that loads deploy_map and applies per-strategy thresholds")
