# -*- coding: utf-8 -*-
"""滑窗 62 特征提取（移植对方管线的窗口层，适配本项目真实时间轴缓存）。

设计（对方规格要点）：
- 240s 窗 / 15s 步长全覆盖滑窗（窗不跨缺口：窗内有效行覆盖率 <0.8 丢弃）
- 标签三态：与餐重叠 >50% → 1；零重叠且距最近餐 ≥300s → 0；其余 -1（不训练）
- 训练负采样：每受试者 ≤3× 其正窗数（seed 固定）
- 62 特征 = acc_y/z/mag/jerk 各 11 统计 + acc_x 姿态 7 + 1s 活动包络时间特征 11
  （本项目 raw 105Hz 直接算；时间特征基于 1s 包络序列与采样率无关）

用法：D:/Anaconda3/envs/bme/python.exe scripts/slide_features.py --fold 0 --mode train|val|all
产物：cache/slide/fold{k}_{mode}.npz（feat (n,62) + label + window_id (sid,s,e)）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests, splits, loader
from src.pipeline.features.macro import extract_macro_features

WIN_MS = 240_000
STRIDE_MS = 15_000
NEG_BUFFER_S = 300
COV_MIN = 0.80
OUT = config.CACHE_DIR / "slide"


def window_rows(t_v, ws, we):
    """窗 [ws, we) 内有效行切片。"""
    lo = int(np.searchsorted(t_v, ws))
    hi = int(np.searchsorted(t_v, we))
    return lo, hi


def _process_session(args_p):
    """会话级特征提取（多进程 worker）。返回 (feats, labels, wids) 或 None。"""
    sid, meals_json, mode = args_p
    meals = json.loads(meals_json) if meals_json else []
    p = config.CACHE_DIR / "sessions" / f"{sid}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        acc = z["acc"][:, z["imu_valid"]].astype(np.float32)
        t_v = z["t_acc"][z["imu_valid"]].astype(np.int64)
    if len(t_v) < int(240 * 100):
        return None
    t_start, t_end = int(t_v[0]), int(t_v[-1])
    ws_all = t_start + np.arange(0, t_end - t_start - WIN_MS + 1, STRIDE_MS, dtype=np.int64)
    row_rate = 105.0
    n_rows_full = WIN_MS / 1000.0 * row_rate
    feats, labels, wids = [], [], []
    for ws in ws_all:
        we = ws + WIN_MS
        lo, hi = window_rows(t_v, ws, we)
        if hi - lo < COV_MIN * n_rows_full:
            continue
        seg = acc[:, lo:hi]
        ov = 0.0
        for m in meals:
            ov = max(ov, min(we, m["after"]) - max(ws, m["before"]))
        if ov > 0.5 * WIN_MS:
            lab = 1
        else:
            dmin = min((abs(ws - m["after"]), abs(we - m["before"]), abs(ws - m["before"]),
                        abs(we - m["after"])) for m in meals) if meals else (1e18,)
            lab = 0 if (ov == 0 and min(dmin) / 1000.0 >= NEG_BUFFER_S) else -1
        if lab == -1 and mode == "train":
            continue
        feats.append(extract_62(seg, row_rate))
        labels.append(lab)
        wids.append((sid, int(ws), int(we)))
    return (feats, labels, wids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--mode", choices=("train", "val", "all", "meal_train", "no_meal_train"), default="train")
    ap.add_argument("--limit", type=int, default=0, help="会话数上限（调试）")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    folds = splits.load_folds()
    f = folds[args.fold]
    meal_meta, _ = manifests.load_meal_meta()
    idx = manifests.load_sensor_index()
    sid_meals = {}
    for _, r in idx.iterrows():
        ext, sid, st, en = r["externalid"], r["session_id"], int(r["timeStamp.startTime"]), int(r["timeStamp.endTime"])
        ms = [m for m in meal_meta.get(ext, []) if m["before"] >= st and m["after"] <= en]
        if ms:
            sid_meals[sid] = ms
    if args.mode == "train":
        sessions = f["train_sessions"]
    elif args.mode == "meal_train":   # 复核器训练：train 的含餐会话全窗（含 -1）
        sessions = [s for s in f["train_sessions"] if s in sid_meals]
    elif args.mode == "no_meal_train":   # 复核负样本：无餐 train 会话抽样（150——60 对 val 190 无餐会话代表性不足）
        rng = np.random.default_rng(20260901)
        no_meal = [s for s in f["train_sessions"] if s not in sid_meals]
        sessions = list(rng.choice(no_meal, min(150, len(no_meal)), replace=False))   # 150 为复核负样本甜点（全量使复核保守 F1 降）
    elif args.mode == "all":
        sessions = f["train_sessions"] + f["val_sessions"]
    else:
        sessions = f["val_sessions"]
    if args.limit:
        sessions = sessions[:args.limit]

    feats, labels, wids = [], [], []
    t0 = __import__("time").time()
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(sid, json.dumps(sid_meals.get(sid, [])), args.mode) for sid in sessions]
    with ProcessPoolExecutor(max_workers=8) as ex:
        for si, res in enumerate(ex.map(_process_session, tasks, chunksize=4)):
            if res:
                f0, l0, w0 = res
                feats.extend(f0); labels.extend(l0); wids.extend(w0)
            if (si + 1) % 100 == 0:
                print(f"  {si + 1}/{len(sessions)} 会话，{len(feats)} 窗，"
                      f"{__import__('time').time() - t0:.0f}s", flush=True)
    # 负采样（train：每会话 ≤3× 其正窗数；meal_train 全量保留）
    if args.mode == "train":
        rng = np.random.default_rng(20260901)
        keep = np.zeros(len(labels), bool)
        sid_windows = {}
        for i, w in enumerate(wids):
            sid_windows.setdefault(w[0], []).append(i)
        for sid, ids in sid_windows.items():
            n_pos = sum(1 for i in ids if labels[i] == 1)
            n_neg_allow = max(3 * n_pos, 1)
            neg_ids = [i for i in ids if labels[i] == 0]
            if len(neg_ids) > n_neg_allow:
                neg_ids = list(rng.choice(neg_ids, n_neg_allow, replace=False))
            for i in ids:
                keep[i] = labels[i] == 1 or i in neg_ids
        labels = [l for l, k in zip(labels, keep) if k]
        feats = [x for x, k in zip(feats, keep) if k]
        wids = [w for w, k in zip(wids, keep) if k]
    X = np.array(feats, np.float32)
    out_p = OUT / f"fold{args.fold}_{args.mode}.npz"
    np.savez_compressed(out_p, feat=X, label=np.array(labels, np.int8),
                        wid=np.array([json.dumps(w) for w in wids]))
    print(f"→ {out_p}: {X.shape} 窗（正 {sum(1 for l in labels if l == 1)}，"
          f"负 {sum(1 for l in labels if l == 0)}，-1 {sum(1 for l in labels if l == -1)}）"
          f" 用时 {__import__('time').time() - t0:.0f}s", flush=True)


def extract_62(seg, fs):
    """Compatibility name; raw macro mathematics now lives in ``src``."""
    return extract_macro_features(seg, fs)


if __name__ == "__main__":
    main()
