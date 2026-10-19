#!/usr/bin/env python
"""events.csv の1イベントについて、局所PSD版の背景テストを走らせる (PE と同じ条件)。
出力:
  {name}_L1_real_localpsd_{all,clean}_band_results_fracional_power_grid.csv
  {name}_L1_sim_localpsd_{all,clean}_band_results_fracional_power_grid.csv
  {name}_L1_bg_windows.json   … 過渡ノイズ窓・エピソード窓の時刻 (pe_campaign.py が PE 窓から除外に使う)
使い方:
  python bg_auto.py --event GW200129_065458
  cut -d, -f1 events.csv | tail -n +2 | xargs -P 4 -I{} python bg_auto.py --event {}
"""
import argparse
import csv
import json
import os
import numpy as np
from gwpy.timeseries import TimeSeries
from bgtest_func import *          # generate_PSD_gauss など (既存)
from bgtest_func_frac import *     # calibrate_thinning_band など (既存)
from bgtest_grid import run_test_localpsd

EXCLUDE = 11.0
HALF = 2048.0
N_SEG_HALF = 100
SEG = 8.0
BANDS = [(0, 30, 40), (1, 40, 50), (2, 50, 60), (3, 60, 80)]
Q = 8
Y_CUT = 100.0
SEG_MEAN_CUT = 1.5


def load_event(name, path="events.csv"):
    for r in csv.DictReader(open(path)):
        if r["name"] == name:
            return r
    raise KeyError(name)


def contiguous_valid(data, t_event):
    """NaN を含まない連続区間のうち、合体時刻を含むもの (無ければ最長) に切り詰める。"""
    ok = np.isfinite(data.value)
    if ok.all():
        return data
    t = data.times.value
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    runs = list(zip(edges[::2], edges[1::2]))               # [start, stop) のインデックス
    best = max(runs, key=lambda r: (t[r[0]] <= t_event < t[r[1] - 1], r[1] - r[0]))
    print(f"  NaN あり: 連続区間 {t[best[0]]:.1f}–{t[best[1]-1]:.1f} ({(best[1]-best[0])/data.sample_rate.value:.0f} s) を使用")
    return data.crop(t[best[0]], t[best[1] - 1])


def make_random_time(t_event, data):
    rt = []
    for j in range(N_SEG_HALF):
        n = ((t_event - EXCLUDE) - (t_event - HALF)) / N_SEG_HALF
        rt.append(n * j + (t_event - HALF))
    for j in range(N_SEG_HALF):
        n = ((t_event + HALF - 1.0) - (t_event + EXCLUDE)) / N_SEG_HALF
        rt.append(n * j + (t_event + EXCLUDE))
    t0, t1 = float(data.span[0]), float(data.span[1])
    return [t for t in rt if t >= t0 + 8.0 and t + SEG <= t1 - 8.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--events", default="events.csv")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    r = load_event(a.event, a.events)
    name, det, t_event = r["name"], r["det"], float(r["gps"])
    prefix = f"{name}_"
    done = f"{prefix}{det}_bg_windows.json"
    if os.path.exists(done) and not a.force:
        print(f"{name}: 済み ({done})"); return

    data = contiguous_valid(TimeSeries.read(r["file"], format="hdf5.gwosc"), t_event)
    random_time = make_random_time(t_event, data)
    print(f"{name}: 窓候補 {len(random_time)}")

    sim = generate_PSD_gauss(data)
    white_sim = sim.whiten(fftlength=4, overlap=2, fduration=4)
    rate = [0] * len(BANDS)
    for i, fmin, fmax in BANDS:
        rate[i], _ = calibrate_thinning_band(Q, white_sim, random_time, fmin, fmax, SEG)
    print(f"  間引き率 {rate}")

    res = run_test_localpsd(data, random_time, SEG, Q, BANDS, rate, prefix, det,
                            type="real_localpsd", y_cut=Y_CUT, seg_mean_cut=SEG_MEAN_CUT,
                            geocent_time=t_event)
    run_test_localpsd(sim, random_time, SEG, Q, BANDS, rate, prefix, det,
                      type="sim_localpsd", y_cut=Y_CUT, seg_mean_cut=SEG_MEAN_CUT,
                      geocent_time=t_event)

    rt = res["rt"]
    out = dict(name=name, det=det, t_event=t_event, seg=SEG, n_windows=int(rt.size),
               bad=[[float(rt[s]), float(rt[s] + SEG)] for s in res["bad"]],
               episode=[[float(rt[s]), float(rt[s] + SEG)] for s in res["episode"]])
    json.dump(out, open(done, "w"), indent=1)
    print(f"{name}: 過渡 {len(out['bad'])} / エピソード {len(out['episode'])} / {out['n_windows']} 窓 -> {done}")


if __name__ == "__main__":
    main()
