#!/usr/bin/env python
"""campaign/*/*/w*/summary.json を集計し、real と gauss のペア比較を
背景テストの非ガウス性指標 (dlogL_exp / 1000タイル, fp) と並べる。"""
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FREE = ["chirp_mass", "mass_ratio", "luminosity_distance", "theta_jn"]
FBAND = (30.0, 60.0)    # 信号がSNRを稼ぐ帯域で非ガウス性を平均する

# 背景テストの CSV (汚染窓を除いた版, PE と同じ検出器)
FP_CSV = {
    "GW200129": "GW200129_065458_L1_real_clean_band_results_fracional_power_grid.csv",
    "GW190412": "GW190412_L1_real_no_seg55_thin_band_results_fracional_power_grid.csv",
}


def load_runs():
    rows = []
    for fn in glob.glob("campaign/*/*/w*/summary.json"):
        s = json.load(open(fn))
        r = dict(event=s["event"], noise=s["noise"], window=s["window"],
                 snr=s["optimal_snr"])
        for k in FREE:
            r[f"z_{k}"] = s[k]["z"]
            r[f"p_{k}"] = s[k]["p"]
            r[f"in90_{k}"] = s[k]["in90"]
            r[f"w90_{k}"] = s[k]["width90"]
        r.update({f"deg_{k}": v for k, v in s["degeneracy"].items()})
        r["n_hit"] = s["contamination"]["n_hit"]
        r["n_windows"] = s["contamination"]["n_windows"]
        rows.append(r)
    return pd.DataFrame(rows)


def background_metric(event):
    df = pd.read_csv(FP_CSV[event]).drop_duplicates(subset=["freqs"], keep="last")
    d = df[(df.freqs >= FBAND[0]) & (df.freqs <= FBAND[1])]
    w = d.n_tiles / d.n_tiles.sum()                       # タイル数で重み付け
    return dict(excess_power=float(np.sum(w * (d.y_mean - 1.0))),
                dlogL_per1k=float(np.mean(d.dlogL_exp / d.n_tiles * 1000)),
                fp_mean=float(d.fp_mean.mean()))


def main():
    df = load_runs()
    df.to_csv("campaign_all.csv", index=False)
    out = []
    for ev, g in df.groupby("event"):
        A = g[g.noise == "gauss"].set_index("window")
        B = g[g.noise == "real"].set_index("window")
        idx = A.index.intersection(B.index)
        A, B = A.loc[idx], B.loc[idx]
        bg = background_metric(ev) if ev in FP_CSV else dict(dlogL_per1k=np.nan, fp_mean=np.nan)
        hit = g.n_hit.iloc[0] / g.n_windows.iloc[0]
        for k in FREE:
            zA, zB = A[f"z_{k}"], B[f"z_{k}"]
            out.append(dict(
                event=ev, param=k, N=len(idx), contam_rate=hit, **bg,
                snr_check=float(np.max(np.abs(A.snr - B.snr))),
                std_z_A=zA.std(ddof=1), std_z_B=zB.std(ddof=1),
                mean_z_A=zA.mean(), mean_z_B=zB.mean(),
                cov90_A=A[f"in90_{k}"].mean(), cov90_B=B[f"in90_{k}"].mean(),
                ks_p_A=stats.kstest(A[f"p_{k}"], "uniform").pvalue,
                ks_p_B=stats.kstest(B[f"p_{k}"], "uniform").pvalue,
                width_ratio_med=float(np.median(B[f"w90_{k}"] / A[f"w90_{k}"])),
                absz_wilcoxon_p=(stats.wilcoxon(zB.abs(), zA.abs()).pvalue
                                 if len(idx) > 5 else np.nan)))
    s = pd.DataFrame(out)
    s.to_csv("campaign_summary.csv", index=False)
    print(s.round(3).to_string())

    # 図: 横軸 背景の非ガウス性, 縦軸 std(z) の比 (real/gauss)
    fig, axes = plt.subplots(1, len(FREE), figsize=(4 * len(FREE), 3.5), sharex=True)
    for ax, k in zip(axes, FREE):
        sk = s[s.param == k]
        ratio = sk.std_z_B / sk.std_z_A
        n = sk.N.values
        err = ratio * np.sqrt(2.0 / (2 * np.maximum(n - 1, 1)))   # 標準偏差比の近似誤差
        ax.errorbar(sk.dlogL_per1k, ratio, yerr=err, fmt="o")
        for x, y, e in zip(sk.dlogL_per1k, ratio, sk.event):
            ax.annotate(e, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.axhline(1.0, color="gray", ls="--")
        ax.set_title(k)
        ax.set_xlabel(f"dlogL_exp / 1000 tiles ({FBAND[0]:.0f}-{FBAND[1]:.0f} Hz)")
    axes[0].set_ylabel("std(z) real / gauss")
    fig.tight_layout()
    fig.savefig("campaign_vs_background.png", dpi=150)


if __name__ == "__main__":
    main()