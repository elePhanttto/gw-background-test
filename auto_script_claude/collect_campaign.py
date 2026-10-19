#!/usr/bin/env python
"""campaign/*/*/w*/summary.json を集計し、real と gauss のペア比較を
背景テストの非ガウス性指標と並べる。

使い方:
  python collect_campaign.py                 # summary.json から集計 (campaign_all.csv も更新)
  python collect_campaign.py --from-csv      # 既存の campaign_all.csv から集計だけやり直す
  python collect_campaign.py --x fp          # 横軸: fp (既定) / excess / dlogL
"""
import argparse
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
N_BOOT = 4000

# 背景テストの CSV (汚染窓を除いた版, PE と同じ検出器)
# PE と条件を揃えるなら局所PSD版 (*_real_localpsd_clean_*) を使う
FP_CSV = {
    "GW200129": "GW200129_065458_L1_real_localpsd_clean_band_results_fracional_power_grid.csv",
    "GW190412": "GW190412_L1_real_localpsd_clean_band_results_fracional_power_grid.csv",
    "GW200224": "GW200224_222234_L1_real_localpsd_clean_band_results_fracional_power_grid.csv",
}

def fp_csv_path(event):
    """手動登録が無ければ、bg_auto.py の命名規則 {event}_L1_real_localpsd_clean_... を使う。"""
    import os
    if event in FP_CSV:
        return FP_CSV[event]
    fn = f"{event}_L1_real_localpsd_clean_band_results_fracional_power_grid.csv"
    return fn if os.path.exists(fn) else None


XLABEL = {
    "fp": f"fractional power ({FBAND[0]:.0f}-{FBAND[1]:.0f} Hz, tile-weighted)",
    "excess": f"excess power  y_mean - 1 ({FBAND[0]:.0f}-{FBAND[1]:.0f} Hz)",
    "dlogL": f"dlogL_exp / 1000 tiles ({FBAND[0]:.0f}-{FBAND[1]:.0f} Hz)",
}


def load_runs():
    rows = []
    for fn in glob.glob("campaign/*/*/w*/summary.json"):
        s = json.load(open(fn))
        r = dict(event=s["event"], noise=s["noise"], window=s["window"],
                 snr=s["optimal_snr"], t_win=s.get("t_win", np.nan),
                 t_minus_event=s.get("t_minus_event", np.nan))
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
    """30-60 Hz の行をタイル数で重み付けして平均する。
    fp の誤差: 各行の 89% 区間 [fp_lo, fp_hi] を ±1.6σ とみなして σ を出し、
    行どうしは独立として重み付き平均の誤差に伝播する (非対称な区間の近似)。
    unidentified(bimodal) の行は fp の数値が信用できないので fp の平均から外す。"""
    df = pd.read_csv(fp_csv_path(event)).drop_duplicates(subset=["freqs"], keep="last")
    d = df[(df.freqs >= FBAND[0]) & (df.freqs <= FBAND[1])]
    w = d.n_tiles / d.n_tiles.sum()
    out = dict(excess=float(np.sum(w * (d.y_mean - 1.0))),
               excess_err=float(np.sqrt(np.sum(w ** 2 / d.n_tiles))),   # Var[y]=1 (ガウス近似)
               dlogL=float(np.mean(d.dlogL_exp / d.n_tiles * 1000)),
               dlogL_err=0.0)
    status = d["status"].fillna("") if "status" in d else pd.Series([""] * len(d), index=d.index)
    ok = ~status.str.contains("unidentified")
    dd, ww = d[ok], d.n_tiles[ok] / d.n_tiles[ok].sum()
    sig = (dd.fp_hi - dd.fp_lo) / (2 * 1.598)        # 89% 区間 → 1σ
    out["fp"] = float(np.sum(ww * dd.fp_mean))
    out["fp_err"] = float(np.sqrt(np.sum(ww ** 2 * sig ** 2)))
    out["fp_rows_used"] = int(ok.sum())
    out["fp_rows_total"] = int(len(d))
    return out


def paired_bootstrap_ratio(zA, zB, rng, n_boot=N_BOOT):
    """std(zB)/std(zA) の 68% / 95% 区間を、窓ペアごと再標本化して求める。"""
    n = len(zA)
    j = rng.integers(0, n, size=(n_boot, n))
    r = zB[j].std(axis=1, ddof=1) / zA[j].std(axis=1, ddof=1)
    return np.percentile(r, [2.5, 16, 84, 97.5])


def summarize(df):
    rng = np.random.default_rng(0)
    out = []
    for ev, g in df.groupby("event"):
        A = g[g.noise == "gauss"].set_index("window")
        B = g[g.noise == "real"].set_index("window")
        idx = A.index.intersection(B.index)
        A, B = A.loc[idx], B.loc[idx]
        bg = background_metric(ev) if fp_csv_path(ev) else {}
        hit = g.n_hit.iloc[0] / g.n_windows.iloc[0]
        for k in FREE:
            zA, zB = A[f"z_{k}"].values, B[f"z_{k}"].values
            q = paired_bootstrap_ratio(zA, zB, rng)
            out.append(dict(
                event=ev, param=k, N=len(idx), contam_rate=hit, **bg,
                snr_check=float(np.max(np.abs(A.snr - B.snr))),
                frac_snr_lt8=float(np.mean(B.snr < 8)),
                std_z_A=zA.std(ddof=1), std_z_B=zB.std(ddof=1),
                ratio=zB.std(ddof=1) / zA.std(ddof=1),
                ratio_lo95=q[0], ratio_lo68=q[1], ratio_hi68=q[2], ratio_hi95=q[3],
                mean_z_A=zA.mean(), mean_z_B=zB.mean(),
                cov90_A=A[f"in90_{k}"].mean(), cov90_B=B[f"in90_{k}"].mean(),
                ks_p_A=stats.kstest(A[f"p_{k}"], "uniform").pvalue,
                ks_p_B=stats.kstest(B[f"p_{k}"], "uniform").pvalue,
                width_ratio_med=float(np.median(B[f"w90_{k}"] / A[f"w90_{k}"])),
                absz_wilcoxon_p=(stats.wilcoxon(np.abs(zB), np.abs(zA)).pvalue
                                 if len(idx) > 5 else np.nan)))
    return pd.DataFrame(out)


def plot(s, xkey, fname):
    fig, axes = plt.subplots(1, len(FREE), figsize=(4 * len(FREE), 3.6), sharex=True)
    for ax, k in zip(axes, FREE):
        sk = s[s.param == k]
        x, xe = sk[xkey].values, sk[f"{xkey}_err"].values
        y = sk.ratio.values
        yerr = np.vstack([y - sk.ratio_lo68.values, sk.ratio_hi68.values - y])     # 68% (1σ相当)
        ax.errorbar(x, y, xerr=xe, yerr=yerr, fmt="o", capsize=3)
        ax.vlines(x, sk.ratio_lo95, sk.ratio_hi95, color="C0", alpha=0.3, lw=6)    # 95% は太い薄線
        if len(sk) <= 12:                                   # イベントが多いときはラベルを省略
            for xi, yi, e in zip(x, y, sk.event):
                ax.annotate(e, (xi, yi), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.axhline(1.0, color="gray", ls="--")
        ax.set_title(k)
        ax.set_xlabel(XLABEL[xkey], fontsize=8)
    axes[0].set_ylabel("std(z) real / gauss\n(bar: 68%, band: 95% bootstrap)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--x", choices=["fp", "excess", "dlogL"], default="fp")
    a = ap.parse_args()
    df = pd.read_csv("campaign_all.csv") if a.from_csv else load_runs()
    if not a.from_csv:
        df.to_csv("campaign_all.csv", index=False)
    s = summarize(df)
    s.to_csv("campaign_summary.csv", index=False)
    cols = ["event", "param", "N", a.x, f"{a.x}_err", "ratio", "ratio_lo95", "ratio_hi95",
            "cov90_A", "cov90_B", "frac_snr_lt8"]
    print(s[cols].round(3).to_string())
    if a.x == "fp":
        print(s.groupby("event")[["fp_rows_used", "fp_rows_total"]].first()
              .rename(columns={"fp_rows_used": "fp に使った行", "fp_rows_total": "30-60Hz の行"}))
    plot(s, a.x, f"campaign_vs_background_{a.x}.png")


if __name__ == "__main__":
    main()