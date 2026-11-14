#!/usr/bin/env python
"""イベントごとの center_factor / zrms_ratio (gauss_est->real) を背景の非ガウス性 (fp, excess) と比べる"""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collect_campaign import background_metric, fp_csv_path   # 既存の関数を再利用

FREE = ["chirp_mass", "mass_ratio", "luminosity_distance", "theta_jn"]
A, B = "gauss_est", "real"
rng = np.random.default_rng(0)
rms = lambda v: np.sqrt(np.mean(v ** 2))

df = pd.read_csv("campaign_all.csv")
for p in FREE:
    df[f"err_{p}"] = df[f"post_med_{p}"] - df[f"inj_{p}"]

rows = []
for ev, g in df.groupby("event"):
    if not fp_csv_path(ev):
        continue                                   # 背景テストの結果が無いイベントは飛ばす
    a = g[g.noise == A].set_index("window"); b = g[g.noise == B].set_index("window")
    idx = a.index.intersection(b.index)
    if len(idx) < 5:
        continue
    bg = background_metric(ev)
    for p in FREE:
        za = a.loc[idx, f"z_{p}"].values; zb = b.loc[idx, f"z_{p}"].values
        c = b.loc[idx, f"err_{p}"].values / a.loc[idx, f"post_std_{p}"].values
        # 窓ペアを再標本化して、イベントごとの誤差棒を出す
        j = rng.integers(0, len(idx), size=(1000, len(idx)))
        cf_b = np.sqrt((c[j] ** 2).mean(1) / (za[j] ** 2).mean(1))
        zr_b = np.sqrt((zb[j] ** 2).mean(1) / (za[j] ** 2).mean(1))
        rows.append(dict(event=ev, param=p, N=len(idx), fp=bg["fp"], fp_err=bg["fp_err"],
                         excess=bg["excess"],
                         center=rms(c) / rms(za), center_lo=np.percentile(cf_b, 16), center_hi=np.percentile(cf_b, 84),
                         zrms=rms(zb) / rms(za), zrms_lo=np.percentile(zr_b, 16), zrms_hi=np.percentile(zr_b, 84)))
s = pd.DataFrame(rows)
s.to_csv("center_vs_fp.csv", index=False)
print(f"背景テストの結果があるイベント数: {s.event.nunique()}")

fig, axes = plt.subplots(2, len(FREE), figsize=(4.2 * len(FREE), 7), sharex=True)
for j, p in enumerate(FREE):
    sp = s[s.param == p]
    for i, (y, lo, hi, lab) in enumerate([("center", "center_lo", "center_hi", "center_factor (ずれの増え方)"),
                                          ("zrms", "zrms_lo", "zrms_hi", "zrms_ratio (較正のずれ)")]):
        ax = axes[i, j]
        ax.errorbar(sp.fp, sp[y], xerr=sp.fp_err, yerr=[sp[y] - sp[lo], sp[hi] - sp[y]],
                    fmt="o", ms=3, alpha=0.6, capsize=0)
        ax.axhline(1, color="k", ls="--", lw=1)
        rho, pv = stats.spearmanr(sp.fp, sp[y])
        ax.set_title(f"{p}\nSpearman ρ={rho:.2f}, p={pv:.2g}", fontsize=9)
        if j == 0: ax.set_ylabel(lab)
        if i == 1: ax.set_xlabel("fp (30-60 Hz)")
fig.tight_layout(); fig.savefig("center_vs_fp.png", dpi=150)

# fp で3グループに分けて中央値を比べる (1点ずつよりも統計が効く)
s["fp_group"] = s.groupby("param")["fp"].transform(lambda x: pd.qcut(x, 3, labels=["low", "mid", "high"]))
print(s.groupby(["param", "fp_group"], observed=True)[["center", "zrms"]].median().round(3))