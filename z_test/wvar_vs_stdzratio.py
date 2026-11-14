#!/usr/bin/env python
"""白色化したノイズの分散の比 (real/gauss_est) と std(z) の比 (real/gauss_est) の関係を描く。
campaign_all.csv と window_whitened_var.csv と同じフォルダで実行する。"""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FREE = ["chirp_mass", "mass_ratio", "luminosity_distance", "theta_jn"]
BAND = "20_60"          # 横軸に使う帯域 ("60_200", "20_1024" も可)
NBINS = 6
rng = np.random.default_rng(0)

# --- 窓ごとに real と gauss_est の z を横に並べ、白色化した分散と結合する
d = pd.read_csv("campaign_all.csv")
w = pd.read_csv("window_whitened_var.csv")
z = d.pivot_table(index=["event", "window"], columns="noise",
                  values=[f"z_{p}" for p in FREE]).reset_index()
z.columns = ["_".join(c for c in col if c) for col in z.columns]   # 例: z_chirp_mass_real
m = w.merge(z, on=["event", "window"])
x_col = f"ratio_real_est_{BAND}"
print(f"窓の数: {len(m)}")


def std_ratio(g, p, n_boot=1000):
    """窓の集まり g について std(z_real)/std(z_est) と 68% ブートストラップ区間"""
    g = g.dropna(subset=[f"z_{p}_real", f"z_{p}_gauss_est"])
    a, b = g[f"z_{p}_real"].values, g[f"z_{p}_gauss_est"].values
    r = a.std(ddof=1) / b.std(ddof=1)
    j = rng.integers(0, len(a), size=(n_boot, len(a)))
    rb = a[j].std(axis=1, ddof=1) / b[j].std(axis=1, ddof=1)
    return r, np.percentile(rb, 16), np.percentile(rb, 84)


fig, axes = plt.subplots(2, len(FREE), figsize=(4.2 * len(FREE), 7.5))
xx = np.linspace(0.8, 3.0, 100)
for j, p in enumerate(FREE):
    # (a) イベントごと
    rows = []
    for ev, g in m.groupby("event"):
        r, lo, hi = std_ratio(g, p)
        rows.append((g[x_col].median(), r, lo, hi))
    ev = np.array(rows)
    ax = axes[0, j]
    ax.errorbar(ev[:, 0], ev[:, 1], yerr=[ev[:, 1] - ev[:, 2], ev[:, 3] - ev[:, 1]],
                fmt="o", ms=3, alpha=0.6, capsize=0)
    rho, pv = stats.spearmanr(ev[:, 0], ev[:, 1])
    ax.set_title(f"{p} (per event)\nSpearman rho={rho:.2f}, p={pv:.2g}", fontsize=9)

    # (b) 窓を横軸の分位点でビン分け
    edges = np.quantile(m[x_col], np.linspace(0, 1, NBINS + 1)); edges[-1] += 1e-9
    rows = []
    for l, r_ in zip(edges[:-1], edges[1:]):
        g = m[(m[x_col] >= l) & (m[x_col] < r_)]
        r, lo, hi = std_ratio(g, p)
        rows.append((g[x_col].median(), r, lo, hi))
    bn = np.array(rows)
    ax = axes[1, j]
    ax.errorbar(bn[:, 0], bn[:, 1], yerr=[bn[:, 1] - bn[:, 2], bn[:, 3] - bn[:, 1]],
                fmt="o-", capsize=3)
    # 窓ごとの検定 (ビンの図とは別に)
    ok = m[[f"z_{p}_real", f"z_{p}_gauss_est"]].notna().all(axis=1)
    dz2 = m.loc[ok, f"z_{p}_real"] ** 2 - m.loc[ok, f"z_{p}_gauss_est"] ** 2
    rho_w, pv_w = stats.spearmanr(m.loc[ok, x_col], dz2)
    ax.set_title(f"{p} (windows binned, {NBINS} bins)\n"
                 f"window-level Spearman rho={rho_w:.3f}, p={pv_w:.2g}", fontsize=9)

    for ax in axes[:, j]:
        ax.axhline(1, color="k", ls="--", lw=1)
        ax.plot(xx, np.sqrt(xx), color="gray", ls=":", lw=1, label=r"$\sqrt{x}$ (upper bound)")
        ax.set_xscale("log")
        ax.set_xlabel(f"whitened noise variance ratio real/gauss_est ({BAND} Hz)", fontsize=8)
axes[0, 0].set_ylabel("std(z) ratio real/gauss_est")
axes[1, 0].set_ylabel("std(z) ratio real/gauss_est")
axes[1, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"zstd_vs_wvar_{BAND}.png", dpi=150)
print(f"-> zstd_vs_wvar_{BAND}.png")
print(stats.spearmanr(w["ratio_real_est_20_60"], w["ratio_real_est_60_200"]))