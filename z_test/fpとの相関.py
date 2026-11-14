import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# データの読み込み
df_all = pd.read_csv("campaign_all.csv")
df_sum = pd.read_csv("campaign_summary_real_vs_gauss_est.csv")

# イベントごとの fp (30-60 Hz) と誤差を抽出
fp_ev = df_sum.groupby("event")[["fp", "fp_err"]].first().reset_index()

FREE = ["chirp_mass", "mass_ratio", "luminosity_distance", "theta_jn"]
sub = df_all[df_all["noise"].isin(["real", "gauss_est"])].copy()

# イベント × ノイズ条件ごとに mean と std を集計
agg_dict = {}
for p in FREE:
    for q in ["post_std", "post_skew", "post_kurt"]:
        col = f"{q}_{p}"
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"] = (col, lambda x: np.std(x, ddof=1))

ev_agg = sub.groupby(["event", "noise"]).agg(**agg_dict).unstack("noise")

# 比 (post_std) と 差 (post_skew, post_kurt) を計算して fp と結合
comp_rows = []
for _, r_fp in fp_ev.iterrows():
    ev = r_fp["event"]
    if ev not in ev_agg.index or np.isnan(r_fp["fp"]):
        continue
    row = {"event": ev, "fp": r_fp["fp"], "fp_err": r_fp["fp_err"]}
    for p in FREE:
        row[f"post_std_mean_ratio_{p}"] = (
            ev_agg.loc[ev, (f"post_std_{p}_mean", "real")]
            / ev_agg.loc[ev, (f"post_std_{p}_mean", "gauss_est")]
        )
        row[f"post_std_std_ratio_{p}"] = (
            ev_agg.loc[ev, (f"post_std_{p}_std", "real")]
            / ev_agg.loc[ev, (f"post_std_{p}_std", "gauss_est")]
        )
        row[f"post_skew_mean_diff_{p}"] = (
            ev_agg.loc[ev, (f"post_skew_{p}_mean", "real")]
            - ev_agg.loc[ev, (f"post_skew_{p}_mean", "gauss_est")]
        )
        row[f"post_skew_std_diff_{p}"] = (
            ev_agg.loc[ev, (f"post_skew_{p}_std", "real")]
            - ev_agg.loc[ev, (f"post_skew_{p}_std", "gauss_est")]
        )
        row[f"post_kurt_mean_diff_{p}"] = (
            ev_agg.loc[ev, (f"post_kurt_{p}_mean", "real")]
            - ev_agg.loc[ev, (f"post_kurt_{p}_mean", "gauss_est")]
        )
        row[f"post_kurt_std_diff_{p}"] = (
            ev_agg.loc[ev, (f"post_kurt_{p}_std", "real")]
            - ev_agg.loc[ev, (f"post_kurt_{p}_std", "gauss_est")]
        )
    comp_rows.append(row)

comp_df = pd.DataFrame(comp_rows)

# 相関係数の出力と描画
metrics = [
    (
        "post_std_mean_ratio",
        r"$\mathrm{mean}(\sigma_{\mathrm{post}})$ ratio"
        + "\n(real / gauss_est)",
        1.0,
    ),
    (
        "post_std_std_ratio",
        r"$\mathrm{std}(\sigma_{\mathrm{post}})$ ratio"
        + "\n(real / gauss_est)",
        1.0,
    ),
    (
        "post_skew_mean_diff",
        r"$\mathrm{mean}(\mathrm{skew})$ diff" + "\n(real - gauss_est)",
        0.0,
    ),
    (
        "post_skew_std_diff",
        r"$\mathrm{std}(\mathrm{skew})$ diff" + "\n(real - gauss_est)",
        0.0,
    ),
    (
        "post_kurt_mean_diff",
        r"$\mathrm{mean}(\mathrm{kurt})$ diff" + "\n(real - gauss_est)",
        0.0,
    ),
    (
        "post_kurt_std_diff",
        r"$\mathrm{std}(\mathrm{kurt})$ diff" + "\n(real - gauss_est)",
        0.0,
    ),
]

PARAM_TITLES = {
    "chirp_mass": r"Chirp mass $\mathcal{M}$",
    "mass_ratio": r"Mass ratio $q$",
    "luminosity_distance": r"Luminosity distance $d_L$",
    "theta_jn": r"Inclination $\theta_{jn}$",
}

fig, axes = plt.subplots(6, 4, figsize=(15, 18), sharex=True)
summary_rows = []

for i, (m_key, ylab, base) in enumerate(metrics):
    for j, p in enumerate(FREE):
        ax = axes[i, j]
        col = f"{m_key}_{p}"
        valid = comp_df[["fp", "fp_err", col]].dropna()
        x, xe, y = (
            valid["fp"].values,
            valid["fp_err"].values,
            valid[col].values,
        )
        rho, pval = stats.spearmanr(x, y)
        summary_rows.append(
            dict(metric=m_key, param=p, N=len(valid), rho=rho, p_value=pval)
        )

        ax.errorbar(
            x,
            y,
            xerr=xe,
            fmt="o",
            ms=4,
            alpha=0.65,
            color="#4C72B0",
            elinewidth=0.8,
        )
        ax.axhline(base, color="crimson", ls="--", lw=1.0)
        if "kurt" in m_key or "skew" in m_key:
            lo, hi = np.percentile(y, [2, 98])
            span = max(hi - lo, 1e-3)
            ax.set_ylim(lo - 0.2 * span, hi + 0.25 * span)

        ax.text(
            0.05,
            0.92,
            rf"$\rho={rho:.2f}$ ($p={pval:.2f}$)",
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.8,
                edgecolor="none",
            ),
        )
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_title(PARAM_TITLES[p], fontsize=11, fontweight="bold")
        if j == 0:
            ax.set_ylabel(ylab, fontsize=8.5)
        if i == 5:
            ax.set_xlabel("fractional power (30-60 Hz)", fontsize=9)

plt.tight_layout()
plt.savefig("fp_vs_posterior_moments.png", dpi=150)
plt.close(fig)

pd.DataFrame(summary_rows).to_csv("fp_vs_moments_spearman.csv", index=False)
print("Saved fp_vs_posterior_moments.png and fp_vs_moments_spearman.csv")