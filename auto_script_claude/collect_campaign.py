#!/usr/bin/env python
"""campaign/*/*/w*/summary.json を集計し、real と gauss のペア比較を
背景テストの非ガウス性指標と並べる。

使い方:
  python collect_campaign.py                 # summary.json から集計 (campaign_all.csv も更新)
  python collect_campaign.py --from-csv      # 既存の campaign_all.csv から集計だけやり直す
  python collect_campaign.py --x fp          # 横軸: fp (既定) / excess / dlogL
  python collect_campaign.py --y mad         # 縦軸: ratio (std比, 既定) / mad (MAD-std比) / dcov (cov90の差)
  python collect_campaign.py --ref gauss --cmp gauss_est   # 比べる2条件 (A=ref, B=cmp)

結果ファイルを約6900本読むので、数十分〜1時間程度かかるかもしれません。最初に python collect_campaign.py --posterior を1回だけ実行して campaign_all.csv を作り直しておけば、あとは --from-csv で一瞬で集計できます。
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
import os

def posterior_stats(outdir):
    """Bilby の結果ファイルから事後分布の統計量を計算する。見つからなければ空の dict を返す。"""
    cand = glob.glob(os.path.join(outdir, "pe_result.*"))
    if not cand:
        return {}
    import bilby
    post = bilby.result.read_in_result(cand[0]).posterior
    out = {}
    for k in FREE:
        x = post[k].values
        out[f"post_med_{k}"] = float(np.median(x))
        out[f"post_std_{k}"] = float(np.std(x))                  # 事後分布そのものの幅
        out[f"post_skew_{k}"] = float(stats.skew(x))             # 3次：左右非対称
        out[f"post_kurt_{k}"] = float(stats.kurtosis(x))         # 4次：裾の重さ（ガウスなら0）
    c = post[FREE].corr().values                                  # パラメータ間の相関係数
    for i in range(len(FREE)):
        for j in range(i + 1, len(FREE)):
            out[f"post_corr_{FREE[i]}__{FREE[j]}"] = float(c[i, j])
    return out


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


def load_runs(with_posterior=False):
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
        r.update({f"inj_{k}": v for k, v in s.get("injection", {}).items()})   # 注入した真の値
        r.update({f"deg_{k}": v for k, v in s["degeneracy"].items()})
        r["n_hit"] = s["contamination"]["n_hit"]
        r["n_windows"] = s["contamination"]["n_windows"]
        if with_posterior:
            r.update(posterior_stats(os.path.dirname(fn)))
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


MAD_K = 1.4826   # 正規分布なら MAD*1.4826 = std


def mad_std(z, axis=None):
    """外れ値に強い std (正規分布なら std と一致)"""
    z = np.asarray(z, float)
    med = np.median(z, axis=axis, keepdims=True)
    return MAD_K * np.median(np.abs(z - med), axis=axis)


def paired_bootstrap_mad(zA, zB, rng, n_boot=N_BOOT):
    """MAD-std(zB)/MAD-std(zA) の区間 (窓ペアごと再標本化)"""
    n = len(zA)
    j = rng.integers(0, n, size=(n_boot, n))
    r = mad_std(zB[j], axis=1) / mad_std(zA[j], axis=1)
    r = r[np.isfinite(r)]
    return np.percentile(r, [2.5, 16, 84, 97.5])


def paired_bootstrap_dcov(cA, cB, rng, n_boot=N_BOOT):
    """cov90(B) - cov90(A) の区間 (窓ペアごと再標本化)"""
    n = len(cA)
    j = rng.integers(0, n, size=(n_boot, n))
    d = cB[j].mean(axis=1) - cA[j].mean(axis=1)
    return np.percentile(d, [2.5, 16, 84, 97.5])


def summarize(df, ref="gauss", cmp="real"):
    rng = np.random.default_rng(0)
    out = []
    for ev, g in df.groupby("event"):
        A = g[g.noise == ref].set_index("window")
        B = g[g.noise == cmp].set_index("window")
        idx = A.index.intersection(B.index)
        if len(idx) < 5:
            continue
        A, B = A.loc[idx], B.loc[idx]
        bg = background_metric(ev) if fp_csv_path(ev) else {}
        hit = g.n_hit.iloc[0] / g.n_windows.iloc[0]
        for k in FREE:
            zA, zB = A[f"z_{k}"].values, B[f"z_{k}"].values
            q = paired_bootstrap_ratio(zA, zB, rng)
            # 外れ値に強い指標
            qm = paired_bootstrap_mad(zA, zB, rng)
            cA = A[f"in90_{k}"].values.astype(float)
            cB = B[f"in90_{k}"].values.astype(float)
            qc = paired_bootstrap_dcov(cA, cB, rng)
            pA, pB = A[f"p_{k}"].values, B[f"p_{k}"].values
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
                                 if len(idx) > 5 else np.nan),
                mad_ratio=mad_std(zB) / mad_std(zA),
                mad_lo95=qm[0], mad_lo68=qm[1], mad_hi68=qm[2], mad_hi95=qm[3],
                dcov90=cB.mean() - cA.mean(),
                dcov90_lo95=qc[0], dcov90_lo68=qc[1], dcov90_hi68=qc[2], dcov90_hi95=qc[3],
                ks_D_A=stats.kstest(pA, "uniform").statistic,
                ks_D_B=stats.kstest(pB, "uniform").statistic,
                ks2_p=stats.ks_2samp(pA, pB).pvalue,
                n_outlier_A=int(np.sum(np.abs(zA) > 4)),
                n_outlier_B=int(np.sum(np.abs(zB) > 4))))
    return pd.DataFrame(out)


YDEF = {   # 中央値, 68%下, 68%上, 95%下, 95%上, 基準線, ラベル
    "ratio": ("ratio", "ratio_lo68", "ratio_hi68", "ratio_lo95", "ratio_hi95", 1.0, "std(z) {B} / {A}"),
    "mad":   ("mad_ratio", "mad_lo68", "mad_hi68", "mad_lo95", "mad_hi95", 1.0, "MAD-std(z) {B} / {A}"),
    "dcov":  ("dcov90", "dcov90_lo68", "dcov90_hi68", "dcov90_lo95", "dcov90_hi95", 0.0,
              "cov90({B}) - cov90({A})"),
}


def plot(s, xkey, fname, ykey="ratio", ref="gauss", cmp="real"):
    yc, l68, h68, l95, h95, base, ylab = YDEF[ykey]
    fig, axes = plt.subplots(1, len(FREE), figsize=(4 * len(FREE), 3.6), sharex=True)
    for ax, k in zip(axes, FREE):
        sk = s[s.param == k]
        x, xe = sk[xkey].values, sk[f"{xkey}_err"].values
        y = sk[yc].values
        yerr = np.vstack([y - sk[l68].values, sk[h68].values - y])     # 68% (1σ相当)
        yerr = np.clip(yerr, 0, None)
        ax.errorbar(x, y, xerr=xe, yerr=yerr, fmt="o", capsize=3)
        ax.vlines(x, sk[l95], sk[h95], color="C0", alpha=0.3, lw=6)    # 95% は太い薄線
        if len(sk) <= 12:                                   # イベントが多いときはラベルを省略
            for xi, yi, e in zip(x, y, sk.event):
                ax.annotate(e, (xi, yi), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.axhline(base, color="gray", ls="--")
        ax.set_title(k)
        ax.set_xlabel(XLABEL[xkey], fontsize=8)
    axes[0].set_ylabel(ylab.format(A=ref, B=cmp) + "\n(bar: 68%, band: 95% bootstrap)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--x", choices=["fp", "excess", "dlogL"], default="fp")
    ap.add_argument("--y", choices=list(YDEF), default="ratio")
    ap.add_argument("--ref", default="gauss", help="基準 (A)")
    ap.add_argument("--cmp", default="real", help="比べる側 (B)")
    ap.add_argument("--posterior", action="store_true",
                    help="結果ファイルから事後分布の統計量も計算する（時間がかかる）")
    a = ap.parse_args()
    df = pd.read_csv("campaign_all.csv") if a.from_csv else load_runs(a.posterior)
    if not a.from_csv:
        df.to_csv("campaign_all.csv", index=False)
    s = summarize(df, a.ref, a.cmp)
    if s.empty:
        print(f"{a.cmp} と {a.ref} のペアが5組以上あるイベントがありません")
        return
    tag = "" if (a.ref, a.cmp) == ("gauss", "real") else f"_{a.cmp}_vs_{a.ref}"
    s.to_csv(f"campaign_summary{tag}.csv", index=False)
    yc, _, _, l95, h95, base, _ = YDEF[a.y]
    cols = ["event", "param", "N", a.x, f"{a.x}_err", yc, l95, h95,
            "cov90_A", "cov90_B", "n_outlier_A", "n_outlier_B", "frac_snr_lt8"]
    print(s[[c for c in cols if c in s]].round(3).to_string())
    if a.x == "fp" and "fp_rows_used" in s:
        print(s.groupby("event")[["fp_rows_used", "fp_rows_total"]].first()
              .rename(columns={"fp_rows_used": "fp に使った行", "fp_rows_total": "30-60Hz の行"}))
    # 全イベントまとめた要約 (外れ値の数と、指標が基準より悪い側にあるイベント数)
    print("\n--- パラメータごとの要約 ---")
    for k, sk in s.groupby("param", sort=False):
        worse = (sk[yc] > base).sum() if a.y != "dcov" else (sk[yc] < base).sum()
        print(f"{k:20s} 中央値 {yc}={sk[yc].median():.3f}  悪い側 {worse}/{len(sk)}  "
              f"|z|>4: {a.ref} {sk.n_outlier_A.sum()} / {a.cmp} {sk.n_outlier_B.sum()}")
    if a.x in s:
        plot(s, a.x, f"campaign_vs_background{tag}_{a.x}_{a.y}.png", a.y, a.ref, a.cmp)
    else:
        print(f"背景指標 {a.x} が無いので図は省略")


if __name__ == "__main__":
    main()