#!/usr/bin/env python
"""
campaign_all.csv (collect_campaign.py --posterior で作ったもの) から、
イベントごとの統計量を作って、ノイズ条件の間で対応のある t 検定を行います。

調べる量 (1 回の PE ごとの値)
  z        : (事後中央値 - 真値) / 事後標準偏差
  abserr   : |事後中央値 - 真値|          (z の分子)
  post_std : 事後分布の標準偏差           (z の分母)
  post_skew: 事後分布の歪度 (3次)
  post_kurt: 事後分布の超過尖度 (4次, ガウスなら 0)

イベントごとに 30 窓ぶんの「平均」と「標準偏差」を作り、
  gauss -> gauss_est (PSD推定の効果)、gauss_est -> real (非ガウス性の効果)
を対応のある t 検定で比べます (イベント単位, 多重比較の補正なし)。
abserr と post_std は値の大きさがイベントで大きく変わるので、log を取ってから検定します。

出力
  hist_{量}_{param}.png       : イベントごとの平均・標準偏差のヒストグラム
  ttest_summary.csv           : 全ての検定結果
  vs_inj_chirp_mass.png       : 横軸 inj_chirp_mass, 縦 = 量, 横 = 推定パラメータ (4x4 等)

使い方
  python z_t_test.py
  python z_t_test.py --csv campaign_all.csv --nbins 6
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FREE = ["chirp_mass", "mass_ratio", "luminosity_distance", "theta_jn"]
NOISES = ["gauss", "gauss_est", "real"]
PAIRS = [("gauss", "gauss_est"), ("gauss_est", "real")]   # (前, 後)
QUANT = ["z", "abserr", "post_std", "post_skew", "post_kurt"]
LOGQ = {"abserr", "post_std"}        # log を取って比べる量
REF = {"z": (0.0, 1.0), "post_skew": (0.0, None), "post_kurt": (0.0, None)}   # (平均の理想値, 標準偏差の理想値)


def add_columns(df):
    """abserr 列を作る (z の分子)"""
    for k in FREE:
        df[f"abserr_{k}"] = (df[f"post_med_{k}"] - df[f"inj_{k}"]).abs()
    return df


def per_event(df, q, param):
    """イベント×ノイズ条件ごとに、30窓の平均と標準偏差を出す。
    戻り値: {noise: DataFrame(index=event, columns=[mean, std])}"""
    col = f"{q}_{param}"
    x = df[col]
    if q in LOGQ:
        x = np.log(x.clip(lower=1e-300))
    g = df.assign(_x=x).groupby(["noise", "event"])["_x"]
    agg = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1)})
    return {n: agg.loc[n] for n in NOISES if n in agg.index.get_level_values(0)}


def ttests(df, q, param):
    """per_event の平均・標準偏差について、PAIRS で対応のある t 検定"""
    d = per_event(df, q, param)
    rows = []
    for a, b in PAIRS:
        if a not in d or b not in d:
            continue
        ev = d[a].index.intersection(d[b].index)          # 両方にあるイベントだけ使う (対応をそろえる)
        for stat in ["mean", "std"]:
            xa, xb = d[a].loc[ev, stat].values, d[b].loc[ev, stat].values
            ok = np.isfinite(xa) & np.isfinite(xb)
            xa, xb = xa[ok], xb[ok]
            t = stats.ttest_rel(xa, xb)
            w = stats.wilcoxon(xa, xb) if len(xa) > 5 else None
            rows.append(dict(quantity=q + ("(log)" if q in LOGQ else ""), param=param,
                             stat=stat, pair=f"{a}->{b}", n_event=int(ok.sum()),
                             median_a=np.median(xa), median_b=np.median(xb),
                             t_p=t.pvalue, wilcoxon_p=(w.pvalue if w else np.nan)))
    return rows, d


def plot_hist(d, q, param):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, stat in zip(axes, ["mean", "std"]):
        allv = np.concatenate([d[n][stat].dropna().values for n in d])
        bins = np.linspace(np.percentile(allv, 1), np.percentile(allv, 99), 21)
        for n in d:
            ax.hist(d[n][stat].dropna(), bins=bins, alpha=0.5, lw=2, label=n)
        ref = REF.get(q, (None, None))[0 if stat == "mean" else 1]
        if ref is not None:
            ax.axvline(ref, color="k", ls="--")
        lab = f"log {q}" if q in LOGQ else q
        ax.set_title(f"{stat} of {lab}  ({param})", fontsize=9)
        ax.set_xlabel(f"per-event {stat}")
    axes[0].set_ylabel("number of events")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"hist_{q}_{param}.png", dpi=130)
    plt.close(fig)


def plot_vs_mc(df, nbins, fname="vs_inj_chirp_mass.png"):
    """横軸 inj_chirp_mass。列 = 推定パラメータ、行 = 量。点は各PE、線はビンごとの中央値。"""
    qs = ["z", "abserr", "post_std", "post_skew", "post_kurt"]
    mc = df["inj_chirp_mass"]
    edges = np.quantile(mc, np.linspace(0, 1, nbins + 1))   # 各ビンに同じ数の PE が入るように分ける
    centers = 0.5 * (edges[1:] + edges[:-1])
    fig, axes = plt.subplots(len(qs), len(FREE), figsize=(4.2 * len(FREE), 2.6 * len(qs)),
                             sharex=True, squeeze=False)
    for j, k in enumerate(FREE):
        for i, q in enumerate(qs):
            ax = axes[i][j]
            for c, n in enumerate(NOISES):
                d = df[df.noise == n]
                y = d[f"{q}_{k}"]
                ax.scatter(d["inj_chirp_mass"], y, s=2, alpha=0.08, color=f"C{c}")
                b = np.clip(np.digitize(d["inj_chirp_mass"], edges[1:-1]), 0, nbins - 1)
                if q == "z":     # z はビンごとの標準偏差 (1 が理想)
                    stat = [y[b == i_].std(ddof=1) for i_ in range(nbins)]
                    lab = "std(z) per bin"
                else:
                    stat = [y[b == i_].median() for i_ in range(nbins)]
                    lab = f"median {q}"
                ax.plot(centers, stat, "-o", color=f"C{c}", ms=3, lw=2, label=n)
            if q == "z":
                ax.set_ylim(-5, 5)
                ax.axhline(1, color="k", ls="--", lw=1)
                ax.axhline(0, color="gray", ls=":", lw=1)
            elif q in LOGQ:
                ax.set_yscale("log")
            else:
                ax.axhline(0, color="k", ls="--", lw=1)
                lo, hi = np.nanpercentile(df[f"{q}_{k}"], [1, 99])
                ax.set_ylim(lo, hi)
            if i == 0:
                ax.set_title(k)
            if j == 0:
                ax.set_ylabel(f"{q}\n(line: {lab})", fontsize=8)
            if i == len(qs) - 1:
                ax.set_xlabel("injected chirp mass")
    axes[0][0].legend(fontsize=8, markerscale=3)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="campaign_all.csv")
    ap.add_argument("--nbins", type=int, default=6, help="inj_chirp_mass のビン数 (等頻度)")
    a = ap.parse_args()
    df = add_columns(pd.read_csv(a.csv, encoding="utf-8"))
    need = [f"post_std_{k}" for k in FREE] + ["inj_chirp_mass"]
    miss = [c for c in need if c not in df]
    if miss:
        raise SystemExit(f"列がありません: {miss}\n"
                         "collect_campaign.py --posterior で campaign_all.csv を作り直してください")

    rows = []
    for q in QUANT:
        for k in FREE:
            r, d = ttests(df, q, k)
            rows += r
            plot_hist(d, q, k)
    t = pd.DataFrame(rows)
    t.to_csv("ttest_summary.csv", index=False)
    pd.set_option("display.width", 200)
    for q, tq in t.groupby("quantity", sort=False):
        print(f"\n=== {q} ===")
        print(tq.drop(columns="quantity").round(4).to_string(index=False))
    plot_vs_mc(df, a.nbins)
    print("\n-> ttest_summary.csv, hist_*.png, vs_inj_chirp_mass.png")


if __name__ == "__main__":
    main()