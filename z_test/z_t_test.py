#!/usr/bin/env python
"""
campaign_all.csv から、ノイズ条件の間 (gauss vs gauss_est, gauss_est vs real) で
イベントごとの「平均」と「標準偏差」を比べる対応のある t 検定を行う。

調べる量
  1. z_{param}              : (事後中央値 - 真値) / 事後std
  2. post_std_{param}       : 事後分布そのものの標準偏差
  3. post_skew_{param}      : 事後分布の歪度 (3次)
  4. post_kurt_{param}      : 事後分布の超過尖度 (4次, ガウスなら 0)
  0. err_{param}            : z の分子 = 事後中央値 - 真値 (post_med - inj から計算)
  5. post_corr_{p1}__{p2}   : 事後分布内のパラメータ間相関係数 (別CSV)
さらに、横軸 inj_chirp_mass、縦軸 z / post_std / post_skew / post_kurt の図を作る。

出力
  test_{param}.png              : z の平均・std のヒストグラム (従来どおり)
  ttest_summary.csv             : 1-4 の検定結果
  ttest_corr.csv                : 5 の検定結果
  vs_inj_chirp_mass.png         : 行 = 量, 列 = 推定パラメータ

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
MIN_PAIRS = 5
NOISES = ["gauss", "gauss_est", "real"]
PAIRS = [("gauss", "gauss_est"), ("gauss_est", "real")]   # (前, 後)
QUANTS = ["z", "post_std", "post_skew", "post_kurt"]          # 図の対象
TEST_QUANTS = ["z", "err", "post_std", "post_skew", "post_kurt"]   # 検定の対象
# err = 事後中央値 - 真値 (z の分子)。z = err / post_std なので
#   std(z) の比 ≈ std(err) の比 / post_std の比  で「中心のぶれ」と「幅」に分解できる


# ---------------------------------------------------------------- 集計
def per_event(df, col):
    """イベント × ノイズ条件ごとに col の平均と標準偏差を出す。
    返り値: 列が MultiIndex (統計量, ノイズ条件)、行がイベントの DataFrame"""
    g = df.groupby(["event", "noise"])[col]
    out = pd.concat({"mean": g.mean(), "std": g.std(ddof=1)}, axis=1)
    return out.unstack("noise")        # 行 = event, 列 = (mean/std, noise)


def paired_tests(table, quantity, param):
    """table = per_event の出力。PAIRS それぞれについて mean と std で t 検定する。
    両方の条件がそろっているイベントだけを使う (イベントの並びをそろえるため)。"""
    rows = []
    for a, b in PAIRS:
        for stat in ["mean", "std"]:
            x = table[(stat, a)]
            y = table[(stat, b)]
            ok = x.notna() & y.notna()
            x, y = x[ok].values, y[ok].values
            t = stats.ttest_rel(x, y)
            w = stats.wilcoxon(x, y) if len(x) > 5 else None   # 外れ値の確認用
            with np.errstate(divide="ignore", invalid="ignore"):
                r = y / x
            r = r[np.isfinite(r)]
            rows.append(dict(param=param, quantity=quantity, pair=f"{a}->{b}", stat=stat,
                             n_event=len(x), median_a=np.median(x), median_b=np.median(y),
                             ratio_med=np.median(r),        # イベントごとの b/a の中央値
                             t=t.statistic, p_t=t.pvalue,
                             p_wilcoxon=(w.pvalue if w is not None else np.nan)))
    return rows


# ---------------------------------------------------------------- 図
def plot_z_hist(table, param):
    """従来の z の平均・std のヒストグラム"""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    bins_m = np.linspace(-1, 1, 21)
    bins_s = np.linspace(0.4, 2.2, 37)
    for n in NOISES:
        if ("mean", n) not in table:
            continue
        axes[0].hist(table[("mean", n)].dropna(), bins=bins_m, histtype="step", lw=2, label=n)
        axes[1].hist(table[("std", n)].dropna(), bins=bins_s, histtype="step", lw=2, label=n)
    axes[0].axvline(0, color="k", ls="--"); axes[1].axvline(1, color="k", ls="--")
    axes[0].set_title(f"z_mean_{param}"); axes[1].set_title(f"z_std_{param}")
    axes[0].set_xlabel("per-event mean of z"); axes[1].set_xlabel("per-event std of z")
    axes[0].set_ylabel("number of events")
    axes[0].legend(); axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"test_{param}.png", dpi=150)
    plt.close(fig)


def binned_median(x, y, edges):
    """x のビンごとに y の中央値と、中央値の 68% ブートストラップ区間を返す"""
    rng = np.random.default_rng(0)
    c, med, lo, hi = [], [], [], []
    for l, r in zip(edges[:-1], edges[1:]):
        m = (x >= l) & (x < r)
        if m.sum() < 10:
            continue
        yy = y[m]
        boot = np.median(rng.choice(yy, size=(500, yy.size)), axis=1)
        c.append(np.median(x[m])); med.append(np.median(yy))
        lo.append(np.percentile(boot, 16)); hi.append(np.percentile(boot, 84))
    return map(np.array, (c, med, lo, hi))


def plot_vs_mc(df, nbins, fname="vs_inj_chirp_mass.png"):
    """行 = QUANTS, 列 = FREE。点は各ラン (薄く)、線はビンごとの中央値"""
    x_all = df["inj_chirp_mass"].values
    edges = np.quantile(x_all, np.linspace(0, 1, nbins + 1))   # 各ビンのラン数がそろうよう分位点で切る
    edges[-1] += 1e-9
    ref = {"z": None, "err": 0.0, "post_std": None, "post_skew": 0.0, "post_kurt": 0.0}
    fig, axes = plt.subplots(len(QUANTS), len(FREE), figsize=(4.2 * len(FREE), 3.0 * len(QUANTS)),
                             sharex=True)
    for j, p in enumerate(FREE):
        for i, q in enumerate(QUANTS):
            ax = axes[i, j]
            col = f"{q}_{p}"
            for c, n in enumerate(NOISES):
                d = df[df.noise == n]
                if d.empty:
                    continue
                x, y = d["inj_chirp_mass"].values, d[col].values
                ax.plot(x, y, ".", ms=1.5, alpha=0.15, color=f"C{c}")
                xc, med, lo, hi = binned_median(x, y, edges)
                ax.errorbar(xc, med, yerr=[med - lo, hi - med], fmt="o-", ms=4, capsize=2,
                            color=f"C{c}", label=n)
                if q == "z":   # z は中央値ではなく「ビンごとの std(z)」も重ねたいので別軸で表示
                    pass
            if ref[q] is not None:
                ax.axhline(ref[q], color="k", ls="--", lw=1)
            if q == "z":
                ax.axhline(0, color="k", ls="--", lw=1)
                ax.set_ylim(-4, 4)
            else:   # 外れ値で軸が伸びないように 1-99% で切る
                v = df[col].values
                ax.set_ylim(*np.nanpercentile(v, [1, 99]))
            if i == 0:
                ax.set_title(p)
            if j == 0:
                ax.set_ylabel(q)
            if i == len(QUANTS) - 1:
                ax.set_xlabel("injected chirp mass")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def plot_zstd_vs_mc(df, nbins, fname="zstd_vs_inj_chirp_mass.png"):
    """ビンごとの std(z) (誤差の過小評価の指標) を横軸 inj_chirp_mass で描く"""
    edges = np.quantile(df["inj_chirp_mass"], np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(FREE), figsize=(4.2 * len(FREE), 3.4), sharex=True)
    for ax, p in zip(axes, FREE):
        for c, n in enumerate(NOISES):
            d = df[df.noise == n]
            if d.empty:
                continue
            xc, s, lo, hi = [], [], [], []
            for l, r in zip(edges[:-1], edges[1:]):
                z = d.loc[(d.inj_chirp_mass >= l) & (d.inj_chirp_mass < r), f"z_{p}"].values
                if z.size < 10:
                    continue
                boot = rng.choice(z, size=(500, z.size)).std(axis=1, ddof=1)
                xc.append((l + r) / 2); s.append(z.std(ddof=1))
                lo.append(np.percentile(boot, 16)); hi.append(np.percentile(boot, 84))
            xc, s, lo, hi = map(np.array, (xc, s, lo, hi))
            ax.errorbar(xc, s, yerr=[s - lo, hi - s], fmt="o-", capsize=2, color=f"C{c}", label=n)
        ax.axhline(1, color="k", ls="--", lw=1)
        ax.set_title(p); ax.set_xlabel("injected chirp mass")
    axes[0].set_ylabel("std(z) in chirp-mass bin"); axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="campaign_all.csv")
    ap.add_argument("--nbins", type=int, default=6, help="inj_chirp_mass のビン数")
    a = ap.parse_args()
    df = pd.read_csv(a.csv, encoding="utf-8")
    # z の分子 (事後中央値 - 真値) を作る。単位はパラメータそのもの (例: chirp_mass なら太陽質量)
    for p in FREE:
        if f"post_med_{p}" in df and f"inj_{p}" in df:
            df[f"err_{p}"] = df[f"post_med_{p}"] - df[f"inj_{p}"]

    # 1-4: z, err, post_std, post_skew, post_kurt
    rows = []
    for p in FREE:
        for q in TEST_QUANTS:
            col = f"{q}_{p}"
            if col not in df:
                print(f"  列 {col} が無いのでスキップ (collect_campaign.py --posterior で作り直してください)")
                continue
            table = per_event(df, col)
            if q == "z":
                plot_z_hist(table, p)
            rows += paired_tests(table, q, p)
    t1 = pd.DataFrame(rows)
    # 統計量ごとにまとめて並べる (z → err → post_std → post_skew → post_kurt、その中でパラメータ順)
    t1["quantity"] = pd.Categorical(t1["quantity"], TEST_QUANTS, ordered=True)
    t1["param"] = pd.Categorical(t1["param"], FREE, ordered=True)
    t1 = t1.sort_values(["quantity", "param", "pair", "stat"]).reset_index(drop=True)

    # 5: 相関係数
    rows = []
    for col in [c for c in df.columns if c.startswith("post_corr_")]:
        p1, p2 = col[len("post_corr_"):].split("__")      # 例: chirp_mass, mass_ratio
        r = paired_tests(per_event(df, col), "post_corr", p1)
        for x in r:
            x["param2"] = p2
        rows += r
    t2 = pd.DataFrame(rows)
    if not t2.empty:   # 列の並びを param, param2, ... にそろえる
        t2 = t2[["param", "param2"] + [c for c in t2.columns if c not in ("param", "param2")]]

    # std(z) の比の分解 (同じ窓 = 同じ注入値どうしをペアにする)
    # root mean square = rms
    #   z_a = err_a/σ_a, z_b = err_b/σ_b のとき、c = err_b/σ_a (b の中心のぶれを a の幅で測る) を作ると
    #   rms(z_b)/rms(z_a) = [rms(c)/rms(z_a)] × [rms(z_b)/rms(c)]
    #                     = 「中心のぶれの変化」 × 「幅の変化の効果」  (掛け算で厳密に一致)
    #   幅の効果 > 1 なら b の事後分布が細くなって z が大きくなった (過信)
    #   std(err) をそのまま比べると、注入値ごとに err の大きさが違うのが混ざるので、σ_a で規格化している
    dec = []
    for p in FREE:
        for a_, b_ in PAIRS:
            rows_ev = []
            for ev, g in df.groupby("event"):
                A = g[g.noise == a_].set_index("window")
                B = g[g.noise == b_].set_index("window")
                idx = A.index.intersection(B.index)
                if len(idx) < MIN_PAIRS:
                    continue
                za = A.loc[idx, f"z_{p}"].values
                zb = B.loc[idx, f"z_{p}"].values
                c = B.loc[idx, f"err_{p}"].values / A.loc[idx, f"post_std_{p}"].values
                rms = lambda v: np.sqrt(np.mean(v ** 2))
                rows_ev.append((rms(zb) / rms(za), rms(c) / rms(za), rms(zb) / rms(c)))
            r = np.array(rows_ev)
            dec.append(dict(param=p, pair=f"{a_}->{b_}", n_event=len(r),
                            zrms_ratio=np.median(r[:, 0]),       # z のばらつきの比 (全体)
                            center_factor=np.median(r[:, 1]),    # 中心のぶれの変化
                            width_factor=np.median(r[:, 2]),     # 幅の変化の効果 (>1 = 細くなった)
                            p_center=stats.wilcoxon(np.log(r[:, 1])).pvalue,
                            p_width=stats.wilcoxon(np.log(r[:, 2])).pvalue))
    t3 = pd.DataFrame(dec)

    pd.set_option("display.width", 200)
    print("=== z / err / post_std / post_skew / post_kurt (対応のある t 検定, イベント単位, 補正なし) ===")
    print(t1.round(4).to_string(index=False))
    print("\n=== 事後分布内のパラメータ間相関 ===")
    print(t2.round(4).to_string(index=False))
    print("\n=== z のばらつきの比の分解 (イベントごとの中央値; zrms = center × width) ===")
    print(t3.round(4).to_string(index=False))
    t1.to_csv("ttest_summary.csv", index=False)
    t3.to_csv("ttest_decompose.csv", index=False)
    t2.to_csv("ttest_corr.csv", index=False)

    # 図: inj_chirp_mass との関係
    if "inj_chirp_mass" in df:
        plot_vs_mc(df, a.nbins)
        plot_zstd_vs_mc(df, a.nbins)
        print("\n-> vs_inj_chirp_mass.png, zstd_vs_inj_chirp_mass.png")
    else:
        print("inj_chirp_mass の列が無いので図は省略")
    print("-> test_{param}.png, ttest_summary.csv, ttest_corr.csv, ttest_decompose.csv")


if __name__ == "__main__":
    main()