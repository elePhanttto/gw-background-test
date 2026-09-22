# 関数を入れているファイルです．消したら駄目だよ～
# this is a function file. Don't delete this!
# https://journals.aps.org/prd/abstract/10.1103/PhysRevD.108.063016 のテストを再現するヨ!
from gwpy.timeseries import TimeSeries
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from random import *
import math
import pymc as pm
from pymc import *
import arviz as az
from scipy.special import gammaln
import pytensor.tensor as pt
def half_t_mean(sigma, nu):
    """
    half Student-T (scale=sigma, df=nu) の平均。nu>1 で有限。
        mean = 2*sigma*sqrt(nu)*Gamma((nu+1)/2) / (sqrt(pi)*Gamma(nu/2)*(nu-1))
    sigma, nu は ndarray でも可(事後サンプルごとに一括計算するため)。
    """
    log = (np.log(2.0 * sigma) + 0.5 * np.log(nu) + gammaln((nu + 1) / 2)
           - 0.5 * np.log(np.pi) - gammaln(nu / 2) - np.log(nu - 1))
    return np.exp(log)

def half_t_mean_expr(sigma, nu):
    """half_t_mean のPyTensor式版(モデル内部で使う用)"""
    return (2*sigma*pt.sqrt(nu)*pt.exp(pt.gammaln((nu+1)/2))
            / (pt.sqrt(np.pi)*pt.exp(pt.gammaln(nu/2))*(nu-1)))

def run_test_band_qavg(Q,random_time,white,skipped,geocent_time,rate_of_mabiki,fmin,fmax,seg_duration):
    all_y = []
    # KS/AD検定と同じ、200個の離散的な2秒窓だけを対象にする
    # (min~maxの連続クロップだと、合体信号や未サンプルの区間まで
    #  混ざってしまい、KS検定とは別のデータを見ることになる)
    e_list, f_list, t_list = [], [], []
    for t0 in random_time:
        qg = white.crop(t0, t0 + seg_duration).q_gram(
            qrange=[Q, Q], frange=[fmin, fmax], snrthresh=0)
        e_list.append(np.asarray(qg["energy"]))
        f_list.append(np.asarray(qg["frequency"]))
        t_list.append(np.asarray(qg["time"]))

    e_tile = np.concatenate(e_list)
    f_tile = np.concatenate(f_list)
    t_tile = np.concatenate(t_list)

    order = np.argsort(t_tile)
    y = e_tile[order][::rate_of_mabiki]
    y = y * np.log(2.0)
    f_thinned = f_tile[order][::rate_of_mabiki]
    y_norm = y / y.mean()

    # ---- average tile power(周波数ごとの時間平均) ---- Claude製
    # 注意: y(GWpyのq_gramが返す正規化済みenergy)をそのまま使う。
    #       y_normのように全体平均で割ってしまうと、
    #       「セグメント全体が過剰ノイズを持つかどうか」という情報が
    #       消えてしまうため、ここでは使わない。
    # ガウスデータなら、各周波数の平均タイルパワーは理論上1になる(Fig.2a, Fig.3aの再現)
    unique_freqs = np.unique(f_thinned)
    avg_power_by_freq = np.array([
        y[f_thinned == f].mean() for f in unique_freqs
    ])
    std_power_by_freq = np.array([
        y[f_thinned == f].std() / np.sqrt((f_thinned == f).sum())  # 平均の標準誤差
        for f in unique_freqs
    ])

    return y_norm, e_tile, f_tile, t_tile, unique_freqs, avg_power_by_freq, std_power_by_freq, y, f_thinned

def plot_average_tile_power(unique_freqs, avg_power_by_freq, std_power_by_freq,
                              label="real", ax=None, color=None):
    """
    論文Fig.2a/Fig.3aと同じ形式:
    横軸=周波数、縦軸=平均タイルパワー。ガウスなら1にhlineが引ける。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(unique_freqs, avg_power_by_freq, yerr=std_power_by_freq,
                fmt='o-', ms=3, lw=1, label=label, color=color, alpha=0.7)
    ax.axhline(1.0, linestyle="--", color="gray", label="Gaussian expectation")
    ax.set_xlabel("f0 [Hz]")
    ax.set_ylabel("Average tile power")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax

def estimate_lam_robust(y):
    """
    ガウス成分(指数分布)のrate λ を、外れ値に強い形で推定する。

    【なぜ必要か】
    論文 式(3) の正規化は「ガウス成分の平均タイルパワー = 1」を要求している。
    ところが GWpy の norm='mean' は「データ全体(外れ値込み)の平均 = 1」に
    正規化するため、実データのように巨大な外れ値があると、
    ガウス成分のバルクは 1 よりずっと小さい位置に押し下げられる。
    (検証: 外れ値2%混入で、ガウス成分の平均が 0.29 まで下がった)
    この状態で lam_gauss=1.0 に固定すると、平均1の指数分布ではバルクが
    全く当てはまらず、モデルは「全部を非ガウス成分で説明する」方に倒れ、
    fractional power が 1 に張り付く。
    指数分布の中央値 = ln(2)/λ という関係を使い、外れ値の影響を受けにくい
    中央値から λ を推定することで、この問題を回避する。
    """
    return np.log(2.0) / np.median(y)


def frac_power(y, lam_gauss=None, threshold_ratio=1.5, penalty_k=1000.0,
               draws=1000, tune=1000, chains=4,
               target_accept=0.9, seed=42, progressbar=False,
               return_idata=False):
    y = np.asarray(y, dtype=float)
    # lam_gauss=None なら、データからロバストに推定する(推奨)
    if lam_gauss is None:
        lam_gauss = estimate_lam_robust(y)
    mu_gauss = 1.0 / lam_gauss
    # 識別性制約の閾値も、ガウス成分の平均に比例させる
    threshold = threshold_ratio * mu_gauss

    with pm.Model() as model:
        F = pm.Uniform("F", 0, 1)
        # sigmaの事前分布のスケールも mu_gauss に合わせる
        sigma = pm.HalfNormal("sigma", sigma=5.0 * mu_gauss)
        nu_raw = pm.Gamma("nu_raw", alpha=2.0, beta=0.1)
        nu = pm.Deterministic("nu", 1.0 + nu_raw)

        # 識別性制約(前述のコメント参照)
        mu_ng = half_t_mean_expr(sigma, nu)
        gap = pt.maximum(threshold - mu_ng, 0.0)
        pm.Potential("identifiability", -penalty_k * gap**2)

        w = pm.math.stack([1.0 - F, F])
        components = [
            pm.Exponential.dist(lam=lam_gauss),      # ← 推定したλを使う
            pm.HalfStudentT.dist(nu=nu, sigma=sigma),
        ]
        like = pm.Mixture("like", w=w, comp_dists=components, observed=y)
        idata = pm.sample(draws, tune=tune, chains=chains, cores=1,
                          target_accept=target_accept, random_seed=seed,
                          progressbar=progressbar,
                          initvals={"F": 0.1, "sigma": 3.0 * mu_gauss, "nu_raw": 5.0})

    post = idata.posterior
    Fs = post["F"].values.ravel()
    sgs = post["sigma"].values.ravel()
    nus = post["nu"].values.ravel()

    mu_nongauss = half_t_mean(sgs, nus)
    fp = (Fs * mu_nongauss) / ((1 - Fs) * mu_gauss + Fs * mu_nongauss)
    summ = az.summary(idata, var_names=["F", "sigma", "nu"])

    return_dict = dict(
        fp_mean=float(np.mean(fp)),
        fp_std=float(np.std(fp)),
        fp_lo=float(np.percentile(fp, 5.5)),
        fp_hi=float(np.percentile(fp, 94.5)),
        F_mean=float(np.mean(Fs)),
        sigma_mean=float(np.mean(sgs)),
        nu_mean=float(np.mean(nus)),
        lam_gauss=float(lam_gauss),   # ← 推定値も記録しておく
        r_hat_max=float(summ["r_hat"].max()),
        n_tiles=len(y),
    )
    if return_idata:
        return return_dict, idata
    return return_dict

def frac_power_by_freq(y,f_thinned,lam_gauss=None,min_tiles=200, **kw):
    y = np.asarray(y, dtype=float)
    f_thinned = np.asarray(f_thinned, dtype=float)
    freqs, fp_m, fp_l, fp_h = [], [], [], []
    for f in np.unique(f_thinned):
        mask = (f_thinned == f)
        if mask.sum() < min_tiles:
            print(f"  f={f:.1f}Hz: タイル数 {mask.sum()} が min_tiles={min_tiles} 未満のためスキップ")
            continue
        # 辞書型のやつが返ってくる
        r = frac_power(y[mask],lam_gauss=lam_gauss,**kw)
        if r["r_hat_max"] > 1.05:
            print(f"  f={f:.1f}Hz: r_hat={r['r_hat_max']:.3f} — 収束せず棄却")
            continue
        freqs.append(f)
        fp_m.append(r["fp_mean"])
        fp_l.append(r["fp_lo"])
        fp_h.append(r["fp_hi"])
        print(f"  f={f:.1f}Hz (N={mask.sum()}): FP={r['fp_mean']:.4f} "
              f"[{r['fp_lo']:.4f}-{r['fp_hi']:.4f}], r_hat={r['r_hat_max']:.3f}")
    return (np.array(freqs),np.array(fp_m),np.array(fp_l),np.array(fp_h))

def plot_fractional_power(BANDS,freqs, fp_mean, fp_lo, fp_hi,freqs_sim, fp_mean_sim, fp_lo_sim, fp_hi_sim, output,
                           axes=None, color=None):
    """論文 Fig.2b / Fig.3b と同じ形式のプロット。"""
    from matplotlib import pyplot as plt
    if axes is None:
        fig, axes = plt.subplots(1,2,figsize=(20, 5))
    for i,fmin,fmax in BANDS:
        yerr = np.vstack([fp_mean[i] - fp_lo[i], fp_hi[i] - fp_mean[i]])
        axes[0].errorbar(freqs[i], fp_mean[i], yerr=yerr, fmt='o-', ms=4, lw=1,
                label="real", color=color, alpha=0.8, capsize=3)
    axes[0].set_xlabel("f0 [Hz]")
    axes[0].set_ylabel("Fractional power of real data")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    for i ,fmin,fmax in BANDS:
        yerr_sim = np.vstack([fp_mean_sim[i]-fp_lo_sim[i],fp_hi_sim[i]-fp_mean_sim[i]])
        axes[1].errorbar(freqs_sim[i], fp_mean_sim[i], yerr=yerr_sim, fmt='o-', ms=4, lw=1,
                    label="sim", color=color, alpha=0.8, capsize=3)
    axes[1].set_xlabel("f0 [Hz]")
    axes[1].set_ylabel("Fractional power of sim data")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.savefig(output)