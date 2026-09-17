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

def half_t_mean(sigma, nu):
    """
    half Student-T (scale=sigma, df=nu) の平均。nu>1 で有限。
        mean = 2*sigma*sqrt(nu)*Gamma((nu+1)/2) / (sqrt(pi)*Gamma(nu/2)*(nu-1))
    sigma, nu は ndarray でも可(事後サンプルごとに一括計算するため)。
    """
    log = (np.log(2.0 * sigma) + 0.5 * np.log(nu) + gammaln((nu + 1) / 2)
           - 0.5 * np.log(np.pi) - gammaln(nu / 2) - np.log(nu - 1))
    return np.exp(log)

import pytensor.tensor as pt

def half_t_mean_expr(sigma, nu):
    """half_t_mean のPyTensor式版(モデル内部で使う用)"""
    return (2*sigma*pt.sqrt(nu)*pt.exp(pt.gammaln((nu+1)/2))
            / (pt.sqrt(np.pi)*pt.exp(pt.gammaln(nu/2))*(nu-1)))

def run_test_band_qavg(Q,random_time,white,skipped,geocent_time,rate_of_mabiki,fmin,fmax,seg_duration):
    all_y = []
    random_time_list = []
    seg = white.crop(min(random_time)+1.0,max(random_time)) #切り出し
    qgram_H1 = seg.q_gram(qrange=[Q,Q], frange=[fmin, fmax], snrthresh=0, norm='mean') #q-gramでq-transform．正規化をmeanにすることで，元ネタ論文と揃える
    y_H1 = np.asarray(qgram_H1["energy"]) #エネルギーの部分を取り出す
    # 時間順にソートしてから等間隔間引き
    t_tile = np.asarray(qgram_H1["time"])
    f_tile = np.asarray(qgram_H1["frequency"])
    e_tile = np.asarray(qgram_H1["energy"])
    order = np.argsort(t_tile)
    y = e_tile[order][::rate_of_mabiki]  # 間引く(時間方向に相関があるため)
    f_thinned = f_tile[order][::rate_of_mabiki]  # yと対応する周波数も同様に間引く
    y_norm = y / y.mean()
    all_y.append(y_norm)

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

def frac_power(y,lam_gauss=1.0, draws=1000, tune=1000, chains=2,
                          target_accept=0.9, seed=42, progressbar=False,
                          return_idata=False):
    with pm.Model() as model:
        #パラメータの事前分布
        F = pm.Uniform("F",0,1)
        sigma = pm.HalfNormal("sigma",sigma=5.0)
        nu_raw = pm.Gamma("nu_raw",alpha=2.0,beta=0.1)
        nu = pm.Deterministic("nu",1.0+nu_raw)
        # 識別性制約(重要): half Student-Tは、sigma,nuの選び方次第で
        # ガウス成分(Exponential(1), 平均1)とほぼ同じ形になれてしまう。
        # そのままだと「F≈1(全部を非ガウス側と誤判定)、sigma,nuで指数分布を
        # 模倣する」という偽の解と、正しい解(F≈0)の尤度が拮抗し、
        # データの非ガウス性がわずかなときほどサンプラーが偽の解に
        # 転がりやすい(fractional powerが1近くに張り付くバグの原因)。
        # 「非ガウス成分はガウス成分より平均パワーが大きいはず」という
        # 物理的に自然な制約を、滑らかなペナルティで課すことで解消する。
        mu_ng = half_t_mean_expr(sigma, nu)
        pm.Potential("identifiability",
                     -10.0 * pt.log1pexp(10.0 * (1.0 - mu_ng)))

        #重みF
        w = pm.math.stack([1.0-F,F])
        components = [
            pm.Exponential.dist(lam = 1.0),
            pm.HalfStudentT.dist(nu=nu,sigma=sigma),
        ]
        like = pm.Mixture("like",w=w,comp_dists=components,observed=y)
        idata = pm.sample(draws, tune=tune, chains=chains, cores=1,
                          target_accept=target_accept, random_seed=seed,
                          progressbar=progressbar)
    post = idata.posterior
    #事後分布
    Fs = post["F"].values.ravel()
    sgs = post["sigma"].values.ravel()
    nus = post["nu"].values.ravel()

    # 各成分の平均
    mu_gauss = 1.0 / lam_gauss
    mu_nongauss = half_t_mean(sgs,nus)
    #fractional power!
    fp = (Fs * mu_nongauss)/((1-Fs)*mu_gauss + Fs*mu_nongauss)
    summ = az.summary(idata, var_names=["F", "sigma", "nu"])

    return_dict = dict(
        fp_mean=float(np.mean(fp)), #平均
        fp_std=float(np.std(fp)),
        fp_lo=float(np.percentile(fp, 5.5)),    # 89% 信用区間
        fp_hi=float(np.percentile(fp, 94.5)),
        F_mean=float(np.mean(Fs)),
        sigma_mean=float(np.mean(sgs)),
        nu_mean=float(np.mean(nus)),
        r_hat_max=float(summ["r_hat"].max()),   # 1.01 を超えたら収束を疑う
        n_tiles=len(y),
    )
    if return_idata:
        return return_dict, idata
    return return_dict

def frac_power_by_freq(y,f_thinned,lam_gauss=1.0,min_tiles=200, **kw):
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