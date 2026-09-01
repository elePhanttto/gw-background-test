# 関数を入れているファイルです．消したら駄目だよ～
# this is a function file. Don't delete this!
from gwpy.timeseries import TimeSeries
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from random import *
import math

def acf_all_rows(qgram, max_lag=20):
    """qgramの全周波数行についてACFを計算し、{周波数: acf配列}の辞書を返す"""
    t = np.asarray(qgram["time"]) #時間
    f = np.asarray(qgram["frequency"]) #周波数
    e = np.asarray(qgram["energy"]) #エネルギー
    freqs = np.unique(f)

    acf_dict = {}
    for f_target in freqs:
        m = (f == f_target) #ターゲットの周波数
        row = e[m][np.argsort(t[m])] #各周波数のeを時刻順に並べている
        n = len(row) #タイルエネルギーの数
        # タイル数が少なすぎる行はACFが不安定なのでスキップ
        if n < 8:
            continue
        row = row - row.mean() #偏差
        ft = np.fft.rfft(row, n=2 * n) #フーリエ変換
        acf = np.fft.irfft(ft * np.conj(ft))[:n // 4] #共役複素数をかけて逆フーリエ変換
        acf = acf / acf[0] #自己相関係数を求める
        L = min(max_lag, len(acf))
        acf_dict[f_target] = acf[:L]
    return acf_dict

# 平均ACF（長さが揃う範囲だけ）
def mean_acf(acf_dict, max_lag=20):
    arrs = [a for a in acf_dict.values() if len(a) == max_lag]
    return np.mean(arrs, axis=0) if arrs else None

# ============================================
# AD検定の実装（論文式3.4準拠）
# ============================================

def ad_statistic(y_norm):
    """A^2 統計量。帰無仮説 y ~ Exp(1)"""
    y = np.sort(np.asarray(y_norm, dtype=float))
    N = len(y)
    # F(y) = 1 - exp(-y) なので
    #   log F     = log1p(-exp(-y))   （y→0 でも精度を保つ）
    #   log(1-F)  = -y                （厳密、オーバーフローしない）
    log_F   = np.log1p(-np.exp(-y))
    log_1mF = -y
    i = np.arange(1, N + 1)
    return -N - np.sum((2*i - 1) / N * (log_F + log_1mF[::-1]))


def build_ad_null_distribution(N, n_sim=1e+6, seed=42):
    """
    帰無仮説下でのA^2分布をモンテカルロで構築する。
    N: サンプル数（タイル数）
    n_sim: シミュレーション回数
    """
    rng = np.random.default_rng(seed)
    A2_sim = np.empty(n_sim)
    for k in range(n_sim):
        y_sim = rng.exponential(scale=1.0, size=N)
        y_sim /= y_sim.mean()      # 実データと同じ正規化を適用
        A2_sim[k] = ad_statistic(y_sim)
    return np.sort(A2_sim)


def ad_pvalue(A2_obs, A2_null):
    """経験分布からp値を計算（右側確率）"""
    n_ge = np.sum(A2_null >= A2_obs)
    if n_ge == 0:
        return 1.0 / len(A2_null)   # 下限値を返す
    return n_ge / len(A2_null)

# chi2乗検定のための関数

def equiprob_bins(n_bins=9):
    # 指数分布(scale=1)で等確率になるビン境界
    return -np.log(1 - np.arange(1, n_bins) / n_bins)

def Y_distribution_equiprob(y_norm, n_bins=9): #等確率になるビン境界での分布関数
    edges = np.concatenate([[0], equiprob_bins(n_bins), [np.inf]])
    obs, _ = np.histogram(y_norm, bins=edges)
    exp = np.full(n_bins, len(y_norm) / n_bins)
    return obs, exp

def kai2jou(f_obs,f_exp): #カイ二乗検定の統計量を直接計算
    goukei = 0.0
    for i in range(0,len(f_exp)):
        goukei += (f_obs[i] - f_exp[i])**2 / f_exp[i]
    return goukei

# シミュレーションデータを使って間引き率を決める
# threads(閾値)は0.1としているが，状況に応じて変えると良い
def calibrate_thinning(white_ts, times,seg_duration, n_seg=20, max_lag=20, threshold=0.1,
                       qkw=dict(qrange=[8,8], frange=[30.0,500.0], snrthresh=0)):
    """複数セグメントのACFを平均して間引き率を決める"""
    acfs = []
    for t in times[:n_seg]:
        qg = white_ts.crop(t, t + seg_duration).q_gram(**qkw)
        d = acf_all_rows(qg, max_lag=max_lag)
        m = mean_acf(d, max_lag=max_lag)
        if m is not None:
            acfs.append(m)
    mean_all = np.mean(acfs, axis=0)
    print(f"平均ACF (n_seg={len(acfs)}): {np.round(mean_all[:8], 3)}")

    rate = 1
    for i in range(1, len(mean_all)):
        if mean_all[i] < threshold:
            rate = i
            break
    else:
        rate = len(mean_all) - 1
    return max(rate, 1), mean_all

#帯域ごとに間引き率を変える
def calibrate_thinning_band(white_ts, times,fmin,fmax,seg_duration,n_seg=20, max_lag=20, threshold=0.1):
    """複数セグメントのACFを平均して間引き率を決める"""
    acfs = []
    qkw=dict(qrange=[8,8], frange=[fmin,fmax], snrthresh=0)
    for t in times[:n_seg]:
        qg = white_ts.crop(t, t + seg_duration).q_gram(**qkw)
        d = acf_all_rows(qg, max_lag=max_lag)
        m = mean_acf(d, max_lag=max_lag)
        if m is not None:
            acfs.append(m)
    mean_all = np.mean(acfs, axis=0)
    print(f"平均ACF (n_seg={len(acfs)}): {np.round(mean_all[:8], 3)}")

    rate = 1
    for i in range(1, len(mean_all)):
        if mean_all[i] < threshold:
            rate = i
            break
    else:
        rate = len(mean_all) - 1
    return max(rate, 1), mean_all

# 背景テストを行う関数
def run_test(random_time,white,skipped,geocent_time,rate_of_mabiki,A2_null,seg_duration):
    kslist = []
    chi2list = []
    adlist = []
    random_time_list = []
    a2list  = []
    all_y = []
    for j in range(len(random_time)):
        seg = white.crop(random_time[j],random_time[j] + seg_duration) #切り出し
        qgram_H1 = seg.q_gram(qrange=[8, 8], frange=[30.0, 500.0], snrthresh=0) #q-gramでq-transform
        y_H1 = np.asarray(qgram_H1["energy"]) #エネルギーの部分を取り出す
        # 時間順にソートしてから等間隔間引き
        t_tile = np.asarray(qgram_H1["time"])
        e_tile = np.asarray(qgram_H1["energy"])
        order = np.argsort(t_tile)
        y = e_tile[order][::rate_of_mabiki]  # 間引く(時間方向に相関があるため)
        y_norm = y / y.mean()
        all_y.append(y_norm)
        print(f"タイル数: {len(y_norm)}")
        obs, exp = Y_distribution_equiprob(y_norm,n_bins=9)
        # ---KS検定---
        ks = stats.kstest(y_norm, "expon")
        chi2_stat, chi2_p_value = stats.chisquare(f_obs=obs,f_exp=exp)
        # ---- AD検定 ----
        A2_obs = ad_statistic(y_norm)
        ad_p = ad_pvalue(A2_obs, A2_null)
        result = stats.anderson(y_norm, dist='expon')
        FLOOR = 1.0e-8
        kslist.append(max(ks.pvalue, FLOOR))
        chi2list.append(max(chi2_p_value, FLOOR))
        adlist.append(max(ad_p, FLOOR))
        random_time_list.append(random_time[j])
        a2list.append(A2_obs)
        print(f"t={random_time[j]-geocent_time:+8.1f}s  "
          f"KS p={ks.pvalue:.2e}  chi2 p={chi2_p_value:.2e}  "
          f"AD A2={A2_obs:.2f} p={ad_p:.2e}")
    print(f"NaNでスキップしたセグメント: {skipped}")
    return kslist,chi2list,adlist,random_time_list,a2list,all_y,qgram_H1,y_norm

# 背景テストを帯域別に行う関数
def run_test_band(random_time,white,skipped,geocent_time,rate_of_mabiki,A2_null,fmin,fmax,seg_duration):
    kslist = []
    chi2list = []
    adlist = []
    random_time_list = []
    a2list  = []
    all_y = []
    for j in range(len(random_time)):
        seg = white.crop(random_time[j],random_time[j] + seg_duration) #切り出し
        qgram_H1 = seg.q_gram(qrange=[8, 8], frange=[fmin, fmax], snrthresh=0) #q-gramでq-transform
        y_H1 = np.asarray(qgram_H1["energy"]) #エネルギーの部分を取り出す
        # 時間順にソートしてから等間隔間引き
        t_tile = np.asarray(qgram_H1["time"])
        e_tile = np.asarray(qgram_H1["energy"])
        order = np.argsort(t_tile)
        y = e_tile[order][::rate_of_mabiki]  # 間引く(時間方向に相関があるため)
        y_norm = y / y.mean()
        all_y.append(y_norm)
        print(f"タイル数: {len(y_norm)}")
        obs, exp = Y_distribution_equiprob(y_norm,n_bins=9)
        # ---KS検定---
        ks = stats.kstest(y_norm, "expon")
        chi2_stat, chi2_p_value = stats.chisquare(f_obs=obs,f_exp=exp)
        # ---- AD検定 ----
        A2_obs = ad_statistic(y_norm)
        ad_p = ad_pvalue(A2_obs, A2_null)
        result = stats.anderson(y_norm, dist='expon')
        FLOOR = 1.0e-8
        kslist.append(max(ks.pvalue, FLOOR))
        chi2list.append(max(chi2_p_value, FLOOR))
        adlist.append(max(ad_p, FLOOR))
        random_time_list.append(random_time[j])
        a2list.append(A2_obs)
        print(f"t={random_time[j]-geocent_time:+8.1f}s  "
          f"KS p={ks.pvalue:.2e}  chi2 p={chi2_p_value:.2e}  "
          f"AD A2={A2_obs:.2f} p={ad_p:.2e}")
    print(f"NaNでスキップしたセグメント: {skipped}")
    return kslist,chi2list,adlist,random_time_list,a2list,all_y,qgram_H1,y_norm

#プロットしまくる
def plot(kslist,chi2list,adlist,random_time_list,a2list,all_y,qgram_H1,y_norm,kslist_sim,chi2list_sim,adlist_sim,random_time_list_sim,a2list_sim,all_y_sim,qgram_H1_sim,y_norm_sim,geocent_time,event,det,A2_null,output1,output2,output3,output4,output5,output6,output7,output8,output9):
    #タイルのエネルギーを見る

    freqs = np.asarray(qgram_H1["frequency"])
    energies = np.asarray(qgram_H1["energy"])
    plt.scatter(freqs, energies / energies.mean(), s=2, alpha=0.3)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("frequency [Hz]"); plt.ylabel("normalized energy")
    plt.savefig(output7)

    # ==========================================
    # p-value のプロット
    # ==========================================

    # geocent_time からの時間差 [s]
    time_offsets = np.array(random_time_list) - geocent_time

    ks_pvalues = np.array(kslist)
    chi2_pvalues = np.array(chi2list)
    ks_pvalues_sim = np.array(kslist_sim)
    chi2_pvalues_sim = np.array(chi2list_sim)

    # ------------------------------------------
    # KS test
    # ------------------------------------------

    fig, axes = plt.subplots(1,2,figsize=(20,5))

    axes[0].scatter(time_offsets, ks_pvalues, s=15)

    axes[0].axhline(1e-8, linestyle="--", label="p = 1e-8")
    axes[0].set_xlabel(f"Time from {event} geocent time [s]")
    axes[0].set_ylabel("KS test p-value")
    axes[0].set_title("KS test p-value vs time")
    axes[0].set_yscale('log')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].scatter(time_offsets, ks_pvalues_sim, s=15)

    axes[1].axhline(1e-8, linestyle="--", label="p = 1e-8")

    axes[1].set_xlabel("Time from {event} geocent time [s]")
    axes[1].set_ylabel("KS test p-value of gaussian noise")
    axes[1].set_title("KS test p-value vs time of gaussian noise")
    axes[1].set_yscale('log')
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output1)


    # ------------------------------------------
    # Chi-square test
    # ------------------------------------------

    fig, axes = plt.subplots(1,2,figsize=(20,5))

    axes[0].scatter(time_offsets, chi2_pvalues, s=15)

    axes[0].axhline(1e-8, linestyle="--", label="p = 1e-8")

    axes[0].set_xlabel("Time from {event} geocent time [s]")
    axes[0].set_ylabel(r"$\chi^2$ test p-value")
    axes[0].set_title(r"$\chi^2$ test p-value vs time")
    axes[0].set_yscale('log')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].scatter(time_offsets, chi2_pvalues_sim, s=15)

    axes[1].axhline(1e-8, linestyle="--", label="p = 1e-8")

    axes[1].set_xlabel("Time from {event} geocent time [s]")
    axes[1].set_ylabel(r"$\chi^2$ test p-value")
    axes[1].set_title(r"$\chi^2$ test p-value vs time of gaussian noise")
    axes[1].set_yscale('log')
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output2)

    # adtestのプロット!

    ad_pvalues = np.array(adlist)
    ad_pvalues_sim = np.array(adlist_sim)

    fig, axes = plt.subplots(1,2,figsize=(20,5))
    axes[0].scatter(time_offsets, ad_pvalues, s=15)
    axes[0].axhline(1e-6, linestyle="--", label="p = 1e-6")
    axes[0].set_xlabel("Time from {event} geocent time [s]")
    axes[0].set_ylabel("AD test p-value")
    axes[0].set_title("Anderson-Darling test p-value vs time")
    axes[0].set_yscale('log')
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].scatter(time_offsets, ad_pvalues_sim, s=15)
    axes[1].axhline(1e-6, linestyle="--", label="p = 1e-6")
    axes[1].set_xlabel("Time from {event} geocent time [s]")
    axes[1].set_ylabel("AD test p-value")
    axes[1].set_title("Anderson-Darling test p-value vs time of gaussian noise")
    axes[1].set_yscale('log')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(output3)

    # histgram

    fig, axes = plt.subplots(figsize=(10,5))

    p_sorted = np.sort(ks_pvalues)
    p_sorted_sim = np.sort(ks_pvalues_sim)
    n = len(p_sorted)
    expected = (np.arange(1, n+1) - 0.5) / n
    axes.plot(expected, p_sorted, 'o', ms=3,label="real")
    axes.plot(expected, p_sorted_sim, 'o',color="orange", ms=3,label="sim")
    axes.plot([0,1], [0,1], 'r--', lw=1)     # 一様なら45度線に乗る
    axes.set_xlabel("expected quantile")
    axes.set_ylabel("observed p-value")
    axes.set_xlim(0,1)
    axes.set_ylim(0,1)
    axes.legend()
    plt.gca().set_aspect('equal')
    fig.savefig(output4)

    # 分布チェック!
    fig, axes = plt.subplots(1,2,figsize=(20,5))
    axes[0].hist(y_norm, bins=60, density=True, alpha=0.6, label="observed")
    x = np.linspace(0, 10, 200)
    axes[0].plot(x, np.exp(-x), 'r-', lw=2, label=r"$e^{-y}$")
    axes[0].set_yscale('log')
    axes[0].set_xlabel("normalized energy")
    axes[0].legend()
    axes[1].hist(y_norm_sim, bins=60, density=True, alpha=0.6, label="observed")
    x = np.linspace(0, 10, 200)
    axes[1].plot(x, np.exp(-x), 'r-', lw=2, label=r"$e^{-y}$")
    axes[1].set_yscale('log')
    axes[1].set_xlabel("normalized energy")
    axes[1].legend()
    fig.savefig(output5)

    # A^2自体をプロットしてもらう

    fig, axes = plt.subplots(1,2,figsize=(20,5))
    axes[0].scatter(time_offsets, a2list, s=15)
    axes[0].axhline(np.percentile(A2_null, 99), ls='--', c='r', label='99% of null')
    axes[0].axhline(np.median(A2_null), ls=':', c='gray', label='null median')
    axes[0].set_xlabel("Time from {event} geocent time [s]")
    axes[0].set_ylabel(r"$A^2$ statistic")
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[1].scatter(time_offsets, a2list_sim, s=15)
    axes[1].axhline(np.percentile(A2_null, 99), ls='--', c='r', label='99% of null')
    axes[1].axhline(np.median(A2_null), ls=':', c='gray', label='null median')
    axes[1].set_xlabel("Time from {event} geocent time [s]")
    axes[1].set_ylabel(r"$A^2$ statistic")
    axes[1].set_yscale('log')
    axes[1].legend()

    fig.savefig(output8)

    with open(output6,mode = "w",encoding='utf-8') as t:
        t.write("裾の部分の過剰な部分を見てみる \n")
    #裾の部分の過剰な部分を見てみる
        y_all = np.concatenate(all_y)  # 全部まとめて
        for thr in [4,5,6,7]:
            obs = (y_all > thr).mean()
            t.write(f"y>{thr}: ratio={obs/np.exp(-thr):.2f} \n")

    t.close()

    with open(output9,mode = "w",encoding='utf-8') as t:
        t.write("simデータについて裾の部分の過剰な部分を見てみる \n")
    #裾の部分の過剰な部分を見てみる
        y_all = np.concatenate(all_y_sim)  # 全部まとめて
        for thr in [4,5,6,7]:
            obs = (y_all > thr).mean()
            t.write(f"y>{thr}: ratio={obs/np.exp(-thr):.2f} \n")

    t.close()

    # CSVファイル出力
    df = pd.DataFrame({
        "time_offset": np.array(random_time_list) - geocent_time,
        "ks_p": kslist, "chi2_p": chi2list, "ad_p": adlist, "a2": a2list
    })
    df.to_csv(f"{event}_{det}_results.csv", index=False)

    df = pd.DataFrame({
        "time_offset": np.array(random_time_list) - geocent_time,
        "ks_p": kslist_sim, "chi2_p": chi2list_sim, "ad_p": adlist_sim, "a2": a2list_sim
        })
    df.to_csv(f"{event}_{det}_results_sim.csv", index=False)

    # 異常セグメントの抽出
    bad = df[(df.ks_p < 1e-3) | (df.chi2_p < 1e-3) | (df.ad_p < 1e-4)]
    print(bad.sort_values("ks_p"))

#PSDを反映した合成ガウスノイズを作る(ChatGPT製)

def generate_PSD_gauss(data):
    # --- 実データのPSDを推定 ---
    psd = data.psd(fftlength=8)

    # --- PSDに従う定常ガウスノイズを生成 ---
    N = len(data)
    fs = float(data.sample_rate.value)

    # FFTの周波数
    freq = np.fft.rfftfreq(N, d=1/fs)

    # 実データPSDを補間
    psd_interp = np.interp(
        freq,
        psd.frequencies.value,
        psd.value,)

    # 再現可能な乱数生成器
    rng = np.random.default_rng(0)

    # 周波数領域で複素ガウス乱数を生成
    re = rng.normal(size=len(freq))
    im = rng.normal(size=len(freq))

    # PSDに対応する振幅を与える
    spectrum = (re + 1j * im) * np.sqrt(psd_interp * fs * N / 4)

    # DC成分・Nyquist成分は実数にする
    spectrum[0] = rng.normal() * np.sqrt(psd_interp[0] * fs * N / 2)

    if N % 2 == 0:
        spectrum[-1] = rng.normal() * np.sqrt(psd_interp[-1] * fs * N / 2)

    # 時間領域へ
    sim_array = np.fft.irfft(spectrum, n=N)

    # GWPy TimeSeriesにする
    sim = TimeSeries(
        sim_array,
        sample_rate=data.sample_rate,
        t0=data.t0,
    )
    return sim