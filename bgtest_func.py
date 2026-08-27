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
    """
    正規化エネルギーに対するAnderson-Darling統計量を計算する。
    帰無仮説: y ~ Exp(1) （F(y) = 1 - exp(-y)）
    """
    y = np.sort(np.asarray(y_norm))
    N = len(y)
    F = 1.0 - np.exp(-y)          # 指数分布(scale=1)のCDF
    
    # log(0)を避けるためのクリップ
    eps = 1e-300
    F = np.clip(F, eps, 1.0 - eps)
    
    i = np.arange(1, N + 1)
    A2 = -N - np.sum((2*i - 1) / N * (np.log(F) + np.log(1.0 - F[::-1])))
    return A2


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