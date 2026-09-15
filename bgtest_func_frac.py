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

def run_test_band_qavg(Q,random_time,white,skipped,geocent_time,rate_of_mabiki,fmin,fmax,seg_duration):
    all_y = []
    random_time_list = []
    seg = white.crop(min(random_time)+1.0,max(random_time)) #切り出し
    qgram_H1 = seg.q_gram(qrange=[Q,Q], frange=[fmin, fmax], snrthresh=0, norm='mean') #q-gramでq-transform
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

    return y_norm, e_tile, f_tile, t_tile, unique_freqs, avg_power_by_freq, std_power_by_freq

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