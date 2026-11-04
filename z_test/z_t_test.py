#!/usr/bin/env python
"""
campaign_all.csvから，zの平均値・標準偏差についてgauss_est vs gaussとreal vs gauss_estについてt検定を行います
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
N_NULL = 20000
input = "campaign_all.csv"

df = pd.read_csv(input,encoding="utf-8")

def per_event_z(df,param):
    event = df["event"] #イベント
    noise = df["noise"] #ノイズの状態
    event_unique = np.unique(event)
    noise_unique = np.unique(noise)
    kekka = [] #結果を入れておく配列
    kentei_mean = {}   # {"gauss": [...], "gauss_est": [...], "real": [...]}
    kentei_std = {}
    for eq in event_unique:
        m = (event == eq) #ターゲットのイベント
        for n in noise_unique:
            k = (noise == n)
            z = df.loc[m & k,f"z_{param}"].values # 各イベント，ノイズのz_param
            z_mean = z.mean() #平均
            z_std = z.std(ddof=1)
            kekka.append(dict(event=eq,noise=n,z_mean=z_mean,z_std=z_std))
    fig, axes = plt.subplots(1,2)
    for n in noise_unique:
        kekka2 = [r["z_mean"] for r in kekka if r["noise"] == n ] #結果のうちz_meanを取り出す
        kekka3 = [r["z_std"] for r in kekka if r["noise"] == n ] #結果のうちz_stdを取り出す
        axes[0].hist(kekka2,bins=10,alpha=0.5,label=f"{n}")
        axes[1].hist(kekka3,bins=10,alpha=0.5,label=f"{n}")
        kentei_mean[n] = np.array(kekka2)
        kentei_std[n]  = np.array(kekka3)
    axes[0].set_title(f"z_mean_{param}")
    axes[1].set_title(f"z_std_{param}")
    axes[0].legend()
    axes[1].legend()
    axes[0].axvline(0, color="k", ls="--"); axes[1].axvline(1, color="k", ls="--")
    output = f"test_{param}.png"
    fig.savefig(output)

    # real vs gauss_est
    p_rm = stats.ttest_rel(kentei_mean["gauss_est"], kentei_mean["real"]).pvalue
    #p_rm = stats.ttest_ind(kentei_mean["gauss_est"], kentei_mean["real"], equal_var=False).pvalue ウェルチのt検定
    p_rs = stats.ttest_rel(kentei_std["gauss_est"],  kentei_std["real"]).pvalue
    #p_rs = stats.ttest_ind(kentei_std["gauss_est"],  kentei_std["real"], equal_var=False).pvalue
    # gauss_est vs gauss
    p_gm = stats.ttest_rel(kentei_mean["gauss"], kentei_mean["gauss_est"]).pvalue
    #p_gm = stats.ttest_ind(kentei_mean["gauss"], kentei_mean["gauss_est"], equal_var=False).pvalue
    p_gs = stats.ttest_rel(kentei_std["gauss"],  kentei_std["gauss_est"]).pvalue
    #p_gs = stats.ttest_ind(kentei_std["gauss"],  kentei_std["gauss_est"], equal_var=False).pvalue
    print(f"{param:20s} gauss↔est: mean p={p_gm:.3g} std p={p_gs:.3g} | est↔real: mean p={p_rm:.3g} std p={p_rs:.3g}")

for param in FREE:
    per_event_z(df,param)