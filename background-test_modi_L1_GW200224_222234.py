#小さい値を一定の値に固定するバージョン
#タイルのエネルギーに時間相関があることから，間引いています

from gwpy.timeseries import TimeSeries
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from random import *
import math
from bgtest_func import *

event = "GW200224_222234_" #イベント名
det = "L1" #検出器
output1 = event + "kstest_L1.png" #ksテスト
output2 = event + "chi2test_L1.png" #アウトプットのファイル
output3 = event + "adtest_L1.png" #adテスト
output4 = event + "pvalue_hist_L1.png" #histgram
output5 = event + "dist_check_L1.png" #分布の比較
output6 = event + "ratio_L1.txt" #裾の部分の定量化
output7 = event + "tile_energy_L1.png" #タイルのエネルギー
output8 = event + "a2_L1.png" #A2自体のプロット
output9 = event + "ratio_L1_sim.txt"

geocent_time = 1266618172.4 #ここではGW200224_222234の合体時刻!
exclude_time = 11 #除外する前後の区間
duration_time = 4096 #調べる区間!
duration_time_half = 2048 #前後2048秒
i = 200
random_time = [] #ランダムに選ぶ区間のための配列
skipped = 0
skipped_sim = 0
auto_correlation_list = []

for j in range(0,100):
    n = ((geocent_time - exclude_time)-(geocent_time-duration_time_half))/100
    second = n * j + (geocent_time-duration_time_half + 1.0)
    random_time.append(second)

for j in range(0,100):
    n = ((geocent_time+duration_time_half-1.0)-(geocent_time + exclude_time))/100
    second = n * j + (geocent_time + exclude_time)
    random_time.append(second)

data_L1 = TimeSeries.read("hdf5/L-L1_GWOSC_4KHZ_R1-1266616125-4096.hdf5",format="hdf5.gwosc") #gwoscからデータを入れておいてください!

# NaNのない範囲を特定(合体前からの連続した区間を選ぶ)
valid = ~np.isnan(data_L1.value)
invalid = np.isnan(data_L1.value) #nanの部分
deltat = 1/4000 #周波数は4000Hz

if not valid.all():
    times = data_L1.times.value
    valid_times = times[valid]
    invalid_times = times[invalid]
    t_start = valid_times.min()
    t_nan_min = invalid_times.min()
    if t_nan_min <= t_start:
        for i in range(0,len(invalid_times)-1):
            sabun = invalid_times[i+1] - invalid_times[i]
            if sabun > deltat:
                t_nan_min = invalid_times[i+1]
                break
    print(f"連続した有効区間 {t_start:.1f} 👉️ {t_nan_min:.1f} ({t_nan_min - t_start:.0f}秒)")
    margin_time = t_nan_min - 1.0 #nanが始まる時間から1秒差し引く
    data_L1 = data_L1.crop(t_start,margin_time)
    margin = 8.0
    random_time = [t for t in random_time if (t_start + margin) < t and (t + 1.0) < (margin_time - margin)]
    print(f"有効な候補数: {len(random_time)}")

white = data_L1.whiten(fftlength=4, overlap=2) #ホワイトニング

#シミュレーションデータ
# --- 合成ガウスノイズ（1つ目のセグメント）---
sim = TimeSeries(
    np.random.normal(size=len(data_L1)),
    sample_rate=data_L1.sample_rate,
    t0=data_L1.t0,
)
white_sim = sim.whiten(fftlength=4, overlap=2)

seg_sim = white_sim.crop(random_time[2],random_time[2] + 2.0)

qspec_sim = seg_sim.q_transform(qrange=[8, 8], frange=[30.0, 500.0])
qgram_sim = seg_sim.q_gram(qrange=[8, 8], frange=[30.0, 500.0], snrthresh=0)

acf_dict_sim  = acf_all_rows(qgram_sim,  max_lag=20)
mean_sim  = mean_acf(acf_dict_sim)
print(mean_sim)

rate_of_mabiki, mean_acf_curve = calibrate_thinning(white_sim, random_time)

print(f"間引き率は…{rate_of_mabiki}")

# ---- 帰無分布の準備 ----
# まず1つ目のセグメントでタイル数を確認
_seg0 = white.crop(random_time[0], random_time[0] + 1.0)
_qg0 = _seg0.q_gram(qrange=[8, 8], frange=[30.0, 500.0], snrthresh=0)
t_tile = np.asarray(_qg0["time"])
e_tile = np.asarray(_qg0["energy"])
order = np.argsort(t_tile)
y = e_tile[order][::rate_of_mabiki]
N_tiles = len(y)
print(f"帰無分布を構築中（N={N_tiles}, n_sim=1e+6）...")
A2_null = build_ad_null_distribution(N_tiles, n_sim=int(1e+6))
print(f"帰無分布の中央値: {np.median(A2_null):.3f}, 99%点: {np.percentile(A2_null, 99):.3f}")

kslist,chi2list,adlist,random_time_list,a2list,all_y,qgram_L1,y_norm = run_test(random_time,white,skipped,geocent_time,rate_of_mabiki,A2_null)

kslist_sim,chi2list_sim,adlist_sim,random_time_list_sim,a2list_sim,all_y_sim,qgram_L1_sim,y_norm_sim = run_test(random_time,white_sim,skipped,geocent_time,rate_of_mabiki,A2_null)

#タイルのエネルギーを見る

freqs = np.asarray(qgram_L1["frequency"])
energies = np.asarray(qgram_L1["energy"])
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
axes[0].set_xlabel("Time from GW200224_222234 geocent time [s]")
axes[0].set_ylabel("KS test p-value")
axes[0].set_title("KS test p-value vs time")
axes[0].set_yscale('log')
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].scatter(time_offsets, ks_pvalues_sim, s=15)

axes[1].axhline(1e-8, linestyle="--", label="p = 1e-8")

axes[1].set_xlabel("Time from GW200224_222234 geocent time [s]")
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

axes[0].set_xlabel("Time from GW200224_222234 geocent time [s]")
axes[0].set_ylabel(r"$\chi^2$ test p-value")
axes[0].set_title(r"$\chi^2$ test p-value vs time")
axes[0].set_yscale('log')
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].scatter(time_offsets, chi2_pvalues_sim, s=15)

axes[1].axhline(1e-8, linestyle="--", label="p = 1e-8")

axes[1].set_xlabel("Time from GW200224_222234 geocent time [s]")
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
axes[0].set_xlabel("Time from GW200224_222234 geocent time [s]")
axes[0].set_ylabel("AD test p-value")
axes[0].set_title("Anderson-Darling test p-value vs time")
axes[0].set_yscale('log')
axes[0].grid(alpha=0.3)
axes[0].legend()
axes[1].scatter(time_offsets, ad_pvalues_sim, s=15)
axes[1].axhline(1e-6, linestyle="--", label="p = 1e-6")
axes[1].set_xlabel("Time from GW200224_222234 geocent time [s]")
axes[1].set_ylabel("AD test p-value")
axes[1].set_title("Anderson-Darling test p-value vs time of gaussian noise")
axes[1].set_yscale('log')
axes[1].grid(alpha=0.3)
axes[1].legend()
plt.tight_layout()
fig.savefig(output3)

# histgram

fig, axes = plt.subplots(1,2,figsize=(20,5))
axes[0].hist(np.log10(ks_pvalues), bins=30, alpha=0.7)
axes[0].set_xlabel("log10(p-value)")
axes[0].set_ylabel("count")
axes[0].set_xlim((-8,0))
axes[0].set_title("KS p-value distribution")
axes[1].hist(np.log10(ks_pvalues_sim), bins=30, alpha=0.7)
axes[1].set_xlabel("log10(p-value)")
axes[1].set_ylabel("count")
axes[1].set_xlim((-8,0))
axes[1].set_title("KS p-value distribution of gaussian noise")
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
axes[0].set_xlabel("Time from GW200224_222234 geocent time [s]")
axes[0].set_ylabel(r"$A^2$ statistic")
axes[0].set_yscale('log')
axes[0].legend()
axes[1].scatter(time_offsets, a2list_sim, s=15)
axes[1].axhline(np.percentile(A2_null, 99), ls='--', c='r', label='99% of null')
axes[1].axhline(np.median(A2_null), ls=':', c='gray', label='null median')
axes[1].set_xlabel("Time from GW200224_222234 geocent time [s]")
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