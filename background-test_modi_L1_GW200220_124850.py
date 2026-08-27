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

event = "GW200220_124850_" #イベント名
det = "L1" #検出器
output1 = event + "kstest_L1.png" #ksテスト
output2 = event + "chi2test_L1.png" #アウトプットのファイル
output3 = event + "adtest_L1.png" #adテスト
output4 = event + "pvalue_hist_L1.png" #histgram
output5 = event + "dist_check_L1.png" #分布の比較
output6 = event + "ratio_L1.txt" #裾の部分の定量化
output7 = event + "tile_energy_L1.png" #タイルのエネルギー
output8 = event + "a2_L1.png" #A2自体のプロット

geocent_time = 1266238148.1 #ここではGW200220_124850の合体時刻!
exclude_time = 11 #除外する前後の区間
duration_time = 4096 #調べる区間!
duration_time_half = 2048 #前後2048秒
i = 200
random_time = [] #ランダムに選ぶ区間のための配列
adlist = []
a2list = []
kslist = []
chi2list = []
random_time_list = []
skipped = 0
all_y = []
auto_correlation_list = []

for j in range(0,100):
    n = ((geocent_time - exclude_time)-(geocent_time-duration_time_half))/100
    second = n * j + (geocent_time-duration_time_half + 1.0)
    random_time.append(second)

for j in range(0,100):
    n = ((geocent_time+duration_time_half-1.0)-(geocent_time + exclude_time))/100
    second = n * j + (geocent_time + exclude_time)
    random_time.append(second)

data_L1 = TimeSeries.read("hdf5/L-L1_GWOSC_4KHZ_R1-1266236101-4096.hdf5",format="hdf5.gwosc") #gwoscからデータを入れておいてください!

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

# シミュレーションデータを使って間引き率を決める

rate_of_mabiki = 1
threshold = 0.1 #閾値

for i in range(len(mean_sim)):
    rate_of_mabiki = i
    if mean_sim[i] < threshold : #閾値を下回ったところを間引き率にする
        break
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

for j in range(len(random_time)):
    seg = white.crop(random_time[j],random_time[j] + 1.0) #切り出し
    qgram_L1 = seg.q_gram(qrange=[8, 8], frange=[30.0, 500.0], snrthresh=0) #q-gramでq-transform
    y_L1 = np.asarray(qgram_L1["energy"]) #エネルギーの部分を取り出す
    # 時間順にソートしてから等間隔間引き
    t_tile = np.asarray(qgram_L1["time"])
    e_tile = np.asarray(qgram_L1["energy"])
    order = np.argsort(t_tile)
    y = e_tile[order][::rate_of_mabiki]        # 間引く(時間方向に相関があるため)
    y_norm = y / y.mean()
    all_y.append(y_norm)
    print(f"タイル数: {len(y_norm)}")
    obs, exp = Y_distribution_equiprob(y_norm,n_bins=9)
    #print(y_dist,expdist)
    # 6. 検定
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
    a2list.append(result.statistic)
    print(f"t={random_time[j]-geocent_time:+8.1f}s  "
          f"KS p={ks.pvalue:.2e}  chi2 p={chi2_p_value:.2e}  "
          f"AD A2={A2_obs:.2f} p={ad_p:.2e}")

print(f"NaNでスキップしたセグメント: {skipped}")

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

# ------------------------------------------
# KS test
# ------------------------------------------
plt.figure(figsize=(10, 5))

plt.scatter(time_offsets, ks_pvalues, s=15)

plt.axhline(1e-8, linestyle="--", label="p = 1e-8")

plt.xlabel("Time from GW200220_124850 geocent time [s]")
plt.ylabel("KS test p-value")
plt.title("KS test p-value vs time")
plt.yscale('log')
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(output1)


# ------------------------------------------
# Chi-square test
# ------------------------------------------
plt.figure(figsize=(10, 5))

plt.scatter(time_offsets, chi2_pvalues, s=15)

plt.axhline(1e-8, linestyle="--", label="p = 1e-8")

plt.xlabel("Time from GW200220_124850 geocent time [s]")
plt.ylabel(r"$\chi^2$ test p-value")
plt.title(r"$\chi^2$ test p-value vs time")
plt.yscale('log')
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(output2)

# adtestのプロット!

ad_pvalues = np.array(adlist)

plt.figure(figsize=(10, 5))
plt.scatter(time_offsets, ad_pvalues, s=15)
plt.axhline(1e-6, linestyle="--", label="p = 1e-6")
plt.xlabel("Time from GW200220_124850 geocent time [s]")
plt.ylabel("AD test p-value")
plt.title("Anderson-Darling test p-value vs time")
plt.yscale('log')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(output3)

# histgram

plt.figure(figsize=(8,5))
plt.hist(np.log10(ks_pvalues), bins=30, alpha=0.7)
plt.xlabel("log10(p-value)")
plt.ylabel("count")
plt.xlim((-8,0))
plt.title("KS p-value distribution")
plt.savefig(output4)

# 分布チェック!

plt.figure(figsize=(8,5))
plt.hist(y_norm, bins=60, density=True, alpha=0.6, label="observed")
x = np.linspace(0, 10, 200)
plt.plot(x, np.exp(-x), 'r-', lw=2, label=r"$e^{-y}$")
plt.yscale('log')
plt.xlabel("normalized energy")
plt.legend()
plt.savefig(output5)

# A^2自体をプロットしてもらう

plt.figure(figsize=(10,5))
plt.scatter(time_offsets, a2list, s=15)
plt.axhline(np.percentile(A2_null, 99), ls='--', c='r', label='99% of null')
plt.axhline(np.median(A2_null), ls=':', c='gray', label='null median')
plt.xlabel("Time from GW200220_124850 geocent time [s]")
plt.ylabel(r"$A^2$ statistic")
plt.yscale('log')
plt.legend()

plt.savefig(output8)

with open(output6,mode = "w",encoding='utf-8') as t:
    t.write("裾の部分の過剰な部分を見てみる \n")
#裾の部分の過剰な部分を見てみる
    y_all = np.concatenate(all_y)  # 全部まとめて
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

# 異常セグメントの抽出
bad = df[(df.ks_p < 1e-3) | (df.chi2_p < 1e-3) | (df.ad_p < 1e-4)]
print(bad.sort_values("ks_p"))