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

event = "GW191215_223052_" #イベント名
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

geocent_time = 1260484270.3 #ここではGW191215_223052の合体時刻!
exclude_time = 11 #除外する前後の区間
duration_time = 4096 #調べる区間!
duration_time_half = 2048 #前後2048秒
i = 200
random_time = [] #ランダムに選ぶ区間のための配列
skipped = 0
skipped_sim = 0
auto_correlation_list = []

n_seg_half = 100 #セグメントの数
seg_duration = 2.0 #セグメントの期間

for j in range(0,n_seg_half):
    n = ((geocent_time - exclude_time)-(geocent_time-duration_time_half))/n_seg_half
    second = n * j + (geocent_time-duration_time_half + 1.0)
    random_time.append(second)

for j in range(0,n_seg_half):
    n = ((geocent_time+duration_time_half-1.0)-(geocent_time + exclude_time))/n_seg_half
    second = n * j + (geocent_time + exclude_time)
    random_time.append(second)

data_L1 = TimeSeries.read("hdf5/L-L1_GWOSC_4KHZ_R1-1260482223-4096.hdf5",format="hdf5.gwosc") #gwoscからデータを入れておいてください!

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
    random_time = [t for t in random_time if (t_start + margin) < t and (t + seg_duration) < (margin_time - margin)]
    print(f"有効な候補数: {len(random_time)}")

white = data_L1.whiten(fftlength=8, overlap=4,fduration=4) #ホワイトニング

#シミュレーションデータ
# --- 合成ガウスノイズ（1つ目のセグメント）---
sim = generate_PSD_gauss(data_L1)
white_sim = sim.whiten(fftlength=8, overlap=4,fduration=4)

seg_sim = white_sim.crop(random_time[2],random_time[2] + seg_duration)W

qspec_sim = seg_sim.q_transform(qrange=[8, 8], frange=[30.0, 500.0])
qgram_sim = seg_sim.q_gram(qrange=[8, 8], frange=[30.0, 500.0], snrthresh=0)

acf_dict_sim  = acf_all_rows(qgram_sim,  max_lag=20)
mean_sim  = mean_acf(acf_dict_sim)
print(mean_sim)

rate_of_mabiki, mean_acf_curve = calibrate_thinning(white_sim, random_time,seg_duration)

print(f"間引き率は…{rate_of_mabiki}")

# ---- 帰無分布の準備 ----
# まず1つ目のセグメントでタイル数を確認
_seg0 = white.crop(random_time[0], random_time[0] + seg_duration)
_qg0 = _seg0.q_gram(qrange=[8, 8], frange=[30.0, 500.0], snrthresh=0)
t_tile = np.asarray(_qg0["time"])
e_tile = np.asarray(_qg0["energy"])
order = np.argsort(t_tile)
y = e_tile[order][::rate_of_mabiki]
N_tiles = len(y)
print(f"帰無分布を構築中（N={N_tiles}, n_sim=1e+4）...")
A2_null = build_ad_null_distribution(N_tiles, n_sim=int(1e+4))
print(f"帰無分布の中央値: {np.median(A2_null):.3f}, 99%点: {np.percentile(A2_null, 99):.3f}")

kslist,chi2list,adlist,random_time_list,a2list,all_y,qgram_L1,y_norm = run_test(random_time,white,skipped,geocent_time,rate_of_mabiki,A2_null,seg_duration)

kslist_sim,chi2list_sim,adlist_sim,random_time_list_sim,a2list_sim,all_y_sim,qgram_L1_sim,y_norm_sim = run_test(random_time,white_sim,skipped,geocent_time,rate_of_mabiki,A2_null,seg_duration)

plot(kslist,chi2list,adlist,random_time_list,a2list,all_y,qgram_L1,y_norm,kslist_sim,chi2list_sim,adlist_sim,random_time_list_sim,a2list_sim,all_y_sim,qgram_L1_sim,y_norm_sim,geocent_time,event,det,A2_null,output1,output2,output3,output4,output5,output6,output7,output8,output9)