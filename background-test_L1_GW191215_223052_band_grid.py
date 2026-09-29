# https://journals.aps.org/prd/abstract/10.1103/PhysRevD.108.063016 のテストを再現するヨ!

from gwpy.timeseries import TimeSeries
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from random import *
import math
from bgtest_func import *
from bgtest_func_frac import * 
import pymc as pm
from pymc import *
from bgtest_grid import *

event = "GW191215_223052_" #イベント名
det = "L1" #検出器
output1 = event + "L1_avg_tilepower.png" #average tile power
output2 = event  + det + "fractional_power.png" #fractional power

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
#BANDS = [(0,30, 80), (1,80, 120), (2,120, 250), (3,250, 500)] #周波数ごとに分けて解析
BANDS = [(0,30,40),(1,40,50),(2,50,60),(3,60,80)]
Q = 8 #q-transformのQ値．どんな異常を見たいかで変える

for j in range(0,n_seg_half):
    n = ((geocent_time - exclude_time)-(geocent_time-duration_time_half))/n_seg_half
    second = n * j + (geocent_time-duration_time_half)
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
    
print(f"最大時間{max(random_time)}")
print(f"最小時間{min(random_time)}")

white = data_L1.whiten(fftlength=4, overlap=2,fduration=2) #ホワイトニング．ここではあえてfduration=2

# ホワイトニング後の実際の範囲に収まるセグメントだけを残す
# (NaNが無いとNaNチェックのブロックが丸ごとスキップされ、
#  データ範囲外のセグメントが残ってしまうため、ここで必ず行う)
w_start = white.t0.value
w_end = white.t0.value + white.duration.value
before = len(random_time)
random_time = [t for t in random_time
               if t >= w_start and (t + seg_duration) <= w_end]
print(f"ホワイトニング後の範囲 {w_start:.1f} 〜 {w_end:.1f}")
print(f"有効なセグメント数: {len(random_time)} (除外 {before - len(random_time)}個)")

print(f"最大時間{max(random_time)}")
print(f"最小時間{min(random_time)}")

#シミュレーションデータ
# --- 合成ガウスノイズ（1つ目のセグメント）---
sim = generate_PSD_gauss(data_L1)
white_sim = sim.whiten(fftlength=4, overlap=2,fduration=2) #Q値に応じて，fftlength,fdurationも変えるべき

seg_sim = white_sim.crop(random_time[2],random_time[2] + seg_duration)

qspec_sim = seg_sim.q_transform(qrange=[Q,Q], frange=[30.0, 500.0])
qgram_sim = seg_sim.q_gram(qrange=[Q,Q], frange=[30.0, 500.0], snrthresh=0)

acf_dict_sim  = acf_all_rows(qgram_sim,  max_lag=20)
mean_sim  = mean_acf(acf_dict_sim)
print(mean_sim)

rate_of_mabiki = [0 for i in range(4)]

for i,fmin,fmax in BANDS:
    rate_of_mabiki[i],mean_acf_curve = calibrate_thinning_band(Q,white_sim,random_time,fmin,fmax,seg_duration)

print(f"間引き率は…{rate_of_mabiki}")

A2_null = [None] * 4

# ---- 帰無分布の準備 ----
# まず1つ目のセグメントでタイル数を確認

for i,fmin,fmax in BANDS:
    _seg0 = white.crop(random_time[0], random_time[0] + 2.0)
    _qg0 = _seg0.q_gram(qrange=[Q,Q], frange=[fmin, fmax], snrthresh=0)
    t_tile = np.asarray(_qg0["time"])
    e_tile = np.asarray(_qg0["energy"])
    order = np.argsort(t_tile)
    y = e_tile[order][::rate_of_mabiki[i]]
    N_tiles = len(y)
    #print(f"帰無分布を構築中（N={N_tiles}, n_sim=1e+4）,(fmin={fmin},fmax={fmax})...")
    #A2_null[i] = build_ad_null_distribution(N_tiles, n_sim=int(1e+4))
    #print(f"帰無分布の中央値: {np.median(A2_null[i]):.3f}, 99%点: {np.percentile(A2_null[i], 99):.3f}")

#出力を定義する

f_thinned = [[] for _ in range(4)]
y_norm = [[] for _ in range(4)]
y = [[] for _ in range(4)]
e_tile = [[] for _ in range(4)]
f_tile = [[] for _ in range(4)]
t_tile = [[] for _ in range(4)]
random_time_list = [[] for _ in range(4)]

unique_freqs = [[] for _ in range(4)]
avg_power_by_freq = [[] for _ in range(4)]
std_power_by_freq = [[] for _ in range(4)]

freqs = [[] for _ in range(4)]
fp_m = [[] for _ in range(4)]
fp_l = [[] for _ in range(4)]
fp_h = [[] for _ in range(4)]

f_thinned_sim = [[] for _ in range(4)]
y_norm_sim = [[] for _ in range(4)]
y_sim = [[] for _ in range(4)]
e_tile_sim = [[] for _ in range(4)]
f_tile_sim = [[] for _ in range(4)]
t_tile_sim = [[] for _ in range(4)]
random_time_list_sim = [[] for _ in range(4)]

unique_freqs_sim = [[] for _ in range(4)]
avg_power_by_freq_sim = [[] for _ in range(4)]
std_power_by_freq_sim = [[] for _ in range(4)]

freqs_sim = [[] for _ in range(4)]
fp_m_sim = [[] for _ in range(4)]
fp_l_sim = [[] for _ in range(4)]
fp_h_sim = [[] for _ in range(4)]

# 見たいところ!
t_center = 560.72
window_duration = 5.0 #前後XX秒!
window_time = [0,0]
window_time[0] = geocent_time + t_center - window_duration #注目するところ最小値
window_time[1] = geocent_time + t_center + window_duration #注目するところ最小値

for i,fmin,fmax in BANDS:
    y_norm[i],e_tile[i],f_tile[i],t_tile[i],unique_freqs[i],avg_power_by_freq[i],std_power_by_freq[i],y[i],f_thinned[i]= run_test_band_qavg(Q,random_time,white,skipped,geocent_time,rate_of_mabiki[i],fmin,fmax,seg_duration)
    run_test_grid_freq(y[i],f_thinned[i],event,det,type="real")
    y_norm_sim[i],e_tile_sim[i],f_tile_sim[i],t_tile_sim[i],unique_freqs_sim[i],avg_power_by_freq_sim[i],std_power_by_freq_sim[i],y_sim[i],f_thinned_sim[i]= run_test_band_qavg(Q,random_time,white_sim,skipped,geocent_time,rate_of_mabiki[i],fmin,fmax,seg_duration)
    run_test_grid_freq(y_sim[i],f_thinned_sim[i],event,det,type="sim")