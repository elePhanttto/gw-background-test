#ホワイトニングした歪データをwavに変換します
#audacityで扱ってみる?

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
output10 = event + "psd.png" #PSDのプロット

geocent_time = 1260484270.3 #ここではGW191215_223052の合体時刻!
exclude_time = 11 #除外する前後の区間
duration_time = 4096 #調べる区間!
duration_time_half = 2048 #前後2048秒
i = 200
random_time = [] #ランダムに選ぶ区間のための配列
skipped = 0
skipped_sim = 0
auto_correlation_list = []
BANDS = [(0,30, 80), (1,80, 120), (2,120, 250), (3,250, 500)]

for j in range(0,100):
    n = ((geocent_time - exclude_time)-(geocent_time-duration_time_half))/100
    second = n * j + (geocent_time-duration_time_half)
    random_time.append(second)

for j in range(0,100):
    n = ((geocent_time+duration_time_half-1.0)-(geocent_time + exclude_time))/100
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
    random_time = [t for t in random_time if (t_start + margin) < t and (t + 2.0) < (margin_time - margin)]
    print(f"有効な候補数: {len(random_time)}")

white = data_L1.whiten(fftlength=4, overlap=2,fduration=4) 

import soundfile as sf
sf.write(f"{event}.wav", white.value, samplerate=4096, subtype='FLOAT')