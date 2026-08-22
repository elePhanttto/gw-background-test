#あるセグメントのバックグラウンドについて，リアルのデータと合成ガウスノイズをヒートマップでプロット
#ついでに自己相関関数も求めています

from gwpy.timeseries import TimeSeries
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from random import *
import math

event = "GW190412_" #イベント名
output1 = event + "heat_map_H1_with_simulation.png" #タイルのエネルギー

geocent_time = 1239082262.2 #ここではGW190412の合体時刻!
exclude_time = 11 #除外する前後の区間
duration_time = 4096 #調べる区間!
duration_time_half = 2048 #前後2048秒
i = 200
random_time = [] #ランダムに選ぶ区間のための配列
random_time_list = []
skipped = 0

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

for j in range(0,100):
    n = ((geocent_time - exclude_time)-(geocent_time-duration_time_half))/100
    second = n * j + (geocent_time-duration_time_half)
    random_time.append(second)

for j in range(0,100):
    n = ((geocent_time+duration_time_half-1.0)-(geocent_time + exclude_time))/100
    second = n * j + (geocent_time + exclude_time)
    random_time.append(second)

data_H1 = TimeSeries.read("H-H1_GWOSC_4KHZ_R1-1239080215-4096.hdf5",format="hdf5.gwosc") #gwoscからデータを入れておいてください!

# NaNのない範囲を特定
valid = ~np.isnan(data_H1.value)
if not valid.all():
    times = data_H1.times.value
    valid_times = times[valid]
    t_start, t_end = valid_times.min(), valid_times.max()
    print(f"⚠️ 有効区間: {t_start:.1f} - {t_end:.1f} ({t_end-t_start:.0f}秒)")
    
    # 有効区間だけ切り出す（端に少し余裕を持たせる）
    data_H1 = data_H1.crop(t_start, t_end)
    
    # 候補時刻も有効区間内に絞る
    margin = 8.0
    random_time = [t for t in random_time
                   if (t_start + margin) < t and (t + 1.0) < (t_end - margin)]
    print(f"有効な候補数: {len(random_time)}")

white = data_H1.whiten(fftlength=4, overlap=2) #ホワイトニング

#実データについて

seg = white.crop(random_time[5],random_time[5] + 2.0) #切り出し
qspec_H1 = seg.q_transform(qrange=[8, 8], frange=[30.0, 500.0])
qgram_H1 = seg.q_gram(qrange=[8, 8], frange=[30.0, 500.0], snrthresh=0)
print(qspec_H1)

print(f"NaNでスキップしたセグメント: {skipped}")

#シミュレーションデータ
# --- 合成ガウスノイズ（同じ長さ・サンプリングレート）---
sim = TimeSeries(
    np.random.normal(size=len(data_H1)),
    sample_rate=data_H1.sample_rate,
    t0=data_H1.t0,
)
white_sim = sim.whiten(fftlength=4, overlap=2)

seg_sim = white_sim.crop(random_time[5],random_time[5] + 2.0)

qspec_sim = seg_sim.q_transform(qrange=[8, 8], frange=[30.0, 500.0])
qgram_sim = seg_sim.q_gram(qrange=[8, 8], frange=[30.0, 500.0], snrthresh=0)
#お試しヒートマップ

#リアルのデータをプロット
fig, axes = plt.subplots(1,3,figsize=(45,10))
mesh = axes[0].pcolormesh(qspec_H1.times.value,qspec_H1.frequencies.value,qspec_H1.value.T,vmin=0,vmax=30,cmap="viridis", shading="auto"
)
fig.colorbar(mesh, ax=axes[0], label="normalized energy")
axes[0].set_yscale("log")
axes[0].set_xlabel("times[s]"); plt.ylabel("frequency[Hz]")
axes[0].set_title("real background")

#シミュレーションデータをプロット
mesh2 = axes[1].pcolormesh(qspec_sim.times.value,qspec_sim.frequencies.value,qspec_sim.value.T,vmin=0,vmax=30,cmap="viridis", shading="auto")
fig.colorbar(mesh2, ax=axes[1], label="normalized energy")
axes[1].set_yscale("log")
axes[1].set_xlabel("times[s]"); plt.ylabel("frequency[Hz]")
axes[1].set_title("gaussian noise background")

acf_dict_real = acf_all_rows(qgram_H1, max_lag=20)
acf_dict_sim  = acf_all_rows(qgram_sim,  max_lag=20)

# --- ACF: 全周波数行を重ね書き ---
for k, (freq, acf) in enumerate(acf_dict_sim.items()):
    axes[2].plot(acf, color="red", alpha=0.25, lw=0.8,
                 label="synthetic" if k == 0 else None)

for k, (freq, acf) in enumerate(acf_dict_real.items()):
    axes[2].plot(acf, color="blue", alpha=0.25, lw=0.8,
                 label="real" if k == 0 else None)

axes[2].axhline(0, c="gray", lw=0.5)
axes[2].set_xlabel("tile lag"); axes[2].set_ylabel("autocorrelation")
axes[2].set_xlim(0, 20)
axes[2].legend()
axes[2].set_title(f"ACF of all frequency rows (n_real={len(acf_dict_real)}, n_sim={len(acf_dict_sim)})")

# 平均ACF（長さが揃う範囲だけ）
def mean_acf(acf_dict, max_lag=20):
    arrs = [a for a in acf_dict.values() if len(a) == max_lag]
    return np.mean(arrs, axis=0) if arrs else None

mean_real = mean_acf(acf_dict_real)
mean_sim  = mean_acf(acf_dict_sim)
if mean_real is not None:
    axes[2].plot(mean_real, color="blue", lw=2.5, label="real (mean)")
if mean_sim is not None:
    axes[2].plot(mean_sim, color="red", lw=2.5, label="synthetic (mean)")

plt.tight_layout()
fig.savefig(output1, dpi=150)