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

event = "GW200129_065458_" #イベント名
det = "L1" #検出器
output1 = event + "kstest_L1_band.png" #ksテスト
output2 = event + "chi2test_L1_band.png" #アウトプットのファイル
output3 = event + "adtest_L1_band.png" #adテスト
output4 = event + "pvalue_hist_L1_band.png" #histgram
output5 = event + "dist_check_L1_band.png" #分布の比較
output6 = event + "ratio_L1_band.txt" #裾の部分の定量化
output7 = event + "tile_energy_L1_band.png" #タイルのエネルギー
output8 = event + "a2_L1_band.png" #A2自体のプロット
output9 = event + "ratio_L1_band_sim.txt"
output10 = event + "psd.png" #PSDのプロット

geocent_time = 1264316116.4 #ここではGW200129_065458の合体時刻!
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
BANDS = [(0,30, 80), (1,80, 120), (2,120, 250), (3,250, 500)] #周波数ごとに分けて解析
Q = 8 #q-transformのQ値．どんな異常を見たいかで変える

for j in range(0,n_seg_half):
    n = ((geocent_time - exclude_time)-(geocent_time-duration_time_half))/n_seg_half
    second = n * j + (geocent_time-duration_time_half)
    random_time.append(second)

for j in range(0,n_seg_half):
    n = ((geocent_time+duration_time_half-1.0)-(geocent_time + exclude_time))/n_seg_half
    second = n * j + (geocent_time + exclude_time)
    random_time.append(second)

data_L1 = TimeSeries.read("hdf5/L-L1_GWOSC_4KHZ_R1-1264314069-4096.hdf5",format="hdf5.gwosc") #gwoscからデータを入れておいてください!

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

white = data_L1.whiten(fftlength=4, overlap=2,fduration=4) #ホワイトニング

#シミュレーションデータ
# --- 合成ガウスノイズ（1つ目のセグメント）---
sim = generate_PSD_gauss(data_L1)
white_sim = sim.whiten(fftlength=4, overlap=2,fduration=4) #Q値に応じて，fftlength,fdurationも変えるべき

seg_sim = white_sim.crop(random_time[2],random_time[2] + 2.0)

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
    print(f"帰無分布を構築中（N={N_tiles}, n_sim=1e+6）,(fmin={fmin},fmax={fmax})...")
    A2_null[i] = build_ad_null_distribution(N_tiles, n_sim=int(1e+6))
    print(f"帰無分布の中央値: {np.median(A2_null[i]):.3f}, 99%点: {np.percentile(A2_null[i], 99):.3f}")

kslist = [[] for _ in range(4)]
chi2list = [[] for _ in range(4)]
adlist = [[] for _ in range(4)]
random_time_list = [[] for _ in range(4)]
a2list = [[] for _ in range(4)]

kslist_sim = [[] for _ in range(4)]
chi2list_sim = [[] for _ in range(4)]
adlist_sim = [[] for _ in range(4)]
random_time_list_sim = [[] for _ in range(4)]
a2list_sim = [[] for _ in range(4)]
time_offsets = [[] for _ in range(4)]
ks_pvalues = [[] for _ in range(4)]
chi2_pvalues = [[] for _ in range(4)]
ks_pvalues_sim = [[] for _ in range(4)]
chi2_pvalues_sim = [[] for _ in range(4)]

for i,fmin,fmax in BANDS:
    kslist[i],chi2list[i],adlist[i],random_time_list[i],a2list[i],all_y,qgram_L1,y_norm = run_test_band(Q,random_time,white,skipped,geocent_time,rate_of_mabiki[i],A2_null[i],fmin,fmax,seg_duration)
    kslist_sim[i],chi2list_sim[i],adlist_sim[i],random_time_list_sim[i],a2list_sim[i],all_y_sim,qgram_L1_sim,y_norm_sim = run_test_band(Q,random_time,white_sim,skipped,geocent_time,rate_of_mabiki[i],A2_null[i],fmin,fmax,seg_duration)

    # geocent_time からの時間差 [s]
    time_offsets[i] = np.array(random_time_list[i]) - geocent_time
    ks_pvalues[i] = np.array(kslist[i])
    chi2_pvalues[i] = np.array(chi2list[i])
    ks_pvalues_sim[i] = np.array(kslist_sim[i])
    chi2_pvalues_sim[i] = np.array(chi2list_sim[i])

fig, axes = plt.subplots(1,2,figsize=(20,5))
for i,fmin,fmax in BANDS:
    axes[0].scatter(time_offsets[i], ks_pvalues[i], s=15,alpha=0.3,label=f"real {fmin}-{fmax}")

axes[0].axhline(1e-8, linestyle="--", label="p = 1e-8")
axes[0].set_xlabel("Time from GW200129_065458 geocent time [s]")
axes[0].set_ylabel("KS test p-value")
axes[0].set_title("KS test p-value vs time")
axes[0].set_yscale('log')
axes[0].grid(alpha=0.3)
axes[0].legend()

for i,fmin,fmax in BANDS:
    axes[1].scatter(time_offsets[i], ks_pvalues_sim[i], s=15,alpha=0.3,label=f"sim {fmin}-{fmax}")

axes[1].axhline(1e-8, linestyle="--", label="p = 1e-8")

axes[1].set_xlabel("Time from GW200129_065458 geocent time [s]")
axes[1].set_ylabel("KS test p-value of gaussian noise")
axes[1].set_title("KS test p-value vs time of gaussian noise")
axes[1].set_yscale('log')
axes[1].grid(alpha=0.3)
axes[1].legend()

plt.tight_layout()
fig.savefig(output1)

# histgram

fig, axes = plt.subplots(figsize=(10,5))

for i,fmin,fmax in BANDS:
    p_sorted = np.sort(ks_pvalues[i][:])
    p_sorted_sim = np.sort(ks_pvalues_sim[i][:])
    n = len(p_sorted)
    expected = (np.arange(1, n+1) - 0.5) / n
    axes.plot(expected, p_sorted, 'o', ms=3,label=f"real {fmin}-{fmax}",alpha=0.3)
    axes.plot(expected, p_sorted_sim, 'o',color="orange", ms=3,label=f"sim {fmin}-{fmax}",alpha=0.3)

axes.plot([0,1], [0,1], 'r--', lw=1) # 一様なら45度線に乗る
axes.set_xlabel("expected quantile")
axes.set_ylabel("observed p-value")
axes.set_xlim(0,1)
axes.set_ylim(0,1)
axes.legend()
plt.gca().set_aspect('equal')
fig.savefig(output4)


# ============================================================
# CSV出力（ロング形式：帯域 × real/sim を1ファイルに）
# ============================================================
rows = []
for i, fmin, fmax in BANDS:
    for tag, ks, chi2, ad, a2 in [
        ("real", kslist[i],     chi2list[i],     adlist[i],     a2list[i]),
        ("sim",  kslist_sim[i], chi2list_sim[i], adlist_sim[i], a2list_sim[i]),
    ]:
        for t, kp, cp, ap, av in zip(time_offsets[i], ks, chi2, ad, a2):
            rows.append(dict(band=i, fmin=fmin, fmax=fmax,
                             thin_rate=rate_of_mabiki[i], dataset=tag,
                             time_offset=t, ks_p=kp, chi2_p=cp, ad_p=ap, a2=av))

df = pd.DataFrame(rows)
df.to_csv(f"{event}{det}_band_results.csv", index=False)
print(f"保存: {event}{det}_band_results.csv  ({len(df)}行)")

# ============================================================
# 帯域別サマリ（端1点を除く）
# ============================================================
lines = [f"{event}{det}  帯域別サマリ", ""]
lines.append(f"{'band [Hz]':<12}{'間引き':>7}{'N_tile':>8}{'N_seg':>7}"
             f"{'KS中央 real':>12}{'sim':>8}{'2標本KS p':>12}"
             f"{'A2max real':>12}{'sim':>8}")
lines.append("-" * 86)

for i, fmin, fmax in BANDS:
    d = df[df.band == i]
    # 端（最小 time_offset）の1点を除外
    t_edge = d.time_offset.min()
    dr = d[(d.dataset == "real") & (d.time_offset > t_edge)]
    ds = d[(d.dataset == "sim")  & (d.time_offset > t_edge)]

    # 間引き後のタイル数を再確認
    qg = white_sim.crop(random_time[0], random_time[0] + 2.0).q_gram(
        qrange=[Q,Q], frange=[fmin, fmax], snrthresh=0)
    n_tile = len(np.asarray(qg["energy"])[::rate_of_mabiki[i]])

    p2 = stats.ks_2samp(dr.a2, ds.a2).pvalue
    lines.append(f"{f'{fmin}-{fmax}':<12}{rate_of_mabiki[i]:>7}{n_tile:>8}{len(dr):>7}"
                 f"{np.median(dr.ks_p):>12.3f}{np.median(ds.ks_p):>8.3f}"
                 f"{p2:>12.2e}{dr.a2.max():>12.1f}{ds.a2.max():>8.1f}")

lines += ["", "一様分布なら KS中央値 = 0.5",
          "2標本KS p は real と sim の A2 分布の差（小さいほど有意）"]

txt = "\n".join(lines)
print("\n" + txt)
with open(f"{event}{det}_band_summary.txt", "w", encoding="utf-8") as f:
    f.write(txt + "\n")

#PSDをプロットしておく

psd_L1 = data_L1.psd(fftlength=8)
psd_sim = sim.psd(fftlength=8)

freq = psd_L1.frequencies.value
power = psd_L1.value
freq_sim = psd_sim.frequencies.value
power_sim = psd_sim.value
fig, axes = plt.subplots(figsize=(10,5))
axes.plot(freq,power)
axes.plot(freq_sim,power_sim)
axes.set_xlabel("frequencies")
axes.set_ylabel("PSD")
axes.set_xlim((400,600))
axes.set_yscale("log")
fig.savefig(output10)