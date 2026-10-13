# GW191215_223052 L1 で、ホワイトニングの位相選択(zero-phase vs minimum-phase)が
# 背景雑音検定の結果(KS/AD/chi2, タイルエネルギー分布)にどれだけ影響するかを検証する。
# 一部Claude製
#   「q-transformする際に、位相の選び方で結果が変わらないのだろうか」
# への回答実験。
#
# 理論的には:
#   - 定常ガウス過程に全域通過(振幅1)フィルタを掛けてもパワースペクトルは不変
#     → 統計検定(KS, AD, chi2)の結果はphaseに依存しないはず
#   - ただし非定常な異常(violin mode由来の振幅非定常性など)がある場合、
#     zero-phase(対称・非因果)だと異常が前後に漏れて薄まり、
#     minimum-phase(因果)だと異常の"後ろ側"に局在するはず
#     → 「劣化がどのタイルに乗るか」という局在パターンは変わりうる

from gwpy.timeseries import TimeSeries
import numpy as np
from bgtest_func import (
    generate_PSD_gauss, acf_all_rows, mean_acf, calibrate_thinning,
    build_ad_null_distribution, run_test, plot,
)
from phase_filter_func import *

event = "GW191215_223052_"
det = "L1"

geocent_time = 1260484270.3
exclude_time = 11
duration_time = 4096
duration_time_half = 2048
random_time = []
skipped = 0

FFTLENGTH = 4      # ASD推定用FFT長[s] (固定。ここでは位相の効果だけを見る)
OVERLAP = 2
FDURATION = 4      # FIRフィルタ長[s]
PHASES = ["zero", "minimum"]
Q = 8 #q-transformのQ値．どんな異常を見たいかで変える

n_seg_half = 100
seg_duration = 2.0

for j in range(0, n_seg_half):
    n = ((geocent_time - exclude_time) - (geocent_time - duration_time_half)) / n_seg_half
    second = n * j + (geocent_time - duration_time_half + 1.0)
    random_time.append(second)

for j in range(0, n_seg_half):
    n = ((geocent_time + duration_time_half - 1.0) - (geocent_time + exclude_time)) / n_seg_half
    second = n * j + (geocent_time + exclude_time)
    random_time.append(second)

data_L1 = TimeSeries.read(
    "hdf5/L-L1_GWOSC_4KHZ_R1-1260482223-4096.hdf5", format="hdf5.gwosc"
)

# NaNのない連続区間を特定(既存スクリプトと同じロジック)
valid = ~np.isnan(data_L1.value)
invalid = np.isnan(data_L1.value)
deltat = 1 / 4000

if not valid.all():
    times = data_L1.times.value
    valid_times = times[valid]
    invalid_times = times[invalid]
    t_start = valid_times.min()
    t_nan_min = invalid_times.min()
    if t_nan_min <= t_start:
        for i in range(0, len(invalid_times) - 1):
            sabun = invalid_times[i + 1] - invalid_times[i]
            if sabun > deltat:
                t_nan_min = invalid_times[i + 1]
                break
    print(f"連続した有効区間 {t_start:.1f} 👉️ {t_nan_min:.1f} ({t_nan_min - t_start:.0f}秒)")
    margin_time = t_nan_min - 1.0
    data_L1 = data_L1.crop(t_start, margin_time)
    margin = 8.0
    random_time = [
        t for t in random_time
        if (t_start + margin) < t and (t + seg_duration) < (margin_time - margin)
    ]
    print(f"有効な候補数: {len(random_time)}")

# --- 合成ガウスノイズ(位相に依らず共通の素材として1回だけ作る) ---
sim = generate_PSD_gauss(data_L1)

results_by_phase = {}
h_by_phase = {}

for phase in PHASES:
    print(f"\n========== phase = {phase} ==========")
    fftsetting = f"_{phase}phase_"
    output1 = event + fftsetting + "kstest_L1.png"
    output2 = event + fftsetting + "chi2test_L1.png"
    output3 = event + fftsetting + "adtest_L1.png"
    output4 = event + fftsetting + "pvalue_hist_L1.png"
    output5 = event + fftsetting + "dist_check_L1.png"
    output6 = event + fftsetting + "ratio_L1.txt"
    output7 = event + fftsetting + "tile_energy_L1.png"
    output8 = event + fftsetting + "a2_L1.png"
    output9 = event + fftsetting + "ratio_L1_sim.txt"
    output10 = fftsetting

    if phase == "zero":
        white = whiten_zero(data_L1,32,16)
        white_sim = whiten_zero(sim,32,16)
    else:
        white = whiten_min(data_L1,32,16)
        white_sim = whiten_min(sim,32,16)

    seg_sim = white_sim.crop(random_time[2], random_time[2] + seg_duration)
    qgram_sim = seg_sim.q_gram(qrange=[8, 8], frange=[30.0, 500.0], snrthresh=0)
    acf_dict_sim = acf_all_rows(qgram_sim, max_lag=20)
    mean_sim = mean_acf(acf_dict_sim)
    print(mean_sim)

    rate_of_mabiki, mean_acf_curve = calibrate_thinning(Q,white_sim, random_time, seg_duration)
    print(f"間引き率は…{rate_of_mabiki}")

    _seg0 = white.crop(random_time[1], random_time[1] + seg_duration)
    print("q_gramに渡すデータの長さ[s]:", _seg0.duration.value)  # 2.0になるべき
    _qg0 = _seg0.q_gram(qrange=[8, 8], frange=[30.0, 500.0], snrthresh=0)
    t_tile = np.asarray(_qg0["time"])
    e_tile = np.asarray(_qg0["energy"])
    order = np.argsort(t_tile)
    y = e_tile[order][::rate_of_mabiki]
    N_tiles = len(y)
    print(f"帰無分布を構築中（N={N_tiles}, n_sim=1e+4）...")
    A2_null = build_ad_null_distribution(N_tiles, n_sim=int(1e+4))
    print(f"帰無分布の中央値: {np.median(A2_null):.3f}, 99%点: {np.percentile(A2_null, 99):.3f}")

    kslist, chi2list, adlist, random_time_list, a2list, all_y, qgram_L1, y_norm = run_test(
        Q,random_time, white, skipped, geocent_time, rate_of_mabiki, A2_null, seg_duration
    )
    kslist_sim, chi2list_sim, adlist_sim, random_time_list_sim, a2list_sim, all_y_sim, qgram_L1_sim, y_norm_sim = run_test(
        Q,random_time, white_sim, skipped, geocent_time, rate_of_mabiki, A2_null, seg_duration
    )
    plot(
        kslist, chi2list, adlist, random_time_list, a2list, all_y, qgram_L1, y_norm,
        kslist_sim, chi2list_sim, adlist_sim, random_time_list_sim, a2list_sim, all_y_sim, qgram_L1_sim, y_norm_sim,
        geocent_time, event, det, A2_null,
        output1, output2, output3, output4, output5, output6, output7, output8, output9, output10,
    )

    results_by_phase[phase] = dict(
        random_time_offset=np.array(random_time_list) - geocent_time,
        ks_p=np.array(kslist), chi2_p=np.array(chi2list), ad_p=np.array(adlist), a2=np.array(a2list),
    )

# --- zero vs minimum のズレを直接比較 ---
z = results_by_phase["zero"]
m = results_by_phase["minimum"]
assert np.allclose(z["random_time_offset"], m["random_time_offset"])

import pandas as pd
diff_df = pd.DataFrame({
    "time_offset": z["random_time_offset"],
    "ks_p_zero": z["ks_p"], "ks_p_min": m["ks_p"],
    "ad_p_zero": z["ad_p"], "ad_p_min": m["ad_p"],
    "a2_zero": z["a2"], "a2_min": m["a2"],
})
diff_df["log10_a2_ratio"] = np.log10(diff_df["a2_min"] / diff_df["a2_zero"])
diff_df.to_csv(f"{event}_{det}_phase_comparison.csv", index=False)

print("\n===== phase比較サマリ =====")
print(f"A2(zero) 中央値: {np.median(z['a2']):.3f}")
print(f"A2(minimum) 中央値: {np.median(m['a2']):.3f}")
print("|log10(A2_min/A2_zero)| の分布 (0に近いほど位相非依存):")
print(diff_df["log10_a2_ratio"].abs().describe())

# 大きくズレているセグメント = 位相選択に敏感な異常
sensitive = diff_df[diff_df["log10_a2_ratio"].abs() > 0.3].sort_values(
    "log10_a2_ratio", key=abs, ascending=False
)
print("\n位相選択に敏感なセグメント(|Δlog10 A2| > 0.3):")
print(sensitive)
