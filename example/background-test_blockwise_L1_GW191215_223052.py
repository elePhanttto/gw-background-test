# 小さい値を一定の値に固定するバージョン
# タイルのエネルギーに時間相関があることから，間引いています
# ChatGPT製
# ============================================================
# GW191215_223052 L1
# Whitening の期間(BLOCK)を変えて背景ノイズテストを比較する版
#
# 元コードからの主な変更点:
#   1. BLOCK = 256, 128, 64, 32 s をそれぞれ独立にwhiten
#   2. random_time は全BLOCKで共通
#   3. 各BLOCKの境界から MARGIN 秒以内の候補は除外
#   4. KS / chi2 / AD / A2 / 裾の過剰率をBLOCKごとに保存
#   5. 同じ time_offset をBLOCK間で比較できるCSVを出力
#
# bgtest_func.py は元のものをそのまま使用します。
# ============================================================

from gwpy.timeseries import TimeSeries
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from random import *
import math
from bgtest_func import *


# ============================================================
# 基本設定
# ============================================================

event = "GW191215_223052_"
det = "L1"

geocent_time = 1260484270.3
exclude_time = 11
duration_time = 4096
duration_time_half = 2048

# 比較する whitening の期間
BLOCKS = [256.0, 128.0, 64.0, 32.0]
fl = [8,16,64]

# whitening
FFT_LENGTH = 4.0
OVERLAP = 2.0

# BLOCK境界・whitening端部を避けるためのmargin
MARGIN = 16.0

# 1秒segment
SEGMENT_DURATION = 1.0

# q-transform
QRANGE = [8, 8]
FRANGE = [30.0, 500.0]
SNRTHRESH = 0

# χ2検定
N_BINS = 9

# A2 null distribution
# 元コードと同じ 1e+6
N_AD_SIM = int(1e+6)

# p-valueを非常に小さい値で固定する
FLOOR = 1.0e-8

# random_time生成数
N_RANDOM_LEFT = 100
N_RANDOM_RIGHT = 100


# ============================================================
# 出力ファイル名
# ============================================================

output1 = event + "blockwise_kstest_L1_block.png"
output2 = event + "blockwise_chi2test_L1_block.png"
output3 = event + "blockwise_adtest_L1_block.png"
output4 = event + "blockwise_pvalue_hist_L1_block.png"
output5 = event + "blockwise_dist_check_L1_block.png"
output6 = event + "blockwise_ratio_L1_block.txt"
output7 = event + "blockwise_tile_energy_L1_block.png"
output8 = event + "blockwise_a2_L1_block.png"

output_csv = event + det + "_blockwise_results.csv"


# ============================================================
# random_time を元コードと同じ方法で作る
# ============================================================

random_time = []

for j in range(0, N_RANDOM_LEFT):
    n = (
        (geocent_time - exclude_time)
        - (geocent_time - duration_time_half)
    ) / N_RANDOM_LEFT

    second = (
        n * j
        + (geocent_time - duration_time_half + 1.0)
    )

    random_time.append(second)


for j in range(0, N_RANDOM_RIGHT):
    n = (
        (geocent_time + duration_time_half - 1.0)
        - (geocent_time + exclude_time)
    ) / N_RANDOM_RIGHT

    second = (
        n * j
        + (geocent_time + exclude_time)
    )

    random_time.append(second)


# ============================================================
# データ読み込み
# ============================================================

data_L1 = TimeSeries.read(
    "hdf5/L-L1_GWOSC_4KHZ_R1-1260482223-4096.hdf5",
    format="hdf5.gwosc",
)


# ============================================================
# NaNのない範囲を特定
# 元コードの処理を維持
# ============================================================

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

            sabun = (
                invalid_times[i + 1]
                - invalid_times[i]
            )

            if sabun > deltat:

                t_nan_min = invalid_times[i + 1]
                break

    print(
        f"連続した有効区間 "
        f"{t_start:.1f} 👉️ {t_nan_min:.1f} "
        f"({t_nan_min - t_start:.0f}秒)"
    )

    margin_time = t_nan_min - 1.0

    data_L1 = data_L1.crop(
        t_start,
        margin_time,
    )

    # 元コードと同じ8秒margin
    margin = 8.0

    random_time = [
        t for t in random_time
        if (
            (t_start + margin) < t
            and
            (t + 1.0) < (margin_time - margin)
        )
    ]

    print(
        f"有効な候補数: {len(random_time)}"
    )


random_time = np.asarray(
    random_time,
    dtype=float,
)


# ============================================================
# BLOCKの境界に近いrandom_timeを除外
#
# 各BLOCKでwhitenしたデータを使うため、
# BLOCK境界 + MARGIN の内側だけを使う。
#
# 全BLOCKで同じrandom_timeを使えるように、
# 最小BLOCK=32 sに対しても十分離れている候補だけを
# 最終的に採用する。
#
# ただしBLOCKごとの結果を最大限残すため、
# 実際の除外判定は各BLOCKごとに行う。
# ============================================================

print("\n============================================================")
print("random_time")
print("============================================================")
print(f"候補数: {len(random_time)}")


# ============================================================
# 合成Gaussian noise
#
# 元コードと同じく、rate_of_mabikiを決めるために
# 4096秒全体を一度whitenする。
#
# ここではBLOCKごとの比較そのものとは独立に、
# 「タイル間相関を落とすための間引き率」を決める。
# ============================================================

print("\n============================================================")
print("合成Gaussian noiseによる間引き率の決定")
print("============================================================")

sim = TimeSeries(
    np.random.normal(
        size=len(data_L1)
    ),
    sample_rate=data_L1.sample_rate,
    t0=data_L1.t0,
)

white_sim = sim.whiten(
    fftlength=FFT_LENGTH,
    overlap=OVERLAP,
)

seg_sim = white_sim.crop(
    random_time[2],
    random_time[2] + 2.0,
)

qspec_sim = seg_sim.q_transform(
    qrange=QRANGE,
    frange=FRANGE,
)

qgram_sim = seg_sim.q_gram(
    qrange=QRANGE,
    frange=FRANGE,
    snrthresh=SNRTHRESH,
)

acf_dict_sim = acf_all_rows(
    qgram_sim,
    max_lag=20,
)

mean_sim = mean_acf(
    acf_dict_sim
)

print(mean_sim)


# ============================================================
# シミュレーションデータを使って間引き率を決める
# 元コードのロジックを維持
# ============================================================

rate_of_mabiki = 1
threshold = 0.1

for i in range(len(mean_sim)):

    rate_of_mabiki = i

    if mean_sim[i] < threshold:

        break

# step=0を防ぐ
if rate_of_mabiki < 1:
    rate_of_mabiki = 1

print(
    f"間引き率は…{rate_of_mabiki}"
)


# ============================================================
# A2帰無分布の準備
#
# 1秒segmentのq_gramについてN_tilesを確認する。
# BLOCKを変えてもq_gramの条件は同じなので、
# A2_nullは共通にする。
#
# 元コードと同じく n_sim=1e+6。
# ============================================================

print("\n============================================================")
print("A2帰無分布の準備")
print("============================================================")

# random_time[0] は各BLOCKで境界条件に入る可能性があるので、
# 最初の候補から順に、全BLOCKで使える候補を探す。

def is_valid_time_for_block(
    t,
    data_t0,
    data_t_end,
    block,
    margin,
):
    """
    指定したrandom_timeが、
    指定BLOCKの内部でmargin以上離れているか判定する。
    """

    block_index = int(
        np.floor(
            (t - data_t0) / block
        )
    )

    block_lo = (
        data_t0
        + block_index * block
    )

    block_hi = (
        block_lo
        + block
    )

    # BLOCK内に1秒segmentが完全に収まっているかだけ確認
    if not (
        block_lo <= t
        and
        t + SEGMENT_DURATION <= block_hi
    ):
        return None

    # raw dataの範囲
    if not (
        data_t0 <= t
        and
        t + SEGMENT_DURATION <= data_t_end
    ):
        return False

    return True


data_t0 = data_L1.t0.value
data_t_end = (
    data_L1.t0.value
    + data_L1.duration.value
)


reference_time = None

for t in random_time:

    ok = True

    for block in BLOCKS:

        if not is_valid_time_for_block(
            t,
            data_t0,
            data_t_end,
            block,
            MARGIN,
        ):
            ok = False
            break

    if ok:
        reference_time = t
        break


if reference_time is None:

    raise RuntimeError(
        "全BLOCKで使用可能なreference_timeが見つかりません。"
    )


print(
    f"A2 null distribution用 reference time = "
    f"{reference_time - geocent_time:+.2f} s"
)


# ============================================================
# reference_timeを256秒BLOCK等に対応するwhitening chunkから
# 切り出すための関数
# ============================================================

def get_block_whitened_segment(
    data,
    t,
    block,
    margin,
):
    """
    指定時刻tを含むBLOCKを取り出し、
    BLOCK +/- margin を含む範囲でwhitenしてから、
    t～t+1秒を返す。
    """

    data_t0 = data.t0.value
    data_t_end = (
        data.t0.value
        + data.duration.value
    )

    # tが属するBLOCK
    block_index = int(
        np.floor(
            (t - data_t0) / block
        )
    )

    block_lo = (
        data_t0
        + block_index * block
    )

    block_hi = (
        block_lo
        + block
    )

    # BLOCK外側にもmarginをつけてwhitening
    lo = max(
        data_t0,
        block_lo - margin,
    )

    hi = min(
        data_t_end,
        block_hi + margin,
    )

    # 1秒segmentがBLOCK内部のmargin領域にあるか確認
    if not (
        block_lo <= t
        and
        t + SEGMENT_DURATION <= block_hi
        ):
        return None

    chunk = data.crop(
        lo,
        hi,
    )

    # NaN / Inf確認
    values = np.asarray(
        chunk.value
    )

    if not np.isfinite(values).all():
        return None

    # BLOCKごとに独立してwhitening
    white = chunk.whiten(
        fftlength=FFT_LENGTH,
        overlap=OVERLAP,
    )

    seg = white.crop(
        t,
        t + SEGMENT_DURATION,
    )

    if len(seg) == 0:
        return None

    seg_values = np.asarray(
        seg.value
    )

    if not np.isfinite(seg_values).all():
        return None

    return seg


# ============================================================
# A2 null distribution
# ============================================================

_seg0 = None

for block in BLOCKS:

    _seg0 = get_block_whitened_segment(
        data_L1,
        reference_time,
        block,
        MARGIN,
    )

    if _seg0 is not None:
        break


if _seg0 is None:

    raise RuntimeError(
        "A2 null distribution用のsegmentを作成できません。"
    )


_qg0 = _seg0.q_gram(
    qrange=QRANGE,
    frange=FRANGE,
    snrthresh=SNRTHRESH,
)

t_tile = np.asarray(
    _qg0["time"]
)

e_tile = np.asarray(
    _qg0["energy"]
)

order = np.argsort(
    t_tile
)

y = e_tile[
    order
][::rate_of_mabiki]

N_tiles = len(y)

print(
    f"帰無分布を構築中 "
    f"(N={N_tiles}, n_sim={N_AD_SIM:.0e})..."
)

A2_null = build_ad_null_distribution(
    N_tiles,
    n_sim=N_AD_SIM,
)

print(
    f"帰無分布の中央値: "
    f"{np.median(A2_null):.3f}, "
    f"99%点: "
    f"{np.percentile(A2_null, 99):.3f}"
)


# ============================================================
# BLOCKごとの結果を格納
#
# columns:
#   block
#   index
#   time
#   time_offset
#   ks_p
#   chi2_p
#   ad_p
#   a2
# ============================================================

all_results = []

# 裾の過剰率用
all_y_by_block = {
    block: []
    for block in BLOCKS
}


# ============================================================
# 各BLOCKについて背景ノイズテスト
# ============================================================

for block in BLOCKS:

    print("\n")
    print("============================================================")
    print(
        f"BLOCK = {block:.0f} s の解析開始"
    )
    print("============================================================")

    skipped_block = 0

    # BLOCKごとにwhitenしたsegmentを入れる
    block_results = []

    for j, t in enumerate(random_time):

        seg = get_block_whitened_segment(
            data_L1,
            t,
            block,
            MARGIN,
        )

        if seg is None:

            skipped_block += 1

            continue

        try:

            # q-gram
            qgram_L1 = seg.q_gram(
                qrange=QRANGE,
                frange=FRANGE,
                snrthresh=SNRTHRESH,
            )

            # energy
            y_L1 = np.asarray(
                qgram_L1["energy"]
            )

            # 時間方向に並べてから等間隔間引き
            t_tile = np.asarray(
                qgram_L1["time"]
            )

            e_tile = np.asarray(
                qgram_L1["energy"]
            )

            order = np.argsort(
                t_tile
            )

            y = e_tile[
                order
            ][::rate_of_mabiki]

            # 念のため有限値のみ
            y = y[
                np.isfinite(y)
            ]

            if len(y) == 0:

                skipped_block += 1
                continue

            # 正規化
            y_norm = (
                y
                / y.mean()
            )

            all_y_by_block[
                block
            ].append(y_norm)

            print(
                f"BLOCK={block:.0f}s  "
                f"tile数={len(y_norm)}"
            )

            # ------------------------------------------------
            # χ2用等確率bin
            # ------------------------------------------------

            obs, exp = (
                Y_distribution_equiprob(
                    y_norm,
                    n_bins=N_BINS,
                )
            )

            # ------------------------------------------------
            # KS
            # ------------------------------------------------

            ks = stats.kstest(
                y_norm,
                "expon",
            )

            # ------------------------------------------------
            # χ2
            # ------------------------------------------------

            chi2_stat, chi2_p_value = (
                stats.chisquare(
                    f_obs=obs,
                    f_exp=exp,
                )
            )

            # ------------------------------------------------
            # AD
            # ------------------------------------------------

            A2_obs = ad_statistic(
                y_norm
            )

            ad_p = ad_pvalue(
                A2_obs,
                A2_null,
            )

            # scipyのAD statisticも保存
            result = stats.anderson(
                y_norm,
                dist="expon",
            )

            # ------------------------------------------------
            # floor
            # ------------------------------------------------

            ks_p = max(
                ks.pvalue,
                FLOOR,
            )

            chi2_p = max(
                chi2_p_value,
                FLOOR,
            )

            ad_p_fixed = max(
                ad_p,
                FLOOR,
            )

            # ------------------------------------------------
            # 保存
            # ------------------------------------------------

            row = {
                "block": block,
                "index": j,
                "time": t,
                "time_offset": (
                    t
                    - geocent_time
                ),
                "ks_p": ks_p,
                "chi2_p": chi2_p,
                "ad_p": ad_p_fixed,
                "a2": result.statistic,
                "n_tiles": len(y_norm),
            }

            block_results.append(
                row
            )

            print(
                f"t={t - geocent_time:+8.1f}s  "
                f"KS p={ks.pvalue:.2e}  "
                f"chi2 p={chi2_p_value:.2e}  "
                f"AD A2={A2_obs:.2f} "
                f"p={ad_p:.2e}"
            )

        except Exception as e:

            skipped_block += 1

            print(
                f"t={t - geocent_time:+8.1f}s  "
                f"解析失敗: {e}"
            )

    print(
        f"\nBLOCK={block:.0f}s:"
        f" {len(block_results)} segments analyzed"
        f", skipped={skipped_block}"
    )

    all_results.extend(
        block_results
    )


print(
    "\n============================================================"
)

print(
    "全BLOCKの解析終了"
)

print(
    "============================================================"
)


# ============================================================
# DataFrame
# ============================================================

df = pd.DataFrame(
    all_results
)

df = df.sort_values(
    [
        "block",
        "time_offset",
    ]
).reset_index(
    drop=True
)


# ============================================================
# CSV
# ============================================================

df.to_csv(
    output_csv,
    index=False,
)

print(
    f"CSVを保存しました: {output_csv}"
)


# ============================================================
# BLOCKごとのsummaryをtxtに保存
# ============================================================

summary_file = (
    event
    + "blockwise_summary_L1.txt"
)

with open(
    summary_file,
    mode="w",
    encoding="utf-8",
) as t:

    t.write(
        "GW191215_223052 L1 "
        "blockwise whitening background test\n"
    )

    t.write(
        "============================================================\n"
    )

    t.write(
        f"BLOCKS = {BLOCKS}\n"
    )

    t.write(
        f"MARGIN = {MARGIN} s\n"
    )

    t.write(
        f"FFT_LENGTH = {FFT_LENGTH} s\n"
    )

    t.write(
        f"OVERLAP = {OVERLAP} s\n"
    )

    t.write(
        f"rate_of_mabiki = {rate_of_mabiki}\n"
    )

    t.write(
        f"N_tiles = {N_tiles}\n"
    )

    t.write("\n")

    for block in BLOCKS:

        sub = df[
            df["block"] == block
        ]

        t.write(
            f"\nBLOCK = {block:.0f} s\n"
        )

        t.write(
            "------------------------------------------------------------\n"
        )

        if len(sub) == 0:

            t.write(
                "結果なし\n"
            )

            continue

        for col in [
            "ks_p",
            "chi2_p",
            "ad_p",
            "a2",
        ]:

            values = sub[
                col
            ].dropna()

            if len(values) == 0:
                continue

            t.write(
                f"{col}: "
                f"median={np.median(values):.6g}, "
                f"min={np.min(values):.6g}, "
                f"max={np.max(values):.6g}\n"
            )

        # A2最大
        idx_max = sub[
            "a2"
        ].idxmax()

        row_max = sub.loc[
            idx_max
        ]

        t.write(
            f"max A2 = "
            f"{row_max['a2']:.6g} "
            f"at dt="
            f"{row_max['time_offset']:+.2f} s\n"
        )

        # p-value最小
        for col in [
            "ks_p",
            "chi2_p",
            "ad_p",
        ]:

            idx_min = sub[
                col
            ].idxmin()

            row_min = sub.loc[
                idx_min
            ]

            t.write(
                f"min {col} = "
                f"{row_min[col]:.6g} "
                f"at dt="
                f"{row_min['time_offset']:+.2f} s\n"
            )


print(
    f"summaryを保存しました: {summary_file}"
)


# ============================================================
# KS p-value
# ============================================================

plt.figure(
    figsize=(10, 5)
)

for block in BLOCKS:

    sub = df[
        df["block"] == block
    ]

    plt.scatter(
        sub["time_offset"],
        sub["ks_p"],
        s=15,
        label=f"BLOCK={block:.0f}s",
    )

plt.axhline(
    FLOOR,
    linestyle="--",
    label=f"p = {FLOOR:.0e}",
)

plt.xlabel(
    "Time from GW191215_223052 geocent time [s]"
)

plt.ylabel(
    "KS test p-value"
)

plt.title(
    "KS test p-value vs time "
    "(blockwise whitening)"
)

plt.yscale(
    "log"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    output1
)

plt.close()


# ============================================================
# Chi-square p-value
# ============================================================

plt.figure(
    figsize=(10, 5)
)

for block in BLOCKS:

    sub = df[
        df["block"] == block
    ]

    plt.scatter(
        sub["time_offset"],
        sub["chi2_p"],
        s=15,
        label=f"BLOCK={block:.0f}s",
    )

plt.axhline(
    FLOOR,
    linestyle="--",
    label=f"p = {FLOOR:.0e}",
)

plt.xlabel(
    "Time from GW191215_223052 geocent time [s]"
)

plt.ylabel(
    r"$\chi^2$ test p-value"
)

plt.title(
    r"$\chi^2$ test p-value vs time "
    "(blockwise whitening)"
)

plt.yscale(
    "log"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    output2
)

plt.close()


# ============================================================
# AD p-value
# ============================================================

plt.figure(
    figsize=(10, 5)
)

for block in BLOCKS:

    sub = df[
        df["block"] == block
    ]

    plt.scatter(
        sub["time_offset"],
        sub["ad_p"],
        s=15,
        label=f"BLOCK={block:.0f}s",
    )

plt.axhline(
    FLOOR,
    linestyle="--",
    label=f"p = {FLOOR:.0e}",
)

plt.xlabel(
    "Time from GW191215_223052 geocent time [s]"
)

plt.ylabel(
    "AD test p-value"
)

plt.title(
    "Anderson-Darling test p-value vs time "
    "(blockwise whitening)"
)

plt.yscale(
    "log"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    output3
)

plt.close()


# ============================================================
# p-value histogram
# ============================================================

plt.figure(
    figsize=(8, 5)
)

for block in BLOCKS:

    sub = df[
        df["block"] == block
    ]

    if len(sub) == 0:
        continue

    plt.hist(
        np.log10(
            sub["ks_p"]
        ),
        bins=30,
        alpha=0.5,
        label=f"BLOCK={block:.0f}s",
    )

plt.xlabel(
    "log10(KS p-value)"
)

plt.ylabel(
    "count"
)

plt.xlim(
    (-8, 0)
)

plt.title(
    "KS p-value distribution "
    "(blockwise whitening)"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    output4
)

plt.close()


# ============================================================
# 分布チェック
#
# 各BLOCKについて最後に処理したsegmentではなく、
# 最も異常なsegment(A2最大)を描画する。
# これによりBLOCK間比較ができる。
# ============================================================

plt.figure(
    figsize=(8, 5)
)

for block in BLOCKS:

    sub = df[
        df["block"] == block
    ]

    if len(sub) == 0:
        continue

    idx = sub[
        "a2"
    ].idxmax()

    t_bad = sub.loc[
        idx,
        "time"
    ]

    seg_bad = get_block_whitened_segment(
        data_L1,
        t_bad,
        block,
        MARGIN,
    )

    if seg_bad is None:
        continue

    qgram_bad = seg_bad.q_gram(
        qrange=QRANGE,
        frange=FRANGE,
        snrthresh=SNRTHRESH,
    )

    e_bad = np.asarray(
        qgram_bad["energy"]
    )

    t_bad_tile = np.asarray(
        qgram_bad["time"]
    )

    order = np.argsort(
        t_bad_tile
    )

    y_bad = e_bad[
        order
    ][::rate_of_mabiki]

    y_bad = y_bad[
        np.isfinite(y_bad)
    ]

    y_bad = (
        y_bad
        / y_bad.mean()
    )

    plt.hist(
        y_bad,
        bins=60,
        density=True,
        alpha=0.35,
        label=(
            f"BLOCK={block:.0f}s, "
            f"dt={t_bad-geocent_time:+.1f}s"
        ),
    )

x = np.linspace(
    0,
    10,
    200,
)

plt.plot(
    x,
    np.exp(-x),
    "k-",
    lw=2,
    label=r"$e^{-y}$",
)

plt.yscale(
    "log"
)

plt.xlabel(
    "normalized energy"
)

plt.ylabel(
    "density"
)

plt.title(
    "Distribution of the most anomalous segment "
    "(each whitening block)"
)

plt.legend(
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    output5
)

plt.close()


# ============================================================
# A2自体をプロット
# ============================================================

plt.figure(
    figsize=(10, 5)
)

for block in BLOCKS:

    sub = df[
        df["block"] == block
    ]

    plt.scatter(
        sub["time_offset"],
        sub["a2"],
        s=15,
        label=f"BLOCK={block:.0f}s",
    )

plt.axhline(
    np.percentile(
        A2_null,
        99,
    ),
    linestyle="--",
    label="99% of null",
)

plt.axhline(
    np.median(
        A2_null
    ),
    linestyle=":",
    label="null median",
)

plt.xlabel(
    "Time from GW191215_223052 geocent time [s]"
)

plt.ylabel(
    r"$A^2$ statistic"
)

plt.title(
    r"$A^2$ vs time "
    "(blockwise whitening)"
)

plt.yscale(
    "log"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    output8
)

plt.close()


# ============================================================
# タイルエネルギー
#
# 各BLOCKについて、A2最大segmentを描画
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for block in BLOCKS:

    sub = df[
        df["block"] == block
    ]

    if len(sub) == 0:
        continue

    idx = sub[
        "a2"
    ].idxmax()

    t_bad = sub.loc[
        idx,
        "time"
    ]

    seg_bad = get_block_whitened_segment(
        data_L1,
        t_bad,
        block,
        MARGIN,
    )

    if seg_bad is None:
        continue

    qgram_bad = seg_bad.q_gram(
        qrange=QRANGE,
        frange=FRANGE,
        snrthresh=SNRTHRESH,
    )

    freqs = np.asarray(
        qgram_bad["frequency"]
    )

    energies = np.asarray(
        qgram_bad["energy"]
    )

    plt.scatter(
        freqs,
        energies / energies.mean(),
        s=2,
        alpha=0.3,
        label=(
            f"BLOCK={block:.0f}s, "
            f"dt={t_bad-geocent_time:+.1f}s"
        ),
    )

plt.xscale(
    "log"
)

plt.yscale(
    "log"
)

plt.xlabel(
    "frequency [Hz]"
)

plt.ylabel(
    "normalized energy"
)

plt.title(
    "Tile energy of the most anomalous segment"
)

plt.legend(
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    output7
)

plt.close()


# ============================================================
# 裾の部分の過剰な部分を定量化
# ============================================================

with open(
    output6,
    mode="w",
    encoding="utf-8",
) as t:

    t.write(
        "GW191215_223052 L1\n"
    )

    t.write(
        "Whitening blockwise tail excess\n\n"
    )

    for block in BLOCKS:

        t.write(
            f"BLOCK = {block:.0f} s\n"
        )

        y_list = (
            all_y_by_block[block]
        )

        if len(y_list) == 0:

            t.write(
                "有効segmentなし\n\n"
            )

            continue

        y_all = np.concatenate(
            y_list
        )

        for thr in [
            4,
            5,
            6,
            7,
        ]:

            obs = (
                y_all > thr
            ).mean()

            expected = np.exp(
                -thr
            )

            ratio = (
                obs
                / expected
            )

            t.write(
                f"y>{thr}: "
                f"ratio={ratio:.2f}\n"
            )

        t.write("\n")


# ============================================================
# 異常セグメント
# ============================================================

print("\n============================================================")
print("異常セグメント")
print("============================================================")

bad = df[
    (df.ks_p < 1e-3)
    |
    (df.chi2_p < 1e-3)
    |
    (df.ad_p < 1e-4)
]

for block in BLOCKS:

    bad_block = bad[
        bad["block"] == block
    ]

    print(
        f"\nBLOCK={block:.0f}s"
    )

    if len(bad_block) == 0:

        print(
            "  該当なし"
        )

    else:

        print(
            bad_block.sort_values(
                "ks_p"
            )[
                [
                    "time_offset",
                    "ks_p",
                    "chi2_p",
                    "ad_p",
                    "a2",
                ]
            ].to_string(
                index=False
            )
        )


# ============================================================
# BLOCKごとのA2最大値を最後に表示
# ============================================================

print("\n============================================================")
print("BLOCKごとのA2最大値")
print("============================================================")

for block in BLOCKS:

    sub = df[
        df["block"] == block
    ]

    if len(sub) == 0:

        print(
            f"BLOCK={block:.0f}s: 結果なし"
        )

        continue

    row = sub.loc[
        sub["a2"].idxmax()
    ]

    print(
        f"BLOCK={block:.0f}s: "
        f"A2={row['a2']:.3f}, "
        f"dt={row['time_offset']:+.2f}s, "
        f"KS p={row['ks_p']:.3e}, "
        f"chi2 p={row['chi2_p']:.3e}, "
        f"AD p={row['ad_p']:.3e}"
    )


print("\n解析終了。")
