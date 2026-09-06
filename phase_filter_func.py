# 位相選択可能なホワイトニング関数群
# bgtest_func.py と同じディレクトリに置いて import してください
#一部Claude製
#
# T先生の疑問「q-transformはphaseで結果が変わるか」を検証するための実装。
#
# 設計方針:
#   1) "zero"    : GWpyのwhiten()と同じ思想。1/ASD(f)をIFFTして対称(線形位相)FIRを作る。
#                  非因果的(未来のデータも使う)、応答は中心対称。
#   2) "minimum" : (1)のFIRと"同じ振幅応答"を保ったまま、scipy.signal.minimum_phase
#                  (Hilbert変換法)で最小位相(因果)化したFIR。
#                  T先生の言う「Hilbert変換で複素平面下側で正則になるように位相を選ぶ」に対応。
#
# 振幅応答|H(f)|が(1)と(2)で完全に同じであることが肝心。これにより
# 「統計的性質(パワースペクトル)は同じだが、時間局在だけが変わる」という
# 状況を作れます。

import numpy as np
from scipy.signal import fftconvolve, minimum_phase
from scipy.signal.windows import tukey
from gwpy.timeseries import TimeSeries


def whiten_zero(data,FFTLENGTH,FDURATION):
    if FFTLENGTH < FDURATION:
        print("FFTLENGTH > FDURATION にしてネ")
    # --- 実データのASDを推定 ---
    asd = data.asd(fftlength=FFTLENGTH) #周波数は1/fftlength Hz 刻みになる(0Hz-2048Hz) 2048*8 = 16384要素+1
    mado = 1.0/(asd.value) #逆ASD(片側)
    fs = 4096 #データは4096Hz!
    freq = asd.frequencies.value
    ref_mask = (freq >= 30) & (freq <= 500)
    ref_gain = np.median(mado[ref_mask])
    print("ref_gain:", ref_gain)
    cap = ref_gain * 10**(30/20)
    floor = ref_gain * 10**(-30/20)
    print(asd.value[0])
    mado[0] = np.median(mado[ref_mask]) 

    full_n = 2 * (len(mado) - 1)
    window = np.fft.irfft(mado, n=full_n) #逆フーリエ変換
    filter = window

    # ゼロ遅延が配列の先頭に来ているので、中心に持ってきたいならfftshiftする
    filter_centered = np.fft.fftshift(filter) #期間はfftlength秒間

    fduration = FDURATION #切り出す秒数
    n_fir = int(round(fduration*fs)) # 要素数を調べる

    if n_fir % 2 == 0:
        n_fir += 1 #対称性のため，要素数を奇数にする

    center = len(filter_centered) // 2 #真ん中の要素の番号 //は割り算の答えの整数値

    half = (n_fir) // 2

    h = filter_centered[center-half:center+half + 1] #切り出す
    h = h * tukey(len(h), alpha=0.5) #窓関数をかけてなだらかにする
    y = fftconvolve(data.value,h,mode="same")

    # 対称フィルタなので前後 fduration/2 分を切り捨てる
    crop_n = int(round((fduration / 2.0) * fs))

    white = TimeSeries(y, sample_rate=data.sample_rate, t0=data.t0)

    if crop_n > 0 and 2 * crop_n < len(white):
        white = white.crop(
            white.t0.value + crop_n / fs,
            white.t0.value + (len(white) - crop_n) / fs,
        )
    return white

def whiten_min(data,FFTLENGTH,FDURATION):
    if FFTLENGTH < FDURATION:
            print("FFTLENGTH > FDURATION にしてネ")
    # --- 実データのASDを推定 ---
    asd = data.asd(fftlength=FFTLENGTH) #周波数は1/fftlength Hz 刻みになる(0Hz-2048Hz) 2048*8 = 16384要素+1
    mado = 1.0/(asd.value) #逆ASD(片側)
    fs = 4096 #データは4096Hz!
    freq = asd.frequencies.value
    ref_mask = (freq >= 30) & (freq <= 500)
    ref_gain = np.median(mado[ref_mask])
    print("ref_gain:", ref_gain)
    cap = ref_gain * 10**(30/20)
    floor = ref_gain * 10**(-30/20)
    print(asd.value[0])
    mado[0] = np.median(mado[ref_mask]) 

    full_n = 2 * (len(mado) - 1)
    window = np.fft.irfft(mado, n=full_n) #逆フーリエ変換
    filter = window

    # ゼロ遅延が配列の先頭に来ているので、中心に持ってきたいならfftshiftする
    filter_centered = np.fft.fftshift(filter) #期間はfftlength秒間

    fduration = FDURATION #切り出す秒数
    n_fir = int(round(fduration*fs)) # 要素数を調べる

    if n_fir % 2 == 0:
        n_fir += 1 #対称性のため，要素数を奇数にする

    center = len(filter_centered) // 2 #真ん中の要素の番号 //は割り算の答えの整数値

    half = (n_fir) // 2

    h = filter_centered[center-half:center+half + 1] #切り出す
    h = h * tukey(len(h), alpha=0.5) #窓関数をかけてなだらかにする

    h_min = minimum_phase(h, method='homomorphic', half=False) #最小位相フィルターを作る

    y = fftconvolve(data.value,h_min,mode="full")[:len(data)]

    # 対称フィルタなので前後 fduration/2 分を切り捨てる
    crop_n = len(h_min)

    white = TimeSeries(y, sample_rate=data.sample_rate, t0=data.t0)
    t0 = data.t0
    if crop_n > 0 and 2 * crop_n < len(white):
        white = white.crop(
            white.t0.value + crop_n / fs,
            white.t0.value + len(white)/fs,
        )
    return white


def plot_impulse_response_comparison(h_zero, h_min, fs, output_path):
    """線形位相 vs 最小位相のインパルス応答を並べてプロットし、
    ringing/時間局在の違いを可視化する。"""
    from matplotlib import pyplot as plt

    t_zero = (np.arange(len(h_zero)) - len(h_zero) // 2) / fs
    t_min = np.arange(len(h_min)) / fs

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(t_zero, h_zero, lw=0.8)
    axes[0].set_title("Linear-phase (zero-phase-like) FIR impulse response")
    axes[0].set_xlabel("time [s] (relative to center)")
    axes[0].set_ylabel("amplitude")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_min, h_min, lw=0.8, color="C1")
    axes[1].set_title("Minimum-phase (causal) FIR impulse response")
    axes[1].set_xlabel("time [s] (relative to filter start)")
    axes[1].set_ylabel("amplitude")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)