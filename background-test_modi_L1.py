#小さい値を一定の値に固定するバージョン

from gwpy.timeseries import TimeSeries
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from random import *
import math

output1 = "L1_kstest.png" #ksテスト
output2 = "L1_chi2test.png" #アウトプットのファイル
output3 = "L1_adtest.png" #adテスト

geocent_time = 1239082262 #ここではGW190412の合体時刻!
exclude_time = 11 #除外する前後の区間
duration_time = 4096 #調べる区間!
duration_time_half = 2048 #前後2048秒
i = 200
random_time = [] #ランダムに選ぶ区間のための配列
adlist = []
kslist = []
chi2list = []
random_time_list = []
skipped = 0

for j in range(0,100):
    n = uniform(geocent_time-duration_time_half,geocent_time - exclude_time)
    random_time.append(n)

for j in range(0,100):
    n = uniform(geocent_time + exclude_time,geocent_time+duration_time_half)
    random_time.append(n)

data_L1 = TimeSeries.read("L-L1_GWOSC_4KHZ_R1-1239080215-4096.hdf5", end = 1239080215 + duration_time,format="hdf5.gwosc") #gwoscからデータを入れておいてください!

# NaNのない範囲を特定
valid = ~np.isnan(data_L1.value)
if not valid.all():
    times = data_L1.times.value
    valid_times = times[valid]
    t_start, t_end = valid_times.min(), valid_times.max()
    print(f"⚠️ 有効区間: {t_start:.1f} - {t_end:.1f} ({t_end-t_start:.0f}秒)")
    
    # 有効区間だけ切り出す（端に少し余裕を持たせる）
    data_L1 = data_L1.crop(t_start, t_end)
    
    # 候補時刻も有効区間内に絞る
    margin = 8.0
    random_time = [t for t in random_time
                   if (t_start + margin) < t and (t + 1.0) < (t_end - margin)]
    print(f"有効な候補数: {len(random_time)}")

white = data_L1.whiten(fftlength=4, overlap=2) #ホワイトニング

# ============================================
# AD検定の実装（論文式3.4準拠）
# ============================================

def ad_statistic(y_norm):
    """
    正規化エネルギーに対するAnderson-Darling統計量を計算する。
    帰無仮説: y ~ Exp(1) （F(y) = 1 - exp(-y)）
    """
    y = np.sort(np.asarray(y_norm))
    N = len(y)
    F = 1.0 - np.exp(-y)          # 指数分布(scale=1)のCDF
    
    # log(0)を避けるためのクリップ
    eps = 1e-300
    F = np.clip(F, eps, 1.0 - eps)
    
    i = np.arange(1, N + 1)
    A2 = -N - np.sum((2*i - 1) / N * (np.log(F) + np.log(1.0 - F[::-1])))
    return A2


def build_ad_null_distribution(N, n_sim=1e+8, seed=42):
    """
    帰無仮説下でのA^2分布をモンテカルロで構築する。
    N: サンプル数（タイル数）
    n_sim: シミュレーション回数
    """
    rng = np.random.default_rng(seed)
    A2_sim = np.empty(n_sim)
    for k in range(n_sim):
        y_sim = rng.exponential(scale=1.0, size=N)
        y_sim /= y_sim.mean()      # 実データと同じ正規化を適用
        A2_sim[k] = ad_statistic(y_sim)
    return np.sort(A2_sim)


def ad_pvalue(A2_obs, A2_null):
    """経験分布からp値を計算（右側確率）"""
    n_ge = np.sum(A2_null >= A2_obs)
    if n_ge == 0:
        return 1.0 / len(A2_null)   # 下限値を返す
    return n_ge / len(A2_null)

def equiprob_bins(n_bins=9):
    # 指数分布(scale=1)で等確率になるビン境界
    return -np.log(1 - np.arange(1, n_bins) / n_bins)

def Y_distribution_equiprob(y_norm, n_bins=9): #等確率になるビン境界での分布関数
    edges = np.concatenate([[0], equiprob_bins(n_bins), [np.inf]])
    obs, _ = np.histogram(y_norm, bins=edges)
    exp = np.full(n_bins, len(y_norm) / n_bins)
    return obs, exp

def kai2jou(f_obs,f_exp): #カイ二乗検定の統計量を直接計算
    goukei = 0.0
    for i in range(0,len(f_exp)):
        goukei += (f_obs[i] - f_exp[i])**2 / f_exp[i]
    return goukei

# ---- 帰無分布の準備 ----
# まず1つ目のセグメントでタイル数を確認
_seg0 = white.crop(random_time[0], random_time[0] + 1.0)
_qg0 = _seg0.q_gram(qrange=[8, 8], frange=[30.0, 500.0], snrthresh=0)
N_tiles = len(_qg0["energy"])
print(f"帰無分布を構築中（N={N_tiles}, n_sim=10000）...")

A2_null = build_ad_null_distribution(N_tiles, n_sim=10000)
print(f"帰無分布の中央値: {np.median(A2_null):.3f}, 99%点: {np.percentile(A2_null, 99):.3f}")

for j in range(len(random_time)):
    seg = white.crop(random_time[j],random_time[j] + 1.0) #切り出し
    qgram_L1 = seg.q_gram(qrange=[8, 8], frange=[30.0, 500.0], snrthresh=0) #q-gramでq-transform
    y_L1 = np.asarray(qgram_L1["energy"]) #エネルギーの部分を取り出す
    y_norm = y_L1 / y_L1.mean() #正規化
    print(f"タイル数: {len(y_norm)}")
    obs, exp = Y_distribution_equiprob(y_norm,n_bins=9)
    #print(y_dist,expdist)
    # 6. 検定
    ks = stats.kstest(y_norm, "expon")
    #ad = stats.anderson(y_norm, dist="expon") //アトで
    chi2_stat, chi2_p_value = stats.chisquare(f_obs=obs,f_exp=exp)
    # ---- AD検定 ----
    A2_obs = ad_statistic(y_norm)
    ad_p = ad_pvalue(A2_obs, A2_null)
    
    FLOOR = 1.0e-8
    kslist.append(max(ks.pvalue, FLOOR))
    chi2list.append(max(chi2_p_value, FLOOR))
    adlist.append(max(ad_p, FLOOR))
    random_time_list.append(random_time[j])
    
    print(f"t={random_time[j]-geocent_time:+8.1f}s  "
          f"KS p={ks.pvalue:.2e}  chi2 p={chi2_p_value:.2e}  "
          f"AD A2={A2_obs:.2f} p={ad_p:.2e}")

print(f"NaNでスキップしたセグメント: {skipped}")

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

plt.xlabel("Time from GW190412 geocent time [s]")
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

plt.xlabel("Time from GW190412 geocent time [s]")
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
plt.axhline(1e-8, linestyle="--", label="p = 1e-8")
plt.xlabel("Time from GW190412 geocent time [s]")
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
plt.title("KS p-value distribution")
plt.savefig("pvalue_hist_L1.png")

# 分布チェック!

plt.figure(figsize=(8,5))
plt.hist(y_norm, bins=60, density=True, alpha=0.6, label="observed")
x = np.linspace(0, 10, 200)
plt.plot(x, np.exp(-x), 'r-', lw=2, label=r"$e^{-y}$")
plt.yscale('log')
plt.xlabel("normalized energy")
plt.legend()
plt.savefig("dist_check_L1.png")

