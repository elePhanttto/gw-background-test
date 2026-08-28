#Claude作成

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from gwpy.timeseries import TimeSeries

event = "GW191215_223052"
det = "L1"
geocent_time = 1260484270.3
BLOCK = 256.0

data = TimeSeries.read("hdf5/L-L1_GWOSC_4KHZ_R1-1260482223-4096.hdf5",
                       format="hdf5.gwosc")

t0 = data.t0.value
t_end = t0 + data.duration.value
edges = np.arange(t0, t_end - BLOCK, BLOCK)

# 解析する周波数帯（散乱光は低周波、ショットノイズは高周波に出る）
SUBBANDS = [(30, 50), (50, 100), (100, 250), (250, 500)]

psds, centers = [], []
for lo in edges:
    chunk = data.crop(lo, lo + BLOCK)
    if np.isnan(chunk.value).any():
        continue
    p = chunk.psd(fftlength=4, overlap=2, method="median")
    psds.append(p.value)
    centers.append(lo + BLOCK / 2 - geocent_time)

psds = np.array(psds)
centers = np.array(centers)
freqs = p.frequencies.value
ref = np.median(psds, axis=0)          # 全ブロックの中央値を基準にする

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # numpy 1.x/2.x 両対応

# --- ブロックごとの帯域パワーを集計 ---
rows = {"time_offset": centers}
for f_lo, f_hi in SUBBANDS:
    m = (freqs >= f_lo) & (freqs <= f_hi)
    pw = trapz(psds[:, m], freqs[m], axis=1)
    pw_ref = trapz(ref[m], freqs[m])
    rows[f"power_{f_lo}_{f_hi}"] = pw
    rows[f"ratio_{f_lo}_{f_hi}"] = pw / pw_ref

band = (freqs >= 30) & (freqs <= 500)
rows["power_full"] = trapz(psds[:, band], freqs[band], axis=1)
rows["ratio_full"] = rows["power_full"] / trapz(ref[band], freqs[band])

# 各ブロックで中央値PSDから最も外れた比（局所的な線ノイズの検出用）
rt = psds[:, band] / ref[band]
rows["ratio_max"] = rt.max(axis=1)
rows["ratio_min"] = rt.min(axis=1)

df = pd.DataFrame(rows)
df.to_csv(f"{event}_{det}_psddrift.csv", index=False)

# --- サマリ ---
lines = [f"event={event} det={det} BLOCK={BLOCK:.0f}s n_block={len(df)}", ""]
lines.append(f"{'band [Hz]':<14}{'変動係数%':>10}{'比 最小':>9}{'比 最大':>9}")
for f_lo, f_hi in SUBBANDS + [(30, 500)]:
    key = "ratio_full" if (f_lo, f_hi) == (30, 500) else f"ratio_{f_lo}_{f_hi}"
    r = df[key]
    lines.append(f"{f'{f_lo}-{f_hi}':<14}{r.std()/r.mean()*100:>10.1f}"
                 f"{r.min():>9.3f}{r.max():>9.3f}")

txt = "\n".join(lines)
print(txt)
with open(f"{event}_{det}_psddrift.txt", "w", encoding="utf-8") as f:
    f.write(txt + "\n")

# --- 図 ---
fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
mesh = ax[0].pcolormesh(centers, freqs[band], rt.T,
                        vmin=0.5, vmax=2.0, cmap="RdBu_r", shading="auto")
ax[0].set_yscale("log"); ax[0].set_ylabel("frequency [Hz]")
fig.colorbar(mesh, ax=ax[0], label="PSD / median PSD")

for f_lo, f_hi in SUBBANDS:
    ax[1].plot(centers, df[f"ratio_{f_lo}_{f_hi}"], marker="o", ms=3,
               label=f"{f_lo}-{f_hi} Hz")
ax[1].axhline(1.0, c="gray", lw=0.5)
ax[1].set_xlabel("time from merger [s]")
ax[1].set_ylabel("band power / median")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(f"{event}_{det}_psddrift.png", dpi=150)