#Claude製
#!/usr/bin/env python
"""各PE窓について、注入前のノイズを PE と同じ推定PSDで白色化したときの分散を測る。
  gauss     : 真のPSDで白色化 → 定義上 1 (式の確認用)
  gauss_est : 合成ガウス + PSD推定 → 推定方法のクセの基準線
  real      : 実ノイズ + PSD推定
real/gauss_est の比が、実ノイズによる「パワーのずれ」の正味の大きさ。
pe_campaign.py と同じフォルダで実行する。"""
import json, glob
import numpy as np, pandas as pd
import pe_campaign as pc

BANDS = [(20, 60), (60, 200), (20, 1024)]


def wvar(ifo):
    """白色化したノイズの分散を帯域ごとに返す (ガウスで期待値 1)"""
    f = ifo.frequency_array
    d = ifo.strain_data.frequency_domain_strain
    S = ifo.power_spectral_density_array
    T = ifo.duration
    wf = ifo.strain_data.window_factor            # bilby の tukey 窓の補正
    ratio = 2 * np.abs(d) ** 2 / (S * T * wf)     # E|n(f)|^2 = S T / 2 (bilby の規約)
    out = {}
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi) & np.isfinite(ratio)
        out[f"{lo}_{hi}"] = float(ratio[m].mean())
    return out


rows = []
events = sorted({json.load(open(f))["event"] for f in glob.glob("campaign/*/real/w*/summary.json")})
for ev_name in events:
    ev = pc.get_event(ev_name)
    data = pc.load_data(ev)
    for fn in sorted(glob.glob(f"campaign/{ev_name}/real/w*/summary.json")):
        s = json.load(open(fn))
        i = s["window"]
        t_c = s["t_win"] + pc.DURATION - pc.POST
        ifo_real = pc.make_real_noise_ifo(data, ev["det"], t_c)
        # PE と同じシードなので、PE で使ったのと同じノイズが再現される
        ifos = dict(real=ifo_real,
                    gauss=pc.make_gauss_noise_ifo_like(ifo_real, ev["det"], seed=50000 + i),
                    gauss_est=pc.make_gauss_est_ifo(ifo_real, ev["det"], seed=50000 + i, t_c=t_c))
        r = dict(event=ev_name, window=i)
        for noise, ifo in ifos.items():
            for band, v in wvar(ifo).items():
                r[f"wvar_{noise}_{band}"] = v
        for lo, hi in BANDS:
            b = f"{lo}_{hi}"
            r[f"ratio_real_est_{b}"] = r[f"wvar_real_{b}"] / r[f"wvar_gauss_est_{b}"]
        rows.append(r)
    print(ev_name, "done")

df = pd.DataFrame(rows)
df.to_csv("window_whitened_var.csv", index=False)
print(df.filter(like="wvar_").median().round(3).to_string())