#claude製
#!/usr/bin/env python
"""
注入キャンペーン: 実ノイズ(real) と 「同じPSDの合成ガウス」(gauss) で同じ注入信号を推論する。

  - 窓 i ごとに、解析と同じ prior から注入パラメータを引く (シード = i, real/gauss 共通)
  - gauss は、その窓の直前から推定した実PSDで合成 → 違いは「ガウスかどうか」だけ
  - 汚染窓 (巨大グリッチ・NaN・データ端) は自動で除外し、汚染率を記録する

使い方:
  # 0) 窓リストと汚染率の確認 (PEは走らない)
  python pe_campaign.py --event GW200129 --list
  # 1) SNR分布の下見 (PEは走らない) → prior の dL 範囲を決める
  python pe_campaign.py --event GW200129 --survey 200
  # 1b) dL の prior をイベントごとに自動調整 (priors/{event}.prior を作り events.csv の prior 列に書く)
  python pe_campaign.py --event GW200129_065458 --autoprior 300
  # 2) 1 run
  python pe_campaign.py --event GW200129 --window 0 --noise real
  # 3) 並列 (npool=1 を10本同時)
  for n in real gauss; do seq 0 19 | xargs -P 10 -I{} python pe_campaign.py --event GW200129 --window {} --noise $n; done
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import numpy as np
import bilby
from gwpy.timeseries import TimeSeries

# ======================= 設定 =======================
DURATION = 8.0          # 解析窓 [s]
POST = 2.0              # 合体後に残す時間 [s]
FS = 2048.0
MIN_FREQ = 20.0
PSD_SPAN = 128.0        # PSD推定に使う長さ [s] (窓の直前)
PSD_GAP = 2.0
NLIVE = 500             # 本番 1000
PRIOR_FILE = "injection.prior"     # 注入と解析で同じものを使う (校正のため)
FREE = ["chirp_mass", "mass_ratio", "luminosity_distance", "theta_jn"]
FIXED = dict(a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0, phi_12=1.7, phi_jl=0.3,
             psi=2.659, phase=1.3, ra=1.375, dec=-1.2108)

# 汚染窓の自動検出
WHITEN = dict(fftlength=4, overlap=2, fduration=4)
EDGE = 4.0              # ホワイトニング端で捨てる長さ [s]
NSIG = 6.0              # ロバストσ の何倍を超えたら汚染とみなすか (ガウスなら 4096 s で誤検出 ~0.02 回)
PAD = 1.0               # 汚染サンプルの前後に付ける余白 [s]
EXCLUDE_EVENT = 11.0    # 本物のイベントの前後 [s]

EVENTS = {
    "GW200129": dict(file="hdf5/L-L1_GWOSC_4KHZ_R1-1264314069-4096.hdf5",
                     det="L1", t_event=1264316116.4),
    "GW190412": dict(file="hdf5/L-L1_GWOSC_4KHZ_R1-1239080215-4096.hdf5",
                     det="L1", t_event=1239082262.2),
    "GW200224": dict(file="hdf5/L-L1_GWOSC_4KHZ_R1-1266616125-4096.hdf5",
                     det="L1", t_event=1266618172.4),
}

# events.csv (make_event_list.py の出力) があれば、そこから自動で追加する。キーはフル名 (例 GW200129_065458)
EXCLUDE_BG_WINDOWS = True   # bg_auto.py の {name}_L1_bg_windows.json にある過渡・エピソード窓も PE から除く


def _load_events_csv(path="events.csv"):
    import csv
    out = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            out[r["name"]] = dict(file=r["file"], det=r.get("det", "L1"),
                                  t_event=float(r["gps"]), prior=r.get("prior") or None)
    return out


EVENTS.update(_load_events_csv())


def bg_intervals(event, ev):
    """背景テストで判定した過渡ノイズ窓・エピソード窓 (絶対GPS)。無ければ空。"""
    fn = f"{event}_{ev['det']}_bg_windows.json"
    if not (EXCLUDE_BG_WINDOWS and os.path.exists(fn)):
        return []
    j = json.load(open(fn))
    return [tuple(x) for x in j["bad"] + j["episode"]]


# ======================= データと窓 =======================
def load_data(ev):
    return TimeSeries.read(ev["file"], format="hdf5.gwosc").resample(FS)


def find_bad_intervals(data):
    """ホワイトニング後の時系列で |x| > NSIG * (ロバストσ) のサンプル、NaN、データ端を汚染とする。
    返り値: [(t_a, t_b), ...] (絶対GPS)"""
    nan = ~np.isfinite(data.value)
    x = TimeSeries(np.where(nan, 0.0, data.value), t0=data.t0, sample_rate=data.sample_rate)
    w = x.whiten(**WHITEN)
    v, t = w.value, w.times.value
    ok = np.isfinite(v) & ~nan
    med = np.median(v[ok])
    sig = 1.4826 * np.median(np.abs(v[ok] - med))
    bad = nan | ~np.isfinite(v) | (np.abs(v - med) > NSIG * sig)

    t0, t1 = float(data.span[0]), float(data.span[1])
    intervals = [(t0, t0 + EDGE), (t1 - EDGE, t1)]
    idx = np.where(bad)[0]
    if idx.size:
        tb = t[idx]
        cuts = np.where(np.diff(tb) > PAD)[0]
        starts = np.r_[tb[0], tb[cuts + 1]]
        ends = np.r_[tb[cuts], tb[-1]]
        intervals += [(a - PAD, b + PAD) for a, b in zip(starts, ends)]
    return intervals


def _overlap(a0, a1, intervals):
    return any(a0 < b1 and a1 > b0 for b0, b1 in intervals)


def candidate_windows(data, ev, verbose=False):
    """重ならない8秒窓のうち、窓とそのPSD区間が [イベント, 汚染区間] と重ならないもの。
    固定シードで並べ替えて返す (窓番号 i が実行ごとに変わらないように)。"""
    t0, t1 = float(data.span[0]), float(data.span[1])
    look_back = PSD_SPAN + PSD_GAP
    grid = np.arange(t0 + EDGE + look_back, t1 - EDGE - DURATION, DURATION)
    te = ev["t_event"]
    no_event = [t for t in grid
                if not _overlap(t - look_back, t + DURATION,
                                [(te - EXCLUDE_EVENT, te + EXCLUDE_EVENT)])]
    bad = find_bad_intervals(data)
    extra = ev.get("_bg", [])
    clean = [t for t in no_event if not _overlap(t - look_back, t + DURATION, bad + extra)]
    # 「窓そのもの」が汚染された割合 (PSD区間は含めない) = PE窓がグリッチに当たる率
    hit = [t for t in no_event if _overlap(t, t + DURATION, bad[2:])]
    if verbose:
        print(f"候補窓 {len(no_event)} 個 / 汚染区間 {len(bad) - 2} 個")
        for a, b in bad[2:]:
            print(f"  汚染: t-t_event = {a - te:+9.2f} 〜 {b - te:+9.2f} s")
        print(f"窓の汚染率: {len(hit)}/{len(no_event)} = {len(hit) / max(len(no_event), 1):.4f}")
        print(f"背景テスト由来の除外窓: {len(extra)} 個")
        print(f"使える clean 窓 (PSD区間込み): {len(clean)} 個")
    return np.random.default_rng(12345).permutation(np.array(clean)), dict(
        n_windows=len(no_event), n_hit=len(hit), n_clean=len(clean),
        bad=[(a - te, b - te) for a, b in bad[2:]])


# ======================= ifo を作る =======================
def make_real_noise_ifo(data, det, t_c):
    t_end = t_c + POST
    t_start = t_end - DURATION
    seg = data.crop(t_start, t_end)
    psd_ts = data.crop(t_start - PSD_GAP - PSD_SPAN, t_start - PSD_GAP)
    psd = psd_ts.psd(fftlength=DURATION, overlap=DURATION / 2,
                     window=("tukey", 0.1), method="median")
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.minimum_frequency = MIN_FREQ
    ifo.maximum_frequency = FS / 2
    ifo.set_strain_data_from_gwpy_timeseries(seg)
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value)
    return ifo


def make_gauss_noise_ifo_like(real_ifo, det, seed):
    bilby.core.utils.random.seed(seed)
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.minimum_frequency = real_ifo.minimum_frequency
    ifo.maximum_frequency = real_ifo.maximum_frequency
    ifo.power_spectral_density = real_ifo.power_spectral_density
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=real_ifo.sampling_frequency,
        duration=real_ifo.duration, start_time=real_ifo.start_time)
    return ifo


def make_waveform_generator():
    return bilby.gw.WaveformGenerator(
        duration=DURATION, sampling_frequency=FS,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant="IMRPhenomPv2",
                                reference_frequency=50.0, minimum_frequency=MIN_FREQ))


# ======================= 注入パラメータ =======================
def draw_injection(priors, i):
    """窓番号 i をシードに、解析と同じ prior から自由パラメータを引く (real/gauss で共通)。"""
    bilby.core.utils.random.seed(1000 + i)
    s = priors.sample()
    inj = dict(FIXED)
    for k in FREE:
        inj[k] = float(s[k])
    m1, m2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
        inj["chirp_mass"], inj["mass_ratio"])
    inj["mass_1"], inj["mass_2"] = float(m1), float(m2)
    return inj


# ======================= 要約 =======================
def degeneracy_metrics(post):
    th = post["theta_jn"].values
    dl = post["luminosity_distance"].values
    return dict(face_away=float(np.mean(th > np.pi / 2)),
                edge_on=float(np.mean(np.abs(np.cos(th)) < 0.5)),
                corr_cos_dL=float(np.corrcoef(np.abs(np.cos(th)), dl)[0, 1]))


def summarize(result, inj):
    out = {}
    post = result.posterior
    for k in FREE:
        x = post[k].values
        t = inj[k]
        lo, hi = np.quantile(x, [0.05, 0.95])
        out[k] = dict(p=float(np.mean(x < t)),
                      z=float((np.median(x) - t) / np.std(x)),
                      in90=bool(lo <= t <= hi),
                      width90=float(hi - lo))
    out["degeneracy"] = degeneracy_metrics(post)
    return out


# ======================= 実行モード =======================
def get_event(event):
    ev = dict(EVENTS[event])
    ev["_bg"] = bg_intervals(event, ev)
    return ev


def run_one(event, i, noise, outroot="campaign"):
    ev = get_event(event)
    data = load_data(ev)
    wins, info = candidate_windows(data, ev)
    if i >= len(wins):
        raise IndexError(f"window {i} は範囲外 (clean 窓は {len(wins)} 個)")
    t_win = float(wins[i])
    t_c = t_win + DURATION - POST

    priors = bilby.gw.prior.BBHPriorDict(filename=ev.get("prior") or PRIOR_FILE)
    inj = draw_injection(priors, i)
    inj["geocent_time"] = t_c

    ifo_real = make_real_noise_ifo(data, ev["det"], t_c)
    if noise == "real":
        ifo = ifo_real
    else:
        ifo = make_gauss_noise_ifo_like(ifo_real, ev["det"], seed=50000 + i)

    wg = make_waveform_generator()
    ifos = bilby.gw.detector.InterferometerList([ifo])
    ifos.inject_signal(waveform_generator=wg, parameters=inj)
    snr = float(ifo.meta_data["optimal_SNR"])

    for k, v in FIXED.items():
        priors[k] = v
    priors["geocent_time"] = t_c
    priors.validate_prior(DURATION, MIN_FREQ)

    outdir = f"{outroot}/{event}/{noise}/w{i:03d}"
    likelihood = bilby.gw.GravitationalWaveTransient(interferometers=ifos,
                                                     waveform_generator=wg)
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler="dynesty",
                               npoints=NLIVE, npool=1, injection_parameters=inj,
                               outdir=outdir, label="pe",
                               result_class=bilby.gw.result.CBCResult)

    summ = dict(event=event, det=ev["det"], noise=noise, window=i, t_win=t_win,
                t_minus_event=t_win - ev["t_event"], optimal_snr=snr,
                injection={k: inj[k] for k in FREE},
                contamination=dict(n_windows=info["n_windows"], n_hit=info["n_hit"]))
    summ.update(summarize(result, inj))
    with open(f"{outdir}/summary.json", "w") as f:
        json.dump(summ, f, indent=1)
    print(f"done {event} {noise} w{i}  SNR={snr:.1f}")


def survey_snr(event, n):
    """PE なしで、prior から n 個引いたときの optimal SNR 分布を見る (dL 範囲の調整用)。"""
    ev = get_event(event)
    data = load_data(ev)
    wins, _ = candidate_windows(data, ev)
    priors = bilby.gw.prior.BBHPriorDict(filename=ev.get("prior") or PRIOR_FILE)
    wg = make_waveform_generator()
    t_c = float(wins[0]) + DURATION - POST
    ifo = make_gauss_noise_ifo_like(make_real_noise_ifo(data, ev["det"], t_c), ev["det"], 0)
    snrs, dls = [], []
    for i in range(n):
        inj = draw_injection(priors, i)
        inj["geocent_time"] = t_c
        pol = wg.frequency_domain_strain(inj)
        sig = ifo.get_detector_response(pol, inj)
        snrs.append(float(np.sqrt(ifo.optimal_snr_squared(sig).real)))
        dls.append(inj["luminosity_distance"])
    snrs, dls = np.array(snrs), np.array(dls)
    print(f"optimal SNR: 5/25/50/75/95% = {np.percentile(snrs, [5, 25, 50, 75, 95]).round(1)}")
    print(f"SNR<8: {np.mean(snrs < 8):.2f}   8-25: {np.mean((snrs >= 8) & (snrs <= 25)):.2f}"
          f"   >25: {np.mean(snrs > 25):.2f}")
    for lo, hi in [(50, 300), (300, 600), (600, 1000), (1000, 1500), (1500, 2000)]:
        m = (dls >= lo) & (dls < hi)
        if m.any():
            print(f"  dL {lo:4d}-{hi:4d} Mpc: N={m.sum():3d}  SNR中央値={np.median(snrs[m]):.1f}")


# ======================= dL prior の自動調整 =======================
import csv
import re

SNR_MIN = 10.0     # この SNR を下回る注入をおおむね Q_LOW 以下に抑える
Q_LOW = 0.10
SNR_MAX = 40.0     # SNR の中央値がこれを超えないよう dL の下限を決める
DL_FLOOR = 50.0


def _snr_amplitudes(event, n):
    """base prior から n 個引き、K = SNR × dL (= 1 Mpc での SNR) を返す。
    SNR は dL に反比例するので、K が分かれば任意の dL 範囲での SNR が計算できる。"""
    ev = get_event(event)
    data = load_data(ev)
    wins, _ = candidate_windows(data, ev)
    priors = bilby.gw.prior.BBHPriorDict(filename=PRIOR_FILE)       # 常にベースの prior から
    wg = make_waveform_generator()
    t_c = float(wins[0]) + DURATION - POST
    ifo = make_gauss_noise_ifo_like(make_real_noise_ifo(data, ev["det"], t_c), ev["det"], 0)
    K = []
    for i in range(n):
        inj = draw_injection(priors, i)
        inj["geocent_time"] = t_c
        sig = ifo.get_detector_response(wg.frequency_domain_strain(inj), inj)
        K.append(float(np.sqrt(ifo.optimal_snr_squared(sig).real)) * inj["luminosity_distance"])
    return np.array(K)


def choose_dl_range(K, snr_min=SNR_MIN, q_low=Q_LOW, snr_max=SNR_MAX, floor=DL_FLOOR):
    """dL 上限: その距離でも (1-q_low) の注入が SNR>=snr_min になる距離。
    dL 下限: 典型的な (中央値の) 注入が SNR=snr_max になる距離。"""
    dmax = np.quantile(K, q_low) / snr_min
    dmin = max(floor, np.median(K) / snr_max)
    if dmin >= 0.8 * dmax:                      # 感度が低すぎて幅が取れないとき
        dmin = 0.5 * dmax
    return float(dmin), float(dmax)


def write_prior(base, out, dmin, dmax):
    """base prior の luminosity_distance 行の minimum/maximum だけを書き換えて保存する。"""
    txt = open(base).read()
    pat = re.compile(r"^(luminosity_distance\s*=.*)$", re.M)
    m = pat.search(txt)
    if m is None:
        raise ValueError(f"{base} に luminosity_distance の行がありません")
    line = m.group(1)
    line = re.sub(r"minimum\s*=\s*[0-9.eE+-]+", f"minimum={dmin:.1f}", line)
    line = re.sub(r"maximum\s*=\s*[0-9.eE+-]+", f"maximum={dmax:.1f}", line)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write(txt[:m.start(1)] + line + txt[m.end(1):])


def set_event_prior(event, prior_path, path="events.csv"):
    """events.csv の prior 列を更新する (列が無ければ追加)。"""
    rows = list(csv.DictReader(open(path)))
    fields = list(rows[0].keys()) if rows else []
    if "prior" not in fields:
        fields.append("prior")
    for r in rows:
        r.setdefault("prior", "")
        if r["name"] == event:
            r["prior"] = prior_path
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def autoprior(event, n):
    K = _snr_amplitudes(event, n)
    dmin, dmax = choose_dl_range(K)
    out = f"priors/{event}.prior"
    write_prior(PRIOR_FILE, out, dmin, dmax)
    # 新しい範囲での SNR 分布を、同じ K から dL を引き直して見積もる (PowerLaw alpha=2)
    rng = np.random.default_rng(0)
    u = rng.uniform(size=K.size)
    dl = (dmin ** 3 + u * (dmax ** 3 - dmin ** 3)) ** (1 / 3)
    snr = K / dl
    print(f"{event}: dL {dmin:.0f}–{dmax:.0f} Mpc -> {out}")
    print(f"  予想 SNR 5/50/95% = {np.percentile(snr, [5, 50, 95]).round(1)}  "
          f"SNR<8: {np.mean(snr < 8):.2f}  SNR<10: {np.mean(snr < 10):.2f}")
    if os.path.exists("events.csv") and event in _load_events_csv():
        set_event_prior(event, out)
        print("  events.csv の prior 列を更新")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--window", type=int)
    ap.add_argument("--noise", choices=["real", "gauss"])
    ap.add_argument("--outroot", default="campaign")
    ap.add_argument("--list", action="store_true", help="窓と汚染率を表示して終了")
    ap.add_argument("--survey", type=int, default=0, help="SNR分布の下見 (個数)")
    ap.add_argument("--autoprior", type=int, default=0, help="dL prior を自動調整 (下見の個数)")
    a = ap.parse_args()
    if a.list:
        ev = get_event(a.event)
        candidate_windows(load_data(ev), ev, verbose=True)
    elif a.autoprior:
        autoprior(a.event, a.autoprior)
    elif a.survey:
        survey_snr(a.event, a.survey)
    else:
        run_one(a.event, a.window, a.noise, a.outroot)