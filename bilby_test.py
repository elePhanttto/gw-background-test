#!/usr/bin/env python
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # ← numpy/bilby の import より前に置く
import numpy as np
import bilby
from gwpy.timeseries import TimeSeries


def make_real_noise_ifo(data, det, t_c, duration=8.0, psd_span=128.0, psd_gap=2.0,
                        fmin=20.0, post=2.0):
    """data: すでに目的のサンプリング周波数にリサンプル済みの gwpy TimeSeries。
    解析窓は [t_c + post - duration, t_c + post]。PSD は窓の直前 psd_span 秒から median Welch。"""
    fs = data.sample_rate.value
    t_end = t_c + post
    t_start = t_end - duration
    t0, t1 = data.span
    if t_start - psd_gap - psd_span < t0 or t_end > t1:
        raise ValueError(f"窓またはPSD区間がデータ外: t_c={t_c:.2f}, data=[{t0}, {t1}]")

    seg = data.crop(t_start, t_end)
    psd_ts = data.crop(t_start - psd_gap - psd_span, t_start - psd_gap)
    psd = psd_ts.psd(fftlength=duration, overlap=duration / 2,
                     window=("tukey", 0.1), method="median")

    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.minimum_frequency = fmin
    ifo.maximum_frequency = fs / 2
    ifo.set_strain_data_from_gwpy_timeseries(seg)
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value)
    return ifo


def make_random_time(data, geocent_time, exclude_time, duration_time, n_seg_half,
                     buffer=140.0, win=8.0):
    """解析窓の開始時刻 (絶対GPS) のリストを返す。窓は [t, t+win]。"""
    t0, t1 = data.span
    t0, t1 = float(t0), float(t1)
    pre = np.linspace(t0 + buffer, geocent_time - exclude_time - win, n_seg_half,
                      endpoint=False)
    post = np.linspace(geocent_time + exclude_time, t1 - win - 1.0, n_seg_half,
                       endpoint=False)
    random_time = np.concatenate([pre, post])

    # NaN を含む窓を除く（PSD区間も含めて）
    if np.isnan(data.value).any():
        times = data.times.value
        bad = times[np.isnan(data.value)]
        ok = [t for t in random_time
              if not np.any((bad >= t - buffer) & (bad <= t + win))]
        random_time = np.array(ok)
        print(f"NaN除外後の候補数: {len(random_time)}")

    print(f"窓開始 最小 {random_time.min():.1f} / 最大 {random_time.max():.1f} "
          f"(データ {t0:.1f}–{t1:.1f})")
    return random_time

def make_gauss_noise_ifo_like(real_ifo, det, seed):
    """real_ifo と同じ PSD・時刻・周波数範囲で、合成ガウスノイズの ifo を作る (条件A)"""
    bilby.core.utils.random.seed(seed)
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.minimum_frequency = real_ifo.minimum_frequency
    ifo.maximum_frequency = real_ifo.maximum_frequency
    ifo.power_spectral_density = real_ifo.power_spectral_density   # ← 実PSDをそのまま使う
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=real_ifo.sampling_frequency,
        duration=real_ifo.duration,
        start_time=real_ifo.start_time)
    return ifo

def takusan_inference(random_time, priors_base, injection_parameters, waveform_generator,
                      data, n_windows, outdir_base, label, nlive, npool, det,
                      duration, min_freq, post=2.0, skip_ranges=()):
    """random_time から n_windows 個を等間隔に選び、各窓に注入して PE する。
    skip_ranges: [(t_a, t_b), ...] 窓と重なったら飛ばす区間 (例: seg#126 のグリッチ)。"""
    step = max(len(random_time) // n_windows, 1)
    chosen = random_time[1::step][:n_windows]

    fixed = ["a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
             "psi", "phase", "ra", "dec"]

    for t_win in chosen:
        t_c = t_win + duration - post          # 窓 [t_win, t_win+8] の中で合体が終わり2秒前
        if any(a < t_win + duration and b > t_win - 130.0 for a, b in skip_ranges):
            print(f"skip t_win={t_win:.1f} (グリッチ区間と重なる)")
            continue

        inj = dict(injection_parameters)
        inj["geocent_time"] = t_c               # ← 注入時刻を窓に合わせる

        ifos = bilby.gw.detector.InterferometerList(
            [make_real_noise_ifo(data, det, t_c, duration=duration, fmin=min_freq, post=post)])
        ifos.inject_signal(waveform_generator=waveform_generator, parameters=inj)

        priors = priors_base.copy()
        for key in fixed:
            priors[key] = inj[key]
        priors["geocent_time"] = t_c            # ← 固定値も窓に合わせる
        priors.validate_prior(duration, min_freq)

        free = [k for k, v in priors.items()
                if isinstance(v, bilby.core.prior.Prior)
                and not isinstance(v, (bilby.core.prior.Constraint,
                                       bilby.core.prior.DeltaFunction))]
        print(f"★ t_win={t_win:.1f}  t_c={t_c:.3f}  sampling: {free}")

        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=ifos, waveform_generator=waveform_generator)

        outdir = f"{outdir_base}/t{t_win:.1f}"      # ← 毎回ベース名から作る
        result = bilby.run_sampler(
            likelihood=likelihood, priors=priors, sampler="dynesty",
            npoints=nlive, npool=npool, injection_parameters=inj,
            outdir=outdir, label=label, result_class=bilby.gw.result.CBCResult)

        post_df = result.posterior
        if "mass_1" not in post_df.columns:
            m1, m2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
                post_df["chirp_mass"].values, post_df["mass_ratio"].values)
            post_df["mass_1"] = m1
            post_df["mass_2"] = m2
        truths = {k: inj[k] for k in ["mass_1", "mass_2", "luminosity_distance", "theta_jn"]}
        result.plot_corner(parameters=truths, filename=f"{outdir}/corner_m1m2.png")
