#!/usr/bin/env python
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import bilby
from gwpy.timeseries import TimeSeries
from bilby_test import make_random_time, takusan_inference

NPOOL = 1
NLIVE = 500          # 試運転。本番1000
DURATION = 8.0
SAMPLING_FREQUENCY = 2048.0
MIN_FREQ = 20.0
det = "L1"


def main():
    outdir = "outdir_injection_real"
    label = "inj"
    bilby.core.utils.setup_logger(outdir=outdir, label=label, log_level="INFO")
    bilby.core.utils.random.seed(88170235)

    injection_parameters = dict(
        mass_1=36.0, mass_2=29.0, a_1=0.4, a_2=0.3,
        tilt_1=0.5, tilt_2=1.0, phi_12=1.7, phi_jl=0.3,
        luminosity_distance=400.0, theta_jn=0.4, psi=2.659, phase=1.3,
        geocent_time=0.0,    # takusan_inference 内で窓ごとに上書き
        ra=1.375, dec=-1.2108,
    )
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant="IMRPhenomPv2",
                                reference_frequency=50.0, minimum_frequency=MIN_FREQ),
    )

    data_L1 = TimeSeries.read("hdf5/L-L1_GWOSC_4KHZ_R1-1264314069-4096.hdf5",
                              format="hdf5.gwosc").resample(SAMPLING_FREQUENCY)  # ← 1回だけ
    geocent_time_event = 1264316116.4
    t_glitch = geocent_time_event + 561.79
    priors = bilby.gw.prior.BBHPriorDict(filename="injection.prior")

    random_time = make_random_time(data_L1, geocent_time_event, exclude_time=11.0,
                                   duration_time=4096.0, n_seg_half=100)
    takusan_inference(random_time, priors, injection_parameters, waveform_generator,
                      data_L1, n_windows=3, outdir_base=outdir, label=label,
                      nlive=NLIVE, npool=NPOOL, det=det,
                      duration=DURATION, min_freq=MIN_FREQ,
                      skip_ranges=[(t_glitch - 1.0, t_glitch + 1.0)])  # 条件B: グリッチ窓は除外


if __name__ == "__main__":
    main()