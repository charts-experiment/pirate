#!/usr/bin/env python3
"""
Synthetic FRB simulation engine with fitburst, following Fonseca et al.
(2024) "Modeling the Morphology of Fast Radio Bursts and Radio Pulsars with
fitburst" (ApJS 271:49, arXiv:2311.05829).

The group-2 and group-3 generators only import the physical constants below
(DM_CONST_SECONDS, DM_INDEX, SCATTERING_INDEX, REF_FREQ); the per-channel model
is in 07_generate_comparison_set.py. generate_frb_fitburst() is the group-1
engine: given the DM/width/scattering/spectrum/flux of ONE FRB, it produces a
.npz with signal + noise using fitburst's SpectrumModeler.

Requires the `fitburst` package (https://github.com/CHIMEFRB/fitburst),
installed normally (pip install / pip install -e).

Physics:
- amplitude=[0.0] (unit) and scaling by the real flux (Section 2.2)
- dm_index=-2.0 (cold plasma law, Equation 5)
- scattering_index=-4.0 (Kolmogorov spectrum, Equation 4)
- ref_freq=400.1953125 MHz (bottom edge of the CHIME band, 400-800 MHz)
- fitburst's dispersion constant (4149.377593360996) gives the delay
  DIRECTLY IN SECONDS (not ms) -- see fitburst/backend/general.yaml
- Since ref_freq is the BOTTOM edge of the band, it is the frequency that
  ARRIVES LAST (low frequencies are delayed more), so arrival_time (defined
  at ref_freq) sits near the end of the time window, not at the start.
- Radiometer-equation noise: sigma = SEFD / sqrt(n_pol * Δν * τ)
"""
import time as pytime
from pathlib import Path

import numpy as np
import fitburst as fb

# ============================================================================
# FIXED PHYSICAL PARAMETERS -- edit HERE. The whole pipeline imports them from
# this module, so a change here is enough.
#
#   DM_CONST_SECONDS  Cold-plasma dispersion constant (Equation 5, Fonseca et
#                     al. 2024). Delay in SECONDS (not ms) -- see
#                     fitburst/backend/general.yaml. Do not change unless
#                     fitburst's own constant changes.
#   DM_INDEX          Index of the law t(ν) ∝ DM·ν^dm_index. -2.0 is the EXACT
#                     law of non-magnetized cold plasma -- not a free parameter
#                     per FRB. It is overridden ONLY for the negatives (on
#                     purpose, to generate control signals with an invalid index).
#   SCATTERING_INDEX  Index of the scattering law used by fitburst:
#                     τ(ν) = scattering_timescale · (ν/ref_freq)^scattering_index
#                     (see fitburst/analysis/model.py, compute_profile()).
#                     -4.0 = Kolmogorov spectrum (Equation 4). Not a free
#                     parameter per FRB either.
#   REF_FREQ          Frequency at which fitburst defines ALL its
#                     frequency-dependent parameters (arrival_time,
#                     scattering_timescale). Fixed at the bottom edge of the
#                     CHIME band (400-800 MHz, Fonseca et al. 2024). NOTE: it is
#                     not the simulated CHARTS band (300-501.6 MHz); it is only
#                     fitburst's definition anchor.
# ============================================================================
DM_CONST_SECONDS = 4149.377593360996  # s * MHz^2 / (pc cm^-3) -- fitburst/backend/general.yaml
DM_INDEX = -2.0          # FIXED: cold plasma law
SCATTERING_INDEX = -4.0  # FIXED: Kolmogorov spectrum
REF_FREQ = 400.1953125   # FIXED: Fonseca et al. (2024), bottom edge of the CHIME band

# Default SEFD if none is given (from the CHIME specifications: T_sys=50K,
# physical area 8000 m², assumed aperture efficiency 0.6 -- see charts_sefd.py
# for the actual CHARTS value).
K_BOLTZMANN = 1.380649e-23      # J/K
T_SYS_K = 50.0
A_PHYSICAL_M2 = 8000.0
APERTURE_EFFICIENCY = 0.6
A_EFFECTIVE_M2 = A_PHYSICAL_M2 * APERTURE_EFFICIENCY
SEFD_JY_DEFAULT = 2.0 * K_BOLTZMANN * T_SYS_K / A_EFFECTIVE_M2 / 1e-26  # 1 Jy = 1e-26 W/m^2/Hz


def dispersion_delay(dm, freq_mhz, ref_freq=REF_FREQ, dm_index=DM_INDEX):
    """Delay (seconds) of freq_mhz relative to ref_freq. Fonseca Equation 5."""
    return dm * DM_CONST_SECONDS * (np.asarray(freq_mhz) ** dm_index - ref_freq ** dm_index)


def generate_frb_fitburst(
    frb_name, dm, width_fitb_s, scat_s_600mhz, sp_idx, sp_run, flux_jy,
    snr_target=None,
    freq_lo=REF_FREQ, freq_hi=800.0, num_freq=1024, res_time_ms=0.5,
    sefd_jy=None, n_pol=2.0, dm_index=DM_INDEX,
    approximate_params=None, output_dir=None, noise_free=False,
):
    """
    Generates a synthetic FRB with fitburst.

    Parameters
    ----------
    frb_name : str
    dm : float
        Dispersion measure [pc/cm³]
    width_fitb_s : float
        Gaussian width of the pulse [s]
    scat_s_600mhz : float
        Scattering timescale at 600 MHz [s]
    sp_idx, sp_run : float
        Spectral index and spectral running (Equation 2)
    flux_jy : float
        Band-averaged peak flux [Jy]
    snr_target : float, optional
        If given, rescales the signal to reach this peak S/N
    freq_lo, freq_hi : float
        Simulated frequency band [MHz]
    num_freq : int
        Number of channels
    res_time_ms : float
        Time resolution [ms]
    sefd_jy : float, optional
        Instrument SEFD [Jy]. None -> SEFD_JY_DEFAULT (approx. CHIME).
        For CHARTS use charts_sefd.CHARTS_SEFD_JY.
    n_pol : float
        Polarizations summed in the radiometer equation. CHIME: 2.0
        (default). CHARTS: 1.0 (charts_sefd.CHARTS_N_POL) -- a single
        polarization, Table 1 of Cassanelli et al. (2024).
    dm_index : float, optional
        Index of the dispersion law, t(ν) ∝ DM·ν^dm_index. FIXED at -2.0
        for any real FRB (non-magnetized cold plasma, Equation 5 of
        Fonseca et al. 2024) -- not a free astrophysical parameter. It is an
        argument only to generate "non-FRB" control signals with a physically
        invalid index (e.g. outside [1.8, 2.2]): since the detectors dedisperse
        assuming the standard index -2.0, a signal with a different index is
        never compensated correctly at any DM trial -- a genuine negative class
        to measure specificity.
    approximate_params : list[str], optional
        Names of parameters that are NOT a real measurement of this FRB (recorded
        in the metadata; it does not affect the computation).
    output_dir : str or Path, optional
        Output folder of the .npz. Default: ./simulations/ relative to this file.
    noise_free : bool, optional
        If True, no radiometer noise is added -- data_full is identical to
        data_clean (pure signal). snr_peak/snr_integrated are inf (there is no
        noise floor to measure S/N against). Incompatible with snr_target.

    Returns
    -------
    dict with filename, path, shape, snr_peak, snr_integrated, dm, flux_jy.
    """
    print(f"\n{'='*80}\nFRB GENERATOR - FITBURST (Fonseca et al. 2024)\n{'='*80}")
    print(f"FRB: {frb_name} | DM: {dm:.2f} pc/cm³ | flux: {flux_jy:.2f} Jy | "
          f"sp_idx: {sp_idx:.2f} | sp_run: {sp_run:.2f}")
    if approximate_params:
        print(f"  ⚠ APPROXIMATE (catalog median, not measured for this FRB): "
              f"{', '.join(approximate_params)}")

    # ---- Band, dispersion sweep and time window ----
    res_freq_mhz = (freq_hi - freq_lo) / (num_freq - 1)
    freqs = np.linspace(freq_lo, freq_hi, num_freq)
    res_time = res_time_ms / 1000.0

    # Frequencies BELOW ref_freq have a POSITIVE delay (they arrive later);
    # ABOVE, a NEGATIVE delay (they arrive earlier). Valid whether ref_freq
    # is at the edge of the band or inside it (e.g. when the observed band
    # extends to lower frequencies).
    delay_lo = dispersion_delay(dm, freq_lo, ref_freq=REF_FREQ, dm_index=dm_index)
    delay_hi = dispersion_delay(dm, freq_hi, ref_freq=REF_FREQ, dm_index=dm_index)
    sweep_s = delay_lo - delay_hi  # always positive (freq_hi arrives before freq_lo)

    margin_start = max(5.0 * width_fitb_s, 0.005)
    margin_end = max(5.0 * max(width_fitb_s, scat_s_600mhz), 0.01)

    # arrival_time is defined AT ref_freq. It is placed so that freq_hi (the
    # first to arrive) falls exactly at margin_start:
    #   t_peak(freq_hi) = margin_start  =>  arrival_time = margin_start - delay_hi
    arrival_time = margin_start - delay_hi
    total_duration = margin_start + sweep_s + margin_end
    num_time = int(np.ceil(total_duration / res_time))
    times = np.arange(num_time) * res_time

    print(f"  Band: {freq_lo:.4f}-{freq_hi:.4f} MHz, {num_freq} channels, "
          f"{res_freq_mhz*1000:.2f} kHz/channel")
    print(f"  Dispersion sweep: {sweep_s:.4f} s | total window: {total_duration:.4f} s "
          f"({num_time} samples @ {res_time*1000:.3f} ms)")

    # ---- fitburst parameters ----
    # fitburst defines scattering_timescale AT ref_freq and scales it to any
    # frequency as sc_time(ν) = scattering_timescale * (ν/ref_freq)**SCATTERING_INDEX
    # (fitburst/analysis/model.py, compute_profile()). scat_s_600mhz is the
    # value measured in the catalog AT 600 MHz, so to recover
    # scattering_timescale that same formula is inverted at ν=600:
    #   scat_s_600mhz = scattering_timescale * (600/ref_freq)**SCATTERING_INDEX
    #   => scattering_timescale = scat_s_600mhz * (600/ref_freq)**(-SCATTERING_INDEX)
    # NOTE: the exponent is -SCATTERING_INDEX (+4), NOT SCATTERING_INDEX (-4).
    # With the sign of SCATTERING_INDEX used directly (bug fixed here on
    # 2026-09-10) the scattering was underestimated ~25x over the whole CHARTS
    # band (checked against fitburst/analysis/model.py:359). The negatives
    # generator does the inverse conversion and depends on the same sign.
    scattering_timescale = scat_s_600mhz * (600.0 / REF_FREQ) ** (-SCATTERING_INDEX)
    params = {
        "amplitude": [0.0],
        "arrival_time": [arrival_time],
        "burst_width": [width_fitb_s],
        "dm": [dm],
        "dm_index": [dm_index],
        "ref_freq": [REF_FREQ],
        "scattering_index": [SCATTERING_INDEX],
        "scattering_timescale": [scattering_timescale],
        "spectral_index": [sp_idx],
        "spectral_running": [sp_run],
    }

    # ---- Unit model ----
    modeler = fb.analysis.model.SpectrumModeler(
        freqs, times, num_components=1, is_dedispersed=False, verbose=False,
    )
    modeler.update_parameters(params)
    t0 = pytime.time()
    unit_model = modeler.compute_model().astype(np.float32)
    print(f"  ✓ Model generated in {pytime.time()-t0:.2f} s | shape {unit_model.shape} (freq×time)")

    n_nonzero = int((unit_model.max(axis=1) > 0).sum())
    if n_nonzero < num_freq:
        print(f"  ⚠ {num_freq - n_nonzero} channels without detectable signal -- check window/arrival_time")

    # ---- Calibrate with the real flux (Section 2.2 of the paper) ----
    # The catalog flux is the band average over the REAL CHIME band
    # (400-800 MHz). If the simulated band extends below 400 MHz, those extra
    # channels must NOT enter the calibration average.
    chime_mask = (freqs >= 400.0) & (freqs <= 800.0)
    band_avg_profile = unit_model[chime_mask].mean(axis=0) if chime_mask.any() else unit_model.mean(axis=0)
    peak_unit = float(band_avg_profile.max())
    scale = flux_jy / peak_unit if peak_unit > 0 else 1.0
    signal_jy = (unit_model * scale).astype(np.float32)

    # ---- Noise (radiometer equation) ----
    if sefd_jy is None:
        sefd_jy = SEFD_JY_DEFAULT
        print(f"  Default SEFD (approx. CHIME): {sefd_jy:.2f} Jy")

    delta_nu_hz = res_freq_mhz * 1e6
    noise_std = sefd_jy / np.sqrt(n_pol * delta_nu_hz * res_time)

    if noise_free:
        if snr_target is not None:
            raise ValueError("snr_target makes no sense with noise_free=True "
                             "(there is no noise floor to rescale against)")
        noise_jy = np.zeros_like(signal_jy)
        data_full = signal_jy.copy()
        snr_peak = float("inf")
        snr_integrated = float("inf")
        print(f"  NOISE-FREE (sefd_jy={sefd_jy:.2f} Jy computed but not applied) "
              f"-> peak/integrated S/N = inf")
    else:
        rng = np.random.default_rng()
        noise_jy = rng.normal(0, noise_std, signal_jy.shape).astype(np.float32)
        data_full = (signal_jy + noise_jy).astype(np.float32)

        snr_peak = float(signal_jy.max() / noise_std)
        band_avg_signal = signal_jy.mean(axis=0)
        noise_band_avg_std = noise_std / np.sqrt(num_freq)
        snr_integrated = float(band_avg_signal.max() / noise_band_avg_std)
        print(f"  SEFD={sefd_jy:.2f} Jy, n_pol={n_pol:.0f} | peak S/N={snr_peak:.2f}, "
              f"integrated S/N={snr_integrated:.2f}")

        if snr_target is not None:
            factor = snr_target / snr_peak
            signal_jy = (signal_jy * factor).astype(np.float32)
            data_full = (signal_jy + noise_jy).astype(np.float32)
            snr_peak = snr_target
            snr_integrated *= factor
            print(f"  -> Rescaled to target S/N {snr_target:.1f} (factor {factor:.3f})")

    metadata = {
        "bad_chans": [], "freqs_bin0": float(freqs[0]), "is_dedispersed": False,
        "num_freq": num_freq, "num_time": num_time, "times_bin0": 0.0,
        "res_freq": float(res_freq_mhz), "res_time": res_time,
        "dm_const_seconds": DM_CONST_SECONDS, "ref_freq": REF_FREQ,
        "dm_index": dm_index, "scattering_index": SCATTERING_INDEX,
        "sweep_s": sweep_s, "sefd_jy": sefd_jy, "n_pol": n_pol, "noise_free": noise_free,
    }
    burst_parameters = {
        "amplitude": [0.0], "arrival_time": [arrival_time], "burst_width": [width_fitb_s],
        "dm": [dm], "dm_index": [dm_index], "ref_freq": [REF_FREQ],
        "scattering_index": [SCATTERING_INDEX], "scattering_timescale": [scattering_timescale],
        "spectral_index": [sp_idx], "spectral_running": [sp_run],
        "is_frb": abs(dm_index) == 2.0, "flux_jy_catalog": flux_jy,
        "snr_peak": snr_peak, "snr_integrated": snr_integrated,
        "frb_source": frb_name,
        "paper": "Fonseca et al. (2024) ApJS 271:49",
        "doi": "10.3847/1538-4365/ad27d6", "arxiv": "2311.05829",
        "approximate_params": approximate_params or [],
    }

    fname = f"{frb_name}_dm{dm:07.2f}_fitburst.npz"
    sim_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)
    npz_path = sim_dir / fname
    np.savez_compressed(npz_path, data_full=data_full, data_clean=signal_jy,
                         metadata=metadata, burst_parameters=burst_parameters)
    print(f"  ✓ Saved: {npz_path} ({npz_path.stat().st_size/1024/1024:.1f} MB)")

    return {
        "filename": fname, "path": npz_path, "shape": data_full.shape,
        "max": float(data_full.max()), "snr_peak": snr_peak,
        "snr_integrated": snr_integrated, "dm": dm, "flux_jy": flux_jy,
    }


def generate_batch_from_catalog(frb_names, normalized_catalog, **common_kwargs):
    """
    Generates simulations for a list of FRBs, taking their parameters from a catalog
    already loaded as a dict {name: row} with normalized column names
    (dm, flux_jy, sp_idx, sp_run, width_s, scat_s).

    Parameters
    ----------
    frb_names : list[str]
        Names to simulate (they must be keys of normalized_catalog)
    normalized_catalog : dict
    **common_kwargs
        Parameters shared by all simulations (freq_lo, freq_hi, num_freq,
        res_time_ms, sefd_jy, n_pol, output_dir, etc.)
    """
    results = []
    missing = [n for n in frb_names if n not in normalized_catalog]
    if missing:
        print(f"⚠ Not found in the catalog: {missing}")

    for name in frb_names:
        if name not in normalized_catalog:
            continue
        row = normalized_catalog[name]
        if row["dm"] is None or row["flux_jy"] is None or row["sp_idx"] is None or row["sp_run"] is None:
            print(f"⚠ {name}: required parameters missing (dm/flux_jy/sp_idx/sp_run) -- skipped")
            continue
        result = generate_frb_fitburst(
            frb_name=name,
            dm=row["dm"],
            width_fitb_s=row["width_s"] if row["width_s"] is not None else 0.001,
            scat_s_600mhz=row["scat_s"] or 0.0,
            sp_idx=row["sp_idx"],
            sp_run=row["sp_run"],
            flux_jy=row["flux_jy"],
            **common_kwargs,
        )
        results.append(result)

    print(f"\n{'='*80}\nBATCH SUMMARY ({len(results)}/{len(frb_names)} generated)\n{'='*80}")
    for r in results:
        print(f"  {r['dm']:8.2f} pc/cm³ | peak S/N={r['snr_peak']:8.1f} | "
              f"integrated S/N={r['snr_integrated']:6.1f} | {r['filename']}")

    return results
