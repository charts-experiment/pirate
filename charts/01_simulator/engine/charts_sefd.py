#!/usr/bin/env python3
"""
SEFD of CHARTS, derived from the specifications published in the design paper
(Cassanelli et al. 2024, "CHARTS: Canadian-Chilean array for radio transient studies", Table 1):

  - T_sys <= 50 K
  - Gain per antenna (zenith): ~7.1 dBi
  - n_a = 256 antennas, ONE polarization (n_p = 1)
  - Band: 300-500 MHz

Explicit assumption: the 256 antennas are combined coherently (phased array / beamforming), i.e.
A_eff_total = n_a * A_eff_single_antenna.

A_eff_antenna = G_linear * lambda^2 / (4*pi), evaluated at ref_freq (400.1953125 MHz, the same fixed
reference used throughout the fitburst pipeline). SEFD = 2*k_B*T_sys / A_eff_total.

CHIME (28.76 Jy, 8000 m^2 physical, T_sys = 50 K) is a radio telescope of large cylinders; CHARTS is
an array of small low-cost antennas that by design favours number of antennas and field of view over
sensitivity (Sec. 2 of the paper). Result: SEFD ~82x worse than CHIME's.
"""
import numpy as np

K_BOLTZMANN = 1.380649e-23   # J/K
C_LIGHT = 299792458.0        # m/s

CHARTS_T_SYS_K = 50.0
CHARTS_ANTENNA_GAIN_DBI = 7.1
CHARTS_N_ANTENNAS = 256
CHARTS_N_POL = 1.0            # Table 1: "Antenna polarization number n_p = 1"
CHARTS_REF_FREQ_MHZ = 400.1953125  # same fixed reference as the rest of the pipeline

_gain_linear = 10 ** (CHARTS_ANTENNA_GAIN_DBI / 10.0)
_wavelength_m = C_LIGHT / (CHARTS_REF_FREQ_MHZ * 1e6)
_a_eff_single_m2 = _gain_linear * _wavelength_m**2 / (4 * np.pi)
CHARTS_A_EFF_SINGLE_M2 = _a_eff_single_m2
CHARTS_A_EFF_ARRAY_M2 = CHARTS_N_ANTENNAS * _a_eff_single_m2

CHARTS_SEFD_JY = 2.0 * K_BOLTZMANN * CHARTS_T_SYS_K / CHARTS_A_EFF_ARRAY_M2 / 1e-26

# Array sizes explored to see how the sensitivity scales with the number of antennas combined
# coherently. Not a physical limitation of CHARTS.
SUPPORTED_N_ANTENNAS = (8, 16, 32, 64, 128, 256)


def charts_sefd(n_antennas=CHARTS_N_ANTENNAS):
    """CHARTS SEFD for an array of `n_antennas` antennas combined coherently
    (A_eff_total = n_antennas * A_eff_single_antenna). With n_antennas=256 (default)
    it gives exactly CHARTS_SEFD_JY."""
    a_eff_total = n_antennas * CHARTS_A_EFF_SINGLE_M2
    return 2.0 * K_BOLTZMANN * CHARTS_T_SYS_K / a_eff_total / 1e-26


def theoretical_noise_std(meta):
    """System noise sigma from the radiometer equation -- the same formula as
    engine/fitburst_sim.py (delta_nu_hz = res_freq_mhz*1e6). A single scalar:
    SEFD/n_pol/res_freq/res_time are uniform across channels in this instrument
    (no simulated bandpass ripple), so this sigma is the same for every channel.
    Used by 03_convert_to_fil.py for the fixed quantization scale."""
    sefd_jy = float(meta["sefd_jy"])
    n_pol = float(meta["n_pol"])
    delta_nu_hz = float(meta["res_freq"]) * 1e6
    res_time = float(meta["res_time"])
    return sefd_jy / np.sqrt(n_pol * delta_nu_hz * res_time)


if __name__ == "__main__":
    print(f"Linear gain: {_gain_linear:.4f}")
    print(f"Wavelength @ {CHARTS_REF_FREQ_MHZ} MHz: {_wavelength_m:.4f} m")
    print(f"A_eff per antenna: {CHARTS_A_EFF_SINGLE_M2:.5f} m²")
    print(f"A_eff array ({CHARTS_N_ANTENNAS} antennas): {CHARTS_A_EFF_ARRAY_M2:.2f} m²")
    print(f"SEFD_CHARTS (256 antennas, n_pol=1): {CHARTS_SEFD_JY:.2f} Jy")
