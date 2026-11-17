#!/usr/bin/env python3
"""Static channel masks from the known site RFI (site_rfi_model.npz).

  m1: persistent lines > 3 dB  or  fast RFI > 1 % of the time      (17.5 % of the band)
  m2: persistent lines > 1 dB  or  fast RFI > 0.2 % of the time    (39 % of the band)
Output: mask_m1.npy, mask_m2.npy (bool, 21 504 channels, order of frequencies_mhz.npy, ascending).
Real-time use: python 08_server/launcher.py local ... --mask 05_site_rfi/mask_m1.npy
"""
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
m = np.load(HERE / "site_rfi_model.npz")
db = 10 * np.log10(1 + np.clip(m["r_p"], 0, None)); occupancy = (m["e"] > 0).mean(0)
for name, mk in (("m1", (db > 3) | (occupancy > 0.01)), ("m2", (db > 1) | (occupancy > 0.002))):
    np.save(HERE / f"mask_{name}.npy", mk)
    print(name, int(mk.sum()), f"{mk.mean() * 100:.1f} %")
