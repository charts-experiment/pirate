"""Reading of the SIGPROC (.fil) header used by the figures in 05_site_rfi/."""
import numpy as np


def read_header(path):
    """SIGPROC header: (header bytes, dict with nchans, nbits, fch1, foff)."""
    with open(path, "rb") as f:
        raw = f.read(4096)
    end = raw.index(b"HEADER_END") + len(b"HEADER_END")
    h, i = {}, raw.index(b"HEADER_START") + len(b"HEADER_START")
    while i < end - len(b"HEADER_END"):
        n = int.from_bytes(raw[i:i + 4], "little"); k = raw[i + 4:i + 4 + n].decode(); i += 4 + n
        if k in ("fch1", "foff", "tsamp", "tstart"):
            h[k] = float(np.frombuffer(raw[i:i + 8], "<f8")[0]); i += 8
        elif k in ("nchans", "nbits", "nifs", "data_type", "machine_id", "telescope_id", "nbeams", "ibeam"):
            h[k] = int(np.frombuffer(raw[i:i + 4], "<i4")[0]); i += 4
        elif k == "source_name":
            n2 = int.from_bytes(raw[i:i + 4], "little"); i += 4 + n2
        elif k in ("src_raj", "src_dej", "az_start", "za_start"):
            i += 8
    return raw[:end], h
