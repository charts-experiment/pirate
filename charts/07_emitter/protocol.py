"""X-engine → pirate network protocol (version 2), in pure Python.

Implements what pirate describes in notes/network_protocol.md, without depending on pirate:

  Per TCP connection (one per 07_emitter/receiver pair), when it opens:
    uint32  0xf4bf4b02                (magic; the 02 is the version)
    uint32  flags                     (FLAG_ACK = 0x1: the receiver answers 1 byte per minichunk)
    uint32  text length, including >= 1 zero padding bytes
    text    YAML metadata (pirate's configs/xengine_metadata.yml format), zero-terminated

  Then a sequence of minichunks of 256 time samples, with no padding between fields:
    uint32  0xf4bf4b02
    uint64  sequence number (seq) of the start of the minichunk
    float16 [nbeams, nfreq, 2]        scale and offset per beam and channel (value = scale·q + offset)
    int4    [nbeams, nfreq, 256]      intensity; −8 = missing sample; two per byte: (x[1] << 4) | x[0]

  nfreq is the number of channels THIS connection sends (the `freq_channels` of its metadata).
  Everything is little-endian. Consecutive minichunks differ in seq by 256 · seq_per_frb_time_sample,
  and the first one must be a multiple of that value.

If FLAG_ACK is set, after each minichunk the receiver returns one byte: 1 if it copied it into pirate's
ring buffer and 0 if it received it but dropped it.
"""
import socket
import struct

import numpy as np

MAGIC = 0xF4BF4B02
FLAG_ACK = 0x1
MINICHUNK_SAMPLES = 256

# Placeholder telescope geometry: pirate requires it and checks it is the same for all senders,
# but the search does not use it (values from pirate's XEngineMetadata::make_fiducial).
FIDUCIAL_TELESCOPE = {
    "tel_origin_itrs_lat_deg": 49.32075144444,
    "tel_origin_itrs_lon_deg": -119.62081125,
    "tel_grid_x_axis": [0.999974342398359362, -0.000037539331442772, -0.007163318767675494],
    "tel_grid_y_axis": [0.000065403387739210, 0.999992433220348809, 0.003889630373557614],
    "tel_dish_elev_axis": [0.99999999838132391, -0.000056897733584327, 0.0],
    "tel_dish_vert_axis": [0.0, 0.0, 1.0],
    "tel_dish_coelev_deg": 0.0,
    "tel_dish_separation_x_m": 6.300156854906823,
    "tel_dish_separation_y_m": 8.500057809796308,
}


def _yaml_list(v):
    return "[" + ", ".join(repr(float(x)) if isinstance(x, float) else str(int(x)) for x in v) + "]"


def metadata_yaml(zone_nfreq, zone_freq_edges, beam_ids, freq_channels, unix_ns_at_seq_0, dt_ns_per_seq,
                  seq_per_frb_time_sample, beamset=0, beam_positions_x=None, beam_positions_y=None,
                  telescope=None, noise_variance=1.0):
    """YAML metadata text of one connection. All senders to the same receiver must send the same text,
    except freq_channels (the channels of each one)."""
    nb = len(beam_ids)
    bx = beam_positions_x if beam_positions_x is not None else [0.01 * (i % 16) for i in range(nb)]
    by = beam_positions_y if beam_positions_y is not None else [0.01 * (i // 16) for i in range(nb)]
    tel = dict(FIDUCIAL_TELESCOPE, **(telescope or {}))
    lines = [
        "version: 2",
        f"zone_nfreq: {_yaml_list(zone_nfreq)}",
        f"zone_freq_edges: {_yaml_list([float(x) for x in zone_freq_edges])}",
        f"freq_channels: {_yaml_list(freq_channels)}",
        f"beamset: {int(beamset)}",
        f"beam_ids: {_yaml_list(beam_ids)}",
        f"beam_positions_x: {_yaml_list([float(x) for x in bx])}",
        f"beam_positions_y: {_yaml_list([float(x) for x in by])}",
        f"unix_ns_at_seq_0: {int(unix_ns_at_seq_0)}",
        f"dt_ns_per_seq: {int(dt_ns_per_seq)}",
        f"seq_per_frb_time_sample: {int(seq_per_frb_time_sample)}",
    ]
    for k, v in tel.items():
        lines.append(f"{k}: {_yaml_list(v) if isinstance(v, (list, tuple)) else repr(float(v))}")
    lines.append(f"noise_variance: {float(noise_variance)!r}")
    return "\n".join(lines) + "\n"


def connection_header(yaml_text, flags=0):
    """Bytes sent when the connection opens: magic, flags, length and the YAML followed by zeros."""
    b = yaml_text.encode("ascii")
    padding = 8 - (len(b) % 8)                      # at least one zero; aligns to 8 bytes
    return struct.pack("<III", MAGIC, flags, len(b) + padding) + b + b"\0" * padding


def pack_int4(codes):
    """codes: uint8 with values 0..15 (two's-complement int4) and an even-length last axis → two per byte."""
    c = codes.astype(np.uint8, copy=False)
    return (c[..., 0::2] & 0x0F) | ((c[..., 1::2] & 0x0F) << 4)


class Connection:
    """One TCP connection to a pirate receiver, sending a fixed subset of channels."""

    def __init__(self, address, yaml_text, ack=False, timeout_s=60.0):
        host, port = address.rsplit(":", 1)
        self.address = address
        self.ack = ack
        self.sock = socket.create_connection((host, int(port)), timeout=timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.sendall(connection_header(yaml_text, FLAG_ACK if ack else 0))
        self.sent = self.assembled = self.dropped = 0

    def send_minichunk(self, seq, scales_offsets, data):
        """scales_offsets: float16 [nbeams, nfreq, 2]; data: packed uint8 [nbeams, nfreq, 128].
        Returns the ACK byte (1 assembled, 0 dropped), or None without ACK."""
        so = np.ascontiguousarray(scales_offsets, dtype=np.float16)
        d = np.ascontiguousarray(data, dtype=np.uint8)
        self.sock.sendall(struct.pack("<IQ", MAGIC, int(seq)))
        self.sock.sendall(so.tobytes())
        self.sock.sendall(memoryview(d).cast("B"))
        self.sent += 1
        if not self.ack:
            return None
        r = self.sock.recv(1)
        if not r:
            raise ConnectionError(f"{self.address}: the receiver closed the connection")
        if r[0]:
            self.assembled += 1
        else:
            self.dropped += 1
        return r[0]

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
