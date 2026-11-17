## pirate - Perimeter Institute RAdio Transient Engine

An experimental GPU-based fast transient search, intended for use in CHORD.

> **CHARTS fork:** the [`charts/`](charts/) folder holds the CHARTS real-time FRB search built on pirate
> (64 beams × 21 504 channels: data emitter, grouper, FRB alerts, evaluation). Start at [`charts/README.md`](charts/README.md).
> Everything outside `charts/` is upstream pirate ([kmsmith137/pirate](https://github.com/kmsmith137/pirate)) unchanged.

Some notes/caveats:

  - This repo contains real-time server code (network receive, ring buffering,
    file-writing RPCs) and GPU dedispersion code, but currently **does not do RFI flagging**.
    Fixing this is my top priority.

  - Currently, this code can only be compiled on a recent ubuntu
    linux machine with **a physical GPU**, and the cuda toolkit installed.
    I hope to improve this in the future!

  - Uses the [`ksgpu`](https://github.com/kmsmith137/ksgpu) helper library.

Documentation: HTML docs are at [kmsmith137.github.io/pirate](https://kmsmith137.github.io/pirate/)
but may not be completely up-to-date. You can generate up-to-date docs with `make docs` and `make docs-serve`,
or browse the `.md` files below:

- [Installation](notes/install.md)
- [Introduction](notes/intro.md)
- [Developer notes](notes/developer.md)
- [Build system](notes/build.md)
- [Hardware](notes/hardware.md)
- [X->FRB network protocol (v2)](notes/network_protocol.md)
- [X->FRB metadata (v2)](configs/xengine_metadata.yml)
- [gRPC protocol definitions](grpc/)

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
