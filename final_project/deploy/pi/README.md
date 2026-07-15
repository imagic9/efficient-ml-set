# Raspberry Pi deployment bundle

Built at E7, exercised on the real Pi at F1. Must install and run on a clean
compatible ARM64 environment **without any access to the training machine**.

Contents once built: the C++ executable, required shared libraries or a
reproducible installer, the final ONNX model, class map, policy JSON, a sample
manifest, `install.sh`, `run_demo.sh`, and checksums.

## The compatibility constraint that shapes this

gx10 runs Ubuntu 24.04 with **glibc 2.39**; Raspberry Pi OS Bookworm ships
**2.36**. A binary built natively on gx10 requests `GLIBC_2.38/2.39` symbols the
target cannot resolve and will not load at all.

So the build happens inside a `debian:bookworm-slim` container pinned by digest:
a binary linked against 2.36 still loads on a newer Pi OS, while the reverse
fails. Before packaging, `ldd` and ELF/required-`GLIBC_*` symbol inspection run
over the executable and every bundled library. If compatibility cannot be
*proved* before the rental, this bundle instead carries pinned source and builds
on the Pi during provisioning.

Empty until E7.
