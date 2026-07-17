# Raspberry Pi deployment bundle

Built at E7 (`scripts/build_bundle.sh`), exercised on the real Pi at F1. Installs and
runs on a clean compatible ARM64 host **without any access to the training machine**.

## Contents

```
bin/wildlife_trigger   the application, built -mcpu=cortex-a76, linked against glibc 2.36
bin/run.sh             launcher: points the loader at the bundled libonnxruntime.so
lib/libonnxruntime.so* the pinned ONNX Runtime (travels in the bundle; no apt package)
models/{M0,M2,M4}.onnx the frozen shortlist (FP32 baseline + two INT8 candidates)
policies/{M0,M2,M4}.json   the bobcat policy bound by sha256 to each model
policies/class_map.json    the 16-class map
data/manifest.jsonl    a stratified sample slice of benchmark_val_1000
data/images/*.jpg      the frames the manifest references (so run_demo works offline)
preflight.sh           fail-closed host check (arch / Bookworm / Cortex-A76 ISA)
install.sh             preflight, verify, apt-install OpenCV, resolve libs, environment.json
run_demo.sh            self-test + infer + short benchmark + run-dataset
run_benchmark.sh       one-command benchmark matrix (M0 baseline + M2 + M4)
BUNDLE.json            provenance: git commit, per-artifact sha256, ORT version, glibc
MANIFEST.sha256        checksums of every file
```

## Fail-closed host preflight (issue #77)

The prebuilt binary is built `-mcpu=cortex-a76` and links OpenCV `.406`; that is proven
for a **Pi 5 on Raspberry Pi OS Bookworm**. `install.sh` runs `preflight.sh` first and
**refuses, before changing anything**, a host outside that contract:

- not `aarch64`;
- not **Bookworm** (a Trixie Pi ships OpenCV `.410`; see the contingency below);
- a CPU without **`asimddp`** — the ARMv8.2 dot-product feature Cortex-A76 (Pi 5) has and
  Cortex-A72 (Pi 4) lacks. The gate is the ISA feature, not the literal CPU part, so it
  also accepts a dev host (e.g. gx10) for the E8 dry run; whether the host is a *literal*
  Pi 5 (`is_pi5_a76`) is recorded in `environment.json`, never confused with a Pi result.

A successful install writes **`environment.json`**: OS/kernel/arch, CPU identity and
features, glibc, the installed OpenCV and bundled ORT versions, and the preflight verdict
— the machine-readable host record Phase F cites.

## Install and run

```sh
cd <bundle>
./install.sh          # verifies checksums, apt-installs the OpenCV runtime
./run_demo.sh         # M0 by default; ./run_demo.sh M2  or  M4
```

## The compatibility constraint that shapes this

gx10 runs Ubuntu 24.04 with **glibc 2.39**; Raspberry Pi OS Bookworm ships **2.36**. A
binary built natively on gx10 requests `GLIBC_2.38/2.39` symbols the target cannot
resolve and will not load at all. So the build happens inside a `debian:bookworm-slim`
container pinned by digest: a binary linked against 2.36 still loads on a newer Pi OS,
while the reverse fails. `install.sh` re-checks the required `GLIBC_*` symbols against
the host before the first run, and `bundle_audit.py` proves it at build time.

## OpenCV

OpenCV is **not** in the bundle. Debian's `libopencv_imgcodecs` drags a ~50-library
GDAL/poppler/database closure that is impractical to carry, so `install.sh` apt-installs
the **OpenCV 4.6.0 runtime** (`libopencv-core406`, `libopencv-imgproc406`,
`libopencv-imgcodecs406`) instead. Raspberry Pi OS Bookworm is Debian bookworm, so those
packages are the **same `.406` soname** the binary was linked against — a byte-compatible
match. This needs network at install time, not the training machine.

**Trixie contingency.** A Raspberry Pi OS Trixie host ships a different OpenCV soname
(`.410`), which the binary linked against `.406` will not load. If the rented Pi is
Trixie, either (a) rebuild the app on the Pi against its apt OpenCV (`cmake` + the
bundled ORT), or (b) build a minimal OpenCV 4.6.0 from source. The bundle is otherwise
unchanged. Verified at E8 (full ARM64 dry run) against the Bookworm target.

## Not bundled, on purpose

The session-optimized ONNX graph. ORT warns a graph serialized above
`ORT_ENABLE_EXTENDED` is only valid in the environment that optimized it (measured in
P0). The Pi optimizes the ordinary model itself at load.
