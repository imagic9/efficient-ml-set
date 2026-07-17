#!/usr/bin/env bash
# Build and stage the ARM64 deployment bundle (PLAN E7).
#
# A bundle is what actually reaches the Pi. Everything here exists to make that
# transfer boring and host-independent:
#
#   - built inside wildlife-trigger-target:bookworm, so every object links against
#     glibc 2.36 — gx10's native 2.39 would produce a binary the Pi's loader refuses;
#   - libonnxruntime.so travels WITH the bundle (P0: needs only GLIBC_2.27, so the
#     same file serves the container and the Pi; no apt package exists for it anyway);
#   - the frozen shortlist M0/M2/M4 (from pre_pi_freeze.json) + their policies + the
#     class map + a small sample manifest and its images, so run_demo works offline;
#   - install.sh, run_demo.sh, README.md, a BUNDLE.json provenance manifest (git
#     commit, per-artifact sha256, ORT version, build glibc), and MANIFEST.sha256 over
#     everything, so what arrives can be proved to be what left.
#
# Deliberately NOT bundled:
#   - the session-optimized ONNX graph (ORT: valid only in the env that optimized it);
#   - OpenCV. Debian's libopencv_imgcodecs drags a ~50-library GDAL/poppler/database
#     closure that is impractical to carry, so install.sh apt-installs the OpenCV 4.6.0
#     runtime instead. Pi OS Bookworm is Debian bookworm — the same .406 soname the
#     binary linked against. (Trixie is a documented contingency; see deploy/pi/README.)
#
# Usage:  scripts/build_bundle.sh [staging_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"
STAGING="${1:-${PROJECT_ROOT}/results/e7/bundle}"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
BUILD_DIR="/work/build/bundle"
REL_STAGING="${STAGING#"${PROJECT_ROOT}"/}"
FREEZE="results/model_selection/pre_pi_freeze.json"
CLASS_MAP="artifacts/class_map.json"
MANIFEST_SRC="data/manifests/benchmark_val_1000.jsonl"
IMAGES_ROOT="data/raw/extracted/eccv_18_all_images_sm"
SAMPLE_N="${SAMPLE_N:-40}"

cd "${PROJECT_ROOT}"
[[ -d "${IMAGES_ROOT}" ]] || { echo "no images root at ${IMAGES_ROOT}" >&2; exit 2; }

echo "Building the deployment bundle in ${TARGET_IMAGE_TAG}"
echo "  staging: ${STAGING}"

rm -rf "${STAGING}"
mkdir -p "${STAGING}/bin" "${STAGING}/lib" "${STAGING}/models" \
         "${STAGING}/policies" "${STAGING}/data/images"

# --- 1. binary + ORT, built and audited in the target container --------------
docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${TARGET_IMAGE_TAG}" bash -lc "
set -euo pipefail
cmake -S /work/cpp -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release \
    -DWILDLIFE_CPU_TARGET=${QEMU_PI5_CPU} >/dev/null
cmake --build ${BUILD_DIR} --target wildlife_trigger -j\"\$(nproc)\" >/dev/null

STAGE=/work/${REL_STAGING}
install -m 0755 ${BUILD_DIR}/wildlife_trigger \"\$STAGE/bin/\"

# The real .so behind the soname chain, then the links the loader follows.
ORT_REAL=\"\$(readlink -f ${ORT_PREFIX}/lib/libonnxruntime.so)\"
install -m 0644 \"\$ORT_REAL\" \"\$STAGE/lib/\$(basename \"\$ORT_REAL\")\"
ln -sf \"\$(basename \"\$ORT_REAL\")\" \"\$STAGE/lib/libonnxruntime.so.1\"
ln -sf \"\$(basename \"\$ORT_REAL\")\" \"\$STAGE/lib/libonnxruntime.so\"

echo '  GLIBC audit against the STAGED binary + ORT:'
/work/scripts/audit_target_compat.sh \"\$STAGE/bin/wildlife_trigger\" \
    \"\$STAGE/lib/\$(basename \"\$ORT_REAL\")\" | sed 's/^/    /'
"

# --- 2. launcher -------------------------------------------------------------
cat > "${STAGING}/bin/run.sh" <<'LAUNCHER'
#!/bin/sh
# Bundle launcher: resolve libonnxruntime.so from the bundle's lib/, keep the system
# path for OpenCV (installed by install.sh).
here="$(cd "$(dirname "$0")/.." && pwd)"
LD_LIBRARY_PATH="$here/lib:${LD_LIBRARY_PATH:-}" exec "$here/bin/wildlife_trigger" "$@"
LAUNCHER
chmod 0755 "${STAGING}/bin/run.sh"

# --- 3. models, policies, class map, and the sample slice (host side) --------
# Resolve the shortlist from the freeze and stage each under a deterministic name so
# run_demo.sh maps MODEL -> models/$MODEL.onnx + policies/$MODEL.json with no jq.
"${PYTHON}" - "${FREEZE}" "${STAGING}" <<'PY'
import json, shutil, sys
from pathlib import Path
freeze, staging = sys.argv[1], Path(sys.argv[2])
d = json.load(open(freeze))
for m in d["models"]:
    mid = m["model_id"]
    if mid not in ("M0", "M2", "M4"):
        continue
    shutil.copyfile(m["onnx"]["artifact"], staging / "models" / f"{mid}.onnx")
    shutil.copyfile(m["policy"]["path"], staging / "policies" / f"{mid}.json")
    print(f"    staged {mid}: {Path(m['onnx']['artifact']).name} + {Path(m['policy']['path']).name}")
PY
cp -f "${CLASS_MAP}" "${STAGING}/policies/class_map.json"

echo "  sampling ${SAMPLE_N} frames (stratified) + copying images"
"${PYTHON}" - "${MANIFEST_SRC}" "${IMAGES_ROOT}" "${STAGING}" "${SAMPLE_N}" <<'PY'
import json, os, shutil, sys
from collections import defaultdict
manifest, images_root, staging, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
from pathlib import Path
staging = Path(staging)
rows = [json.loads(l) for l in open(manifest) if l.strip()]
per = {"bobcat": n // 3, "threshold_adjacent": n // 4, "empty": n // 4,
       "other": n // 6, "rare": n // 6, "multi_label": 2}
by = defaultdict(list)
for r in rows:
    if os.path.exists(os.path.join(images_root, r["file_name"])):
        by[r["benchmark_stratum"]].append(r)
chosen = []
for s in ("bobcat", "threshold_adjacent", "empty", "other", "rare", "multi_label"):
    chosen.extend(sorted(by.get(s, []), key=lambda r: r["image_id"])[: per.get(s, 0)])
# dedupe by image_id, stable order
seen, uniq = set(), []
for r in sorted(chosen, key=lambda r: r["image_id"]):
    if r["image_id"] not in seen:
        seen.add(r["image_id"]); uniq.append(r)
out = staging / "data" / "manifest.jsonl"
with open(out, "w") as f:
    for r in uniq:
        src = os.path.join(images_root, r["file_name"])
        base = r["image_id"] + ".jpg"
        shutil.copyfile(src, staging / "data" / "images" / base)
        f.write(json.dumps({"file_name": base, "image_id": r["image_id"],
                            "seq_id": r.get("seq_id", ""), "labels": r["labels"],
                            "benchmark_stratum": r["benchmark_stratum"]}) + "\n")
print(f"    {len(uniq)} frames across {len({r['benchmark_stratum'] for r in uniq})} strata")
PY

# --- 4. installer, demo, readme ----------------------------------------------
install -m 0755 "${PROJECT_ROOT}/deploy/pi/install.sh" "${STAGING}/install.sh"
install -m 0755 "${PROJECT_ROOT}/deploy/pi/run_demo.sh" "${STAGING}/run_demo.sh"
install -m 0644 "${PROJECT_ROOT}/deploy/pi/README.md" "${STAGING}/README.md"

# --- 5. provenance manifest (git commit, hashes, ORT, glibc) -----------------
"${PYTHON}" - "${STAGING}" "${FREEZE}" "${ORT_VERSION}" "${TARGET_GLIBC}" \
             "${TARGET_BASE_DIGEST}" "${QEMU_PI5_CPU}" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
staging = Path(sys.argv[1]); freeze = json.load(open(sys.argv[2]))
ort_version, glibc, base_digest, cpu = sys.argv[3:7]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

models = {}
for m in freeze["models"]:
    mid = m["model_id"]
    if mid not in ("M0", "M2", "M4"):
        continue
    pol = json.load(open(staging / "policies" / f"{mid}.json"))
    models[mid] = {
        "kind": m.get("kind"),
        "onnx": f"models/{mid}.onnx",
        "onnx_sha256": sha(staging / "models" / f"{mid}.onnx"),
        "onnx_sha256_frozen": m["onnx"].get("sha256"),
        "policy": f"policies/{mid}.json",
        "policy_id": pol.get("policy_id"),
        "threshold": m["policy"]["threshold"],
    }
    if models[mid]["onnx_sha256"] != m["onnx"].get("sha256"):
        raise SystemExit(f"{mid} staged onnx sha != frozen sha")

bundle = {
    "bundle": "wildlife-trigger deployment bundle (PLAN E7)",
    "git_commit": commit,
    "built_in": {"image": "wildlife-trigger-target:bookworm",
                 "base_digest": base_digest, "cpu_target": cpu, "glibc": glibc},
    "onnxruntime_version": ort_version,
    "class_map_sha256": sha(staging / "policies" / "class_map.json"),
    "models": models,
    "opencv": {"bundled": False, "runtime": "libopencv-{core,imgproc,imgcodecs}406",
               "installed_by": "install.sh (apt); Pi OS Bookworm ships the matching 4.6.0"},
    "provenance": "Latency is a Pi result only when measured ON a Pi (DESIGN §12.4). "
                  "The session-optimized graph is never bundled.",
}
(staging / "BUNDLE.json").write_text(json.dumps(bundle, indent=2) + "\n")
print(f"    BUNDLE.json: commit {commit[:12]}, {len(models)} models, ORT {ort_version}")
PY

# --- 6. checksums over everything staged -------------------------------------
( cd "${STAGING}" && find . -type f ! -name MANIFEST.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )

echo
echo "Bundle contents:"
find "${STAGING}" \( -type f -o -type l \) | sort | sed "s|${STAGING}|  .|"
echo
echo "Bundle size: $(du -sh "${STAGING}" | cut -f1) ($(wc -l < "${STAGING}/MANIFEST.sha256") files)"
echo
echo "Install + demo on the target with:"
echo "  cd <bundle> && ./install.sh && ./run_demo.sh"
