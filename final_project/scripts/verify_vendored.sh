#!/usr/bin/env bash
# Verify every vendored third-party source against its recorded SHA-256.
#
# A vendored header is a dependency that looks like our own code: it is in our tree,
# it compiles with our flags, and nothing about it announces an upstream version. An
# accidental edit, a bad merge, or a well-meant "fix" to a warning would be
# invisible. This turns that into a failing gate.
#
# The hashes below are the ones recorded in cpp/third_party/README.md. Both must
# change together, deliberately, when a dependency is bumped.
#
# Usage:  scripts/verify_vendored.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="$(dirname "${HERE}")/cpp/third_party"

# path-relative-to-third_party : expected sha256
VENDORED=(
    "nlohmann/json.hpp:aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63"
)

# Licences must travel with the source they licence; a vendored MIT header without
# its licence file is a compliance problem, not an untidy directory.
REQUIRED_FILES=(
    "nlohmann/LICENSE.MIT"
)

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

failures=0

for entry in "${VENDORED[@]}"; do
    path="${entry%%:*}"
    expected="${entry##*:}"
    file="${THIRD_PARTY}/${path}"

    if [[ ! -f "${file}" ]]; then
        echo "MISSING: ${path}"
        failures=$((failures + 1))
        continue
    fi

    actual="$(sha256_of "${file}")"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "HASH MISMATCH: ${path}"
        echo "  recorded: ${expected}"
        echo "  on disk : ${actual}"
        echo "  A vendored source changed without its manifest. Either restore the"
        echo "  pinned file or bump the version and its hash together, on purpose."
        failures=$((failures + 1))
        continue
    fi

    echo "ok  ${path}  ${actual:0:16}..."
done

for path in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${THIRD_PARTY}/${path}" ]]; then
        echo "MISSING LICENCE: ${path}"
        failures=$((failures + 1))
    else
        echo "ok  ${path}"
    fi
done

if [[ ${failures} -gt 0 ]]; then
    echo
    echo "FAILED: ${failures} vendored-source problem(s)"
    exit 1
fi

echo
echo "PASS: every vendored source matches its recorded hash"
