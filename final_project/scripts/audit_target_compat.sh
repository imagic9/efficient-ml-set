#!/usr/bin/env bash
# Prove a binary can load on the Raspberry Pi (PLAN A2, gate for E7 packaging).
#
# The failure this prevents is total, not subtle: a binary built on gx10 natively
# requests GLIBC_2.38/2.39 symbols that Pi OS Bookworm's 2.36 cannot resolve, and
# the loader refuses it outright. Discovering that during a five-day rental would
# cost a day; discovering it here costs a second.
#
# Checks, for the binary and every library it carries:
#   1. no required GLIBC_* symbol version exceeds the target's glibc;
#   2. no NEEDED library is unresolved;
#   3. the ELF really is aarch64.
#
# Usage:  scripts/audit_target_compat.sh <binary-or-so> [more...]
#         TARGET_GLIBC=2.36 scripts/audit_target_compat.sh build/wildlife_trigger

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

if [[ -f "${PROJECT_ROOT}/configs/env/pins.env" ]]; then
    # shellcheck source=../configs/env/pins.env
    source "${PROJECT_ROOT}/configs/env/pins.env"
fi
TARGET_GLIBC="${TARGET_GLIBC:-2.36}"

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <binary-or-so> [more...]" >&2
    exit 2
fi

# Sort two dotted versions and check the first is not greater than the second.
version_le() {
    [[ "$1" == "$2" ]] && return 0
    [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" == "$1" ]]
}

failures=0

for target in "$@"; do
    echo "=== ${target}"

    if [[ ! -f "${target}" ]]; then
        echo "  FAIL: not found"
        failures=$((failures + 1))
        continue
    fi

    arch="$(file -b "${target}" | grep -oE 'ARM aarch64|x86-64' || echo unknown)"
    echo "  arch: ${arch}"
    if [[ "${arch}" != "ARM aarch64" ]]; then
        echo "  FAIL: not an aarch64 ELF; the Pi cannot run this"
        failures=$((failures + 1))
        continue
    fi

    # Highest GLIBC_x.y this object demands. Absent means it needs none, which is
    # fine (a static or symbol-free object), not suspicious.
    highest="$(objdump -T "${target}" 2>/dev/null \
        | grep -oE 'GLIBC_[0-9]+\.[0-9]+' \
        | sed 's/GLIBC_//' \
        | sort -uV \
        | tail -1)"

    if [[ -z "${highest}" ]]; then
        echo "  glibc: no versioned GLIBC symbols required"
    elif version_le "${highest}" "${TARGET_GLIBC}"; then
        echo "  glibc: requires <= ${highest}, target has ${TARGET_GLIBC}  OK"
    else
        echo "  glibc: FAIL — requires ${highest}, but the target has only ${TARGET_GLIBC}"
        echo "         symbols above the target:"
        objdump -T "${target}" 2>/dev/null \
            | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sed 's/GLIBC_//' | sort -uV \
            | while read -r v; do
                version_le "${v}" "${TARGET_GLIBC}" || echo "           GLIBC_${v}"
              done
        failures=$((failures + 1))
    fi

    # Unresolved NEEDED entries. `ldd` resolves against the *current* system, so
    # this is only meaningful inside the target container -- which is exactly
    # where E7 runs it.
    if missing="$(ldd "${target}" 2>/dev/null | grep 'not found')"; then
        echo "  deps: FAIL — unresolved:"
        echo "${missing}" | sed 's/^/           /'
        failures=$((failures + 1))
    else
        echo "  deps: all NEEDED libraries resolve"
    fi
done

echo
if [[ ${failures} -eq 0 ]]; then
    echo "PASS: every object can load on a glibc ${TARGET_GLIBC} target"
    exit 0
fi
echo "FAIL: ${failures} problem(s); this must not be packaged for the Pi"
exit 1
