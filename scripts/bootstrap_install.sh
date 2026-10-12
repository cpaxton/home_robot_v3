#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
# Standalone entrypoint: clone Emet, then run the checkout's installer.
set -euo pipefail

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            cat <<'HELP'
Usage: bash bootstrap_install.sh [install.sh options]
       curl -fsSL https://raw.githubusercontent.com/cpaxton/home_robot_v3/main/scripts/bootstrap_install.sh | bash -s -- -y

Requires Bash and Git; the checkout installer provisions uv and dependencies.
All arguments are forwarded unchanged to install.sh (e.g. -y --no-sim).
The installer defaults to profile=full; use --no-sim for a lighter installation.
Without -y/--yes, a controlling terminal is required for installer prompts.

Environment:
  EMET_INSTALL_DIR   New checkout directory (default: ./home_robot_v3)
  EMET_INSTALL_REPO  Git URL (default: https://github.com/cpaxton/home_robot_v3.git)
  EMET_INSTALL_REF   Branch or tag to clone (default: main)

Existing destinations are never reused or overwritten. To update an existing
checkout, run its install.sh directly. Private repositories need working Git
authentication; downloading this script also requires access to the repository.
HELP
            exit 0
            ;;
    esac
done

command -v git >/dev/null 2>&1 || { echo "Git is required; install git and retry." >&2; exit 1; }
emet_install_dir="${EMET_INSTALL_DIR:-$PWD/home_robot_v3}"
emet_install_repo="${EMET_INSTALL_REPO:-https://github.com/cpaxton/home_robot_v3.git}"
emet_install_ref="${EMET_INSTALL_REF:-main}"
# Make relative destinations absolute, including names beginning with a dash.
case "$emet_install_dir" in
    /*) ;;
    *) emet_install_dir="$PWD/$emet_install_dir" ;;
esac
if [[ -e "$emet_install_dir" || -L "$emet_install_dir" ]]; then
    echo "Destination already exists: $emet_install_dir. Run its install.sh or choose a new EMET_INSTALL_DIR." >&2
    exit 1
fi

emet_install_yes=false
for arg in "$@"; do
    case "$arg" in
        -y|--yes)
            emet_install_yes=true
            ;;
    esac
done
# A piped script occupies stdin. Give prompts a terminal, never the script pipe.
if [[ "$emet_install_yes" == false ]]; then
    if ! { exec 3<>/dev/tty; } 2>/dev/null; then
        echo "No controlling terminal. Pass -y/--yes for non-interactive installation." >&2
        exit 1
    fi
fi

git clone --branch "$emet_install_ref" -- "$emet_install_repo" "$emet_install_dir"
if [[ ! -f "$emet_install_dir/install.sh" || ! -f "$emet_install_dir/pyproject.toml" ]]; then
    echo "Cloned ref is missing install.sh or pyproject.toml: $emet_install_dir" >&2
    exit 1
fi
cd "$emet_install_dir"
if [[ "$emet_install_yes" == true ]]; then
    exec bash ./install.sh "$@" </dev/null
else
    exec bash ./install.sh "$@" <&3 3>&-
fi
