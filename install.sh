#!/usr/bin/env bash
# This script (c) 2024 Hello Robot under the MIT license: https://opensource.org/licenses/MIT
# Installs Stretch AI using uv (https://docs.astral.sh/uv/) - no conda required.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
cd "$ROOT_DIR"

# Parse options
# Default profile is "full" (sim + MolmoSpaces wrapper when packages exist). Use --no-sim or --profile=minimal for CI/light installs.
# Profiles: EMET_INSTALL_PROFILE or --profile=  (minimal|standard|full). "full" = sim-on-by-default (legacy).
CPU_ONLY="false"
SKIP_ASKING="false"
NO_SAM2="false"
INSTALL_SIM="false"
CLEAN_SIM="false"
FORCE_DOWNLOAD="false"
INSTALL_MOLMOSPACES="false"
NO_MOLMOSPACES="false"
INSTALL_LINGBOT_MAP="false"
# standard | minimal | full | jetson (full = install sim without passing --sim; default profile is full)
PROFILE="${EMET_INSTALL_PROFILE:-full}"
# Set when user passes --sim, --no-sim, or --all (all implies sim)
SIM_EXPLICIT=""
PREV_PROFILE=""
# Jetson Orin / Tegra: lean sync (no sim/SAM2/Molmo); set by --profile=jetson or --jetson
JETSON_INSTALL="false"

for arg in "$@"; do
    if [ "$PREV_PROFILE" = "1" ]; then
        PROFILE="$arg"
        PREV_PROFILE=""
        continue
    fi
    case $arg in
        -y|--yes)
            SKIP_ASKING="true"
            ;;
        --all)
            INSTALL_SIM="true"
            NO_SAM2="false"
            SIM_EXPLICIT="1"
            if [ "$NO_MOLMOSPACES" != "true" ]; then
                INSTALL_MOLMOSPACES="true"
            fi
            ;;
        --cpu)
            CPU_ONLY="true"
            NO_SAM2="true"
            ;;
        --jetson)
            PROFILE="jetson"
            ;;
        --no-sam2)
            NO_SAM2="true"
            ;;
        --no-sim)
            INSTALL_SIM="false"
            SIM_EXPLICIT="1"
            ;;
        --sim)
            INSTALL_SIM="true"
            SIM_EXPLICIT="1"
            ;;
        --clean)
            CLEAN_SIM="true"
            ;;
        --force-download)
            FORCE_DOWNLOAD="true"
            ;;
        --molmospaces)
            INSTALL_MOLMOSPACES="true"
            NO_MOLMOSPACES="false"
            ;;
        --no-molmospaces)
            NO_MOLMOSPACES="true"
            INSTALL_MOLMOSPACES="false"
            ;;
        --lingbot-map)
            INSTALL_LINGBOT_MAP="true"
            ;;
        --profile=*)
            PROFILE="${arg#--profile=}"
            ;;
        --profile)
            PREV_PROFILE="1"
            ;;
        *)
            ;;
    esac
done

# Apply profile when --sim / --no-sim / --all were not used
if [ -z "$SIM_EXPLICIT" ]; then
    case "$(printf '%s' "$PROFILE" | tr '[:upper:]' '[:lower:]')" in
        full|legacy)
            INSTALL_SIM="true"
            ;;
        jetson|orin|tegra)
            PROFILE="jetson"
            JETSON_INSTALL="true"
            INSTALL_SIM="false"
            NO_SAM2="true"
            NO_MOLMOSPACES="true"
            INSTALL_MOLMOSPACES="false"
            CPU_ONLY="true"
            ;;
        minimal|standard|"" )
            ;;
        *)
            echo "WARNING: unknown EMET_INSTALL_PROFILE/--profile=$PROFILE (use minimal, standard, full, or jetson). Using standard."
            ;;
    esac
elif [ "$(printf '%s' "$PROFILE" | tr '[:upper:]' '[:lower:]')" = "jetson" ] \
    || [ "$(printf '%s' "$PROFILE" | tr '[:upper:]' '[:lower:]')" = "orin" ] \
    || [ "$(printf '%s' "$PROFILE" | tr '[:upper:]' '[:lower:]')" = "tegra" ]; then
    # Explicit --sim with --profile=jetson is unusual; still mark jetson packaging mode.
    PROFILE="jetson"
    JETSON_INSTALL="true"
    NO_SAM2="true"
fi

# With sim on, install MolmoSpaces wrapper by default (separate venv) so `emet serve mujoco
# --molmospaces-*` works without a second step. Opt out: --no-molmospaces. Skip if wrapper package
# is not in this checkout (e.g. sparse clone).
if [ "$INSTALL_SIM" = "true" ] && [ "$NO_MOLMOSPACES" != "true" ] && [ -d "$ROOT_DIR/packages/emet_molmospaces" ]; then
    INSTALL_MOLMOSPACES="true"
fi

echo "=============================================="
echo "         INSTALLING STRETCH AI (uv)"
echo "=============================================="
echo "Options: PROFILE=$PROFILE CPU_ONLY=$CPU_ONLY NO_SAM2=$NO_SAM2 INSTALL_SIM=$INSTALL_SIM MOLMOSPACES=$INSTALL_MOLMOSPACES LINGBOT_MAP=$INSTALL_LINGBOT_MAP NO_MOLMOSPACES=$NO_MOLMOSPACES JETSON=$JETSON_INSTALL"
echo "         Defaults: profile=full enables sim when third_party/robocasa exists (use --no-sim or --profile=minimal to skip)."
echo "         EMET_INSTALL_PROFILE=standard  or  --profile=minimal  = no sim unless you also pass --sim."
echo "         --profile=jetson / --jetson = Orin/Tegra lean install (MuJoCo pip + dev; no SAM2/Molmo/Robocasa clone)."
echo "         -y/--yes    = non-interactive (apt, link emet); does NOT imply MolmoSpaces — pass --molmospaces or use --all"
echo "         --all       = sim + molmospaces + dynamem bundle (same as --sim --molmospaces when wrapper package exists)"
echo "         --sim       = uv sim extra + install_simulation.sh (Robocasa + robosuite)"
echo "         --no-sim    = force sim off even if PROFILE=full"
echo "         --molmospaces = create .venv-molmospaces for MolmoSpaces (scenes + rby1 robot)"
echo "         --no-molmospaces = skip MolmoSpaces venv even when sim is installed (lighter / CI)"
echo "         --lingbot-map = create .venv-lingbot-map for LingBot-Map streaming depth (see docs/lingbot_map.md)"
echo "         --clean     = remove and re-clone third_party/robosuite, robosuite_models, robocasa (only if needed; normally we update in place)"
echo "         --force-download = re-download sim assets even if they already exist (use with --sim/--all)"
echo "         Rich menu:  uv sync && uv run emet install menu"
echo "Root: $ROOT_DIR"
echo "---------------------------------------------"

# Optional: remove sim third_party dirs only when explicitly requested (--clean). Normally install_simulation.sh updates in place (fetch/pull).
if [ "$CLEAN_SIM" = "true" ]; then
    echo ""
    echo "Cleaning sim third_party (robosuite, robosuite_models, robocasa) before re-install..."
    for d in third_party/robosuite third_party/robosuite_models third_party/robocasa; do
        if [ -d "$d" ]; then
            rm -rf "$d"
            echo "  -> Removed $d"
        fi
    done
fi

# Step 1: Init required git submodules (segment-anything-2 for SAM-2/dynamem).
# ok-robot is optional (docs/advanced workflows only); use: emet install submodules
echo ""
echo "[1/5] Initializing required git submodules (segment-anything-2)..."
git submodule update --init --recursive third_party/segment-anything-2
if [ ! -d "third_party/segment-anything-2" ]; then
    echo "ERROR: third_party/segment-anything-2 missing after submodule update. Check git and .gitmodules."
    exit 1
fi
echo "  -> Verified third_party/segment-anything-2 exists."

# Ensure uv is installed
echo ""
echo "[2/5] Checking uv..."
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Using uv: $(uv --version)"

# System dependencies (apt-get update may warn about other repos e.g. ROS2 keys; we continue)
# Skip when DOCKER=1 (image already has deps; container often has no sudo)
echo ""
echo "[3/5] Checking system dependencies..."
APT_PKGS=(libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg build-essential wget unzip libsndfile1 gh)
if [ "$JETSON_INSTALL" = "true" ]; then
    # sdist builds on aarch64: sophuspy, scikit-fmm, pyliblzfse, PyAudio, simpleaudio, etc.
    APT_PKGS+=(libopenblas-dev libopenmpi-dev libomp-dev cmake ninja-build pkg-config python3-dev libavformat-dev libavcodec-dev libavutil-dev libavdevice-dev libavfilter-dev)
fi
if [ -n "${DOCKER:-}" ]; then
    echo "  -> Skipping apt (DOCKER=1); image has deps."
elif [ "$SKIP_ASKING" = "true" ]; then
    if sudo -n true 2>/dev/null; then
        sudo apt-get update || true
        sudo apt-get install -y "${APT_PKGS[@]}"
    else
        echo "  -> sudo not available non-interactively; skipping apt."
        echo "     If builds fail, install manually: sudo apt-get install ${APT_PKGS[*]}"
        missing=0
        for pkg in "${APT_PKGS[@]}"; do
            if ! dpkg -s "$pkg" >/dev/null 2>&1; then
                echo "     missing: $pkg"
                missing=1
            fi
        done
        if [ "$missing" -eq 1 ]; then
            echo "  -> Continuing anyway; compile-from-sdist steps may fail without the packages above."
        fi
    fi
else
    echo "Required packages: ${APT_PKGS[*]}"
    echo "Install with: sudo apt-get install ${APT_PKGS[*]}"
    read -p "Install these now? (y/n) " yn
    case $yn in
        y|Y) sudo apt-get update || true; sudo apt-get install -y "${APT_PKGS[@]}" ;;
        *) echo "Skipping. You may need to install these manually." ;;
    esac
fi

# Git LFS
echo ""
echo "[4/5] Setting up git-lfs..."
git lfs install || { echo "Install git-lfs: sudo apt-get install git-lfs"; exit 1; }

# Create venv and install with uv (default-groups in pyproject.toml: dev, sim, hand_tracker, dynamem, da3)
echo ""
echo "[5/5] Creating virtual environment and installing dependencies..."
UV_SYNC=(uv sync)
if [ "$JETSON_INSTALL" = "true" ]; then
    # Lean Orin install: skip SAM-2 / mediapipe / da3; keep pytest + MuJoCo (CLI imports sim paths).
    export UV_PYTHON="${UV_PYTHON:-3.10}"
    export EMET_ALLOW_SDPA_ATTN="${EMET_ALLOW_SDPA_ATTN:-1}"
    UV_SYNC+=(--no-default-groups --group dev --group sim)
    echo "  -> Jetson profile: UV_PYTHON=$UV_PYTHON  EMET_ALLOW_SDPA_ATTN=$EMET_ALLOW_SDPA_ATTN"
    echo "  -> PyPI torch on aarch64 is CPU (or server-CUDA) — Tegra CUDA needs an NVIDIA Jetson wheel or build-from-source."
elif [[ "$NO_SAM2" == "true" ]]; then
    UV_SYNC+=(--no-group dynamem)
fi
if [ ! -d "third_party/robocasa" ] || [ ! -d "third_party/robosuite" ]; then
    echo "Note: third_party/robocasa or robosuite missing — MuJoCo/sim pip deps still install; run ./scripts/install_simulation.sh for editable robosuite/robocasa."
elif ! grep -q '^robocasa = {' pyproject.toml 2>/dev/null; then
    echo "Note: sim block not enabled in pyproject.toml — run: python scripts/enable_sim_pyproject.py  then  uv lock && uv sync"
fi

echo "  -> Running: ${UV_SYNC[*]}"
"${UV_SYNC[@]}"
echo "  -> uv sync completed."

# OpenCV sanity/repair:
# Some dependency combinations (e.g. mediapipe + opencv wheels) can leave cv2 as a namespace
# package without the compiled extension. Detect and repair proactively.
if ! uv run python - <<'PY'
import cv2
required = ["resize", "imencode", "INTER_AREA", "IMWRITE_JPEG_QUALITY"]
missing = [name for name in required if not hasattr(cv2, name)]
raise SystemExit(0 if not missing and getattr(cv2, "__file__", None) else 1)
PY
then
    echo "  -> OpenCV appears broken (cv2 stub/namespace). Reinstalling opencv-contrib-python..."
    uv pip install --python .venv/bin/python --reinstall opencv-contrib-python
    uv run python - <<'PY'
import cv2
required = ["resize", "imencode", "INTER_AREA", "IMWRITE_JPEG_QUALITY"]
missing = [name for name in required if not hasattr(cv2, name)]
if missing or not getattr(cv2, "__file__", None):
    raise SystemExit(f"ERROR: OpenCV repair failed. Missing attributes: {missing}, cv2.__file__={getattr(cv2, '__file__', None)}")
print(f"  -> OpenCV repaired: {cv2.__file__}")
PY
fi

# Uninstall av to avoid conflict (from old install.sh)
source .venv/bin/activate
uv pip uninstall av -y 2>/dev/null || true

# Sim: clone third_party/robosuite and robocasa, install editable, run macros and (optionally) download assets
if [ "$INSTALL_SIM" = "true" ]; then
    echo ""
    echo "Installing simulation (Robocasa + robosuite)..."
    export EMET_USE_UV=1
    export EMET_PYTHON="$ROOT_DIR/.venv/bin/python"
    SIM_SCRIPT="$ROOT_DIR/scripts/install_simulation.sh"
    if [ "$SKIP_ASKING" = "true" ]; then
        if [ "$FORCE_DOWNLOAD" = "true" ]; then
            bash "$SIM_SCRIPT" -y --force-download
        else
            bash "$SIM_SCRIPT" -y
        fi
    else
        if [ "$FORCE_DOWNLOAD" = "true" ]; then
            bash "$SIM_SCRIPT" --force-download
        else
            bash "$SIM_SCRIPT"
        fi
    fi
fi

# MolmoSpaces: separate venv with emet-molmospaces wrapper (molmo-spaces needs mujoco 3.4 + numpy>=2.2)
# emet is not on PyPI — install with --no-deps -e . then --no-deps -e packages/emet_molmospaces, then molmo deps.
# Prefer uv pip install --python .venv-molmospaces/bin/python (uv venvs often have no pip module).
if [ "$INSTALL_MOLMOSPACES" = "true" ]; then
    echo ""
    echo "Setting up MolmoSpaces wrapper venv (.venv-molmospaces)..."
    XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
    MLSPACES_CACHE="${MLSPACES_ASSETS_DIR:-$XDG_CACHE_HOME/molmospaces/assets}"
    export MLSPACES_ASSETS_DIR="$MLSPACES_CACHE"
    export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-$XDG_CACHE_HOME/molmospaces/resource_cache}"
    mkdir -p "$MLSPACES_ASSETS_DIR" "$MLSPACES_CACHE_DIR"
    PY_MOLMO=".venv-molmospaces/bin/python"
    molmo_pip_install() {
        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "$PY_MOLMO" "$@"
        else
            "$PY_MOLMO" -m pip install "$@"
        fi
    }
    molmo_install_editable_chain() {
        molmo_pip_install --upgrade pip 2>/dev/null || true
        molmo_pip_install --no-deps -e .
        if [ -d "packages/emet_molmospaces" ]; then
            # Resolves molmo-spaces from GitHub + mujoco/numpy per packages/emet_molmospaces/pyproject.toml
            molmo_pip_install -e packages/emet_molmospaces
        else
            molmo_pip_install "molmo-spaces @ git+https://github.com/allenai/molmospaces.git@62b416089b2eddff339e52a32106a6bc08ed92b1" "mujoco>=3.4" "numpy>=2.2"
        fi
    }
    _molmospaces_drop_venv_if_python_too_old() {
        if [ ! -d ".venv-molmospaces" ] || [ ! -x "$PY_MOLMO" ]; then
            return 0
        fi
        if "$PY_MOLMO" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)" 2>/dev/null; then
            return 0
        fi
        echo "  -> Replacing .venv-molmospaces (molmo-spaces requires Python >=3.11)..."
        rm -rf .venv-molmospaces
    }
    _molmospaces_drop_venv_if_python_too_old
    if [ ! -d ".venv-molmospaces" ]; then
        if uv venv .venv-molmospaces --python 3.11 2>/dev/null; then
            :
        elif uv venv .venv-molmospaces --python 3.12 2>/dev/null; then
            :
        else
            echo "ERROR: Need Python 3.11+ for MolmoSpaces. Install it or run: uv python install 3.11"
            exit 1
        fi
        molmo_install_editable_chain
        echo "  -> Created .venv-molmospaces with emet and emet-molmospaces wrapper"
    else
        echo "  -> .venv-molmospaces already exists"
        NEED_REPAIR=0
        if [ ! -x "$PY_MOLMO" ]; then
            NEED_REPAIR=1
        elif ! "$PY_MOLMO" -c "import emet" 2>/dev/null; then
            NEED_REPAIR=1
        elif ! "$PY_MOLMO" -c "import emet_molmospaces" 2>/dev/null; then
            NEED_REPAIR=1
        elif ! "$PY_MOLMO" -c "from molmo_spaces.molmo_spaces_constants import get_scenes; from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path" 2>/dev/null; then
            NEED_REPAIR=1
        elif [ ! -x ".venv-molmospaces/bin/emet-molmospaces" ]; then
            NEED_REPAIR=1
        fi
        if [ "$NEED_REPAIR" -eq 1 ]; then
            echo "  -> Repairing MolmoSpaces venv (editable emet + wrapper + molmo-spaces from GitHub)..."
            molmo_install_editable_chain
        fi
    fi
    if ! "$PY_MOLMO" -c "import mujoco; import emet_molmospaces; from molmo_spaces.molmo_spaces_constants import get_scenes; from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path" 2>/dev/null; then
        echo ""
        echo "ERROR: .venv-molmospaces failed verification (molmo-spaces API + mujoco + wrapper)."
        echo "  Fix: rm -rf .venv-molmospaces && ./install.sh --molmospaces -y"
        echo "  Requires Python 3.11+ (uv: uv python install 3.11)."
        exit 1
    fi
    echo "  -> Verified: molmo_spaces imports in .venv-molmospaces"
    echo "  -> MLSPACES_ASSETS_DIR=$MLSPACES_ASSETS_DIR  MLSPACES_CACHE_DIR=$MLSPACES_CACHE_DIR"
    echo "     (must differ; emet defaults cache to …/molmospaces/resource_cache next to …/assets)"
    echo "  -> Run: emet molmospaces list-robots  &&  emet molmospaces serve --viewer"
fi

if [ "$INSTALL_LINGBOT_MAP" = "true" ]; then
    echo ""
    echo "Setting up LingBot-Map venv (.venv-lingbot-map)..."
    bash "$ROOT_DIR/scripts/install_lingbot_map.sh"
fi

# Quick sanity check
if ! uv run python -c "import emet; print('emet:', emet.__file__)" 2>/dev/null; then
    echo "WARNING: emet import check failed. You may need to run: uv sync"
fi

# Put emet CLI in a reasonable place (~/.local/bin so it's on PATH when present)
# Skip when running inside Docker (PATH is set via Dockerfile instead)
echo ""
LINK_EMET="false"
if [ -n "${DOCKER:-}" ]; then
    echo "  -> Skipping link (DOCKER=1); use PATH=/app/.venv/bin or uv run emet"
elif [ "$SKIP_ASKING" = "true" ]; then
    LINK_EMET="true"
else
    read -p "Link 'emet' to ~/.local/bin so you can run it from anywhere? (y/n) " yn
    case $yn in
        y|Y) LINK_EMET="true" ;;
        *) ;;
    esac
fi
if [ "$LINK_EMET" = "true" ]; then
    mkdir -p "$HOME/.local/bin"
    ln -sf "$ROOT_DIR/.venv/bin/emet" "$HOME/.local/bin/emet"
    echo "  -> emet linked to $HOME/.local/bin/emet"
    if ! echo ":$PATH:" | grep -q ":${HOME}/.local/bin:"; then
        echo "  -> Add to your PATH: export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
fi

echo ""
echo "=============================================="
echo "         INSTALLATION COMPLETE"
echo "=============================================="
echo ""
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Run the CLI:  emet  (if linked above) or  uv run emet"
echo ""
if [ "$JETSON_INSTALL" = "true" ]; then
    echo "Jetson profile notes:"
    echo "  - EMET_ALLOW_SDPA_ATTN=1 (no Triton/flash-attn on Tegra)"
    echo "  - PyPI torch on aarch64 is not Tegra-CUDA; see docs/jetson.md"
    echo "  - depth-anything-3 / pycolmap are skipped on aarch64 (use workstation DA3)"
    echo ""
fi
if [ "$INSTALL_SIM" = "true" ] && [ "$INSTALL_MOLMOSPACES" != "true" ]; then
    if [ "$NO_MOLMOSPACES" = "true" ]; then
        echo "MolmoSpaces: skipped (--no-molmospaces). For \`emet serve mujoco --molmospaces-*\` run:  ./install.sh --molmospaces -y"
    elif [ ! -d "$ROOT_DIR/packages/emet_molmospaces" ]; then
        echo "MolmoSpaces: packages/emet_molmospaces not in tree — wrapper not installed. Clone full repo or run:  ./install.sh --molmospaces -y"
    fi
    echo ""
fi
echo "Optional: init all submodules (including ok-robot for advanced workflows):"
echo "  emet install submodules"
echo ""
