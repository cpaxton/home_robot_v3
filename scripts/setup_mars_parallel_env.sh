#!/usr/bin/env bash
# Isolated compute-only Caliban environment. No sudo, drivers, service changes or robot RPCs.
set -euo pipefail

backend=${1:?usage: setup_mars_parallel_env.sh native\|warp NEW_ENV TORCH_SITE_PACKAGES}
target=${2:?new environment path required}
torch_site=${3:?existing read-only Torch site-packages required}
case "$backend" in native|warp) ;; *) echo 'backend must be native or warp' >&2; exit 2 ;; esac
if [[ -e "$target" || ! -d "$torch_site/torch" ]]; then
    echo 'Use a fresh environment path and an existing Torch installation.' >&2
    exit 2
fi
if [[ $(uname -m) != aarch64 ]]; then
    echo 'This setup is pinned for the Caliban aarch64/Python 3.12 host.' >&2
    exit 2
fi
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
python3 -m venv "$target"
site=$("$target/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$torch_site" > "$site/caliban_torch.pth"
export TMPDIR=/dev/shm
export PIP_NO_CACHE_DIR=1
if [[ "$backend" == native ]]; then
    "$target/bin/python" -m pip install \
        numpy==1.26.4 mujoco==3.5.0 stable-baselines3==2.9.0 gymnasium==0.29.1 \
        scipy==1.15.3 trimesh==4.11.3 cloudpickle==3.1.2 wandb==0.30.0 \
        matplotlib==3.10.8 contourpy==1.3.2 termcolor==3.2.0 pyzmq==27.1.0 \
        sophuspy==1.2.0 yacs==0.1.8 omegaconf==2.3.0 pin==2.7.0 \
        opencv-python-headless==4.10.0.84
    "$target/bin/python" -c 'import mujoco, numpy, sophuspy, torch; print(mujoco.__version__, numpy.__version__, torch.__version__)'
else
    "$target/bin/python" -m pip install numpy==1.26.4 mujoco==3.17.0 mujoco-warp==3.17.0 warp-lang==1.18.0
    WARP_CACHE_PATH=/dev/shm/cpaxton-mars-warp-cache "$target/bin/python" -c 'import warp as wp; wp.init(); print(wp.get_device("cuda:0"))'
fi
echo "Created isolated $backend environment: $target"
echo 'RAM-backed environments disappear at reboot; copy result reports to persistent storage.'
