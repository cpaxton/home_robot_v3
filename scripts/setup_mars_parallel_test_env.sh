#!/usr/bin/env bash
# Additional imports required by the full controller/TAMP gate on ARM.
# Run after setup_mars_parallel_env.sh native; never install into the VLM env.
set -euo pipefail
env_prefix=${1:?usage: setup_mars_parallel_test_env.sh ISOLATED_NATIVE_ENV}
if [[ ! -f "$env_prefix/pyvenv.cfg" || "$env_prefix" != /dev/shm/* ]]; then
    echo 'Expected an isolated RAM-backed native environment.' >&2
    exit 2
fi
export TMPDIR=/dev/shm PIP_NO_CACHE_DIR=1
"$env_prefix/bin/python" -m pip install \
    numpy==1.26.4 pytest==8.4.2 lxml==6.0.2 overrides==7.7.0 \
    pyliblzfse==0.4.1 open3d==0.20.0 draccus==0.11.5 scikit-image==0.25.2 \
    pandas==2.3.3 rerun-sdk==0.22.1 pyarrow==19.0.1 scikit-learn==1.7.2 \
    urchin==0.0.30 hello-robot-stretch-urdf==0.1.2 openai==2.26.0 scikit-fmm==2025.6.23
echo 'Run fresh regression and teacher gates on this host before training.'
