# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import setuptools

__version__ = None
with open("emet/version.py") as f:
    exec(f.read())  # overrides __version__

with open("../README.md") as fh:
    long_description = fh.read()

setuptools.setup(
    name="emet",
    version=__version__,
    author="Hello Robot Inc.",
    author_email="support@hello-robot.com",
    description="Stretch Python API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hello-robot/stretchpy",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        "emet": [
            "config/**/*.yaml",
            "perception/*.tsv",
            "assets/**/*",
        ]
    },
    install_requires=[
        # Machine learning code, we will install these packages in install.sh instead
        "torch>=2.6",
        "torchvision",
        "torchaudio",
        # General utilities
        "pyyaml",
        "pyzmq",
        "numpy<2",
        "numba",
        "opencv-python",
        "scipy",
        "matplotlib",
        "trimesh>=3.10.0",
        "yacs",
        "scikit-image>=0.21.0",
        "sophuspy",
        "pin",  # Pinocchio IK solver
        "pynput",
        "pyusb",
        "schema",
        "overrides",
        "wget",
        # From openai
        "openai >= 1.88.0",
        # For gemini
        "google-genai",
        # For Yolo
        "ultralytics==8.3.161",
        # Hardware dependencies
        "hello-robot-stretch-urdf",
        "pyrealsense2",
        "urchin",
        # Visualization
        "rerun-sdk>=0.21.0,<0.23.0",
        # For siglip encoder
        "sentencepiece",
        # For git tools
        "gitpython",
        # Configuration tools and neural networks
        "hydra-core",
        "draccus>=0.11.0",
        "timm>1.0.0",
        "huggingface_hub>=0.28.0",
        "safetensors>=0.4.5",
        # For mobile clip
        "open-clip-torch>=2.32.0",
        "transformers>=4.55.0",
        "retry",
        "qwen_vl_utils",
        "bitsandbytes",
        "triton >= 2.3.1",
        "accelerate >= 1.6.0",
        "einops",
        "protobuf",
        # Compression tools
        "pyliblzfse",
        "webp>=0.3.0",
        # UI tools
        "termcolor",
        # Audio
        "librosa",  # audio analysis (e.g., spectral similarity)
        "PyAudio>=0.2.14",  # the version specification is necessary because apt has 0.2.12 which is incompatible with recent numpy
        "openai-whisper",
        "overrides",  # better inheritance of docstrings
        "pydub",  # playback audio
        "simpleaudio",  # playback audio
        # "wave",
        # These are not supported > python 3.11
        "scikit-fmm",
        "open3d",
        "click>=8.1.8",
        "discord.py",
        "python-dotenv",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pytest",
            "flake8",
            "black",
            "mypy",
            "lark",
            "rich>=13.0.0",
        ],
        "sim": [
            "mujoco>=3.4.0",
            "hello-robot-stretch-urdf",
            "grpcio",
            "click>=8.1.8",
            "inputs>=0.5",
        ],
        "hand_tracker": [
            "mediapipe",
            "webcam",
        ],
        "da3": [
            "depth-anything-3>=0.1.1",
        ],
    },
)
