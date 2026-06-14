# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""SigLIP 2 encoder aliases (https://arxiv.org/abs/2502.14786).

Thin shims over :class:`SiglipEncoder` / :class:`MaskSiglipEncoder` so the
``siglip2`` registry name keeps its short version names (``base``, ``large``,
``so400m``, ``giant``) while sharing one implementation (incl. ``dtype`` support
and fp32 output casting) with the SigLIP 1 path.
"""

from .siglip_encoder import MaskSiglipEncoder, SiglipEncoder

# Registry-facing short names -> unified SIGLIP_CHECKPOINTS keys (high-res variants).
_SIGLIP2_VERSION_ALIASES = {
    "base": "siglip2_base_512",
    "large": "siglip2_large_512",
    "so400m": "siglip2_so400m_512",
    "giant": "siglip2_giant",
}


def _resolve_siglip2_version(version: str | None) -> str:
    version = version or "so400m"
    if version not in _SIGLIP2_VERSION_ALIASES:
        raise ValueError(
            f"Invalid version {version}: must be one of {sorted(_SIGLIP2_VERSION_ALIASES)}"
        )
    return _SIGLIP2_VERSION_ALIASES[version]


class Siglip2Encoder(SiglipEncoder):
    """SigLIP 2 image/text encoder (``get_encoder('siglip2')``)."""

    def __init__(
        self,
        normalize: bool = True,
        device: str | None = None,
        version: str | None = None,
        feature_matching_threshold: float = 0.05,
        dtype: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            normalize=normalize,
            device=device,
            version=_resolve_siglip2_version(version),
            feature_matching_threshold=feature_matching_threshold,
            dtype=dtype,
            **kwargs,
        )


class MaskSiglip2Encoder(MaskSiglipEncoder):
    """Per-pixel SigLIP 2 features (same head surgery as :class:`MaskSiglipEncoder`)."""

    def __init__(
        self,
        device: str | None = None,
        version: str | None = None,
        feature_matching_threshold: float = 0.01,
        dtype: str | None = None,
    ) -> None:
        super().__init__(
            device=device,
            version=_resolve_siglip2_version(version),
            feature_matching_threshold=feature_matching_threshold,
            dtype=dtype,
        )
