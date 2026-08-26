# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

import io
import os

import cv2
import liblzfse
import numpy as np
from PIL import Image


## Compress Python Object to Bytes
def zip_depth(obj: np.ndarray):
    """
    Compresses a Python object to bytes using pickle.

    Args:
        obj: The Python object to be compressed.

    Returns:
        bytes: The compressed bytes representation of the object.
    """
    # compressed_bytes = pickle.dumps(obj)
    compressed_bytes = liblzfse.compress(obj.astype(np.uint16).tobytes())
    # depth_bytes = liblzfse.compress(depth_array.astype(np.float32).tobytes())
    return compressed_bytes


## Decompress Bytes to Python Object
def unzip_depth(compressed_bytes, shape: tuple[int, int] | None = None) -> np.ndarray:
    """
    Decompresses bytes to a Python object using pickle.

    Args:
        compressed_bytes: The compressed bytes representation of the object.

    Returns:
        The decompressed Python object.
    """
    # obj = pickle.loads(compressed_bytes)
    buffer = np.frombuffer(liblzfse.decompress(compressed_bytes), dtype=np.uint16)
    if shape is not None:
        buffer = buffer.reshape(*shape)
    return buffer


def to_webp(img: np.ndarray):
    """
    Converts a NumPy array to a WebP image (bytes).

    Args:
        arr (numpy.ndarray): The input NumPy array.

    Returns:
        bytes: The WebP image data as bytes.
    """
    # Convert the NumPy array to a PIL Image
    pil_img = Image.fromarray(img)

    # Create a BytesIO object to store the WebP image data
    webp_bytes = io.BytesIO()

    # Save the image as WebP format to the BytesIO object
    pil_img.save(webp_bytes, format="WebP", lossless=False)

    # Get the bytes from the BytesIO object
    webp_bytes_data = webp_bytes.getvalue()
    return webp_bytes_data


def from_webp(webp_data) -> np.ndarray:
    # Create a BytesIO object from the WebP image data
    webp_io = io.BytesIO(webp_data)

    # Open the WebP image from the BytesIO object
    img = Image.open(webp_io)

    # Convert the PIL Image to a NumPy array
    arr = np.array(img)
    return arr


def to_jp2(image: np.ndarray, quality: int = 800):
    """Depth is better encoded as jp2"""
    _, compressed_image = cv2.imencode(".jp2", image, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, quality])
    return compressed_image


def to_jpg(image: np.ndarray, quality: int = 90):
    """Encode as JPEG. Input must be **RGB** uint8 (H,W,3); OpenCV expects BGR for ``imencode``."""
    if os.environ.get("EMET_ZMQ_TURBOJPEG", "").strip().lower() in ("1", "true", "yes", "on"):
        try:
            from turbojpeg import TJPF_RGB, TurboJPEG

            return TurboJPEG().encode(image, quality=quality, pixel_format=TJPF_RGB)
        except ImportError:
            pass
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, compressed_image = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return compressed_image


def from_jpg(compressed_image: bytes | np.ndarray) -> np.ndarray:
    """Decode JPEG to **RGB** uint8 (H,W,3). ``imdecode`` yields BGR; we convert back."""
    if isinstance(compressed_image, bytes):
        compressed_image = np.frombuffer(compressed_image, dtype=np.uint8)
    bgr = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed for JPEG bytes")
    if bgr.ndim == 3 and bgr.shape[2] == 3:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr


def from_jp2(compressed_image: bytes | np.ndarray) -> np.ndarray:
    """Convert compressed image to numpy array"""
    if isinstance(compressed_image, bytes):
        compressed_image = np.frombuffer(compressed_image, dtype=np.uint8)
    return cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)


def from_h264(nal_bytes: bytes | np.ndarray) -> np.ndarray:
    """Decode one H.264 access-unit (NAL bundle) to RGB uint8 via PyAV when installed."""
    try:
        import av
    except ImportError as exc:
        raise ImportError("PyAV is required for H.264 ZMQ decode (pip install av)") from exc

    if isinstance(nal_bytes, np.ndarray):
        nal_bytes = bytes(np.asarray(nal_bytes).tobytes())
    container = av.open(io.BytesIO(nal_bytes), format="h264")
    for frame in container.decode(video=0):
        rgb = frame.to_ndarray(format="rgb24")
        return np.ascontiguousarray(rgb)
    raise ValueError("no video frame in H.264 NAL bytes")


def to_h264(image: np.ndarray) -> bytes:
    """Encode one RGB frame to a single H.264 access unit (PyAV)."""
    try:
        import av
    except ImportError as exc:
        raise ImportError("PyAV is required for H.264 ZMQ encode (pip install av)") from exc

    h, w = image.shape[:2]
    output = io.BytesIO()
    container = av.open(output, mode="w", format="h264")
    stream = container.add_stream("h264", rate=30)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
    for packet in stream.encode(frame):
        output.write(bytes(packet))
    for packet in stream.encode():
        output.write(bytes(packet))
    container.close()
    return output.getvalue()
