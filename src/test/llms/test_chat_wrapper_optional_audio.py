"""Headless simulator imports must not require microphone/speech packages."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("voice", [False, True])
def test_audio_is_imported_only_when_voice_is_requested(voice):
    script = """
import sys
class BlockAudio:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'emet.audio' or fullname.startswith(('emet.audio.', 'pyaudio', 'whisper')):
            raise RuntimeError('optional audio requested')
sys.meta_path.insert(0, BlockAudio())
from emet.llms import LLMChatWrapper
try:
    wrapper = LLMChatWrapper(object(), object(), voice=VOICE)
except RuntimeError as exc:
    assert VOICE and str(exc) == 'optional audio requested'
else:
    assert not VOICE and not hasattr(wrapper, 'audio_recorder')
""".replace("VOICE", repr(voice))
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
