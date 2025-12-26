from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile

model = WhisperModel("small", compute_type="int8")
def transcribe_audio(audio_path: str) -> str:
    segments, _ = model.transcribe(audio_path)
    return " ".join(seg.text for seg in segments)


SAMPLE_RATE = 16000
def record_from_mic(duration: int = 6) -> str:
    print(f"[Mic] Recording for {duration} seconds...")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        channels=1,
        dtype="float32"
    )
    sd.wait()

    tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    write(tmp_file.name, SAMPLE_RATE, audio)

    print("[Mic] Recording saved.")
    return tmp_file.name