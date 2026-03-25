import sounddevice as sd
import soundfile as sf
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

client = OpenAI()

def record_audio(duration: int = 5, sample_rate: int = 16000) -> str:
    filepath = "input.wav"
    print(f"Sprich jetzt... ({duration} Sekunden)")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    sf.write(filepath, audio, sample_rate)
    print("Aufnahme gespeichert.")
    return filepath

def transcribe(filepath: str) -> str:
    with open(filepath, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="de"
        )
    return result.text

# --- Main ---
filepath = record_audio(duration=5)
text = transcribe(filepath)
print(f"\nDu hast gesagt: {text}")