# Vergleich Text to Speech
# Das Ergebnis (result.text vs result['text']) ist im Idealfall identisch, weil beide auf der Whisper-Architektur basieren.
# Aber die API-Version nutzt vermutlich ein größeres Modell als whisper-small, deswegen kann die Qualität abweichen — besonders bei Deutsch.

# Option A
# baut einen OpenAI()-Client und ruft client.audio.transcriptions.create() auf — das ist ein HTTP-Request an eine REST-API

from openai import OpenAI
import time
import dotenv

dotenv.load_dotenv()

client = OpenAI()

start = time.time()
with open("../day1_tts/results/tts_benchmark_results/bench_openai.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        language="de"
    )
elapsed = time.time() - start

print(f"Transkription: {result.text}")
print(f"Dauer: {elapsed:.2f}s")

# Option B
# baut eine pipeline() aus der transformers-Bibliothek — das ist ein lokales PyTorch-Modell, das direkt auf deinem Rechner rechnet

from transformers import pipeline
import time

transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device="cpu"  # oder "cuda" falls GPU vorhanden
)

start = time.time()
result = transcriber("tts_benchmark_results/bench_openai.mp3")
elapsed = time.time() - start

print(f"Transkription: {result['text']}")
print(f"Dauer: {elapsed:.2f}s")