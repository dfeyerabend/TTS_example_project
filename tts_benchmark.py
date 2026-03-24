import time
import os
from pathlib import Path
import dotenv

dotenv.load_dotenv()

# Testtext — gleicher Text fuer alle drei
test_text = "Machine Learning ist ein Teilgebiet der kuenstlichen Intelligenz. Neuronale Netze lernen Muster aus Daten."

results = []

# Ensure output directory exists
out_dir = Path("tts_benchmark_results")
out_dir.mkdir(parents=True, exist_ok=True)

# ─── OpenAI TTS ───────────────────────────────────
from openai import OpenAI
client = OpenAI()

start = time.time()
response = client.audio.speech.create(model="tts-1", voice="nova", input=test_text)
openai_path = out_dir / "bench_openai.mp3"
response.write_to_file(openai_path)
t_openai = time.time() - start
size_openai = os.path.getsize(openai_path)
cost_openai = len(test_text) * 0.015 / 1000
results.append(("OpenAI tts-1", t_openai, size_openai, cost_openai, "Closed Source", "Unbekannt"))

# ─── SpeechT5 ────────────────────────────────────
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch, soundfile as sf

proc = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
mdl = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
voc = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
emb_ds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
spk = torch.tensor(emb_ds[7306]["xvector"]).unsqueeze(0)

start = time.time()
inp = proc(text=test_text, return_tensors="pt")
speech = mdl.generate_speech(inp["input_ids"], spk, vocoder=voc)
speecht5_path = out_dir / "bench_speecht5.wav"
sf.write(speecht5_path, speech.numpy(), samplerate=16000)
t_speecht5 = time.time() - start
size_speecht5 = os.path.getsize(speecht5_path)
results.append(("SpeechT5", t_speecht5, size_speecht5, 0.0, "Encoder-Decoder", "JA"))

# ─── Bark ─────────────────────────────────────────
from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile

bproc = AutoProcessor.from_pretrained("suno/bark-small")
bmdl = BarkModel.from_pretrained("suno/bark-small")

start = time.time()
binp = bproc(test_text, voice_preset="v2/de_speaker_3")
baudio = bmdl.generate(**binp).cpu().numpy().squeeze()
bark_path = out_dir / "bench_bark.wav"
scipy.io.wavfile.write(bark_path, rate=24000, data=baudio)
t_bark = time.time() - start
size_bark = os.path.getsize(bark_path)
results.append(("Bark", t_bark, size_bark, 0.0, "3x Decoder-Only", "NEIN"))

# ─── Ergebnisse ───────────────────────────────────
print(f"\n{'Modell':<15} {'Latenz':<10} {'Groesse':<12} {'Kosten':<10} {'Architektur':<20} {'Cross-Att'}")
print("=" * 85)
for name, t, s, c, arch, ca in results:
    print(f"{name:<15} {t:.1f}s      {s/1024:.0f} KB       ${c:.4f}     {arch:<20} {ca}")

# ─── Kostenrechnung 10.000 Requests ──────────────
print(f"\n--- Kosten fuer 10.000 Requests ---")
print(f"OpenAI:   ${cost_openai * 10000:.2f}/Monat")
print(f"SpeechT5: $0 (aber GPU-Server ~$50-150/Monat)")
print(f"Bark:     $0 (aber GPU-Server ~$100-300/Monat, langsamer)")
print(f"\nBreak-Even OpenAI vs Self-Hosted: ~{cost_openai * 10000 / 100:.0f} Monate bei $100/Monat GPU")