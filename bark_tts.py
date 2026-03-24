from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile
import time

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small")

# ─── Test 1: Deutsch ──────────────────────────────
text_de = "Hallo, ich bin ein KI-Agent und ich lerne gerade Text-to-Speech."
inputs_de = processor(text_de, voice_preset="v2/de_speaker_3")

start = time.time()
audio_de = model.generate(**inputs_de)
print(f"Deutsch: {time.time() - start:.1f}s")

audio_de = audio_de.cpu().numpy().squeeze()
scipy.io.wavfile.write("bark_results/bark_deutsch.wav", rate=24000, data=audio_de)

# ─── Test 2: Englisch ─────────────────────────────
text_en = "Hello, I am an AI agent and I am learning text to speech."
inputs_en = processor(text_en, voice_preset="v2/en_speaker_6")

start = time.time()
audio_en = model.generate(**inputs_en)
print(f"Englisch: {time.time() - start:.1f}s")

audio_en = audio_en.cpu().numpy().squeeze()
scipy.io.wavfile.write("bark_results/bark_english.wav", rate=24000, data=audio_en)

# ─── Test 3: Kreativ (Bark kann lachen!) ──────────
text_creative = "Das ist witzig! [laughs] Nein, wirklich. [sighs] Okay, zurueck zur Arbeit."
inputs_creative = processor(text_creative, voice_preset="v2/de_speaker_5")

start = time.time()
audio_creative = model.generate(**inputs_creative)
print(f"Kreativ: {time.time() - start:.1f}s")

audio_creative = audio_creative.cpu().numpy().squeeze()
scipy.io.wavfile.write("bark_results/bark_creative.wav", rate=24000, data=audio_creative)

print("\nHoert euch alle drei Dateien an.")
print("Bark kann [laughs], [sighs], [clears throat], [gasps] und Musik.")
print("Das kann SpeechT5 NICHT — weil SpeechT5 nur Mel-Spectrograms fuer Sprache generiert.")
print("Bark generiert allgemeine Audio-Tokens, nicht nur Sprache.")