from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

# --- Loads models ---

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")                 # Tokenizer: wandelt Text → Token-IDs (Integer-Sequenz)
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")               # Encoder-Decoder: Token-IDs → Mel-Spectrogram
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")                 # Vocoder: Mel-Spectrogram → hörbares Audio-Signal

# --- Text vorbereiten ---

inputs = processor(text="Hello, explain to me promises in TS", return_tensors="pt")     # Text → Token-IDs als PyTorch-Tensor, "pt" = PyTorch-Format

# --- Speaker Embedding laden ---
# CMU Arctic xvectors: ein Datensatz mit ~7900 vorberechneten Stimm-Vektoren aus echten Sprachaufnahmen
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)

speaker_embeddings = torch.tensor(                # wandelt den xvector (Python-Liste) in einen PyTorch-Tensor
    embeddings_dataset[7306]["xvector"]           # holt den 512-dim Vektor von Speaker #7306 (eine bestimmte Stimme)
).unsqueeze(0)                                    # fügt Batch-Dimension hinzu: Shape [512] → [1, 512], weil das Modell einen Batch erwartet


# --- Audio generieren ---

speech = model.generate_speech(       # führt die komplette Generierung aus
    inputs["input_ids"],              # Encoder-Input: die Token-IDs des Textes
    speaker_embeddings,               # steuert WIE die Stimme klingt (nicht WAS gesagt wird)
    vocoder=vocoder                   # wenn vocoder übergeben → Output ist direkt Audio, nicht Mel-Spectrogram
)

# --- Speichern ---

sf.write("output.wav", speech.numpy(), samplerate=16000)                                        # schreibt den Audio-Tensor als WAV-Datei, 16kHz Samplerate
print(f"Audio gespeichert: output.wav {len(speech)} samples, {len(speech)/16000:.1f} Sekunden")     # zeigt Länge in Samples und Sekunden



#embeddings_dataset = embeddings_dataset.map()

# Maennliche Stimme:
#speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Weibliche Stimme:
#speaker_embeddings = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)

# Noch eine andere:
#speaker_embeddings = torch.tensor(embeddings_dataset[3000]["xvector"]).unsqueeze(0)