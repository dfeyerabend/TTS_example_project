from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

text = "This is a test of different speaker voices using the same model."
inputs = processor(text=text, return_tensors="pt")

# 5 verschiedene Stimmen
speaker_ids = [0, 100, 3000, 5000, 7306]

for sid in speaker_ids:
    speaker_emb = torch.tensor(embeddings_dataset[sid]["xvector"]).unsqueeze(0)
    speech = model.generate_speech(inputs["input_ids"], speaker_emb, vocoder=vocoder)
    filename = f"speaker_embedding_results/speaker_{sid}.wav"
    sf.write(filename, speech.numpy(), samplerate=16000)
    print(f"Speaker {sid}: {filename}")

print("\nHoert euch alle 5 an.")
print("Gleicher Text, gleiches Modell, verschiedene Speaker Embeddings.")
print("Der Embedding-Vektor (512 Dimensionen) kodiert NUR die Stimme.")
print("Er veraendert den Decoder-Zustand → anderer Query in Cross-Attention → anderer Sound.")