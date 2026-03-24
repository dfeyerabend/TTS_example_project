import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa
import librosa.display

# ─── Schritt 1: Audio laden ──────────────────────

# Benutzt eine eurer generierten Dateien:
audio, sr = librosa.load("speaker_embedding_results/speaker_5000.wav", sr=16000)

# ─── Schritt 2: Mel-Spectrogram berechnen ────────
mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=80,         # 80 Frequenzbaender (Standard fuer TTS)
    n_fft=1024,         # FFT Fenstergroesse
    hop_length=256      # Schrittweite zwischen Frames
)

# In Dezibel umrechnen (fuer bessere Visualisierung)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# ─── Schritt 3: Plotten ─────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Oben: Wellenform (was ihr hoert)
axes[0].set_title("Wellenform (Audio Signal)")
librosa.display.waveshow(audio, sr=sr, ax=axes[0])
axes[0].set_xlabel("Zeit (Sekunden)")
axes[0].set_ylabel("Amplitude")

# Unten: Mel-Spectrogram (was der Decoder generiert)
axes[1].set_title("Mel-Spectrogram (was der TTS-Decoder generiert)")
img = librosa.display.specshow(
    mel_spec_db,
    x_axis='time',
    y_axis='mel',
    sr=sr,
    hop_length=256,
    ax=axes[1]
)
axes[1].set_xlabel("Zeit (Sekunden)")
axes[1].set_ylabel("Frequenz (Mel)")
fig.colorbar(img, ax=axes[1], format='%+2.0f dB')

plt.tight_layout()
plt.savefig("mel_spectrogram.png", dpi=150)
plt.show()
print("Gespeichert: mel_spectrogram.png")

