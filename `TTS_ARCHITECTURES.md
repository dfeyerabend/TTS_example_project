
### Pipeline-Diagramme

SpeechT5:
Text → [Tokenizer] → Token-IDs → [Encoder] → Hidden States
                                                   ↓ (Cross-Attention)
Speaker Embedding → [Decoder] → Mel-Spectrogram → [HiFi-GAN Vocoder] → Audio

Bark:
Text + Voice Preset → [Semantic Decoder] → Semantic Tokens
                      → [Coarse Decoder] → Coarse Audio Tokens
                      → [Fine Decoder]   → Fine Audio Tokens → Audio

OpenAI TTS:
Text + Voice Parameter → [API / Black-Box] → Audio


### Wie löst Bark das Alignment-Problem ohne Cross-Attention?
- Bark kodiert die Reihenfolge der Wörter im Audio direkt anhand der Reihenfolge der Wöter im Text. Keine semantischen Änderungen, der Text bleibt exakt gleich.


### Warum braucht SpeechT5 einen Vocoder, Bark nicht
- Weil SpeechT5 ein Mel-Spectrogram als Zwischenschritt generiert. Das ist nur eine vereinfachte Darstellung von Audio, kein hörbarer Sound. Deshalb braucht man HiFi-GAN um daraus echtes Audio zu machen.
- Bark generiert direkt Audio-Tokens und braucht daher keinen Vocoder

### Wie ändert man die Stimme bei Bark vs SpeechT5?
- Bei SpeechT5 tauscht man den Speaker Embedding Vektor aus
- Bei Bark ändert man den voice_preset Parameter

### Vor-/Nachteile Encoder-Decoder vs Decoder-Only TTS?
- SpeechT5 (Encoder-Decoder) ist schneller weil der Encoder den Text in einem Durchgang verarbeitet.
- Der Text kann außerdem dem jeweiligen Sprachmodel angepasst werden weil Input nicht direkt Output sein muss

- Bark (Decoder-Only) ist langsamer weil es drei Decoder nacheinander durchlaufen muss.
- Dafür ist Bark flexibler und kann auch Lachen oder Seufzen generieren

### Unterschied Text-Token-Generierung (Qwen) vs Audio-Token-Generierung (Bark)?
- Qwen und Bark machen beide das Gleiche — nächsten Token vorhersagen basierend auf allem davor. 
- Der einzige echte Unterschied: was die Tokens bedeuten. 
  - Bei Qwen steht ein Token für ein Wort oder Wortteil. 
  - Bei Bark steht ein Token für ein kleines Stück Klang. Und weil Audio viel mehr Daten hat als Text (du brauchst tausende Samples pro Sekunde), reicht bei Bark ein Decoder nicht — deshalb hat es drei hintereinander, die jeweils mehr Detail hinzufügen.

**Vergleichstabelle**

| | SpeechT5        | Bark            | OpenAI TTS | Euer Qwen LLM |
|---|-----------------|-----------------|------------|---|
| Architektur | Encoder-Decoder | 3x Decoder      | KA         | ? |
| Input | Text            | Text            | Text       | ? |
| Output | Mel-Spectogram  | Audio Tokens    | Audio      | ? |
| Cross-Attention | JA              | Nein            | KA         | ? |
| Vocoder noetig | JA              | Nein            | KA         | ? |
| Trainiert auf | Sprach-Daten    | Sprache und Sound | KA         | ? |
| Staerke | Schnell         | Flexibler       | Qualität   | ? |
| Schwaeche | Nur Sprache     | Langsam         | Kosten     | ? |

