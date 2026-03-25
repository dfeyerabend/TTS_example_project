### Teil 1: Bark — Decoder-Only TTS

Was klingt besser — SpeechT5 oder Bark? Warum?
- bark klingt besser, die stimmte ist kohärenter in ihrer stimmung und klingt nicht so robotisch
- SpeechT5 generiert ein Mel-Spectrogram (80 Frequenzbänder), und dann muss ein separater Vocoder (HiFi-GAN) daraus Audio machen. Dabei gehen informationen verloren und der Ton ist eher neutral/sachlich.


### Teil 2: Speaker Embedding Vergleich

Welcher Speaker klingt am natuerlichsten? Welcher am wenigsten?
- Speaker 5000 klingt am natürlichsten. Der Ton ist fließend und klingt natürlich
- Speaker 0 klingt sehr unnatürlich. Die Sprache ist abgehackt und es entstehen große Lücken ohne Ton

Erklaerung in eigenen Worten: Wie aendert der Speaker Embedding den Sound ohne die Weights zu aendern?
- Der Speaker Embedding ist ein 512-dimensionaler Vektor — einfach eine Liste von 512 Zahlen, die eine bestimmte Stimme beschreiben (Tonhöhe, Klangfarbe, Tempo, etc.)
- Der Decoder berechnet mit Cross-Attention welches Wort grade in Ton umgewandelt wird. Diese Vektoren werden aber direkt vom Speaker Embedding verändert. Darum ist auch der Inhalt zwischen den Aufnahmen anders,

### Teil 3: Mel-Spectrogram Visualisierung

Was seht ihr im Spectrogram? Wo sind die Pausen? Wo die Vokale?
- Pausen: Die dunklen vertikalen Streifen, wo das Spectrogram fast komplett schwarz wird
- Vokale: Die hellen horizontalen Bänder im unteren Frequenzbereich (unter ~512 Mel).