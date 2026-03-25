import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from anthropic import Anthropic
from pathlib import Path
import dotenv
import numpy as np

# Künstliche Intelligenz verändert die Arbeitswelt im Jahr 2026

dotenv.load_dotenv()

openai_client = OpenAI()
anthropic_client = Anthropic()

def record_audio(max_duration: int = 30, sample_rate: int = 16000, silence_threshold: float = 0.01, silence_duration: float = 2.0) -> str:
    """Nimmt Audio auf bis 3s Stille oder max_duration erreicht."""
    filepath = "./recordings/test_audio.wav"  # dein Pfad beibehalten
    chunk_size = int(sample_rate * 0.1)
    chunks = []
    silent_time = 0.0
    total_time = 0.0

    print(f"\nSprich jetzt... (max {max_duration}s, stoppt nach {int(silence_duration)}s Stille)")

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
        while total_time < max_duration:
            data, _ = stream.read(chunk_size)
            chunks.append(data.copy())

            volume = np.abs(data).mean()

            if volume < silence_threshold:
                silent_time += 0.1
            else:
                silent_time = 0.0

            total_time += 0.1

            if silent_time >= silence_duration:
                print(f"Stille erkannt, stoppe nach {total_time:.1f}s")
                break

    if total_time >= max_duration:
        print(f"Max-Dauer erreicht ({max_duration}s)")

    audio = np.concatenate(chunks)
    sf.write(filepath, audio, sample_rate)
    print("Aufnahme gespeichert.")
    return filepath

def transcribe_to_text(filepath: str) -> str:
    """Whisper STT: Audio --> Text"""
    with open(filepath, 'rb') as f:
        result = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="de"
        )
    return result.text

def ask_agent(user_text: str, history: list) -> str:
    """Claude Agent: Text --> Text"""
    history.append({"role": "user", "content": user_text})

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system="Du bist ein hilfreicher assistant. Antworte kurz und präzise auf Deutsch.",
        messages=history
    )

    assistant_text = response.content[0].text
    history.append({"role": "assistant", "content": assistant_text})
    return assistant_text

def agent_speak(text: str) -> None:
    """TTS: Text --> Audio --> Playback"""
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )

    output_path = "./outputs/response.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)

    # Audio abspielen
    data, samplerate = sf.read(output_path)
    sd.play(data, samplerate)
    sd.wait()


# --- Main ---
print("=== Continuous Voice Agent ===")
print("Sage 'stop' oder 'ende' zum Beenden.\n")
print("Single Loop Pipeline: Mikrofon --> Whisper(STT) --> Claude(Agent) --> TTS --> Lautsprecher\n")

conversation_history = []
running = True

while running:
    filepath = record_audio(max_duration=20)
    user_text = transcribe_to_text(filepath)
    print(f"[USER]  {user_text}")

    stop_words = ["stop", "stopp", "ende", "aufhören", "tschüss", "quit"]
    if any (word in user_text.lower() for word in stop_words):
        print("\n[Agent beendet. Tschüss!]")
        agent_speak("Tschüss! Bis zum nächsten Mal.")
        running = False
        continue

    agent_text = ask_agent(user_text=user_text, history=conversation_history)
    print(f"[AGENT] {agent_text}")
    agent_speak(agent_text)

print(f"\nGespräch beendet. {len(conversation_history) // 2} Austausche.")

