# ============================================================
# IMPORTS
# ============================================================
import gradio as gr
import torch
import numpy as np
import time
import socket
import random
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# ============================================================
# PORT FINDER — picks random port, tries next if taken
# ============================================================
def find_open_port(start_port=None):
    if start_port is None:
        start_port = random.randint(3000, 9000)

    port = start_port

    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                s.close()
                print(f"[SERVER] Using port {port}")
                return port
            except OSError:
                print(f"[SERVER] Port {port} in use, trying {port+1}...")
                port += 1

# ============================================================
# ENGINE BOOT + GPU DETECTION (NO FUNCTION, NO SCOPING BUGS)
# ============================================================
print("Launching Ace-Step Lite...")
print("==========================================")
print("[ENGINE] Ace MusicGen Booting...")
print("==========================================")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = round(props.total_memory / (1024**3))
    print(f"[GPU] {gpu_name} detected ({vram_gb}GB VRAM)")
    print("[GPU] Backend: CUDA")
    device = "cuda"
else:
    print("[GPU] No GPU detected — using CPU (slow)")
    device = "cpu"

# ============================================================
# DSP — Bass boost + limiter
# ============================================================
def bass_boost_and_limit(audio, sr):
    print("[DSP] Applying bass boost...")
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9

    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)

    gain = np.ones_like(spec, dtype=np.float32)
    gain[freqs < 120] *= 1.6
    gain[(freqs >= 120) & (freqs < 250)] *= 1.2

    spec *= gain
    boosted = np.fft.irfft(spec)

    boosted = boosted * 1.3
    boosted = np.clip(boosted, -1.0, 1.0)

    if np.max(np.abs(boosted)) > 0:
        boosted = boosted / np.max(np.abs(boosted)) * 0.98

    return boosted.astype(np.float32)

# ============================================================
# DSP — End cleaner
# ============================================================
def clean_end(audio, sr):
    print("[DSP] Cleaning end...")
    trim_samples = int(sr * 0.5)
    if len(audio) > trim_samples:
        audio = audio[:-trim_samples]

    fade_samples = int(sr * 0.3)
    if len(audio) > fade_samples:
        fade_curve = np.linspace(1.0, 0.0, fade_samples)
        audio[-fade_samples:] *= fade_curve

    return audio

# ============================================================
# Generate one 25s segment
# ============================================================
def generate_segment(prompt, duration_sec):
    print("[GEN] Generating tokens...")
    max_new_tokens = max(int(duration_sec * 50), 400)

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    audio = audio_values[0, 0].cpu().numpy()
    sr = model.config.audio_encoder.sampling_rate

    if np.max(np.abs(audio)) < 1e-4:
        audio = np.random.normal(0, 1e-4, size=audio.shape)

    audio = bass_boost_and_limit(audio, sr)
    audio = clean_end(audio, sr)

    return audio, sr

# ============================================================
# Generate long song EXACT length
# ============================================================
def generate_long_song(prompt, total_minutes, genre, mood, theme, tempo, creativity):
    target_seconds = int(total_minutes * 60)
    segment_length = 25

    full_prompt = f"{genre} {mood} {theme}, tempo {tempo}, creativity {creativity}. {prompt}"

    print(f"[GEN] Starting long generation: target {total_minutes} minutes")

    final_audio = []
    total_len = 0
    segment_index = 1

    while total_len < target_seconds:
        print(f"[GEN] Segment {segment_index} starting...")
        seg, sr = generate_segment(full_prompt, segment_length)
        print(f"[GEN] Segment {segment_index} complete ({len(seg)/sr:.1f}s)")
        final_audio.append(seg)
        total_len += len(seg) / sr
        segment_index += 1

    print(f"[SONG] Final track length: {total_len:.1f}s")
    print("[SONG] Generation complete.")

    return (sr, np.concatenate(final_audio))

# ============================================================
# MODEL LOAD
# ============================================================
print("[ENGINE] Loading MusicGen-Small...")
t0 = time.time()
MODEL_NAME = "facebook/musicgen-small"
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print(f"[ENGINE] Model loaded in {time.time() - t0:.2f}s")

# ============================================================
# UI — Neon Synthwave (external CSS)
# ============================================================
with gr.Blocks(css="style.css", title="Ace MusicGen - Neon Synthwave") as demo:

    gr.Markdown("<h1 class='title'>AI MUSIC GENERATOR</h1>")

    with gr.Row():
        genre = gr.Dropdown(
            ["Synthwave", "Drum & Bass", "Cyberpunk", "Ambient", "EDM"],
            value="Synthwave",
            label="Genre",
            elem_classes="neon-dropdown"
        )
        mood = gr.Dropdown(
            ["Dreamy", "Dark", "Energetic", "Atmospheric"],
            value="Dreamy",
            label="Mood",
            elem_classes="neon-dropdown"
        )
        theme = gr.Dropdown(
            ["Neon City", "Retro Future", "Space", "Night Drive"],
            value="Neon City",
            label="Theme",
            elem_classes="neon-dropdown"
        )
        tempo = gr.Dropdown(
            ["Slow", "Medium", "Fast"],
            value="Medium",
            label="Tempo",
            elem_classes="neon-dropdown"
        )

    prompt = gr.Textbox(
        label="Enter your prompt...",
        value="A nostalgic synthwave track with lush melodies, driving bass, and a retro 80s vibe.",
        lines=3,
        elem_classes="neon-textbox"
    )

    with gr.Row():
        minutes = gr.Slider(1, 3, value=1, label="Track Length (minutes)", elem_classes="neon-slider")
        creativity = gr.Slider(0, 10, value=5, label="Creativity Level", elem_classes="neon-slider")

    generate_btn = gr.Button("Generate", elem_classes="neon-button")

    audio_out = gr.Audio(label="Preview Track", type="numpy", elem_classes="neon-audio")

    generate_btn.click(
        fn=generate_long_song,
        inputs=[prompt, minutes, genre, mood, theme, tempo, creativity],
        outputs=audio_out
    )

# ============================================================
# AUTO PORT + LAUNCH
# ============================================================
port = find_open_port()
demo.launch(server_name="127.0.0.1", server_port=port)


