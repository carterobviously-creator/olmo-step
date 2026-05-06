"""
Local Text-to-Music Generator using MusicGen-small
Gradio UI + CLI mode
Chunk-based generation with crossfade (1-3 minute songs)
"""

import argparse
import os
import sys

import numpy as np

# ──────────────────────────────────────────────
# GPU / CPU detection
# ──────────────────────────────────────────────
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    print("CUDA not found, using CPU (slower)")

# ──────────────────────────────────────────────
# Model loading  (done once, at import time so
# Gradio reuses the same objects)
# ──────────────────────────────────────────────
MODEL_NAME = "facebook/musicgen-small"
model = None
processor = None


def load_model():
    """Load MusicGen-small once and move it to the detected device."""
    global model, processor
    if model is not None:
        return  # already loaded

    print("Loading MusicGen-small…")
    try:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
    except Exception as exc:
        msg = (
            f"\n[ERROR] Could not load the model: {exc}\n\n"
            "Fix checklist:\n"
            "  1. Check your internet connection.\n"
            "  2. Make sure Hugging Face is reachable "
            "(https://huggingface.co).\n"
            "  3. Install / reinstall dependencies:\n"
            "       pip install -r requirements.txt\n"
            "  4. Install ffmpeg (needed by audiocraft):\n"
            "       Windows : winget install ffmpeg\n"
            "       macOS   : brew install ffmpeg\n"
            "       Linux   : sudo apt install ffmpeg\n"
            "  5. Clear the Hugging Face cache and retry:\n"
            "       Windows : rd /s %USERPROFILE%\\.cache\\huggingface\n"
            "       macOS/Linux: rm -rf ~/.cache/huggingface\n"
        )
        print(msg)
        raise RuntimeError(msg) from exc

    status = "Using GPU" if device == "cuda" else "Using CPU"
    print(status)


# ──────────────────────────────────────────────
# Audio helpers
# ──────────────────────────────────────────────
CHUNK_DURATION = 25  # seconds per chunk (fits well within 20-30 s spec)


def crossfade(a: np.ndarray, b: np.ndarray, sr: int, fade_sec: float = 0.5) -> np.ndarray:
    """Blend the tail of chunk *a* into the head of chunk *b*."""
    fade_samples = int(sr * fade_sec)
    fade_samples = min(fade_samples, len(a), len(b))

    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)

    a_body = a[:-fade_samples] if len(a) > fade_samples else np.array([], dtype=np.float32)
    overlap = a[-fade_samples:] * fade_out + b[:fade_samples] * fade_in
    b_body = b[fade_samples:]

    return np.concatenate([a_body, overlap, b_body])


# ──────────────────────────────────────────────
# Core generation
# ──────────────────────────────────────────────
def generate_music(prompt: str, total_seconds: int, status_cb=None):
    """
    Generate a WAV of approximately *total_seconds* for the given *prompt*.

    *status_cb* is an optional callable(str) used to stream status messages
    back to the Gradio UI.
    """

    def status(msg: str):
        print(msg)
        if status_cb is not None:
            status_cb(msg)

    load_model()
    try:
        sr = model.config.audio_encoder.sampling_rate
    except AttributeError:
        sr = 32000  # MusicGen-small default

    num_chunks = max(1, round(total_seconds / CHUNK_DURATION))
    status(f"{'Using GPU' if device == 'cuda' else 'Using CPU'}")
    status(f"Generating {num_chunks} chunk(s) × ~{CHUNK_DURATION}s …")

    chunks = []
    for i in range(num_chunks):
        status(f"Generating chunk {i + 1}/{num_chunks}…")
        max_new_tokens = max(int(CHUNK_DURATION * 50), 400)

        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

        chunk = audio_values[0, 0].cpu().numpy().astype(np.float32)
        chunks.append(chunk)

    status("Stitching audio…")
    if len(chunks) == 1:
        final = chunks[0]
    else:
        final = chunks[0]
        for chunk in chunks[1:]:
            final = crossfade(final, chunk, sr)

    # Trim or pad to exactly total_seconds
    target_samples = total_seconds * sr
    if len(final) > target_samples:
        # Fade out the last 0.5 s before trimming
        fade_s = min(int(sr * 0.5), len(final))
        final[-fade_s:] *= np.linspace(1.0, 0.0, fade_s)
        final = final[:target_samples]
    elif len(final) < target_samples:
        final = np.pad(final, (0, target_samples - len(final)))

    # Normalise
    peak = np.max(np.abs(final))
    if peak > 0:
        final = final / peak * 0.95

    # Save
    output_path = "output.wav"
    import scipy.io.wavfile as wavfile

    wavfile.write(output_path, sr, final)
    status(f"Saved → {os.path.abspath(output_path)}")
    return output_path, sr, final


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────
def build_ui():
    import gradio as gr

    def run_generation(prompt, duration_minutes):
        total_seconds = int(duration_minutes * 60)
        messages = []

        def collect(msg):
            messages.append(msg)

        try:
            output_path, sr, audio = generate_music(prompt, total_seconds, status_cb=collect)
            log_text = "\n".join(messages)
            # Return (sr, audio_array) for gr.Audio with type="numpy"
            return (sr, audio), log_text
        except Exception as exc:
            log_text = "\n".join(messages) + f"\n\n[ERROR] {exc}"
            return None, log_text

    with gr.Blocks(title="MusicGen-small — Text to Music") as demo:
        gr.Markdown("# 🎵 MusicGen-small — Text-to-Music Generator")
        gr.Markdown(
            "Generate 1–3 minute music clips from a text description using "
            "**MusicGen-small** (chunk-based, crossfaded)."
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt_box = gr.Textbox(
                    label="Prompt",
                    placeholder="e.g. upbeat synthwave with driving bass and neon vibes",
                    lines=3,
                )
                duration_slider = gr.Slider(
                    minimum=1,
                    maximum=3,
                    step=0.5,
                    value=1,
                    label="Duration (minutes)",
                )
                gen_btn = gr.Button("🎶 Generate", variant="primary")
            with gr.Column(scale=2):
                audio_out = gr.Audio(label="Output", type="numpy")
                status_box = gr.Textbox(
                    label="Status",
                    lines=8,
                    interactive=False,
                )

        gen_btn.click(
            fn=run_generation,
            inputs=[prompt_box, duration_slider],
            outputs=[audio_out, status_box],
        )

    return demo


# ──────────────────────────────────────────────
# CLI mode
# ──────────────────────────────────────────────
def cli_main():
    parser = argparse.ArgumentParser(
        description="MusicGen-small text-to-music generator (CLI mode)"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text description of the music")
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Total duration in seconds (60–180). Default: 60",
    )
    args = parser.parse_args()

    duration = max(60, min(180, args.duration))
    if duration != args.duration:
        print(f"[INFO] Duration clamped to {duration}s (valid range: 60–180)")

    try:
        output_path, _, _ = generate_music(args.prompt, duration)
        print(f"Done! Output saved to: {output_path}")
    except Exception as exc:
        print(f"[FATAL] {exc}")
        sys.exit(1)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # If --prompt flag is present → CLI mode, else → Gradio UI
    if "--prompt" in sys.argv:
        cli_main()
    else:
        demo = build_ui()
        demo.launch()
