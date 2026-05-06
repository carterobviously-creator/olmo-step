# 🎵 MusicGen-small — Local Text-to-Music Generator

Generate **1–3 minute music clips** from plain text using
[MusicGen-small](https://huggingface.co/facebook/musicgen-small) entirely on
your own machine — no cloud API, no subscription.  
Comes with a **Gradio web UI** and a **CLI mode**.

---

## Requirements

| Item | Minimum |
|------|---------|
| Python | 3.10, 3.11, or 3.12 |
| RAM | 8 GB |
| Disk (free) | 4 GB (model ≈ 2 GB + working space) |
| GPU (optional) | CUDA-capable NVIDIA GPU (any VRAM ≥ 4 GB) |

---

## 1 · Install Python

Download Python **3.10, 3.11, or 3.12** from <https://www.python.org/downloads/>.

- **Windows**: tick *"Add Python to PATH"* during installation.
- **macOS / Linux**: Python is often pre-installed; check with `python3 --version`.

---

## 2 · Clone the repository

```bash
git clone https://github.com/carterobviously-creator/olmo-step
cd olmo-step
```

---

## 3 · Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your prompt.

---

## 4 · Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users (NVIDIA / CUDA)** — if `pip install torch` installs the CPU
> build, force the CUDA build first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt
> ```

Also install **ffmpeg** (required by the audio stack):

| OS | Command |
|----|---------|
| Windows | `winget install ffmpeg` |
| macOS | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt install ffmpeg` |

---

## 5 · Run the app

### Gradio web UI (recommended)

```bash
python app.py
```

Open the URL printed in the terminal (e.g. `http://127.0.0.1:7860`) in your
browser.

### CLI mode

```bash
python app.py --prompt "upbeat synthwave with driving bass" --duration 120
```

`--duration` is in **seconds** (60–180). The output is saved as `output.wav`
in the current directory.

---

## GPU vs CPU behaviour

The app detects your hardware automatically using PyTorch:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

| Situation | What you see | Speed |
|-----------|--------------|-------|
| NVIDIA GPU with CUDA | `Using GPU` | Fast (seconds per chunk) |
| No GPU / unsupported | `CUDA not found, using CPU (slower)` | Slow (minutes per chunk) |

GPU generation is **strongly recommended** for 3-minute tracks.

---

## Disk space notes

| Item | Size |
|------|------|
| MusicGen-small weights | ≈ 2 GB |
| Hugging Face model cache | included above |
| `output.wav` (3 min, 32 kHz) | ≈ 23 MB |
| **Total needed (free SSD)** | **≈ 3–3.5 GB** |

The model is downloaded **once** into `~/.cache/huggingface` on first run and
reused on every subsequent run.

---

## How chunking works

MusicGen is designed for short generations (≤ 30 s).  
To produce longer music this app:

1. Divides the requested duration into **≈ 25-second chunks**.
2. Generates each chunk independently with the same prompt.
3. **Crossfades** adjacent chunks (0.5 s overlap) so transitions are smooth.
4. Concatenates everything into one final WAV and normalises the peak level.

For example, a 2-minute (120 s) request → **5 chunks** × 25 s each.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Model download fails | Check internet; try again; clear `~/.cache/huggingface` |
| `ffmpeg not found` | Install ffmpeg (see step 4 above) |
| Out of memory on GPU | Close other GPU apps; lower duration |
| Very slow on CPU | Normal — a 1-min track can take 10–20 min on CPU |
