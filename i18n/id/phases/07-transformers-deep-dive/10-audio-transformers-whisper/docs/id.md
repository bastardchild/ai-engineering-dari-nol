# Transformer Audio — Arsitektur Whisper

> Audio adalah gambar frekuensi terhadap waktu. Whisper adalah ViT yang memakan spektrogram mel dan berbicara kembali.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time:** ~45 minutes

## Masalahnya

Sebelum Whisper (OpenAI, Radford et al. 2022), pengenalan speech otomatis (ASR) state-of-the-art berarti wav2vec 2.0 dan HuBERT — ekstraktor fitur self-supervised plus head yang di-fine-tune. Kualitas tinggi, pipeline data mahal, rapuh-domain. Pengenalan speech multibahasa butuh model terpisah per rumpun bahasa.

Whisper membuat tiga taruhan:

1. **Latih pada segalanya.** 680.000 jam audio berlabel-lemah yang dikeruk dari internet dalam 97 bahasa. Tanpa korpus akademik bersih. Tanpa label fonem.
2. **Satu model multi-task.** Satu decoder dilatih bersama pada transkripsi, penerjemahan, deteksi aktivitas suara, ID bahasa, dan penandaan waktu via token task.
3. **Transformer encoder-decoder standar.** Encoder mengonsumsi spektrogram log-mel. Decoder menghasilkan token teks secara autoregresif. Tanpa vocoder, tanpa CTC, tanpa HMM.

Hasilnya: Whisper large-v3 tangguh lintas aksen, noise, dan bahasa yang punya nol data berlabel bersih. Ia adalah front-end speech bawaan untuk setiap asisten suara open-source dan sebagian besar yang komersial pada 2026.

## Konsepnya

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### Langkah 1 — resample + window

Audio pada 16 kHz. Clip/pad menjadi 30 detik. Hitung spektrogram log-mel: 80 mel bin, stride 10 ms → ~3.000 frame × 80 fitur. Inilah "gambar input" yang dilihat Whisper.

### Langkah 2 — stem convolutional

Dua lapis Conv1D dengan kernel 3 dan stride 2 mereduksi 3.000 frame menjadi 1.500. Memangkas panjang sequence tanpa menambah banyak parameter.

### Langkah 3 — encoder

Encoder transformer 24 lapis (untuk large) atas 1.500 timestep. Positional encoding sinusoidal, self-attention, FFN GELU. Menghasilkan 1.500 × 1.280 hidden state.

### Langkah 4 — decoder

Decoder transformer 24 lapis. Ia menghasilkan token secara autoregresif dari vocabulary BPE yang merupakan superset GPT-2 dengan beberapa token khusus audio.

### Langkah 5 — token task

Prompt decoder dimulai dengan token kontrol yang memberi tahu model apa yang harus dilakukan:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

atau

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

Model dilatih pada konvensi ini. Kamu mengendalikan task lewat prefix. Setara instruction-tuning 2026, tapi diterapkan ke speech.

### Langkah 6 — output

Beam search (lebar 5) dengan ambang log-prob. Timestamp diprediksi setiap 0.02 detik audio ketika token `<|notimestamps|>` tidak ada.

### Ukuran Whisper

| Model | Params | Layers | d_model | Heads | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-layer decoder) |

Large-v3-turbo (2024) memangkas decoder dari 32 lapis menjadi 4. Decoding 8× lebih cepat dengan regresi WER <1 poin. Unlock kecepatan decode itulah mengapa Whisper-turbo menjadi bawaan agen voice real-time pada 2026.

### Apa yang tidak dilakukan Whisper

- Tanpa diarization (siapa yang berbicara). Pasangkan dengan pyannote untuk itu.
- Tanpa streaming real-time secara native — window 30 detik itu tetap. Wrapper modern (`faster-whisper`, `WhisperX`) menempelkan streaming via VAD + overlap.
- Tanpa konteks long-form melampaui 30 dtk tanpa chunking eksternal. Bekerja baik dalam praktik karena speech manusia jarang butuh konteks jarak jauh untuk transkripsi.

### Lanskap 2026

| Task | Model | Notes |
|------|-------|-------|
| English ASR | Whisper-turbo, Moonshine | Moonshine is 4× faster on edge |
| Multilingual ASR | Whisper-large-v3 | 97 languages |
| Streaming ASR | faster-whisper + VAD | 150 ms latency targets achievable |
| TTS | Piper, XTTS-v2, Kokoro | Encoder-decoder pattern, but Whisper-shaped |
| Audio + language | AudioLM, SeamlessM4T | Text tokens + audio tokens in one transformer |

```figure
n5-mel-decode
```

## Bangun Sendiri

Lihat `code/main.py`. Kita tidak melatih Whisper — kita membangun pipeline spektrogram log-mel + pemformat prompt token-task. Itulah bagian yang benar-benar kamu sentuh di produksi.

### Langkah 1: sintesis audio

Bangkitkan gelombang sinus 1 detik pada 440 Hz yang disampel pada 16 kHz. 16.000 sampel.

### Langkah 2: spektrogram log-mel (disederhanakan)

Spektrogram mel penuh butuh FFT. Kita lakukan versi framing + energi per-frame yang disederhanakan yang menunjukkan pipeline tanpa membutuhkan `librosa`:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Frame = 25 ms, hop = 10 ms. Cocok dengan windowing Whisper. Energi per-frame menggantikan mel bin untuk pedagogi.

### Langkah 3: pad menjadi 30 dtk

Whisper selalu memproses chunk 30 detik. Pad (atau clip) spektrogram menjadi 3.000 frame.

### Langkah 4: bangun token prompt

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

Itulah seluruh permukaan kontrol task. Prefix 4 token.

## Gunakan

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

Lebih cepat, kompatibel-OpenAI:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**Kapan memilih Whisper pada 2026:**

- ASR multibahasa dengan satu model.
- Transkripsi tangguh atas audio yang bising dan beragam.
- ASR riset / prototipe — titik mulai tercepat.

**Kapan memilih yang lain:**

- Streaming latensi-ultra-rendah di edge — Moonshine mengalahkan Whisper pada kualitas sebanding.
- AI percakapan real-time yang butuh <200 ms — ASR streaming khusus.
- Diarization pembicara — Whisper tidak melakukan ini; tempelkan pyannote.

## Rilis

Lihat `outputs/skill-asr-configurator.md`. Skill ini memilih model ASR, parameter decoding, dan pipeline preprocessing untuk aplikasi speech baru.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Konfirmasi jumlah frame untuk sinyal 1 detik pada 16 kHz dengan hop 10 ms adalah ~100 frame. Untuk 30 detik: ~3.000 frame.
2. **Sedang.** Bangun spektrogram log-mel penuh memakai `numpy.fft`. Verifikasi 80 mel bin cocok dengan `librosa.feature.melspectrogram(n_mels=80)` dalam galat numerik.
3. **Sulit.** Implementasikan inference streaming: potong audio menjadi window 10 dtk dengan overlap 2 dtk, jalankan Whisper pada setiap chunk, gabungkan transkrip. Ukur word-error rate vs single-pass pada sampel podcast 5 menit.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Mel spectrogram | "Audio image" | 2D representation: frequency bins on one axis, time frames on the other; log-scaled energy per cell. |
| Log-mel | "What Whisper sees" | Mel spectrogram passed through log; approximates human perception of loudness. |
| Frame | "One time slice" | A 25 ms window of samples; overlapping at 10 ms stride. |
| Task token | "Prompt prefix for speech" | Special tokens like `<\|transcribe\|>` / `<\|translate\|>` in the decoder prompt. |
| Voice activity detection (VAD) | "Find the speech" | Gate that removes silence before ASR; cuts cost massively. |
| CTC | "Connectionist Temporal Classification" | Classic ASR loss for alignment-free training; Whisper does NOT use it. |
| Whisper-turbo | "Small decoder, full encoder" | large-v3 encoder + 4-layer decoder; 8× faster decoding. |
| Faster-whisper | "The production wrapper" | CTranslate2 reimplementation; int8 quantization; 4× faster than OpenAI's reference. |

## Bacaan Lanjutan

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — makalah Whisper.
- [OpenAI Whisper repo](https://github.com/openai/whisper) — kode referensi + bobot model. Baca `whisper/model.py` untuk melihat stem Conv1D + encoder + decoder atas-ke-bawah dalam ~400 baris.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — logika beam-search + token-task yang dijelaskan di Langkah 5–6 ada di sini; 500 baris, sepenuhnya terbaca.
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — pendahulu; masih fitur SOTA di sebagian setting.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — wrapper produksi, 4× lebih cepat daripada referensi.
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — ASR 2024 yang ramah-edge, berbentuk-Whisper tapi lebih kecil.
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — resep fine-tuning kanonis termasuk preprocessor spektrogram mel dan penanganan token-timestamp.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — implementasi penuh (encoder, decoder, cross-attention, generation) yang mencerminkan diagram arsitektur pelajaran ini.
