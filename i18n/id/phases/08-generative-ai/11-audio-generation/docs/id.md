# Pembangkitan Audio

> Audio adalah sinyal 1-D pada 16-48 kHz. Klip lima detik adalah 80-240k sampel. Tidak ada transformer yang attend ke sequence itu secara langsung. Solusi untuk setiap model audio produksi pada 2026 sama: codec neural (Encodec, SoundStream, DAC) mengompresi audio menjadi token diskret pada 50-75 Hz, dan model transformer atau diffusion membangkitkan token.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## Masalahnya

Tiga task pembangkitan audio:

1. **Text-to-speech.** Diberikan teks, hasilkan speech. Speech bersih narrow-band dan punya struktur fonetik kuat — terpecahkan baik oleh transformer-atas-token. VALL-E (Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **Pembangkitan musik.** Diberikan prompt (teks, melodi, progresi chord, genre), hasilkan musik. Distribusi jauh lebih luas. MusicGen (Meta), Stable Audio 2.5, Suno v4, Udio, Riffusion.
3. **Efek audio / desain suara.** Diberikan prompt, hasilkan suara ambient atau Foley. AudioGen, AudioLDM 2, Stable Audio Open.

Ketiganya berjalan pada substrat yang sama: codec audio neural + generator token-AR atau diffusion.

## Konsepnya

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### Codec audio neural

Encodec (Meta, 2022), SoundStream (Google, 2021), Descript Audio Codec (DAC, 2023). Encoder convolutional mengompresi waveform menjadi vektor per-timestep; kuantisasi vektor residual (RVQ) mengubah setiap vektor menjadi kaskade K indeks codebook. Decoder membalikkannya. Audio 24 kHz pada 2 kbps memakai 8 codebook RVQ pada 75 Hz = 600 token/dtk.

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### Dua paradigma generatif di atasnya

**Token-autoregresif.** Ratakan token RVQ menjadi sequence, jalankan transformer decoder-only. MusicGen memakai "delayed parallel" untuk mengemisikan K stream codebook secara paralel dengan offset per-stream. VALL-E membangkitkan token speech dari prompt teks + sampel suara 3 detik.

**Latent diffusion.** Kemas token codec sebagai laten kontinu atau modelkan dengan diffusion kategorikal. Stable Audio 2.5 memakai flow matching pada laten audio kontinu. AudioLDM 2 memakai diffusion teks-ke-mel-ke-audio.

Tren 2024-2026: flow matching menang untuk musik (inference lebih cepat, sampel lebih bersih) sementara token-AR masih mendominasi speech karena ia kausal secara alami dan streaming dengan baik.

## Lanskap produksi

| System | Task | Backbone | Latency |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms first token |
| OpenAI GPT-4o audio | Full-duplex speech | End-to-end multimodal AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | Non-streaming |
| Stable Audio 2.5 | Music / SFX | DiT + flow matching on audio latents | ~10s for 1-minute clip |
| Suno v4 | Full songs | Undisclosed; token-AR suspected | ~30s per song |
| Udio v1.5 | Full songs | Undisclosed | ~30s per song |
| MusicGen 3.3B | Music | Token-AR on Encodec 32kHz | Real-time |
| AudioCraft 2 | Music + SFX | Flow matching | ~5s for 5s clip |
| Riffusion v2 | Music | Spectrogram diffusion | ~10s |

```figure
score-matching
```

## Bangun Sendiri

`code/main.py` mensimulasikan ide inti: latih transformer next-token mungil pada sequence "token audio" sintetis yang dibangkitkan dari dua "style" berbeda (token rendah dan tinggi bergantian untuk style A, ramp monoton untuk style B). Kondisikan pada style dan sampel.

### Langkah 1: token audio sintetis

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### Langkah 2: latih prediktor token mungil

Prediktor ala-bigram yang dikondisikan style. Intinya polanya: token codec → training cross-entropy → sampling autoregresif.

### Langkah 3: sampel kondisional

Diberikan token style dan token awal, sampel token berikutnya dari distribusi prediksi. Lanjutkan untuk 20-40 token.

## Jebakan

- **Kualitas codec membatasi kualitas output.** Kalau codec tidak bisa merepresentasikan suara dengan setia, kualitas generator sebanyak apa pun tidak membantu. DAC adalah yang terbaik open saat ini.
- **Akumulasi galat RVQ.** Setiap lapis RVQ memodelkan residual sebelumnya. Galat pada lapis 1 merambat. Sampling dengan temperature 0 pada lapis lebih tinggi membantu.
- **Struktur musikal.** 30 detik token adalah 20k+ token pada 75 Hz. Sulit untuk transformer. MusicGen memakai sliding window + lanjutan prompt; Stable Audio memakai klip lebih pendek + crossfading.
- **Artefak di batas.** Crossfading antar klip yang dibangkitkan butuh overlap-add yang hati-hati.
- **Selera data-bersih.** Generator musik butuh puluhan ribu jam musik berlisensi. Gugatan RIAA Suno / Udio (2024) membawa ini ke permukaan.
- **Etika kloning suara.** Sampel 3 detik plus prompt teks cukup bagi VALL-E / XTTS / ElevenLabs untuk mengkloning suara. Setiap model produksi butuh deteksi penyalahgunaan + daftar opt-out.

## Gunakan

| Task | 2026 stack |
|------|------------|
| Commercial TTS | ElevenLabs, OpenAI TTS, or Azure Neural |
| Voice cloning (consent-verified) | XTTS v2 (open) or ElevenLabs Pro |
| Background music, fast | Stable Audio 2.5 API, Suno, or Udio |
| Music with lyrics | Suno v4 or Udio v1.5 |
| Sound effects / Foley | AudioCraft 2, ElevenLabs SFX, or Stable Audio Open |
| Real-time voice agent | GPT-4o realtime or Gemini Live |
| Open-weights music research | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| Dubbing / translation | HeyGen, ElevenLabs Dubbing |

## Rilis

Simpan `outputs/skill-audio-brief.md`. Skill menerima brief audio (task, durasi, style, suara, lisensi) dan mengeluarkan: model + hosting, format prompt (tag genre, deskriptor style, penanda struktural), rantai codec + generator + vocoder, protokol seed, dan rencana eval (skor MOS / CLAP / CER untuk TTS / A/B pengguna).

## Latihan

1. **Mudah.** Jalankan `code/main.py` dan atur style eksplisit. Verifikasi sequence yang dibangkitkan cocok dengan pola style-nya.
2. **Sedang.** Tambahkan decoding delayed parallel: simulasikan 2 stream token yang harus tetap offset 1 langkah. Latih prediktor bersama.
3. **Sulit.** Pakai HuggingFace transformers untuk menjalankan MusicGen-small secara lokal. Bangkitkan klip 10 detik dengan tiga prompt berbeda; A/B untuk kepatuhan style.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Codec | "Neural compression" | Encoder / decoder for audio; typical output is 50-75 Hz tokens. |
| RVQ | "Residual VQ" | Cascade of K quantizers; each models the residual of the previous. |
| Token | "One codec symbol" | Discrete index into a codebook; 1024 or 2048 typical. |
| Delayed parallel | "Offset codebooks" | Emit K token streams with staggered offsets to reduce sequence length. |
| Flow matching | "The 2024 win for audio" | Straighter-path alternative to diffusion; faster sampling. |
| Voice prompt | "3-second sample" | Speaker embedding or token prefix that steers the cloned voice. |
| Mel spectrogram | "The visual" | Log-magnitude perceptual spectrogram; used by many TTS systems. |
| Vocoder | "Mel to wave" | Neural component that converts mel spectrograms back to audio. |

## Catatan produksi: audio adalah problem streaming

Audio adalah satu-satunya modalitas output yang diharapkan pengguna tiba *saat dibangkitkan*, bukan sekaligus. Dalam istilah produksi ini berarti TPOT penting (Time Per Output Token) karena kecepatan mendengar pengguna adalah throughput target — bukan kecepatan membaca mereka. Untuk audio 16kHz yang ditokenisasi pada ~75 token/detik (Encodec), server harus membangkitkan ≥75 token/dtk per pengguna agar playback mulus.

Dua konsekuensi arsitektural:

- **Model audio flow-matching tidak bisa streaming secara trivial.** Stable Audio 2.5 dan AudioCraft 2 merender panjang klip tetap dalam satu pass. Untuk streaming, kamu memotong klip dan menumpuk batas — anggap diffusion sliding-window — menambah overhead latensi 100-300ms vs model AR codec.

Kalau produknya "chat suara live" atau "lanjutan musik real-time", pilih jalur AR codec. Kalau "render klip 30 detik saat submit", flow-matching menang pada kualitas dan latensi total.

## Bacaan Lanjutan

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — standar codec.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — codec audio neural pertama yang dipakai luas.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2.
- [Stability AI (2025). Stable Audio 2.5](https://stability.ai/news-updates/stability-ai-introduces-stable-audio-25-the-first-audio-model-built-for-enterprise-sound-production-at-scale) — teks-ke-musik 2025 dengan flow matching.
