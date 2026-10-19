# Generasi Musik — MusicGen, Stable Audio, Suno, dan Gempa Bumi Lisensi

> Generasi musik 2026: Suno v5 dan Udio v4 mendominasi komersial; MusicGen, Stable Audio Open, dan ACE-Step memimpin open-source. Masalah teknisnya sebagian besar sudah selesai. Masalah hukumnya (setelmen Warner Music $500M, setelmen UMG) membentuk ulang bidang ini pada 2025-2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time:** ~75 minutes

## Masalahnya

Teks → cuplikan musik 30 detik sampai 4 menit, dengan lirik, vokal, dan struktur. Tiga sub-masalah:

1. **Generasi instrumental.** Teks seperti "drum hip-hop lo-fi dengan kibor hangat" → audio. MusicGen, Stable Audio, AudioLDM.
2. **Generasi lagu (dengan vokal + lirik).** "Lagu country tentang malam Texas yang hujan" → lagu penuh. Suno, Udio, YuE, ACE-Step.
3. **Kondisional / terkontrol.** Perpanjang cuplikan yang ada, regenerasi bridge, tukar genre, pisahkan stem, atau inpainting. Inpainting + pemisahan stem Udio adalah fitur 2026 yang harus ditandingi.

## Konsepnya

![Generasi musik: token-LM vs diffusion, peta model 2026](../assets/music-generation.svg)

### Token LM di atas token neural-codec

**MusicGen** Meta (2023, MIT) dan banyak turunannya: memberi kondisi pada embedding teks/melodi, memprediksi token EnCodec secara autoregresif (32 kHz, 4 codebook), mendekode dengan EnCodec. 300M - 3,3B parameter. Baseline yang kuat; kesulitan melewati 30 detik.

**ACE-Step** (open-source, 4B XL dirilis April 2026) mengembangkan ini untuk generasi lagu penuh berbasis lirik. Hal terdekat dari komunitas terbuka terhadap Suno.

### Diffusion di atas mels atau latents

**Stable Audio (2023)** dan **Stable Audio Open (2024)**: latent diffusion pada audio terkompresi. Unggul untuk loop, sound design, tekstur ambient. Tidak hebat untuk lagu penuh yang terstruktur.

**AudioLDM / AudioLDM2**: text-to-audio lewat latent diffusion gaya T2I, digeneralisasi ke musik, efek suara, dan bicara.

### Hibrida (produksi) — Suno, Udio, Lyria

Bobot tertutup. Kemungkinan besar AR codec LM + vocoder berbasis diffusion dengan head vokal / drum / melodi khusus. Suno v5 (2026) adalah pemimpin kualitas dengan ELO 1293. Udio v4 menambahkan inpainting + pemisahan stem (bass, drum, vokal diunduh terpisah).

### Evaluasi

- **FAD (Fréchet Audio Distance).** Jarak level-embedding antara distribusi audio hasil vs nyata memakai fitur VGGish atau PANNs. Makin rendah makin baik. MusicGen small: 4,5 FAD di MusicCaps; SOTA ~3,0.
- **Musicality (subjektif).** Preferensi manusia. ELO Suno v5 1293 memimpin.
- **Alignment teks-audio.** Skor CLAP antara prompt dan output.
- **Artefak musicality.** Transisi di luar ketukan, drift frasa vokal, hilangnya struktur setelah 30 detik.

## Peta model 2026

| Model | Params | Length | Vocals | License |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | no | MIT |
| Stable Audio Open | 1.2B | 47 s | no | Stability non-commercial |
| ACE-Step XL (Apr 2026) | 4B | &gt; 2 min | yes | Apache-2.0 |
| YuE | 7B | &gt; 2 min | yes, multilingual | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | yes, ELO 1293 | commercial |
| Udio v4 (closed) | ? | 4 min | yes + stems | commercial |
| Google Lyria 3 (closed) | ? | real-time | yes | commercial |
| MiniMax Music 2.5 | ? | 4 min | yes | commercial API |

## Lanskap hukum (2025-2026)

- **Setelmen Warner Music vs Suno.** $500M. WMG kini punya pengawasan atas kemiripan-AI, hak musik, dan trek buatan pengguna di Suno. Setelmen serupa UMG di Udio.
- **EU AI Act** + **California SB 942**: musik yang dihasilkan AI wajib diungkapkan.
- **Riffusion / MusicGen** di bawah MIT tidak punya beban kepatuhan tetapi juga tidak punya vokal komersial.

Pola aman-dirilis:

1. Hanya menghasilkan instrumental (MusicGen, Stable Audio Open, output MIT/CC0).
2. Memakai API komersial (Suno, Udio, ElevenLabs Music) dengan lisensi per-generasi.
3. Melatih pada katalog yang dimiliki atau dilisensikan (kebanyakan perusahaan berakhir di sini).
4. Menandai generasi dengan watermark + metadata.

```figure
sp-codec-tokens
```

## Bangun Sendiri

### Langkah 1: generate dengan MusicGen

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

Tiga ukuran: `small` (300M, cepat), `medium` (1,5B), `large` (3,3B). Small sudah cukup untuk "apakah idenya sampai."

### Langkah 2: kondisi melodi

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody menerima chromagram dan mempertahankan lagu sambil menukar timbre. Berguna untuk "berikan melodiku ini sebagai kuartet gesek."

### Langkah 3: evaluasi FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

Menghitung jarak embedding VGGish. Berguna untuk regression test tingkat genre; bukan pengganti pendengar manusia.

### Langkah 4: menambahkan ke alur kerja LLM-musik

Gabungkan dengan ide dari Pelajaran 7-8:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## Gunakan

| Goal | Stack |
|------|-------|
| Instrumental sound design | Stable Audio Open |
| Game / adaptive music | Google Lyria RealTime (closed) |
| Full songs with vocals (commercial) | Suno v5 or Udio v4 with explicit license |
| Full songs with vocals (open) | ACE-Step XL or YuE |
| Short ad jingle | MusicGen melody-conditioned on a hummed reference |
| Music-video background | MusicGen + Stable Video Diffusion |

## Jebakan yang masih rilis di 2026

- **Prompt pencucian hak cipta.** "Lagu bergaya Taylor Swift" — Suno/Udio komersial kini menyaring ini, model terbuka tidak. Tambahkan daftar filtermu sendiri.
- **Pengulangan / drift setelah 30 detik.** Model AR berputar-putar. Lakukan crossfade pada beberapa generasi, atau pakai ACE-Step untuk koherensi struktural.
- **Drift tempo.** Model menyimpang dari BPM-nya. Pakai tag BPM di prompt dan saring pascaproses dengan `beat_track` milik librosa.
- **Inteligibilitas vokal.** Suno sangat bagus; model terbuka sering tidak jelas pada kata-katanya. Jika lirik penting, pakai API komersial atau lakukan fine-tuning.
- **Output mono.** Model terbukan menghasilkan mono atau stereo palsu. Tingkatkan dengan rekonstruksi stereo yang benar (ezst, stereo diffusion milik Cartesia).

## Rilis

Simpan sebagai `outputs/skill-music-designer.md`. Pilih model, strategi lisensi, rencana panjang / struktur, dan metadata pengungkapan untuk deployment music-gen.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ia menghasilkan progresi akor + pola drum "generatif" sebagai simbol ASCII — kartun music-gen. Putar lewat renderer MIDI mana pun jika kamu mau.
2. **Sedang.** Instal `audiocraft`, generate cuplikan 10 detik di empat prompt genre dengan MusicGen-small, ukur FAD terhadap set genre referensi.
3. **Sulit.** Dengan ACE-Step (atau MusicGen-melody), generate tiga variasi lagu yang sama dengan prompt timbre berbeda. Hitung kemiripan CLAP ke prompt untuk memverifikasi alignment.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| FAD | Audio FID | Fréchet distance between embedding distributions of real vs generated. |
| Chromagram | Melody as pitches | 12-dim per-frame vector; input to melody conditioning. |
| Stems | Instrument tracks | Separated bass / drums / vocals / melody as WAV. |
| Inpainting | Regen a section | Mask a time window; model regenerates just that. |
| CLAP | Text-audio CLIP | Contrastive audio-text embedding; eval text-audio alignment. |
| EnCodec | Music codec | Meta's neural codec used by MusicGen; 32 kHz, 4 codebooks. |

## Bacaan Lanjutan

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — the open autoregressive benchmark.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — the sound-design default.
- [ACE-Step](https://github.com/ace-step/ACE-Step) — open 4B full-song generator, April 2026.
- [Suno v5 platform docs](https://suno.com) — the commercial quality leader.
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — latent diffusion for music + sound effects.
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/warner-music-group-settles-with-suno-strikes-first-of-its-kind-deal-with-ai-song-generator/) — Nov 2025 precedent.
