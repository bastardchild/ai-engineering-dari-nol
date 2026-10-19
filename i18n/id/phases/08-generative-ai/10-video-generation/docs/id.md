# Pembangkitan Video

> Citra adalah tensor 2-D. Video adalah 3-D. Teorinya sama; komputasinya 10-100x lebih sulit. Sora milik OpenAI (Feb 2024) membuktikan itu mungkin. Pada 2026 Veo 2, Kling 1.5, Runway Gen-3, Pika 2.0, dan WAN 2.2 mengirim video produksi dari teks pada 1080p — dan stack open-weight (CogVideoX, HunyuanVideo, Mochi-1, WAN 2.2) tertinggal 12 bulan.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## Masalahnya

Video 10 detik 1080p pada 24fps adalah 240 frame berisi 1920×1080×3 piksel. Itu ~1.5 GB data mentah per klip. Diffusion ruang-piksel tidak layak. Kamu butuh:

1. **Kompresi spatiotemporal.** VAE yang mengenkode video, bukan frame, menjadi sequence patch spasial-temporal.
2. **Koherensi temporal.** Frame perlu berbagi konten, pencahayaan, dan identitas objek selama hitungan detik. Net harus memodelkan gerak.
3. **Budget komputasi.** Training video 10-100x lebih mahal daripada gambar untuk ukuran model yang sama.
4. **Conditioning.** Teks, gambar (frame-pertama), audio, atau video lain. Sebagian besar model produksi menerima keempatnya.

Arsitektur yang memecahkan ini adalah **Diffusion Transformer (DiT)** yang diterapkan ke patch spatiotemporal, dilatih pada dataset (prompt, caption, video) raksasa. Loss diffusion yang sama dengan Pelajaran 06.

## Konsepnya

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### Patchify

Enkode video dengan VAE 3D (kompresi spatiotemporal terpelajari). Laten ber-shape `[T_latent, H_latent, W_latent, C_latent]`. Bagi menjadi patch berukuran `[t_p, h_p, w_p]`. Untuk model ala-Sora, `t_p = 1` (patch per-frame) atau `t_p = 2` (setiap dua frame). Video 10 detik 1080p terkompresi menjadi ~20.000-100.000 patch.

### DiT Spatiotemporal

Transformer memproses sequence datar patch. Setiap patch punya embedding posisional 3D (waktu + y + x). Attention biasanya difaktorisasi:

- **Attention spasial** dalam patch setiap frame.
- **Attention temporal** lintas frame pada lokasi spasial yang sama.
- **Attention 3D penuh** 16-100x lebih mahal; dipakai hanya pada resolusi rendah atau dalam riset.

### Conditioning teks

Cross-attention dengan encoder teks besar (T5-XXL untuk Sora, CogVideoX-5B memakai T5-XXL). Prompt panjang penting — set training Sora punya re-caption padat buatan-GPT rata-rata 200 token per klip.

### Training

Loss diffusion standar (prediksi ε atau v) atas laten spatiotemporal. Data: video web + ~100M klip terkurasi + caption teks sintetis. Komputasi: 10.000+ jam GPU bahkan untuk run riset kecil; skala-Sora 100.000+.

## Lanskap produksi 2026

| Model | Date | Max duration | Max res | Open weights? | Notable |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | No | First model to show world simulator properties at scale |
| Sora Turbo | 2024-12 | 20s | 1080p | No | Production Sora at 5x faster inference |
| Veo 2 (Google) | 2024-12 | 8s | 4K | No | Highest quality + physics in 2025 |
| Veo 3 | 2025 Q3 | 15s | 4K | No | Native audio and stronger camera control |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | No | Best human motion in 2025 Q1 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | No | Professional video tools on top |
| Pika 2.0 | 2024-10 | 5s | 1080p | No | Strongest character consistency |
| CogVideoX (THUDM) | 2024 | 10s | 720p | Yes (2B, 5B) | First open 5B-scale video |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | Yes (13B) | Open SOTA late 2024 |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | Yes (10B) | Most permissively licensed |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | Yes | Strongest open model mid-2025 |

Open weight menutup kesenjangan lebih cepat daripada di ruang gambar: LoRA HunyuanVideo + WAN 2.2 sudah menggerakkan sebagian besar workflow open-source pada pertengahan 2026.

```figure
video-diffusion-denoise
```

## Bangun Sendiri

`code/main.py` mensimulasikan ide DiT spatiotemporal inti: patchify video sintetis kecil, tambahkan embedding posisi per-patch, dan denoise seluruh sequence dengan attention ala-transformer atas patch. Tanpa numpy; Python murni. Kita menunjukkan koherensi temporal muncul bahkan dalam 1-D ketika patch frame-bersebelahan berbagi denoiser dan embedding posisi.

### Langkah 1: patchify "video" 1-D sintetis

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Langkah 2: embedding posisi per frame

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Langkah 3: denoiser melihat seluruh sequence

Alih-alih mendenoise setiap frame independen, net mungil kita menggabungkan semua nilai frame + embedding posisinya dan memprediksi noise untuk semua frame bersama.

### Langkah 4: uji koherensi temporal

Setelah training, sampel video. Ukur delta frame-ke-frame. Kalau model telah mempelajari struktur temporal, delta tetap lebih kecil daripada menyampel setiap frame independen.

## Jebakan

- **Sampling per-frame independen = flicker.** Kalau kamu menjalankan diffusion gambar pada setiap frame terpisah, output flicker karena noise setiap frame independen. Diffusion video memperbaikinya dengan menggandeng frame melalui attention atau noise bersama.
- **Attention 3D naif = OOM.** Attention 3D penuh pada laten 1080p 10 detik adalah ratusan miliar operasi. Faktorisasi menjadi spasial + temporal.
- **Captioning data lebih penting daripada ukuran.** Upgrade utama Sora atas karya sebelumnya adalah training pada caption ~10x lebih detail (klip dilabel-ulang GPT-4). Laporan teknis OpenAI eksplisit soal ini.
- **Conditioning frame-pertama.** Sebagian besar model produksi juga menerima gambar sebagai frame pertama. Ini mode "image-to-video"; training menyertakan varian ini.
- **Drift fisika.** Klip panjang (>10 dtk) mengakumulasi inkonsistensi halus. Pembangkitan sliding-window + penjangkaran keyframe membantu.

## Gunakan

| Use case | 2026 pick |
|----------|-----------|
| Highest-quality text-to-video, hosted | Veo 3 or Sora |
| Camera-controlled cinematic | Runway Gen-3 with motion brushes |
| Character consistency across clips | Pika 2.0 or Kling 2.1 |
| Open weights, fast fine-tune | WAN 2.2 + LoRA |
| Image-to-video | WAN 2.2-I2V, Kling 2.1 I2V, or Runway |
| Audio-to-video lip sync | Veo 3 (native audio) or a dedicated lip-sync model |
| Video editing | Runway Act-Two, Kling Motion Brush, Flux-Kontext (still-frame) |

Biaya per detik video pada paritas kualitas turun 20x antara 2024 dan 2026.

## Rilis

Simpan `outputs/skill-video-brief.md`. Skill menerima brief video (durasi, rasio aspek, style, rencana kamera, konsistensi subjek, audio) dan mengeluarkan: model + hosting, scaffolding prompt (bahasa kamera, deskripsi subjek, deskriptor gerak), protokol seed + reproduksibilitas, dan checklist QA level-frame.

## Latihan

1. **Mudah.** Dalam `code/main.py`, bandingkan delta frame-ke-frame untuk (a) sampling per-frame independen, (b) sampling sequence bersama. Laporkan rata-rata dan variansi delta.
2. **Sedang.** Tambahkan condition frame-pertama: pin frame 0 ke nilai tertentu dan sampel sisanya. Ukur bagaimana nilai ter-pin merambat.
3. **Sulit.** Pakai HuggingFace diffusers untuk menjalankan CogVideoX-2B pada GPU lokal. Ukur waktu 20 langkah inference pada 720p untuk klip 6 detik. Profil attention spatiotemporal untuk mengidentifikasi bottleneck.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Video VAE | "3-D VAE" | Encoder that compresses `(T, H, W, C)` → spatiotemporal latent. |
| Patches | "The tokens" | Fixed-size 3-D blocks of the latent; input to the DiT. |
| Factorized attention | "Spatial + temporal" | Run attention over space, then over time; skip full 3-D attention. |
| Image-to-video (I2V) | "Animate this photo" | Model takes an image + text, outputs a video that starts from it. |
| Keyframe conditioning | "Anchor frames" | Pin specific frames to control the video's arc. |
| Motion brush | "Directional hint" | UI input where the user paints motion vectors onto the image. |
| Re-captioning | "Dense captions" | Using an LLM to re-label training clips with detailed prompts. |
| Flicker | "Temporal artifact" | Frame-to-frame inconsistency; fixed with coupled denoising. |

## Catatan produksi: laten video adalah problem memori-bandwidth

Klip 1080p 10 detik pada 24 fps adalah 240 frame × 1920 × 1080 × 3 ≈ 1.5 GB piksel mentah. Setelah kompresi VAE video 4× (`2 × spasial × 2 × temporal`) latennya ~100 MB per request. Jalankan ini melalui DiT spatiotemporal selama 30 langkah pada batch 1 dan kamu memindahkan ~3 GB/langkah melalui HBM — bandwidth memori, bukan FLOPs, adalah bottleneck.

Tiga knob produksi, semuanya langsung dari bab inference literatur production-inference:

- **TP melintasi DiT.** Model teks-ke-video rutin ≥10B param. TP=4 melintasi 4 H100 standar; PP=2 × TP=2 untuk model kelas-405B. Latensi per langkah turun kira-kira linear dengan TP sampai tembok all-reduce.
- **Batching frame = continuous batching.** Saat pembangkitan, video secara konseptual adalah batch frame yang ditautkan attention. Continuous batching (scheduling in-flight) berlaku: mulai render frame `t+1` sementara frame `t-1` dikembalikan, kalau arsitektur model memungkinkan pembangkitan sliding-window.
- **Cache prefill level-klip.** Untuk image-to-video, conditioning frame-pertama analog dengan prefill prompt LLM: hitung sekali, pakai ulang melintasi pass decoder temporal. Ini efektif KV-cache untuk video.

## Bacaan Lanjutan

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — laporan teknis Sora.
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX.
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo.
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1.
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — SOTA open pertengahan-2025.
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — makalah diffusion video seminal.
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — leluhur Stable Video Diffusion.
