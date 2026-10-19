# Model Video-Language: Token Temporal dan Grounding

> Video bukan tumpukan foto. Klip 5 detik punya urutan kausal, kata kerja aksi, dan timing peristiwa yang tidak bisa direpresentasikan oleh model gambar. Video-LLaMA (Zhang dkk., Juni 2023) merilis video-LLM open pertama dengan grounding audio-visual. VideoChat dan Video-LLaVA menskalakan pola itu. Pada 2025 TMRoPE milik Qwen2.5-VL menutup jarak dengan model proprietary frontier. Setiap sistem memecahkan token temporal dengan cara berbeda — Q-former per klip, concat-pool per frame, TMRoPE per token. Pelajaran ini membaca pola-pola itu, membangun sampler frame uniform-vs-dynamic, dan mengevaluasi pada tugas temporal grounding.

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Menjelaskan mengapa positional encoding temporal mengubah performa video VLM secara independen dari encoder visi.
- Membandingkan sampling frame uniform, dynamic-FPS, dan event-driven dari sisi token-per-detik vs akurasi grounding.
- Mendeskripsikan desain Q-former-per-klip (Video-LLaMA) vs pooled-per-frame (Video-LLaVA) vs M-RoPE-per-token (Qwen2.5-VL).
- Menyebut empat benchmark video: VideoMME, TempCompass, EgoSchema, Video-MMMU.

## Masalahnya

Video 1 menit pada 30 FPS berjumlah 1800 frame. Pada 196 token visual per frame (ViT-B di 224), itu 352k token — lebih besar daripada context LLM era-2024 mana pun.

Ada tiga strategi reduksi:

1. Subsample frame (1-8 FPS tergantung konten).
2. Pool patch token setiap frame secara agresif (pool bilinear 3x3 atau 4x4).
3. Kompres via Q-former yang mengambil klip 16 frame dan mengeluarkan 64 token.

Setiap trade-off berbeda. Subsampling kehilangan detail temporal. Pooling kehilangan detail spasial. Q-former kehilangan sedikit keduanya tapi menghemat token.

Positional encoding temporal adalah sumbu lainnya: bagaimana model tahu frame 5 datang sebelum frame 6? Opsi termasuk temporal RoPE 1D sederhana (Video-LLaMA), embedding temporal yang dipelajari (Video-LLaVA), dan TMRoPE (Qwen2.5-VL, penuh 3D).

## Konsepnya

### Video-LLaMA: Q-former per klip + cabang audio

Video-LLaMA (2023) adalah video-LLM open pertama. Arsitektur:

- Klip 16 frame pada 2 FPS (jadi 8 detik).
- Fitur ViT per-frame -> Video Q-former yang cross-attend ke seluruh 16 frame -> 32 query yang dipelajari -> LLM.
- Cabang audio paralel: gelombang -> encoder audio ImageBind -> Audio Q-former -> 32 query -> LLM.

Kekuatan: penalaran joint audio-visual. Kelemahan: panjang klip tetap, tanpa grounding waktu sembarang.

### VideoChat dan Video-LLaVA

VideoChat mempertahankan ide Video-LLaMA tapi membuang audio dan menyederhanakannya. Video-LLaVA (Lin dkk., 2023) melatih satu encoder visi pada gambar dan frame video ("alignment before projection"), memberikan representasi terpadu. Keduanya berbasis encoder-CLIP-dibekukan + MLP + LLM.

Tidak satu pun menangani video panjang. Keduanya sistem 8-16 frame.

### Qwen2.5-VL dan TMRoPE

Qwen2.5-VL memperkenalkan TMRoPE — Temporal-Modality Rotary Position Embedding. Setiap token patch membawa posisi (t, h, w) di mana t adalah timestamp sebenarnya (bukan indeks frame).

Perbedaan utama dari embedding temporal sederhana:

- Waktu absolut, bukan indeks. Model melihat "pada 4.2 detik" bukan "pada frame 15."
- Rotasi per-token, bukan per-klip. Setiap token visual berputar secara independen menurut timestamp-nya.
- Kompatibel dengan dynamic FPS. Jika kamu sampling pada 2 FPS di sini dan 4 FPS di sana, TMRoPE menangani jarak yang timpang secara native.

TMRoPE memungkinkan query "pada detik ke berapa kucing melompat?". Model dapat menghasilkan "pada 4.2 detik." Video-LLaMA hanya bisa berkata "di awal klip."

### Strategi sampling frame

Uniform: sample N frame secara merata sepanjang durasi. Sederhana, kehilangan puncak gerakan.

Dynamic FPS: sample secara adaptif berdasarkan intensitas gerakan. Optical flow atau frame differencing memilih segmen gerakan-tinggi untuk sampling lebih dens. Qwen2.5-VL melatih dengan ini.

Event-driven: jalankan detektor ringan, sample lebih banyak di tempat aksi terjadi. Dipakai oleh VideoAgent.

Keyframe + context: sample di batas shot + beberapa frame sebelahnya. Dipakai untuk konten sinematik.

### Pooling per frame

Pada 1 FPS dan 576 token per frame, klip 5 menit berjumlah 172,800 token. Bisa ditangani context 128k Qwen2.5-VL-72B tapi mahal.

Pool bilinear 3x3 mereduksi ke 64 token per frame -> 19,200 token untuk 5 menit. Sweet spot untuk kebanyakan tugas.

Pool lebih agresif (6x6 -> 16 token per frame) untuk alur kerja agen di mana detail spasial kurang penting.

### Empat benchmark video

- VideoMME: pemahaman video komprehensif, pendek + menengah + panjang.
- TempCompass: penalaran temporal halus-halus, pertanyaan "sebelum" / "sesudah".
- EgoSchema: video orang-pertama horizon-panjang.
- Video-MMMU: pertanyaan video multi-disiplin multimodal.

Evaluasi video-VLM yang lengkap memukul keempatnya. Mereka menekan sumbu berbeda — TempCompass seluruhnya tentang urutan, EgoSchema tentang penalaran 3+ menit, VideoMME membentang durasi.

### Format output grounding

Format output untuk temporal grounding:

- Teks bebas: "Kucing melompat di sekitar tanda 4 detik." Mudah diparse tapi tidak presisi.
- JSON terstruktur: `{"event": "jump", "start": 4.1, "end": 4.3}`. Qwen2.5-VL melatih ini.
- Berbasis token: token khusus `<time>4.1</time>` yang selang-seling dengan jawaban. Format internal milik Qwen2.5-VL.

Berbasis token paling akurat untuk penggunaan hilir. Format output JSON milik Qwen2.5-VL dapat diparse langsung.

### Best practice 2026

Untuk video VLM pada 2026:

- Encoder: SigLIP 2 dengan M-RoPE atau TMRoPE (Qwen2.5-VL).
- Sampling frame: dynamic FPS (1-4 tergantung gerakan) dengan batas frame-maksimum.
- Pooling per-frame: bilinear 3x3.
- Output: JSON terstruktur dengan field waktu + peristiwa.
- Benchmark: VideoMME + TempCompass untuk umum; EgoSchema untuk horizon-panjang.

```figure
video-temporal-patches
```

## Gunakan

`code/main.py` memuat:

- Sampler frame uniform dan dynamic-FPS.
- Evaluator temporal-grounding mainan: diberi peristiwa "ground truth" pada waktu T dan output model, skor akurasi dengan toleransi.
- Perbandingan antara Video-LLaMA (16 frame, Q-former), Video-LLaVA (8 frame, MLP), Qwen2.5-VL (dynamic FPS + TMRoPE).

## Rilis

Pelajaran ini menghasilkan `outputs/skill-video-vlm-frame-planner.md`. Diberi tugas video (pengawasan, pengenalan aksi, temporal grounding, ringkasan), ia memilih sampler frame, faktor pooling, format output, dan tier akurasi yang diharapkan.

## Latihan

1. Untuk demo memasak 3 menit, pilih uniform vs dynamic FPS. Justifikasi dengan jumlah token.

2. TMRoPE menambahkan apa secara spesifik yang tidak bisa dilakukan tabel embedding temporal sederhana?

3. Tulis schema JSON untuk temporal grounding yang dapat dipelajari VLM untuk dipancarkan. Sertakan kasus kesalahan.

4. Baca Bagian 3 Video-LLaVA tentang "Alignment Before Projection." Mengapa ini lebih baik daripada melatih encoder gambar dan video terpisah?

5. Mengingat leaderboard VideoMME, berapa jarak antara model open teratas dan model proprietary teratas per 2026? Berapa banyak dari jarak itu dapat dikaitkan pada encoding temporal vs skala basis LLM?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Temporal grounding | "Time-localized answers" | VLM outputs a specific timestamp range for when an event happens |
| TMRoPE | "Time-Multimodal RoPE" | 3D rotary position with absolute timestamps, used by Qwen2.5-VL |
| Dynamic FPS | "Motion-aware sampling" | Sample more frames in high-motion segments, fewer in static ones |
| Frame pooling | "Spatial compress per frame" | Reduce patches per frame with bilinear interpolation before the LLM |
| Video Q-former | "Clip compressor" | Cross-attention bottleneck mapping N frames to K learned queries |
| VideoMME | "Video bench" | Comprehensive short/medium/long video benchmark, 2500+ samples |

## Bacaan Lanjutan

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
