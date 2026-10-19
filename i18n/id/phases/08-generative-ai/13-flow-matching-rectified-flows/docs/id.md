# Flow Matching & Rectified Flow

> Model diffusion butuh 20-50 langkah sampling karena mereka menyusuri jalur melengkung dari noise ke data. Flow matching (Lipman et al., 2023) dan rectified flow (Liu et al., 2022) melatih jalur lurus. Jalur lebih lurus berarti langkah lebih sedikit berarti inference lebih cepat. Stable Diffusion 3, Flux.1, dan AudioCraft 2 semuanya beralih ke flow matching pada 2024.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 06 (DDPM), Phase 1 · Calculus
**Time:** ~45 minutes

## Masalahnya

Proses reverse DDPM adalah jalan acak stokastik 1000 langkah dari `N(0, I)` kembali ke distribusi data. DDIM meruntuhkannya menjadi 20-50 langkah deterministik. Kamu mau langkah lebih sedikit — idealnya satu. Penghalangnya adalah ODE yang memecahkan proses reverse itu stiff; jalurnya melengkung.

Kalau kamu bisa melatih model sehingga jalur dari noise ke data adalah *garis lurus*, satu langkah Euler dari `t=1` ke `t=0` akan bekerja. Flow matching membangun ini langsung: definisikan interpolasi garis-lurus dari `x_1 ∼ N(0, I)` ke `x_0 ∼ data`, latih medan vektor `v_θ(x, t)` agar cocok dengan turunan waktunya, integralkan saat inference.

Rectified flow (Liu 2022) melangkah lebih jauh: luruskan iteratif jalur dengan prosedur reflow yang menghasilkan ODE yang progresif lebih-dekat-ke-linear. Setelah dua iterasi reflow, sampler 2 langkah menyamai kualitas DDPM 50 langkah.

## Konsepnya

![Flow matching: straight-line interpolation between noise and data](../assets/flow-matching.svg)

### Aliran garis-lurus

Definisikan:

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

dengan `x_0 ~ data` dan `x_1 ~ N(0, I)`. Turunan waktu sepanjang garis lurus ini konstan:

```
dx_t / dt = x_1 - x_0
```

Definisikan medan vektor neural `v_θ(x_t, t)` dan latih agar cocok dengan turunan ini:

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

Inilah loss **conditional flow matching** (Lipman 2023). Training bebas-simulasi: kamu tidak pernah meng-unroll ODE. Cukup sampel `(x_0, x_1, t)` dan regresi.

### Sampling

Saat inference, integralkan medan vektor terpelajari *mundur* dalam waktu:

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

Mulai pada `x_1 ~ N(0, I)`, langkah-Euler turun ke `t=0`.

### Rectified flow (Liu 2022)

Aliran garis-lurus bekerja tapi jalur terpelajari *tidak benar-benar lurus* — mereka melengkung karena banyak `x_0` bisa memetakan ke `x_1` yang sama. Langkah reflow milik rectified flow:

1. Latih model flow v_1 dengan pairing acak.
2. Sampel N pasangan `(x_1, x_0)` dengan mengintegrasikan v_1 dari `x_1` ke pendaratan `x_0`-nya.
3. Latih v_2 pada contoh berpasangan itu. Karena pasangan kini "ODE-matched", interpolan garis-lurus di antaranya sungguh lebih datar.
4. Ulangi.

Dalam praktik 2 iterasi reflow membawamu ke nyaris-linear, memungkinkan inference 2-4 langkah. SDXL-Turbo, SD3-Turbo, LCM semuanya model distilled-dari-flow-matching.

### Mengapa ini menang untuk gambar pada 2024

Tiga alasan:

1. **Training bebas-simulasi** — tanpa unrolling ODE saat training, trivial diimplementasikan.
2. **Geometri loss lebih baik** — jalur lurus punya signal-to-noise konsisten, sedangkan loss-ε DDPM punya SNR buruk di tepi schedule.
3. **Inference lebih cepat** — 4-8 langkah pada kualitas SDXL-Turbo; 1 langkah dengan distilasi consistency.

## Flow matching vs DDPM — koneksi eksak

Flow matching dengan jalur kondisional-Gaussian adalah diffusion *dengan schedule noise spesifik*. Pilih schedule `x_t = α(t) x_0 + σ(t) x_1` dan flow matching memulihkan diffusion yang dirumuskan-ulang-Stratonovich dengan `v = α'·x_0 - σ'·x_1`. Keduanya ekuivalen secara aljabar untuk jalur Gaussian.

Apa yang ditambahkan flow matching: *kejelasan* target (velositas polos), loss yang lebih bersih, dan lisensi untuk bereksperimen dengan interpolan non-Gaussian.

```figure
normalizing-flow
```

## Bangun Sendiri

`code/main.py` mengimplementasikan flow matching 1-D pada campuran Gaussian dua-mode. Medan vektor `v_θ(x, t)` adalah MLP mungil yang dilatih dengan target garis-lurus. Saat inference, integralkan 1, 2, 4, dan 20 langkah Euler dan bandingkan kualitas sampel.

### Langkah 1: loss training

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Langkah 2: inference multi-langkah

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Langkah 3: bandingkan jumlah langkah

Harapkan sampler 4 langkah sudah menyamai kualitas 20 langkah — masalah besar untuk latensi.

## Jebakan

- **Parametrisasi waktu.** Flow matching memakai `t ∈ [0, 1]` dengan `t=0` di data, `t=1` di noise. DDPM memakai `t ∈ [0, T]` dengan `t=0` di data, `t=T` di noise. Arah sama, skala berbeda. Makalah salah soal ini terus-menerus.
- **Pilihan schedule.** Garis lurus rectified flow adalah schedule flow-matching "itu", tapi kamu bisa memakai t-sampling cosine atau logit-normal (SD3 melakukan ini) untuk cakupan skala lebih baik.
- **Biaya reflow.** Membangkitkan dataset berpasangan untuk reflow adalah satu pass inference penuh per sampel. Lakukan reflow hanya ketika kamu benar-benar butuh inference 1-2 langkah.
- **Classifier-free guidance tetap berlaku.** Cukup tukar ε dengan v dalam kombinasi linear: `v_cfg = (1+w) v_cond - w v_uncond`.

## Gunakan

| Use case | 2026 stack |
|----------|-----------|
| Text-to-image, best quality | Flow matching: SD3, Flux.1-dev |
| Text-to-image, 1-4 steps | Distilled flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| Real-time inference | Consistency distillation from a flow-matched base (LCM, PCM) |
| Audio generation | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| Video generation | Flow matching mixed with diffusion (Sora, Veo, Stable Video) |
| Science / physics (particle trajectories, molecules) | Flow matching + equivariant vector field |

Kapan pun makalah berkata "lebih cepat daripada diffusion" pada 2025-2026, hampir selalu flow matching + distilasi.

## Rilis

Simpan `outputs/skill-fm-tuner.md`. Skill menerima spek model ala-diffusion dan mengonversinya menjadi config training flow-matching: pilihan schedule, distribusi sampling waktu (seragam / logit-normal), optimizer, rencana reflow, jumlah langkah target, protokol eval.

## Latihan

1. **Mudah.** Jalankan `code/main.py` dan bandingkan MSE 1-langkah vs 20-langkah terhadap distribusi data sejati.
2. **Sedang.** Beralih dari sampling `t` seragam ke logit-normal (mengonsentrasikan sampling di mid-t). Apakah kualitas model membaik?
3. **Sulit.** Implementasikan satu iterasi reflow: bangkitkan pasangan (x_0, x_1) dengan mengintegrasikan model pertama, latih model kedua pada pasangan itu, dan bandingkan kualitas sampel 1-langkah.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Flow matching | "Straight-line diffusion" | Train `v_θ(x, t)` to match `x_1 - x_0` along an interpolant. |
| Rectified flow | "Reflow" | Iterative procedure that straightens learned flows. |
| Velocity field | "v_θ" | Output of the model — the direction to move `x_t`. |
| Straight-line interpolant | "The path" | `x_t = (1-t)·x_0 + t·x_1`; trivial target derivative. |
| Euler sampler | "1st order ODE solver" | Simplest integrator; works well when paths are straight. |
| Logit-normal t | "SD3 sampling" | Concentrate `t` sampling toward mid-values where gradients are strongest. |
| Consistency distillation | "1-step sampler" | Train a student to map any `x_t` directly to `x_0`. |
| CFG with velocity | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`; same trick, new variable. |

## Catatan produksi: Flux.1-schnell adalah flow matching pada titik tercepat

Kemenangan produksi flow matching adalah Flux.1-schnell — DiT flow-matched yang didistilasi menjadi 1-4 langkah inference sambil menjaga kualitas kelas-Flux-dev. Notebook "Run Flux on an 8GB machine" milik Niels adalah resep deployment referensi: enkode T5 + CLIP, denoise MMDiT terkuantisasi (dalam 4 langkah untuk schnell vs 50 untuk dev), decode VAE. Akuntansi biayanya:

| Variant | Steps | Latency at 1024² on L4 | Total FLOPs (relative) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (raw) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (12× faster) |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

Aturan produksi: **basis flow-matched + distilasi = bawaan 2026 untuk teks-ke-gambar cepat.** Setiap vendor besar mengirim kombo ini: SD3-Turbo (SD3 + flow + distilasi), Flux-schnell (Flux-dev + pelurusan rectified-flow), CogView-4-Flash. Basis diffusion murni ada hanya untuk checkpoint legacy.

## Bacaan Lanjutan

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, rectified flow pada skala besar.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — kerangka umum yang mencakup FM + diffusion.
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — distilasi 1-langkah diffusion / flow.
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — varian turbo.
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — flow matching di produksi.
