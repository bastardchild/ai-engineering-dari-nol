# Autoencoder & Variational Autoencoder (VAE)

> Autoencoder polos mengompresi lalu merekonstruksi. Ia menghafal. Ia tidak membangkitkan. Tambahkan satu trik — paksa kodenya tampak Gaussian — dan kamu dapat sampler. Trik tunggal itu, reparametrisasi `z = μ + σ·ε`, adalah mengapa setiap model gambar latent-diffusion dan flow-matching yang kamu pakai pada 2026 punya VAE di input.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Time:** ~75 minutes

## Masalahnya

Kompresi digit MNIST 784 piksel menjadi kode 16 angka, lalu rekonstruksi. Autoencoder polos akan menguasai MSE rekonstruksi tapi ruang kodenya berantakan bergelombang. Pilih titik acak di ruang kode, decode, dan kamu dapat noise. Ia tidak punya sampler. Ia adalah model kompresi yang menyamar.

Yang sebenarnya kamu mau adalah: (a) ruang kodenya distribusi bersih dan mulus yang bisa kamu sampel — katakan Gaussian isotropik `N(0, I)`, (b) mendecode sampel apa pun menghasilkan digit yang masuk akal, dan (c) encoder dan decoder tetap mengompresi dengan baik. Tiga tujuan, satu arsitektur, satu loss.

VAE Kingma 2013 memecahkan ini dengan melatih encoder untuk mengeluarkan *distribusi* `q(z|x) = N(μ(x), σ(x)²)`, menarik distribusi itu ke prior `N(0, I)` via penalti KL, lalu menyampel `z` dari `q(z|x)` sebelum decoding. Saat inference, buang encoder, sampel `z ~ N(0, I)`, decode. Penalti KL inilah yang memaksa ruang kode menjadi terstruktur.

Pada 2026 VAE jarang dikirim mandiri — mereka sudah dikalahkan diffusion untuk kualitas gambar mentah — tapi mereka adalah encoder pilihan untuk setiap model latent-diffusion (SD 1/2/XL/3, Flux, AudioCraft). Pelajari VAE dan kamu mempelajari lapisan pertama yang tak terlihat dari setiap pipeline gambar yang kamu pakai.

## Konsepnya

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**Autoencoder.** `z = encoder(x)`, `x̂ = decoder(z)`, loss = `||x - x̂||²`. Ruang kode tak terstruktur.

**Encoder VAE.** Mengeluarkan dua vektor: `μ(x)` dan `log σ²(x)`. Keduanya mendefinisikan `q(z|x) = N(μ, diag(σ²))`.

**Trik reparametrisasi.** Menyampel dari `q(z|x)` tidak terdiferensiasi. Tulis ulang sampel sebagai `z = μ + σ·ε` dengan `ε ~ N(0, I)`. Kini `z` adalah fungsi deterministik dari `(μ, σ)` plus noise non-parameter — gradien mengalir melalui `μ` dan `σ`.

**Loss.** Evidence Lower BOund (ELBO), dua suku:

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

Rekonstruksi mendorong `x̂` ke `x`. KL mendorong `q(z|x)` ke prior. Keduanya bertukar. β kecil (<1) = sampel lebih tajam, ruang kode kurang Gaussian. β besar (>1) = ruang kode lebih bersih, sampel lebih blur. β-VAE (Higgins 2017) mempopulerkan knob ini dan memicu riset disentanglement.

**Sampling.** Saat inference: ambil `z ~ N(0, I)`, forward melalui decoder. Satu forward pass — tanpa sampling iteratif seperti diffusion.

```figure
vae-latent-grid
```

## Bangun Sendiri

`code/main.py` mengimplementasikan VAE mungil tanpa numpy atau torch. Input adalah data sintetis 8 dimensi yang diambil dari campuran Gaussian 2 komponen dalam 8-D. Encoder dan decoder adalah MLP satu hidden-layer. Kita mengimplementasikan aktivasi tanh, forward pass, loss, dan backward pass tulisan-tangan. Bukan produksi — pedagogi.

### Langkah 1: forward encoder

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`log σ²` alih-alih `σ` supaya output network tak terbatas (softplus dari σ adalah jebakan — gradien mati pada σ ≈ 0).

### Langkah 2: reparametrisasi dan decode

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Langkah 3: ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

KL closed-form eksak karena kedua distribusi Gaussian. Jangan mengintegralkan secara numerik. Orang masih mengirim kode dengan estimasi KL monte-carlo pada 2026 — 3x lebih lambat tanpa alasan.

### Langkah 4: bangkitkan

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

Itulah model generatifnya. Lima baris.

## Jebakan

- **Posterior collapse.** Suku KL mendorong `q(z|x) → N(0, I)` begitu agresif sehingga `z` tidak membawa info soal `x`. Perbaiki: β-annealing (mulai β=0, naikkan ke 1), free bits, atau lewati KL pada dimensi tak aktif.
- **Sampel blur.** Likelihood decoder Gaussian mengimplikasikan rekonstruksi MSE, yang Bayes-optimal untuk L2 (rata-rata) — rata-rata dari sekumpulan digit yang masuk akal adalah digit kabur. Perbaiki: decoder diskret (VQ-VAE, NVAE), atau pakai VAE hanya sebagai encoder dan tumpuk diffusion pada latennya (inilah yang dilakukan Stable Diffusion).
- **β terlalu besar, terlalu dini.** Lihat posterior collapse. Mulai pada β≈0.01 dan naikkan.
- **Dim laten terlalu kecil.** 16-D bekerja untuk MNIST, 256-D untuk ImageNet 256², 2048-D untuk ImageNet 1024². VAE Stable Diffusion mengompresi 512×512×3 → 64×64×4 (faktor downsample 32x dalam luas spasial, 32x dalam channel).

## Gunakan

Stack VAE 2026:

| Situation | Pick |
|-----------|------|
| Image-latent encoder for diffusion | Stable Diffusion VAE (`sd-vae-ft-ema`) or Flux VAE |
| Audio-latent encoder | Encodec (Meta), SoundStream, or DAC (Descript) |
| Video latents | Sora's spatiotemporal patches, Latte VAE, WAN VAE |
| Disentangled representation learning | β-VAE, FactorVAE, TCVAE |
| Discrete latents (for transformer modelling) | VQ-VAE, RVQ (ResidualVQ) |
| Continuous latents for generation | Plain VAE, then condition a flow/diffusion model in that latent space |

Model latent-diffusion adalah VAE dengan model diffusion yang hidup antara encoder dan decoder. VAE melakukan kompresi kasar, model diffusion melakukan kerja berat. Pola sama untuk video (VAE + DiT video-diffusion) dan audio (Encodec + transformer MusicGen).

## Rilis

Simpan `outputs/skill-vae-trainer.md`.

Skill menerima: profil dataset + target dim-laten + pemakaian downstream (rekonstruksi, sampling, atau input latent-diffusion) dan mengeluarkan: pilihan arsitektur (plain/β/VQ/RVQ), schedule β, dim laten, likelihood decoder (Gaussian vs kategorikal), dan rencana evaluasi (MSE recon, KL per dim, jarak Fréchet antara `q(z|x)` dan `N(0, I)`).

## Latihan

1. **Mudah.** Ubah `β` di `code/main.py` menjadi `0.01`, `0.1`, `1.0`, `5.0`. Catat MSE rekonstruksi final dan KL. β mana yang Pareto-terbaik untuk data sintetismu?
2. **Sedang.** Ganti likelihood decoder Gaussian dengan likelihood Bernoulli (loss cross-entropy). Bandingkan kualitas sampel pada versi biner dari data sintetis yang sama.
3. **Sulit.** Perluas `code/main.py` menjadi mini VQ-VAE: ganti `z` kontinu dengan lookup tetangga-terdekat dalam codebook K=32 entri. Bandingkan MSE rekonstruksi dan laporkan berapa entri codebook yang terpakai (codebook collapse itu nyata).

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Autoencoder | Encode-decode network | `x → z → x̂`, learn MSE. Not generative. |
| VAE | AE with a sampler | Encoder outputs a distribution, KL penalty shapes code space. |
| ELBO | Evidence lower bound | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`; tight when `q = p(z\|x)`. |
| Reparameterization | `z = μ + σ·ε` | Rewrites stochastic node as deterministic + pure noise. Enables backprop through sampling. |
| Prior | `p(z)` | Target distribution for the latent, typically `N(0, I)`. |
| Posterior collapse | "KL term wins" | Encoder ignores `x`, outputs the prior; decoder must hallucinate. |
| β-VAE | Tunable KL weight | `loss = recon + β·KL`. Higher β = more disentangled but blurrier. |
| VQ-VAE | Discrete latent | Replace continuous `z` with nearest codebook vector; enables transformer modelling. |

## Catatan produksi: VAE adalah jalur terpanas di server diffusion

Dalam pipeline Stable Diffusion / Flux / SD3, VAE dipanggil dua kali per request — sekali untuk mengenkode (kalau img2img / inpainting) dan sekali untuk mendecode. Pada 1024² pass decoder sering menjadi puncak memori-aktivasi tunggal terbesar di seluruh pipeline karena ia mengupsample laten `128×128×16` kembali menjadi `1024×1024×3`. Dua konsekuensi praktis:

- **Slice atau tile decode-nya.** `diffusers` mengekspos `pipe.vae.enable_slicing()` dan `pipe.vae.enable_tiling()`. Tiling menukar artefak seam kecil dengan memori `O(tile²)` alih-alih `O(H·W)`. Esensial untuk 1024²+ pada GPU konsumen.
- **Decoder bf16, numerik fp32 untuk resize final.** VAE SD 1.x dirilis dalam fp32 dan *diam-diam menghasilkan NaN* ketika dicast ke fp16 pada 1024²+. SDXL menyertakan `madebyollin/sdxl-vae-fp16-fix` — selalu pilih varian fp16-fix atau pakai bf16.

## Bacaan Lanjutan

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — makalah VAE.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — β-VAE disentangled.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — VAE gambar state-of-the-art.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion; VAE sebagai encoder.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec, standar VAE audio.
