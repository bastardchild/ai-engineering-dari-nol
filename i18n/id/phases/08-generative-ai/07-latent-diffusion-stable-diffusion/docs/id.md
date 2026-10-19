# Latent Diffusion & Stable Diffusion

> Diffusion ruang-piksel pada gambar 512×512 adalah kejahatan perang komputasional. Rombach et al. (2022) menyadari kamu tidak butuh seluruh 786k dimensi untuk membangkitkan gambar — kamu butuh cukup untuk menangkap struktur semantik, dan decoder terpisah untuk sisanya. Jalankan diffusion di dalam ruang laten VAE. Satu ide itu adalah Stable Diffusion.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**Time:** ~75 minutes

## Masalahnya

Diffusion ruang-piksel pada 512² berarti U-Net berjalan pada tensor ber-shape `[B, 3, 512, 512]`. Setiap langkah sampling ~100 GFLOPS untuk U-Net 500M-param. Lima puluh langkah berarti 5 TFLOPS per gambar. Latih pada satu miliar gambar dan tagihan komputasinya absurd.

Sebagian besar FLOPs itu dipakai mendorong detail tak penting-perseptual melalui net — tekstur frekuensi-tinggi yang bisa dikompresi VAE lossy. Ide Rombach: latih VAE sekali (*tahap pertama*), bekukan, dan jalankan diffusion sepenuhnya di ruang laten 4-channel 64×64 (*tahap kedua*). U-Net yang sama. 1/16 piksel. ~64x FLOPs lebih sedikit untuk kualitas sebanding.

Inilah resep Stable Diffusion. SD 1.x / 2.x memakai U-Net 860M atas laten `64×64×4`, SDXL memakai U-Net 2.6B atas `128×128×4`, SD3 menukar U-Net dengan Diffusion Transformer (DiT) plus flow matching. Flux.1-dev (Black Forest Labs, 2024) menyertakan DiT-MMDiT 12B-param. Semua berjalan pada substrat dua-tahap yang sama.

## Konsepnya

![Latent diffusion: VAE compression + diffusion in latent space](../assets/latent-diffusion.svg)

**Dua tahap, dilatih terpisah.**

1. **Tahap 1 — VAE.** Encoder `E(x) → z`, decoder `D(z) → x`. Target kompresi: downsample 8× di setiap sumbu spasial + sesuaikan channel supaya ukuran laten total ~1/16 jumlah piksel. Loss = rekonstruksi (L1 + LPIPS perseptual) + KL (bobot kecil supaya `z` tidak dipaksa terlalu Gaussian, karena kita tidak butuh sampling eksak dari `z`). Sering dilatih dengan loss adversarial supaya gambar ter-decode tajam.

2. **Tahap 2 — diffusion pada `z`.** Perlakukan `z = E(x_real)` sebagai data. Latih U-Net (atau DiT) untuk mendenoise `z_t`. Saat inference: sampel `z_0` via diffusion, lalu `x = D(z_0)`.

**Conditioning teks.** Dua komponen tambahan. Encoder teks beku (CLIP-L untuk SD 1.x, CLIP-L+OpenCLIP-G untuk SD 2/XL, T5-XXL untuk SD3 dan Flux). Injeksi cross-attention: setiap block U-Net mengambil `[Q = fitur gambar, K = V = token teks]` dan mencampurnya. Token adalah satu-satunya jalan teks memengaruhi gambar.

**Fungsi loss identik dengan Pelajaran 06.** MSE DDPM / flow matching yang sama pada noise. Kamu hanya menukar domain data.

## Varian arsitektur

| Model | Year | Backbone | Latent shape | Text encoder | Params |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | same | 1-4 step sampling |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 step |

Trennya: ganti U-Net dengan DiT (transformer atas patch laten), skalakan encoder teks (T5 mengalahkan CLIP untuk kepatuhan prompt), naikkan channel laten (4 → 16 memberi lebih banyak ruang detail).

```figure
noise-schedule
```

## Bangun Sendiri

`code/main.py` menumpuk "VAE" 1-D mainan (encoder + decoder identitas, untuk demonstrasi; VAE nyata adalah conv net) di atas DDPM dari Pelajaran 06 dan menambahkan conditioning kelas dengan classifier-free guidance. Ia menunjukkan loss diffusion yang sama bekerja baik saat dijalankan pada nilai 1-D mentah maupun pada nilai terenkode — wawasan kuncinya.

### Langkah 1: encoder/decoder

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

VAE nyata punya bobot terlatih. Untuk pedagogi, peta linear ini cukup untuk menunjukkan diffusion beroperasi pada `z` tanpa peduli ruang data orisinal.

### Langkah 2: diffusion di ruang-`z`

DDPM yang sama dengan Pelajaran 06. Data yang dilihat net adalah `z = E(x)`. Setelah menyampel `z_0`, decode dengan `D(z_0)`.

### Langkah 3: classifier-free guidance

Saat training, drop label kelas 10% (ganti dengan token null). Saat inference, hitung `ε_cond` maupun `ε_uncond`, lalu:

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = tanpa guidance (diversitas penuh), `w = 3` = bawaan, `w = 7+` = saturasi / terlalu-tajam.

### Langkah 4: conditioning teks (konsep, bukan kode)

Ganti label kelas dengan output encoder teks beku. Umpankan embedding teks ke U-Net via cross-attention:

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

Inilah satu-satunya perbedaan substantif antara model diffusion kondisional-kelas dan Stable Diffusion.

## Jebakan

- **Ketidakcocokan skala-VAE.** VAE SD 1.x punya konstanta scaling (`scaling_factor ≈ 0.18215`) yang diterapkan setelah encoding. Melupakannya membuat U-Net berlatih pada laten dengan variansi yang salah total. Setiap checkpoint menyertakan satu.
- **Encoder teks diam-diam salah.** SD3 butuh T5-XXL dengan >=128 token, dan fallback ke CLIP-saja itu lossy. Selalu cek `use_t5=True` atau fidelitas prompt anjlok.
- **Mencampur ruang laten.** SDXL, SD3, Flux semuanya memakai VAE berbeda. LoRA yang dilatih pada laten SDXL tidak akan bekerja pada SD3. Hugging Face diffusers 0.30+ menolak memuat checkpoint yang tak cocok.
- **CFG terlalu tinggi.** `w > 10` menghasilkan gambar saturasi dan berminyak dan over-fit ke prompt dengan mengorbankan diversitas. Sweet spot-nya `w = 3-7`.
- **Negative prompt bocor.** Negative prompt kosong menjadi token null; negative prompt terisi menjadi `ε_uncond`. Keduanya tidak sama; sebagian pipeline diam-diam default ke null.

## Gunakan

Stack produksi pada 2026:

| Target | Recommended backbone |
|--------|----------------------|
| Narrow domain, paired data, training a model from scratch | SDXL fine-tune (LoRA / full) — fastest to ship |
| Open-domain text-to-image, open weights | Flux.1-dev (12B, Apache / non-commercial) or SD3.5-Large |
| Fastest inference, open weights | Flux.1-schnell (1-4 step, Apache) or SDXL-Lightning |
| Best prompt adherence, hosted | GPT-Image / DALL-E 3 (still), Midjourney v7, Imagen 4 |
| Edit workflows | Flux.1-Kontext (Dec 2024) — natively accepts image + text |
| Research, baseline | SD 1.5 — ancient but well-studied |

## Rilis

Simpan `outputs/skill-sd-prompter.md`. Skill menerima prompt teks + style target dan mengeluarkan: model + checkpoint, skala CFG, sampler, negative prompt, resolusi, kombo ControlNet/IP-Adapter opsional, dan checklist QA per-langkah.

## Latihan

1. **Mudah.** Jalankan `code/main.py` dengan guidance `w ∈ {0, 1, 3, 7, 15}`. Catat rata-rata sampel per kelas. Pada `w` berapa rata-rata kelas divergen melewati rata-rata data nyata?
2. **Sedang.** Tukar encoder linear mainan dengan pasangan encoder/decoder MLP-tanh plus loss rekonstruksi. Latih ulang diffusion pada laten baru. Apakah kualitas sampel berubah?
3. **Sulit.** Siapkan inference Stable Diffusion nyata dengan diffusers: muat `sdxl-base`, jalankan 30 langkah Euler dengan CFG=7, ukur waktunya. Kini beralih ke `sdxl-turbo` dengan 4 langkah dan CFG=0. Subjek sama, kualitas berbeda — jelaskan apa yang berubah dan mengapa.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| First stage | "The VAE" | Trained encoder/decoder pair; compresses 512² to 64². |
| Second stage | "The U-Net" | Diffusion model over the latent space. |
| CFG | "Guidance scale" | `(1+w)·ε_cond - w·ε_uncond`; tunes conditioning strength. |
| Null token | "Empty prompt embed" | Unconditional embed used for `ε_uncond`. |
| Cross-attention | "How text gets in" | Each U-Net block attends to text tokens as K and V. |
| DiT | "Diffusion Transformer" | Replace U-Net with a transformer over latent patches; scales better. |
| MMDiT | "Multi-modal DiT" | SD3's architecture: text and image streams with joint attention. |
| VAE scaling factor | "Magic number" | Divides latents by ~5.4 so diffusion operates in unit-variance space. |

## Catatan produksi: menjalankan Flux-12B pada GPU konsumen 8GB

Integrasi Flux referensi adalah resep kanonis "Aku punya GPU konsumen, bisakah aku mengirim ini?" Triknya adalah resep tiga-knob yang sama yang didaftar literatur production inference diterapkan ke DiT diffusion:

1. **Loading bertahap.** Flux punya tiga network yang tidak pernah perlu koeksis di VRAM: encoder teks T5-XXL (~10 GB dalam fp32), CLIP-L (kecil), MMDiT 12B, dan VAE. Enkode prompt dulu, *hapus* encoder, muat DiT, denoise, *hapus* DiT, muat VAE, decode. GPU konsumen 8GB hanya muat satu tahap dalam satu waktu.
2. **Kuantisasi 4-bit via bitsandbytes.** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` pada encoder T5 maupun DiT. Memotong memori 8×, penurunan kualitas tak terasa untuk teks-ke-gambar per benchmark Aritra (tertaut di notebook).
3. **Offload CPU.** `pipe.enable_model_cpu_offload()` menukar modul otomatis antara CPU dan GPU saat setiap forward pass maju. Menambah latensi 10-20% tapi membuat pipeline berjalan.

Akuntansi memorinya: `10 GB T5 / 8 = 1.25 GB` terkuantisasi, `12 B param × 0.5 byte = ~6 GB` DiT terkuantisasi, plus aktivasi. Dalam istilah stas00 ini adalah ujung-ekstrem inference TP=1 — tanpa paralelisme model, kuantisasi maksimum. Untuk produksi kamu akan menjalankan TP=2 atau TP=4 pada H100; untuk laptop dev tunggal, inilah resepnya.

## Bacaan Lanjutan

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — keluarga Flux.1.
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — implementasi referensi untuk setiap checkpoint di atas.
