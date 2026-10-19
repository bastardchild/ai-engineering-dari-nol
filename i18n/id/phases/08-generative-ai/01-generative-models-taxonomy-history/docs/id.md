# Model Generatif — Taksonomi & Sejarah

> Setiap model gambar, model teks, model video, dan model 3D muat dalam satu dari lima ember. Pilih ember yang salah dan kamu akan bergumul dengan matematikanya berminggu-minggu. Pilih yang benar dan dua belas tahun kemajuan bidang ini bertumpuk rapi di kepalamu.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 2 (ML Fundamentals), Phase 3 (Deep Learning Core), Phase 7 · 14 (Transformers)
**Time:** ~45 minutes

## Masalahnya

Model generatif melakukan satu pekerjaan: diberikan sampel training yang diambil dari distribusi tak dikenal `p_data(x)`, keluarkan sampel baru yang tampak berasal dari distribusi yang sama. Wajah, kalimat, file MIDI, struktur protein — semua problem yang sama kalau kamu menyipitkan mata.

Soalnya `p_data` hidup di ruang berjuta dimensi (gambar RGB 512x512 adalah ~786k dimensi), sampel duduk pada manifold tipis di dalam ruang itu, dan kamu hanya punya mungkin 10M contoh. Memaksakan densitas secara brute-force itu hopeless. Setiap model generatif adalah kompromi yang menukar satu problem sulit dengan problem yang sedikit kurang sulit.

Lima keluarga bertahan dua belas tahun terakhir. Mengetahui kompromi apa yang dibuat setiap keluarga memberi tahu mengapa ia menang pada sebagian task dan runtuh pada yang lain.

## Konsepnya

![Five families of generative models — taxonomy by what they model](../assets/taxonomy.svg)

**1. Densitas eksplisit, tractable.** Tulis `log p(x)` sebagai jumlah yang benar-benar bisa dievaluasi. Model autoregresif (PixelCNN, WaveNet, GPT) memfaktorkan `p(x) = ∏ p(x_i | x_<i)`. Normalizing flow (RealNVP, Glow) membangun `p(x)` sebagai transform invertible dari basis sederhana. Pro: likelihood eksak, loss training bersih. Kontra: inference autoregresif itu sekuensial (lambat untuk sequence panjang), flow butuh arsitektur invertible (restriktif secara arsitektural).

**2. Densitas eksplisit, aproksimasi.** Batasi `log p(x)` dari bawah (ELBO) dan optimasi batasnya. VAE (Kingma 2013) memakai encoder-decoder dengan posterior variasional. Model diffusion (DDPM, Ho 2020) melatih denoiser yang secara implisit mengoptimasi ELBO berbobot. Diffusion adalah backbone citra, video, dan 3D yang dominan pada 2026.

**3. Densitas implisit.** Lewati densitas sepenuhnya; pelajari generator `G(z)` yang menghasilkan sampel dan diskriminator `D(x)` yang membedakan nyata dari palsu. GAN (Goodfellow 2014). Cepat saat inference (satu forward pass) tapi terkenal tidak stabil saat training. StyleGAN 1/2/3 tetap state of the art untuk fotorealisme domain-tetap (wajah, kamar tidur) bahkan pada 2026.

**4. Berbasis-skor / waktu-kontinu.** Pelajari gradien log-densitas `∇_x log p(x)` (skor) secara langsung. Song & Ermon (2019) menunjukkan score matching menggeneralisasi diffusion ke SDE. Flow matching (Lipman 2023) adalah primadona 2024-2026: training bebas-simulasi, jalur lebih lurus, sampling 4-10x lebih cepat daripada DDPM. Stable Diffusion 3, Flux, AudioCraft 2 semuanya memakai flow matching.

**5. Autoregresif berbasis-token atas kode diskret.** Kompresi data berdimensi-tinggi dengan VQ-VAE atau quantizer residual menjadi sequence pendek token diskret, lalu pakai Transformer untuk memodelkan sequence token. Parti, MuseNet, AudioLM, VALL-E, tokenizer patch Sora semuanya memakai ini. Ini adalah ember 1 plus tokenizer terpelajari.

## Sejarah singkat

| Year | Model | Why it mattered |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | First deep generative model with a usable training loss. |
| 2014 | GAN (Goodfellow) | Implicit density, no likelihood — shockingly sharp samples. |
| 2015 | DRAW, PixelCNN | Sequential image generation. |
| 2017 | Glow, RealNVP | Invertible flows; exact likelihood with depth. |
| 2017 | Progressive GAN | First megapixel faces. |
| 2019 | StyleGAN / StyleGAN2 | Photorealistic faces still hard to beat for that one domain. |
| 2020 | DDPM (Ho) | Diffusion becomes practical. |
| 2021 | CLIP, DALL-E 1, VQGAN | Text-to-image goes mainstream. |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Latent diffusion + text conditioning = commodity. |
| 2022 | ControlNet, LoRA | Fine control over pretrained diffusion. |
| 2023 | SDXL, Midjourney v5, Flow matching | Scale + better training dynamics. |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | Video diffusion; flow matching wins. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | Production-grade video. |
| 2026 | Consistency + Rectified Flow | One-step sampling from diffusion backbones. |

## Triase lima pertanyaan

Ketika makalah model generatif baru terbit, jawab lima pertanyaan ini sebelum membaca bagian metode.

1. **Apa yang dimodelkan?** Piksel, laten, token diskret, Gaussian 3D, mesh, waveform?
2. **Apakah densitas eksplisit atau implisit?** Apakah mereka menulis `log p(x)`?
3. **Sampling: sekali-jalan atau iteratif?** Iteratif berarti inference lebih lambat; sekali-jalan biasanya berarti adversarial atau distilled.
4. **Conditioning: tanpa-syarat, kelas, teks, gambar, pose?** Ini menentukan loss dan perancah arsitektur.
5. **Evaluasi: FID, CLIP score, IS, preferensi manusia, akurasi task?** Masing-masing punya mode kegagalan yang diketahui (lihat Pelajaran 14).

Kamu akan menjawab ulang kelimanya untuk setiap pelajaran di fase ini. Pada akhirnya, mereka akan menjadi refleks.

```figure
autoencoder-bottleneck
```

## Bangun Sendiri

Kode untuk pelajaran ini adalah visualisasi ringan: fit campuran Gaussian 1-D dari sampel memakai tiga pendekatan mainan (densitas kernel, histogram diskret, dan generator "ala-GAN" sampel-terdekat) supaya kamu bisa melihat perbedaan antara densitas eksplisit vs implisit pada problem yang bisa dicetak dalam satu layar.

Jalankan `code/main.py`. Ia mengambil 2000 sampel dari campuran Gaussian dua-mode, lalu mencetak:

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

Perhatikan: dua yang pertama membuatmu bisa bertanya "seberapa mungkin titik ini?" Yang ketiga tidak bisa. Inilah distingsi *eksplisit vs implisit* yang akan penting untuk setiap pelajaran berikutnya.

## Gunakan

Keluarga mana, untuk task apa, pada 2026?

| Task | Best family | Why |
|------|-------------|-----|
| Photoreal faces, narrow domain | StyleGAN 2/3 | Still sharpest, fastest inference. |
| General text-to-image | Latent diffusion + flow matching | SD3, Flux.1, DALL-E 3. |
| Fast text-to-image | Rectified flow + distillation | SDXL-Turbo, SD3-Turbo, LCM. |
| Text-to-video | Diffusion Transformer + flow matching | Sora, Veo 2, Kling. |
| Speech + music | Token-based AR (AudioLM, VALL-E, MusicGen) or flow matching (AudioCraft 2) | Discrete tokens scale cheaply. |
| 3D scenes | Gaussian Splatting fit, diffusion prior | 3D-GS for reconstruction, diffusion for novel-view. |
| Density estimation (no sampling) | Flows | Only family with exact `log p(x)`. |
| Simulation / physics | Flow matching, score SDE | Straight-line paths, smooth vector fields. |

## Rilis

Simpan sebagai `outputs/skill-model-chooser.md`.

Skill menerima deskripsi task dan mengeluarkan: (1) keluarga mana yang dipakai, (2) daftar peringkat tiga opsi open dan tiga opsi hosted, (3) mode kegagalan yang mungkin harus kamu waspadai, dan (4) budget komputasi/waktu.

## Latihan

1. **Mudah.** Untuk masing-masing dari lima produk ini, identifikasi keluarga dan backbone: ChatGPT image, Midjourney v7, Sora, Runway Gen-3, ElevenLabs. Bukti harus dari laporan teknis publik.
2. **Sedang.** Makalah yang akan kamu baca besok mengklaim sampling 100x lebih cepat daripada diffusion. Tulis tiga pertanyaan untuk memeriksa apakah speedup bertahan pada conditioning dan resolusi tinggi.
3. **Sulit.** Ambil satu domain yang kamu pedulikan (mis. struktur protein, CAD, molekul, trajektori). Jawab triase lima pertanyaan untuk model SOTA saat ini di domain itu dan sketsa apa yang akan diubah model yang lebih baik.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Generative model | "It makes new stuff" | Learns a sampler for `p_data(x)`, optionally exposes `log p(x)`. |
| Explicit density | "You can evaluate it" | Model provides a closed-form or tractable `log p(x)`. |
| Implicit density | "GAN-style" | Only a sampler — no way to evaluate `p(x)` of a given point. |
| ELBO | "Evidence lower bound" | A tractable lower bound on `log p(x)`; VAEs and diffusion optimize it. |
| Score | "Gradient of log-density" | `∇_x log p(x)`; diffusion and SDE models learn this field. |
| Manifold hypothesis | "Data lives on a surface" | High-dim data concentrates on a low-dim manifold; why dimensionality reduction works. |
| Autoregressive | "Predict the next piece" | Factorize joint as product of conditionals. |
| Latent | "Compressed code" | Low-dim representation from which a decoder can reconstruct the input. |

## Catatan produksi: lima keluarga, lima bentuk inference

Setiap keluarga memetakan ke kurva biaya server-inference yang berbeda. Literatur production-inference membingkai inference LLM sebagai prefill + decode; dekomposisi yang sama berlaku di sini:

- **Autoregresif (ember 1 dan 5).** Decode sekuensial mendominasi latensi; KV-cache, continuous batching, dan speculative decoding semuanya berlaku langsung.
- **VAE / diffusion / flow-matching (ember 2 dan 4).** Tidak ada decode dalam arti LLM. Biaya = `num_steps × step_cost`, dan `step_cost` adalah forward transformer atau U-Net pada resolusi laten penuh. Knob produksinya adalah jumlah langkah (DDIM / DPM-Solver / distilasi), ukuran batch, dan presisi (bf16 / fp8 / int4).
- **GAN (ember 3).** Satu forward pass. Tanpa schedule, tanpa KV-cache. TTFT ≈ latensi total. Inilah mengapa StyleGAN masih menang pada UX domain-sempit.

Ketika kamu melihat "lebih cepat daripada diffusion" dalam abstrak makalah, terjemahkan menjadi "langkah lebih sedikit × biaya langkah sama" atau "langkah sama × biaya langkah lebih murah". Sisanya adalah marketing.

## Bacaan Lanjutan

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — makalah GAN.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — makalah VAE.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — makalah DDPM.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — diffusion sebagai SDE.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — makalah flow matching.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3.
