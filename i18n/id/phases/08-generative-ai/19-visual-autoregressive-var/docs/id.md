# Visual Autoregressive Modeling (VAR): Prediksi Next-Scale

> Model diffusion menyampel iteratif dalam waktu (langkah denoising). VAR menyampel iteratif dalam skala — ia memprediksi token 1x1, lalu 2x2, lalu 4x4, sampai resolusi final, setiap skala dikondisikan pada sebelumnya. Makalah 2024 menunjukkan VAR menyamai hukum scaling ala-GPT untuk pembangkitan gambar dan mengalahkan DiT pada budget komputasi yang sama. Pelajaran ini membangun mekanisme intinya.

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## Masalahnya

Pembangkitan autoregresif mendominasi pemodelan bahasa karena ia berskala terprediksi: lebih banyak komputasi, lebih banyak parameter, perplexity lebih rendah, output lebih baik. Pembangkitan gambar punya dua upaya AR utama sebelum 2024: PixelRNN/PixelCNN (piksel-per-piksel) dan DALL-E 1 / Parti / MuseGAN (token-per-token pada kode VQ-VAE).

Keduanya menderita problem urutan-pembangkitan. Piksel dan token tersusun dalam grid 2D, tapi model AR harus mengunjunginya dalam urutan raster 1D. Piksel sudut awal tidak tahu gambar akhirnya menjadi apa. Kualitas pembangkitan berskala lebih buruk daripada GPT-pada-teks dan tidak pernah mencapai kualitas model-diffusion pada komputasi sebanding.

VAR memperbaiki problem urutan-pembangkitan dengan mengubah apa yang dibangkitkan. Alih-alih memprediksi token gambar satu per satu dalam ruang, VAR memprediksi seluruh gambar pada resolusi yang meningkat. Langkah 1: prediksi token 1x1 ("ringkasan" gambar keseluruhan). Langkah 2: prediksi grid token 2x2 (fitur lebih kasar). Langkah 3: prediksi grid 4x4. Langkah K: prediksi grid final (H/8)x(W/8).

Setiap skala attend ke semua skala sebelumnya (kausal dalam "urutan skala") dan paralel dalam skalanya sendiri. Problem urutan menghilang: seluruh gambar pada skala k dihasilkan dalam satu pass transformer.

## Konsepnya

### Tokenizer VQ Multi-Skala

VAR butuh **tokenizer diskret multi-skala**. Untuk gambar x, ia menghasilkan sequence grid token beresolusi-progresif-lebih-tinggi:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

Setiap z_k memakai codebook yang sama (ukuran tipikal 4096-16384). Tokenisasi pada setiap skala tidak independen — ia dilatih sehingga penjumlahan residual pada setiap skala merekonstruksi f:

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

Ini adalah varian **VQ residual**. Skala k menangkap apa yang dilewatkan skala 1..k-1. Decoder mengambil jumlah semua embedding skala dan menghasilkan gambar.

Tokenizer VQ multi-skala dilatih sekali (seperti VQGAN) lalu dibekukan. Semua kerja generatif dilakukan model autoregresif di atasnya.

### Prediksi Next-Scale

Model generatif adalah transformer yang melihat token dari semua skala sebelumnya dan memprediksi token pada skala berikutnya.

Struktur sequence input:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Embedding posisi mengenkode indeks skala maupun posisi spasial dalam skala. Attention kausal dalam urutan skala: token pada skala k, posisi (i, j) bisa attend ke semua token pada skala 1..k dan ke token pada skala k itu sendiri yang datang lebih dulu dalam urutan intra-skala apa pun yang dipakai (VAR memakai attention posisional tetap tanpa kausalitas intra-skala — semua posisi dalam skala diprediksi paralel).

Loss training: pada setiap skala k, prediksi token z_k diberikan semua token skala-sebelumnya. Loss cross-entropy pada kode VQ diskret. Struktur sama dengan GPT kecuali "sequence"-nya kini terstruktur-skala.

### Pembangkitan

Saat inference:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

Untuk K = 10 skala, pembangkitan adalah 10 forward pass transformer. Setiap pass menghasilkan seluruh skalanya secara paralel — tanpa autoregresi per-token dalam skala. Untuk gambar 256x256 ini kira-kira 10 pass vs 28-50 milik DiT.

### Mengapa Next-Scale Mengalahkan Next-Token

Tiga kemenangan struktural:
1. **Coarse-to-fine selaras dengan statistik gambar alami.** Persepsi visual manusia dan dataset gambar sama-sama menunjukkan keteraturan bergantung-skala: struktur frekuensi-rendah stabil dan terprediksi; detail frekuensi-tinggi kondisional pada konten frekuensi-rendah. Prediksi next-scale mengeksploitasi ini.
2. **Pembangkitan paralel dalam skala.** Tidak seperti AR token ala-GPT, VAR menghasilkan semua token pada skala dalam satu langkah. Panjang pembangkitan efektif skala-log alih-alih linear.
3. **Tanpa bias urutan-pembangkitan.** Token pada skala k melihat seluruh skala k-1; tidak ada bias "kiri-dari" atau "di atas" yang memaksa token awal berkomitmen sebelum konteks akhir tersedia.

### Hukum Scaling

Tian et al. mendemonstrasikan VAR mengikuti kurva scaling power-law untuk FID pada ImageNet — persis seperti GPT untuk perplexity. Menggandakan parameter atau komputasi andal memangkas galat separuh. Ini adalah model generatif-gambar pertama yang menunjukkan perilaku scaling semacam ini sebersih model bahasa. Hasilnya prediksi skala-VAR menjadi terprediksi dari komputasi, bukan tebakan empiris per arsitektur.

### Hubungan dengan Diffusion

VAR dan diffusion berbagi cerita kompresi-data yang sama: keduanya memecah problem pembangkitan menjadi sequence subproblem yang lebih mudah.

- Diffusion: tambahkan noise bertahap, pelajari mengurungkan satu langkah.
- VAR: tambahkan resolusi bertahap, pelajari memprediksi skala berikutnya.

Mereka adalah sumbu berbeda melalui problem. Keduanya menghasilkan distribusi kondisional tractable. Secara empiris VAR lebih cepat saat inference (pass lebih sedikit, semua paralel dalam skala) dan menyamai atau mengalahkan DiT pada ImageNet kondisional-kelas. VAR kondisional-teks (VARclip, HART) adalah arah riset aktif.

```figure
gx-var-next-scale
```

## Bangun Sendiri

Dalam `code/main.py` kamu akan:
1. Bangun **tokenizer VQ multi-skala** mungil pada data "gambar" sintetis (cincin Gaussian 2D).
2. Latih **transformer ala-VAR** untuk next-scale-predict token.
3. Sampel dengan memanggil transformer 4 kali (4 skala) dan decoding.
4. Verifikasi bahwa training berurutan-skala membuat pembangkitan paralel dalam skala.

Ini implementasi mainan. Intinya melihat mask attention terstruktur-skala dan pembangkitan paralel-dalam-skala benar-benar bekerja.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-var-tokenizer-designer.md` — skill untuk merancang tokenizer multi-skala: jumlah skala, rasio skala, ukuran codebook, berbagi residual, arsitektur decoder.

## Latihan

1. **Ablasi jumlah skala.** Latih VAR dengan 4, 6, 8, 10 skala. Ukur kualitas rekonstruksi vs jumlah pass autoregresif. Skala lebih banyak = residual lebih halus = kualitas lebih baik tapi pass lebih banyak.

2. **Ukuran codebook.** Latih tokenizer dengan ukuran codebook 512, 4096, 16384. Codebook lebih besar memberi rekonstruksi lebih baik tapi prediksi lebih sulit. Temukan lututnya.

3. **Cek paralel-dalam-skala.** Untuk VAR terlatih, ukur pola attention eksplisit. Dalam skala k, apakah model attend ke posisi lintas-skala tapi bukan intra-skala? Verifikasi implementasi mask.

4. **Scaling VAR vs DiT.** Untuk task kondisional-kelas ImageNet yang sama, latih VAR dan DiT pada budget param sebanding (mis. 33M, 130M, 458M). Plot FID vs komputasi. VAR seharusnya menarik di depan DiT pada setiap ukuran — reproduksi hasil makalah pada skala kecil.

5. **Conditioning teks.** Perluas VAR untuk mengambil embedding teks (pooled CLIP) sebagai input conditioning ekstra via adaLN. Inilah resep HART. Seberapa besar FID membaik pada sampling selaras-teks?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | Image generation by next-scale prediction over a pyramid of VQ token grids |
| Next-scale prediction | "Predict coarser, then finer" | The model predicts tokens at increasing resolution scales, conditioning on all previous scales |
| Multi-scale VQ tokenizer | "Residual VQ" | VQ-VAE that produces K token grids of increasing resolution, with decoder summing all scales |
| Scale k | "Pyramid level k" | One of K resolution levels, from 1x1 at k=1 up to (H/p)x(W/p) at k=K |
| Parallel-within-scale | "One forward per scale" | All tokens at scale k are predicted in one transformer pass, not autoregressively |
| Causal-across-scales | "Scale-ordered attention" | Token at scale k can attend to all of scales 1..k but not scales k+1..K |
| Residual VQ | "Additive tokenization" | Each scale's tokens encode the residual left by lower scales; decoder sums all scale embeddings |
| VAR scaling law | "Image GPT scaling" | FID follows a predictable power law in compute, like language models' perplexity |
| HART | "Hybrid VAR + text" | Text-conditional VAR variant combining MaskGIT-style iterative decoding with VAR's scale structure |
| Scale position embedding | "(scale, row, col) triple" | Positional encoding carries both the scale index and spatial coordinates within the scale |

## Bacaan Lanjutan

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — makalah VAR, referensi kanonis
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT, baseline perbandingan diffusion
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN, keluarga tokenizer yang diperluas tokenizer multi-skala VAR
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE, fondasi tokenisasi gambar diskret
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — VAR kondisional-teks
