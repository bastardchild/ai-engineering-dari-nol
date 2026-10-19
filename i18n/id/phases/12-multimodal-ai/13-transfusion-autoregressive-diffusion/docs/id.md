# Transfusion: Teks Autoregressive + Diffusion Gambar dalam Satu Transformer

> Chameleon dan Emu3 mempertaruhkan segalanya pada token diskrit. Keduanya bekerja, tapi bottleneck kuantisasi terlihat jelas — kualitas gambar berplatok di bawah model diffusion ruang continuous. Transfusion (Meta, Zhou dkk., Agustus 2024) mengambil taruhan berlawanan: pertahankan gambar continuous, buang VQ-VAE sama sekali, dan latih satu transformer dengan dua loss. Token teks mendapat next-token-prediction. Patch gambar mendapat loss flow-matching / diffusion. Kedua objective mengoptimalkan bobot yang sama. Arsitektur yang mendasari Stable Diffusion 3 (MMDiT) adalah kerabat dekatnya. Pelajaran ini membaca tesis Transfusion, membangun trainer dua-loss mainan, dan menelusuri attention mask yang memungkinkan satu transformer mengerjakan kedua pekerjaan sekaligus.

**Type:** Build
**Languages:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Merangkai transformer yang menjalankan dua loss (NTP pada token teks, diffusion MSE pada patch gambar) di satu backbone.
- Menjelaskan mengapa attention bidirectional antar patch gambar digabung attention kausal atas token teks adalah pilihan mask yang tepat.
- Membandingkan gaya Transfusion (gambar continuous, loss diffusion) dengan gaya Chameleon (gambar diskrit, NTP) dari sisi komputasi, kualitas, dan kompleksitas kode.
- Menyebut kontribusi MMDiT: bobot spesifik-modalitas di setiap blok, attention bersama di residual stream.

## Masalahnya

Debat token gambar diskrit vs continuous lebih tua dari LLM. Representasi continuous (piksel mentah, latensi VAE) mempertahankan detail. Token diskrit (indeks VQ) cocok dengan vocabulary native transformer tapi kehilangan detail pada langkah kuantisasi.

Chameleon / Emu3 memilih jalur diskrit: satu loss, satu arsitektur, tapi fidelitas gambar dibatasi kualitas tokenizer.

Model diffusion memilih continuous: kualitas gambar luar biasa, tapi model terpisah dari LLM, rekayasa noise-schedule yang rumit, dan integrasi yang tidak mulus dengan generasi teks.

Transfusion bertanya: bisakah kita mendapat keduanya? Pertahankan gambar continuous, tetap latih satu model, pakai dua loss yang dijahit dalam satu langkah gradien.

## Konsepnya

### Arsitektur dua-loss

Satu transformer decoder-only memproses sekuens yang berisi:

- Token teks (diskrit, dari vocabulary BPE).
- Patch gambar (continuous, blok piksel 16x16 diproyeksikan ke hidden dim via embedding linear — sama seperti input encoder ViT).
- Tag `<image>` dan `</image>` yang menandai di mana patch continuous berada.

Forward pass berjalan sekali. Loss memilih salah satu dari dua head per token:

- Untuk token teks: cross-entropy standar pada head vocab-logits.
- Untuk patch gambar: loss diffusion pada patch continuous — memprediksi noise yang ditambahkan ke setiap patch.

Gradien mengalir melalui badan transformer bersama. Kedua loss memperbaiki bobot bersama secara bersamaan.

### Attention mask: teks kausal + gambar bidirectional

Token teks harus kausal — kamu tidak boleh membiarkan token teks menghadiri teks masa depan, atau teacher forcing rusak. Namun patch gambar merepresentasikan satu snapshot; mereka seharusnya saling menghadiri secara bidirectional dalam blok gambar yang sama.

Mask-nya:

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

Diimplementasikan sebagai mask segitiga-blok saat training dan inference.

### Loss diffusion di dalam transformer

Loss diffusion standar: tambahkan noise ke patch gambar, minta model memprediksi noise (atau patch yang bersih, secara ekuivalen). Versi Transfusion memakai flow matching — memprediksi medan velocity dari noisy ke bersih.

Selama training:
1. Untuk setiap patch gambar x0, sampelkan timestep acak t.
2. Sampelkan noise ε, hitung xt = (1-t) * x0 + t * ε (interpolasi linear untuk flow matching).
3. Transformer memprediksi v_theta(xt, t); loss = MSE(v_theta(xt, t), ε - x0).
4. Backprop bersama loss NTP teks dari sekuens yang sama.

Saat inference, generasinya:
- Token teks: sampling autoregressive standar.
- Patch gambar: loop sampling diffusion (10-30 langkah tipikal) yang di-kondisikan pada token teks sebelumnya.

### MMDiT: varian milik Stable Diffusion 3

Stable Diffusion 3 (Esser dkk., Maret 2024) merilis MMDiT (Multimodal Diffusion Transformer) hampir bersamaan dengan Transfusion. Arsitektur keduanya bersaudara.

Perbedaan utama MMDiT:

- Bobot spesifik-modalitas per blok. Setiap blok transformer memiliki bobot Q, K, V, dan MLP terpisah untuk token teks vs patch gambar. Attention bersama (cross-modality); sisanya spesifik-modalitas.
- Rectified flow training. Varian flow-matching khusus yang sampling-nya sudah diketahui dan matematikanya lebih sederhana daripada DDPM.
- Skala. MMDiT adalah backbone untuk SD3 (varian 2B dan 8B param). Paper Transfusion menskalakan hingga 7B.

Keduanya menyimpulkan pada ide inti yang sama: satu transformer menjalankan NTP pada teks dan diffusion pada representasi gambar continuous.

### Mengapa ini mengalahkan gaya Chameleon

Jarak kualitas antara diffusion continuous dan NTP diskrit pada generasi gambar dapat diukur. Paper Transfusion melaporkan:

- Pada 7B param, mengalahkan model berukuran sama bergaya Chameleon pada FID sebesar 3-5 poin.
- Tidak perlu training tokenizer — encoder gambar lebih sederhana (proyeksi Linear ke hidden, sama seperti layer input ViT).
- Inference dapat memparalelkan denoising patch gambar, tidak seperti token gambar autoregressive.

Kekurangannya: Transfusion adalah model dual-loss, sehingga dinamika training lebih rumit. Bobot loss perlu di-tuning. Ketidakcocokan schedule antara NTP dan diffusion dapat menyebabkan satu head mendominasi.

### Yang ada di hilir

Janus-Pro (Pelajaran 12.15) menyempurnakan ide Transfusion dengan mendecouple-kan encoder visi untuk pemahaman dan generasi — SigLIP untuk satu, VQ untuk yang lain — sambil berbagi badan transformer. Show-o (Pelajaran 12.14) menukar diffusion dengan discrete-diffusion (masked prediction). Keluarga generasi-terpadu bercabang cepat setelah Transfusion.

VLM produksi 2026 yang memancarkan gambar — Gemini 3 Pro, GPT-5, jalur generasi gambar Claude Opus 4.7 — hampir pasti memakai semacam keturunan dari keluarga ini. Detailnya proprietary.

```figure
cfg-guidance-scale
```

## Gunakan

`code/main.py` membangun Transfusion mainan pada masalah kecil bergaya MNIST:

- Caption teks adalah sekuens bilangan pendek yang mendeskripsikan sebuah digit (0-9).
- Gambar adalah grid 4x4 byte.
- Pasangan proyeksi linear berbobot-bersama bertindak sebagai pengganti transformer; loss NTP pada teks, loss MSE pada patch noisy.
- Loop training mengalirkan kedua loss secara bergantian, attention mask eksplisit.
- Generasi menghasilkan caption teks dan gambar 4x4 dalam satu forward pass.

Transformernya mainan. Plumbing dua-loss, konstruksi attention mask, dan loop inference adalah artefak yang sesungguhnya.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-two-loss-trainer-designer.md`. Diberi tugas multimodal baru (teks + gambar, teks + audio, teks + video), ia merancang schedule dua-loss (bobot loss, bentuk mask, blok bersama vs spesifik-modalitas) dan menandai risiko implementasi.

## Latihan

1. Model bergaya Transfusion melatih 70% token teks dan 30% patch gambar. Loss diffusion gambar ~10x loss NTP teks dalam magnitudo. Bobot loss apa yang menyeimbangkan keduanya?

2. Implementasikan mask segitiga-blok untuk sekuens: `[T, T, <image>, P, P, P, P, </image>, T]`. Tandai setiap entri 0 atau 1.

3. MMDiT memiliki bobot QKV spesifik-modalitas. Berapa overhead jumlah parameter yang ditambahkan ini dibanding transformer fully-shared Transfusion? Pada 7B param, apakah itu sepadan?

4. Generasi: diberi prompt teks, model menjalankan NTP selama 50 token, lalu menyentuh `<image>`, lalu menjalankan diffusion pada 256 patch selama 20 langkah denoise. Berapa total forward pass?

5. Baca Bagian 3 paper SD3. Jelaskan rectified flow dan mengapa ia konvergen dalam lebih sedikit langkah inference daripada DDPM.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Two-loss training | "NTP + diffusion" | A single transformer optimizes both cross-entropy on text tokens and MSE on continuous image patches in the same gradient step |
| Flow matching | "Rectified flow" | Diffusion variant that predicts a velocity field from noise to clean data; simpler math than DDPM |
| MMDiT | "Multimodal DiT" | Stable Diffusion 3's architecture: joint attention, modality-specific MLPs and norms |
| Block-triangular mask | "Causal text + bidirectional image" | Attention mask that is causal across text but bidirectional within image regions |
| Continuous image representation | "No VQ" | Image patches as real-valued vectors, not integer codebook indices |
| Velocity prediction | "v-parameterization" | Network output is the velocity field between noise and data, not the noise itself |

## Bacaan Lanjutan

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
