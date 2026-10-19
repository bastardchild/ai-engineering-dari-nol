# Conditional GAN & Pix2Pix

> Unlock besar pertama 2014-2017 adalah mengendalikan apa yang dibuat GAN. Tempelkan label, atau gambar, atau kalimat. Pix2Pix melakukan versi gambarnya dan ia masih mengalahkan setiap model teks-ke-gambar generik pada task gambar-ke-gambar yang sempit.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**Time:** ~75 minutes

## Masalahnya

GAN tanpa-syarat menyampel wajah sembarang. Berguna untuk demo, tak berguna di produksi. Kamu mau: *petakan sketsa ke foto*, *petakan peta ke foto udara*, *petakan scene siang ke malam*, *warnai gambar grayscale*. Dalam semuanya, kamu diberi gambar input `x` dan harus mengeluarkan `y` dengan korespondensi semantik. Ada banyak `y` yang masuk akal per `x`. Mean-squared error meratakannya menjadi bubur. Loss adversarial tidak, karena "tampak nyata" itu tajam.

Conditional GAN (Mirza & Osindero, 2014) menambahkan condition `c` sebagai input ke `G` maupun `D`. Pix2Pix (Isola et al., 2017) mengkhususkannya: condition adalah gambar input penuh, generator adalah U-Net, diskriminator adalah classifier *berbasis-patch* (PatchGAN), dan loss adalah adversarial + L1. Resep itu mengungguli model teks-ke-gambar dari-nol pada domain gambar-ke-gambar yang sempit bahkan pada 2026 karena ia dilatih pada *data berpasangan* — kamu punya persis sinyal yang dibutuhkan.

## Konsepnya

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**G Kondisional.** `G(x, z) → y`. Dalam Pix2Pix, `z` adalah dropout di dalam G (tanpa noise input — Isola menemukan noise eksplisit diabaikan).

**D Kondisional.** `D(x, y) → [0, 1]`. Input adalah *pasangan* (condition, output). Inilah perbedaan kunci: D harus menilai apakah `y` konsisten dengan `x`, bukan hanya apakah `y` tampak nyata.

**Generator U-Net.** Encoder-decoder dengan skip connection melintasi bottleneck. Kritis untuk task di mana input dan output berbagi struktur level-rendah (tepi, siluet). Tanpa skip, detail frekuensi-tinggi menghilang.

**Diskriminator PatchGAN.** Alih-alih mengeluarkan satu skor nyata/palsu, D mengeluarkan grid `N×N` di mana setiap sel menilai receptive field ~70×70 piksel. Dirata-ratakan. Ini adalah asumsi Markov random field: realisme itu lokal. Jauh lebih cepat dilatih, parameter lebih sedikit, output lebih tajam.

**Loss.**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

Suku L1 menstabilkan training dan mendorong G ke target yang diketahui. L1 memberi tepi lebih tajam daripada L2 (median, bukan mean). `λ = 100` adalah bawaan Pix2Pix.

## CycleGAN — ketika kamu tidak punya pasangan

Pix2Pix butuh data `(x, y)` berpasangan. CycleGAN (Zhu et al., 2017) membuang syarat ini dengan biaya loss ekstra: loss *konsistensi-siklus*. Dua generator `G: X → Y` dan `F: Y → X`. Latih sehingga `F(G(x)) ≈ x` dan `G(F(y)) ≈ y`. Ini membuatmu menerjemahkan kuda ke zebra, musim panas ke dingin, tanpa contoh berpasangan.

Pada 2026, gambar-ke-gambar tanpa-pasangan kebanyakan dilakukan via diffusion (ControlNet, IP-Adapter) alih-alih CycleGAN, tapi ide konsistensi-siklus bertahan di hampir setiap makalah adaptasi-domain tanpa-pasangan.

```figure
gx-patchgan
```

## Bangun Sendiri

`code/main.py` mengimplementasikan conditional GAN mungil pada data 1-D. Condition `c` adalah label kelas (0 atau 1). Task: hasilkan sampel dari distribusi kondisional untuk kelas yang diberikan.

### Langkah 1: tempelkan condition ke input G maupun D

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot encoding adalah cara paling sederhana. Model lebih besar memakai embedding terpelajari, modulasi FiLM, atau cross-attention.

### Langkah 2: latih kondisional

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

Generator harus cocok dengan distribusi nyata *untuk condition yang diberikan*, bukan marginalnya.

### Langkah 3: verifikasi output per-kelas

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## Jebakan

- **Condition diabaikan.** G belajar memarginalkan, D tidak pernah menghukum karena sinyal condition lemah. Perbaiki: kondisikan D lebih agresif (lapis awal, bukan hanya akhir), pakai diskriminator proyeksi (Miyato & Koyama 2018).
- **Bobot L1 terlalu rendah.** G melayang ke output tampak-nyata yang sembarang, bukan yang setia. Mulai λ≈100 untuk task ala-Pix2Pix.
- **Bobot L1 terlalu tinggi.** G menghasilkan output blur karena L1 tetap norm L_p. Anneal turun begitu training stabil.
- **Kebocoran ground-truth di D.** Gabungkan `(x, y)` sebagai input D, bukan hanya `y`. Tanpa ini D tidak bisa memeriksa konsistensi.
- **Mode collapse per kelas.** Setiap kelas bisa collapse independen. Jalankan cek diversitas kondisional-kelas.

## Gunakan

State 2026 task gambar-ke-gambar:

| Task | Best approach |
|------|---------------|
| Sketch → photo, same domain, paired data | Pix2Pix / Pix2PixHD (still fast, still sharp) |
| Sketch → photo, unpaired | ControlNet with a Scribble conditioning model |
| Semantic seg → photo | SPADE / GauGAN2 or SD + ControlNet-Seg |
| Style transfer | Diffusion with IP-Adapter or LoRA; GAN methods are legacy |
| Depth → photo | ControlNet-Depth over Stable Diffusion |
| Super-resolution | Real-ESRGAN (GAN), ESRGAN-Plus, or SD-Upscale (diffusion) |
| Colorization | ColTran, diffusion-based colorizers, or Pix2Pix-color |
| Daytime → nighttime, seasons, weather | CycleGAN or ControlNet-based |

Pix2Pix tetap tool yang tepat ketika (a) kamu punya ribuan contoh berpasangan, (b) task-nya sempit dan berulang, dan (c) kamu butuh inference cepat. Pada task terbuka generik, diffusion menang.

## Rilis

Simpan `outputs/skill-img2img-chooser.md`. Skill menerima deskripsi task, ketersediaan data (berpasangan vs tanpa-pasangan, N sampel), dan budget latensi/kualitas, lalu mengeluarkan: pendekatan (Pix2Pix, CycleGAN, varian ControlNet, SDXL + IP-Adapter), kebutuhan data training, biaya inference, dan protokol eval (LPIPS, FID, khusus-task).

## Latihan

1. **Mudah.** Modifikasi `code/main.py` untuk menambah kelas ketiga. Konfirmasi G tetap memetakan noise setiap kelas ke mode yang benar.
2. **Sedang.** Ganti L1 dengan loss ala-perseptual dalam setting 1-D (mis. D kecil beku yang bertindak sebagai ekstraktor fitur). Apakah ia mengubah ketajaman distribusi kondisional?
3. **Sulit.** Sketsa CycleGAN dalam setting 1-D: dua distribusi, dua generator, loss siklus. Tunjukkan bahwa ia belajar memetakan di antara keduanya tanpa data berpasangan.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Conditional GAN | "GAN with labels" | G(z, c), D(x, c). Both networks see the condition. |
| Pix2Pix | "Image-to-image GAN" | Paired cGAN with U-Net G and PatchGAN D + L1 loss. |
| U-Net | "Encoder-decoder with skips" | Symmetric conv network; skips preserve high-freq. |
| PatchGAN | "Local-realism classifier" | D outputs per-patch score instead of global score. |
| CycleGAN | "Unpaired image translation" | Two G's + cycle-consistency loss; no paired data. |
| SPADE | "GauGAN" | Normalizes intermediate activations with the semantic map; segmentation-to-image. |
| FiLM | "Feature-wise linear modulation" | Per-feature affine transform from the condition; cheap conditioning. |

## Catatan produksi: Pix2Pix sebagai baseline yang terikat-latensi

Ketika kamu punya data berpasangan dan task sempit (sketsa → render, peta semantik → foto, siang → malam), inference sekali-jalan Pix2Pix mengalahkan diffusion satu orde magnitudo pada latensi. Perbandingan produksinya biasanya:

| Path | Steps | Typical latency at 512² on a single L4 |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net forward) | 1 | ~30 ms |
| SD-Inpaint or SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix menang pada throughput dalam batch statis (setiap request FLOPs-nya sama). Diffusion menang pada kualitas dan generalisasi. Langkah modern sering kali mengirim model distilled ala-Pix2Pix untuk task sempit dan fallback diffusion untuk input ekor.

## Bacaan Lanjutan

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — makalah cGAN.
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix.
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN.
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD.
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN.
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — D proyeksi.
