# StyleGAN

> Sebagian besar generator mengaduk `z` ke setiap lapis pada saat yang sama. StyleGAN memisahkannya: pertama petakan `z` ke `w` intermediate, lalu *suntikkan* `w` di setiap level resolusi melalui AdaIN. Satu perubahan itu menguraikan ruang laten dan membuat wajah fotorealistik menjadi problem yang terpecahkan selama tujuh tahun berturut-turut.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 08 (Normalization), Phase 3 · 07 (CNNs)
**Time:** ~45 minutes

## Masalahnya

DCGAN memetakan `z` ke gambar melalui tumpukan transposed convolution. Problemnya: `z` mengendalikan segalanya — pose, pencahayaan, identitas, latar — terjalin bersama. Bergerak sepanjang satu sumbu `z`, keempatnya berubah. Kamu tidak bisa meminta model "orang yang sama, pose berbeda" karena representasinya tidak terfaktor seperti itu.

Karras et al. (2019, NVIDIA) mengusulkan: berhenti mengumpankan `z` langsung ke lapis conv. Umpankan tensor konstanta `4×4×512` sebagai input network. Pelajari MLP 8 lapis yang memetakan `z ∈ Z → w ∈ W`. Suntikkan `w` di setiap resolusi via *adaptive instance normalization* (AdaIN): normalkan setiap peta fitur conv, lalu skala dan geser dengan proyeksi affine dari `w`. Tambahkan noise per-lapis untuk detail stokastik (pori kulit, helai rambut).

Hasilnya: `W` punya sumbu kira-kira ortogonal untuk "style level-tinggi" (pose, identitas) vs "style halus" (pencahayaan, warna). Kamu bisa menukar style antara dua gambar dengan memakai `w` gambar A untuk level resolusi-rendah dan `w` gambar B untuk yang tinggi. Ini membuka editing, stilasi lintas-domain, dan seluruh lini riset "inversi-StyleGAN".

## Konsepnya

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**Mapping network.** `f: Z → W`, MLP 8 lapis. `Z = N(0, I)^512`. `W` tidak dipaksa Gaussian — ia mempelajari bentuk yang diadaptasi-data.

**Synthesis network.** Mulai dari konstanta terpelajari `4×4×512`. Setiap block resolusi: `upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`. Resolusi berlipat dua: 4, 8, 16, 32, 64, 128, 256, 512, 1024.

**AdaIN.**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

dengan `y_scale` dan `y_bias` berasal dari proyeksi affine `w`. Normalkan per peta fitur, lalu restyle. "Style" di sini adalah statistik orde-pertama dan kedua dari peta fitur.

**Noise per-lapis.** Noise Gaussian satu-channel ditambahkan ke setiap peta fitur, diskalakan dengan faktor terpelajari per-channel. Mengendalikan detail stokastik tanpa memengaruhi struktur global.

**Trik truncation.** Saat inference, sampel `z`, hitung `w = mapping(z)`, lalu `w' = ŵ + ψ·(w - ŵ)` dengan `ŵ` adalah rata-rata `w` atas banyak sampel. `ψ < 1` menukar diversitas dengan kualitas. Hampir setiap demo StyleGAN memakai `ψ ≈ 0.7`.

## StyleGAN 1 → 2 → 3

| Version | Year | Innovation |
|---------|------|------------|
| StyleGAN | 2019 | Mapping network + AdaIN + noise + progressive growing. |
| StyleGAN2 | 2020 | Weight demodulation replaces AdaIN (fixes droplet artifacts); skip/residual architecture; path-length regularization. |
| StyleGAN3 | 2021 | Alias-free convolution + equivariant kernels; eliminates texture sticking to pixel grid. |
| StyleGAN-XL | 2022 | Class-conditional, 1024², ImageNet. |
| R3GAN | 2024 | Rebrands with stronger reg; closes gap to diffusion on FFHQ-1024 with 20x fewer params. |

Pada 2026 StyleGAN3 tetap bawaan untuk (a) fotorealisme domain-sempit pada FPS tinggi, (b) adaptasi-domain few-shot (latih pada dataset baru dengan 100 gambar, bekukan mapping), (c) editing berbasis-inversi (temukan `w` yang merekonstruksi foto nyata, lalu edit `w` itu). Untuk teks-ke-gambar domain-terbuka, ia bukan tool-nya — diffusion-lah.

```figure
gx-stylegan-mapping
```

## Bangun Sendiri

`code/main.py` mengimplementasikan mainan "style-GAN lite" dalam 1-D: MLP mapping, fungsi sintesis yang mengambil vektor konstanta terpelajari dan memodulasinya dengan scale/bias turunan-`w`, dan noise per-lapis. Ia menunjukkan bahwa menyuntikkan `w` via affine-modulation menyamai atau mengalahkan penggabungan `z` ke input generator.

### Langkah 1: mapping network

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Langkah 2: adaptive instance normalization

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

Scale dan bias per-peta-fitur berasal dari `w` via proyeksi linear.

### Langkah 3: noise per-lapis

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

Sigma per-channel bisa dipelajari.

## Jebakan

- **Artefak droplet.** StyleGAN 1 menghasilkan droplet blobby di peta fitur karena AdaIN menolkan mean. Demodulasi bobot StyleGAN 2 memperbaikinya dengan menskalakan bobot convolution sebagai gantinya.
- **Texture sticking.** Tekstur StyleGAN 1 dan 2 mengikuti koordinat piksel, bukan koordinat objek (terlihat saat interpolasi). Convolution bebas-alias StyleGAN 3 memperbaikinya dengan filter sinc berjendela.
- **Cakupan mode.** Truncation `ψ < 0.7` tampak bersih tapi menyampel dari kerucut sempit; pakai `ψ = 1.0` kalau kamu butuh diversitas.
- **Inversi itu lossy.** Menginversi foto nyata ke `W` biasanya dilakukan lewat optimasi atau encoder (e4e, ReStyle, HyperStyle). Hasil melayang setelah banyak iterasi.

## Gunakan

| Use case | Approach |
|----------|----------|
| Photoreal human faces (anime, product, narrow) | StyleGAN3 FFHQ / custom fine-tune |
| Face editing from a photo | e4e inversion + StyleSpace / InterFaceGAN directions |
| Face swap / reenactment | StyleGAN + encoder + blending |
| Avatar pipelines | StyleGAN3 w/ ADA for low-data fine-tune |
| Domain adaptation from a few images | Freeze mapping network, fine-tune synthesis |
| Multi-modal or text-conditioned generation | Don't — use diffusion |

Untuk demo kelas-produk di mana jawabannya adalah "foto wajah seseorang", StyleGAN mengalahkan diffusion pada biaya inference (satu forward pass, <10ms pada 4090) dan ketajaman untuk bar kualitas yang sama.

## Rilis

Simpan `outputs/skill-stylegan-inversion.md`. Skill menerima foto nyata dan mengeluarkan: metode inversi (e4e / ReStyle / HyperStyle), loss laten harapan, budget editing (seberapa jauh di `W` kamu bisa bergerak sebelum artefak), dan daftar arah editing yang terbukti-bagus (usia, ekspresi, pose).

## Latihan

1. **Mudah.** Jalankan `code/main.py` dengan `adain_on=True` dan `adain_on=False`. Bandingkan sebaran output untuk laten tetap vs laten terpeturbasi.
2. **Sedang.** Implementasikan mixing regularization: untuk batch training, hitung `w_a`, `w_b`, dan terapkan `w_a` untuk paruh pertama sintesis dan `w_b` untuk paruh kedua. Apakah decoder mempelajari style yang disentangled?
3. **Sulit.** Ambil model StyleGAN3 FFHQ terlatih (ffhq-1024.pkl). Temukan arah `w` yang mengendalikan "smile" dengan melatih SVM pada sampel berlabel; laporkan seberapa jauh kamu bisa mendorong sebelum identitas melayang.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Mapping network | "The MLP" | `f: Z → W`, 8 layers, decouples latent geometry from data statistics. |
| W space | "The style space" | Output of the mapping network; roughly disentangled. |
| AdaIN | "Adaptive instance norm" | Normalize feature map, then scale + shift by `w`-projection. |
| Truncation trick | "Psi" | `w = mean + ψ·(w - mean)`, ψ<1 trades diversity for quality. |
| Path-length regularization | "PL reg" | Penalizes large changes in image per unit change in `w`; makes `W` smoother. |
| Weight demodulation | "The StyleGAN2 fix" | Normalize conv weights instead of activations; kills droplet artifacts. |
| Alias-free | "StyleGAN3's trick" | Windowed sinc filters; eliminates texture sticking to the pixel grid. |
| Inversion | "Find w for a real image" | Optimize or encode `x → w` so `G(w) ≈ x`. |

## Catatan produksi: mengapa StyleGAN masih dikirim pada 2026

StyleGAN3 pada 4090 membangkitkan wajah FFHQ 1024² dalam di bawah 10 ms — `num_steps = 1`, tanpa decode VAE, tanpa pass cross-attention. Dalam istilah produksi ini adalah latensi lantai untuk generator gambar apa pun. Pipeline SDXL + VAE-decode 50 langkah pada resolusi yang sama adalah ~3 detik. Itu adalah kesenjangan **300×**, dan untuk produk domain-sempit (layanan avatar, pipeline dokumen ID, pembangkitan wajah stok) ia menang pada TCO.

Dua konsekuensi operasional:

- **Tanpa scheduler, tanpa batcher.** Batch statis pada okupansi target optimal. Continuous batching (esensial untuk LLM dan diffusion) memberi nol manfaat karena setiap request memakan FLOPs yang sama.
- **Truncation `ψ` adalah knob keamanan.** `ψ < 0.7` menyampel dari kerucut sempit rentang mapping network. Ini satu-satunya tuas yang dimiliki lapisan serving atas variansi sampel. Turunkan `ψ` pada beban puncak, naikkan untuk pengguna premium.

## Bacaan Lanjutan

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — inversi e4e.
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL.
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — resep GAN minimal modern.
