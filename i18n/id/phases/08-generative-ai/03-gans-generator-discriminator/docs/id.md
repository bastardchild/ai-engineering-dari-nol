# GAN — Generator vs Diskriminator

> Trik Goodfellow pada 2014 adalah melewatkan densitas sepenuhnya. Dua network. Satu membuat yang palsu. Satu menangkapnya. Mereka bertarung sampai yang palsu tak terbedakan dari yang nyata. Seharusnya tidak bekerja. Sering kali tidak. Ketika berhasil, sampelnya masih yang paling tajam di literatur untuk domain sempit.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 08 (Optimizers), Phase 8 · 02 (VAE)
**Time:** ~75 minutes

## Masalahnya

VAE menghasilkan sampel blur karena loss decoder MSE-nya Bayes-optimal untuk citra *rata-rata* — dan rata-rata banyak digit yang masuk akal adalah digit kabur. Kamu mau loss yang menghargai *plausibilitas*, bukan kedekatan per-piksel dengan target mana pun. Tidak ada closed-form untuk plausibilitas. Kamu harus mempelajarinya.

Ide Goodfellow: latih classifier `D(x)` untuk membedakan gambar nyata dari yang palsu. Latih generator `G(z)` untuk menipu `D`. Sinyal loss untuk `G` adalah apa pun yang saat ini dianggap `D` membuat sesuatu tampak nyata. Sinyal ini terupdate saat `G` membaik, mengejar target bergerak. Kalau kedua network konvergen, `G` telah mempelajari distribusi data tanpa pernah menulis `log p(x)`.

Ini adalah training adversarial. Matematikanya adalah game minimax:

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

Pada 2026 GAN bukan lagi generator SOTA (diffusion dan flow matching memakan mahkota itu). Tapi StyleGAN 2/3 tetap model wajah tertajam yang pernah dikirim, diskriminator GAN dipakai sebagai *perceptual loss* dalam training diffusion, dan training adversarial menggerakkan distilasi 1-langkah yang cepat (SDXL-Turbo, SD3-Turbo, LCM) yang membuatmu bisa mengirim diffusion real-time.

## Konsepnya

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**Generator `G(z)`.** Memetakan vektor noise `z ~ N(0, I)` ke sampel `x̂`. Network berbentuk-decoder (dense atau transposed conv).

**Diskriminator `D(x)`.** Memetakan sampel ke probabilitas (atau skor) skalar. Nyata → 1, palsu → 0.

**Loss.** Dua update bergantian:

- **Latih `D`:** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`. Cross-entropy biner pada nyata=1, palsu=0.
- **Latih `G`:** `loss_G = -log D(G(z))`. Ini adalah bentuk *non-saturating* yang dipakai Goodfellow (`log(1 - D(G(z)))` orisinal saturasi dan membunuh gradien ketika `D` yakin).

**Loop training.** Satu langkah `D`, satu langkah `G`. Ulangi.

**Mengapa bekerja.** Kalau `G` cocok sempurna dengan `p_data`, maka `D` tidak bisa lebih baik dari tebakan dan mengeluarkan 0.5 di mana-mana; `G` tidak mendapat gradien lagi. Ekuilibrium.

**Mengapa rusak.** Mode collapse (`G` menemukan satu mode yang tak bisa diklasifikasi `D` dan mencetaknya selamanya), gradien menghilang (`D` belajar terlalu cepat dan `log D` saturasi), ketidakstabilan training (learning rate, ukuran batch, apa pun).

## Varian yang membuat GAN bekerja

| Year | Innovation | Fix |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv, batch norm, LeakyReLU — the first stable architecture. |
| 2017 | WGAN, WGAN-GP | Replace BCE with Wasserstein distance + gradient penalty. Fixes vanishing gradient. |
| 2017 | Spectral normalization | Lipschitz-bound the discriminator. Still used in 2026 discriminators. |
| 2018 | Progressive GAN | Train low-res first, add layers. First megapixel results. |
| 2019 | StyleGAN / StyleGAN2 | Mapping network + adaptive instance norm. State of the art for fixed-domain photorealism. |
| 2021 | StyleGAN3 | Alias-free, translation-equivariant — still the face gold standard in 2026. |
| 2022 | StyleGAN-XL | Conditional, class-aware, larger scale. |
| 2024 | R3GAN | Rebrands with stronger regularization; works on 1024² without tricks. |

```figure
gan-minimax
```

## Bangun Sendiri

`code/main.py` melatih GAN mungil pada data 1-D: campuran dua Gaussian. Generator dan diskriminator adalah MLP satu-hidden-layer. Kita mengimplementasikan forward, backward, dan loop minimax dengan tangan. Tujuannya melihat dua mode kegagalan kunci (mode collapse + gradien menghilang) saat terjadi.

### Langkah 1: loss non-saturating

Loss Goodfellow vanilla `log(1 - D(G(z)))` menuju 0 ketika D mengklasifikasi palsu milik G sebagai palsu dengan keyakinan tinggi. Pada titik itu gradien untuk G pada dasarnya nol — G tidak bisa membaik. Bentuk non-saturating `-log D(G(z))` punya asimtot sebaliknya: ia meledak ketika D yakin, memberi G sinyal kuat.

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Langkah 2: satu langkah diskriminator per langkah generator

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

Fake segar untuk G, kalau tidak gradiennya basi.

### Langkah 3: waspadai mode collapse

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

Gejala kanonis: satu dari dua mode nyata berhenti dibangkitkan. Diskriminator berhenti mengoreksinya karena ia tidak pernah dilihat sebagai palsu.

## Jebakan

- **Diskriminator terlalu kuat.** Potong learning rate D 2-5x, atau tambahkan noise instance/lapis. Kalau D mencapai akurasi >95%, G mati.
- **Generator menghafal satu mode.** Tambahkan noise ke input D, pakai lapis minibatch-discriminator, atau beralih ke WGAN-GP.
- **Batch norm membocorkan statistik.** Batch nyata + batch palsu mengalir melalui lapis BN yang sama mencampur statistiknya. Pakai instance norm atau spectral norm sebagai gantinya.
- **Gaming inception-score.** FID dan IS berisik pada jumlah sampel rendah. Pakai ≥10k sampel saat eval.
- **Sampling sekali-jalan adalah kebohongan untuk task kondisional.** Kamu tetap butuh skala CFG, trik truncation, dan re-sampling untuk mendapat output yang bisa dipakai.

## Gunakan

Stack GAN 2026:

| Situation | Pick |
|-----------|------|
| Photoreal human faces, fixed pose | StyleGAN3 (sharpest, smallest) |
| Anime / stylized faces | StyleGAN-XL or Stable Diffusion LoRA |
| Image-to-image translation | Pix2Pix / CycleGAN (Phase 8 · 04) or ControlNet (Phase 8 · 08) |
| Fast 1-step text-to-image | Adversarial distillation of diffusion (SDXL-Turbo, SD3-Turbo) |
| Perceptual loss inside a diffusion trainer | Small GAN discriminator on image crops |
| Anything multi-modal, open-ended | Don't — use diffusion or flow matching |

GAN itu tajam tapi sempit. Begitu domain-mu terbuka — foto, prompt teks sembarang, video — beralih ke diffusion. Trik adversarial hidup sebagai komponen (perceptual loss, distilasi), bukan generator mandiri.

## Rilis

Simpan `outputs/skill-gan-debugger.md`. Skill menerima run GAN yang gagal (kurva loss, grid sampel, ukuran dataset) dan mengeluarkan daftar peringkat penyebab yang mungkin, perbaikan satu-baris, dan protokol rerun.

## Latihan

1. **Mudah.** Jalankan `code/main.py` dengan setting standar. Lalu atur `D_LR = 5 * G_LR` dan jalankan ulang. Seberapa cepat loss G runtuh menjadi konstanta?
2. **Sedang.** Ganti loss BCE Goodfellow dengan loss WGAN: `loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]`, dan clip bobot D ke `[-0.01, 0.01]`. Apakah training lebih stabil? Bandingkan konvergensi wall-clock.
3. **Sulit.** Perluas contoh 1-D ke data 2-D (campuran 8 Gaussian pada cincin). Lacak berapa dari 8 mode yang ditangkap generator pada langkah 1k, 5k, 10k. Implementasikan minibatch discrimination dan ukur ulang.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Generator | "G" | Noise-to-sample network, `G: z → x̂`. |
| Discriminator | "D" | Classifier `D: x → [0, 1]`, real vs fake. |
| Minimax | "The game" | `min_G max_D` of a joint objective. |
| Non-saturating loss | "The fix" | Use `-log D(G(z))` for G instead of `log(1 - D(G(z)))`. |
| Mode collapse | "G memorized one thing" | Generator produces few distinct outputs despite diverse data. |
| WGAN | "Wasserstein" | Replace BCE with Earth-Mover distance + gradient penalty; smoother gradient. |
| Spectral norm | "Lipschitz trick" | Constrain D's weight norms to bound its slope; stabilizes training. |
| StyleGAN | "The one that works" | Mapping network + AdaIN; best-in-class for faces, still in 2026. |

## Catatan produksi: inference sekali-jalan adalah keunggulan abadi GAN

GAN tidak lagi menang pada kualitas sampel untuk pembangkitan domain-terbuka, tapi mereka tetap menang pada biaya inference. Dalam kosakata literatur production-inference, GAN punya:

- **Tanpa tahap prefill, tanpa tahap decode.** Satu forward pass `G(z)`. TTFT ≈ latensi total.
- **Tanpa tekanan KV-cache.** Satu-satunya state adalah bobot. Ukuran batch dibatasi memori aktivasi, bukan cache.
- **Continuous batching trivial.** Karena setiap request memakan FLOPs tetap yang sama, batch statis pada okupansi target server biasanya optimal. Tanpa scheduler in-flight.

Inilah mengapa distilasi GAN (SDXL-Turbo, SD3-Turbo, ADD, LCM) adalah teknik dominan untuk teks-ke-gambar cepat pada 2026: ia meruntuhkan pipeline diffusion 20-50 langkah menjadi 1-4 forward pass ala-GAN sambil menjaga distribusi basis diffusion. Loss adversarial bertahan sebagai knob waktu-training untuk mengubah generator lambat menjadi yang cepat.

## Bacaan Lanjutan

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — makalah GAN orisinal.
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — arsitektur stabil pertama.
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN.
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo.
