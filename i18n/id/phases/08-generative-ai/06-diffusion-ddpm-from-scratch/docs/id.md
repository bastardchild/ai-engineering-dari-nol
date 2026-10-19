# Model Diffusion — DDPM dari Nol

> Ho, Jain, Abbeel (2020) memberi bidang ini resep yang tak bisa ditinggalkan. Hancurkan data dengan noise dalam seribu langkah kecil. Latih satu neural net untuk memprediksi noise. Balikkan prosesnya saat inference. Hari ini setiap model citra, video, 3D, dan musik arus utama berjalan pada loop ini, mungkin dengan flow matching atau trik consistency di atasnya.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 8 · 02 (VAE)
**Time:** ~75 minutes

## Masalahnya

Kamu mau sampler untuk `p_data(x)`. GAN memainkan game minimax yang sering divergen. VAE menghasilkan sampel blur dari decoder Gaussian. Yang benar-benar kamu mau adalah objektif training yang (a) satu loss stabil (tanpa saddle point, tanpa minimax), (b) lower bound atas `log p(x)` (supaya kamu punya likelihood), dan (c) sampel yang menyamai kualitas SOTA.

Sohl-Dickstein et al. (2015) punya jawaban teoretis: definisikan rantai Markov `q(x_t | x_{t-1})` yang bertahap menambahkan noise Gaussian, dan latih rantai reverse `p_θ(x_{t-1} | x_t)` untuk mendenoise. Ho, Jain, Abbeel (2020) menunjukkan loss bisa disederhanakan menjadi satu baris — prediksi noise — dan merapikan matematikanya. Pada 2020 ini keingintahuan. Pada 2021 ia menghasilkan sampel state-of-the-art. Pada 2022 ia menjadi Stable Diffusion. Pada 2026 ia adalah substrat.

## Konsepnya

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**Proses forward `q`.** Tambahkan noise Gaussian dalam `T` langkah kecil. Bentuk closed — alasan matematikanya tractable — adalah langkah kumulatif juga Gaussian:

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

dengan `α̅_t = ∏_{s=1..t} (1 - β_s)` untuk schedule `β_t`. Pilih `β_t` dari 1e-4 ke 0.02 secara linear selama T=1000 langkah dan `x_T` kira-kira `N(0, I)`.

**Proses reverse `p_θ`.** Pelajari neural net `ε_θ(x_t, t)` yang memprediksi noise yang ditambahkan. Diberikan `x_t`, denoise dengan:

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

dengan `σ_t` adalah `sqrt(β_t)` atau variansi terpelajari. Ekspresinya jelek tapi itu hanya aljabar — memecahkan `x_{t-1}` dari posterior `q(x_{t-1} | x_t, x_0)` dan mensubstitusi `x_0` dengan estimasi prediksi-noise-nya.

**Loss training.**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

Sampel `x_0` dari data, pilih `t` acak, sampel `ε ~ N(0, I)`, hitung `x_t` ber-noise dalam sekali jalan via closed form, dan regresi pada noise. Satu loss, tanpa minimax, tanpa KL, tanpa trik reparametrisasi.

**Sampling.** Mulai `x_T ~ N(0, I)`. Iterasikan langkah reverse dari `t = T` ke `1`. Selesai.

## Mengapa bekerja

Tiga intuisi:

1. **Denoising itu mudah; membangkitkan itu sulit.** Pada `t=T`, data adalah noise murni — net harus memecahkan problem trivial. Pada `t=0`, net hanya harus membersihkan beberapa piksel. Pada `t` intermediate, problemnya sulit tapi net punya banyak gradien mengalir melalui bobot yang sama dari setiap level noise.

2. **Score matching yang menyamar.** Vincent (2011) membuktikan memprediksi noise setara dengan mengestimasi `∇_x log q(x_t | x_0)`, *skor*-nya. SDE reverse memakai skor ini untuk menaiki gradien densitas — random walk terpandu ke wilayah berprobabilitas-tinggi.

3. **ELBO tereduksi menjadi MSE sederhana.** Lower bound variasional penuh punya suku KL per timestep. Dengan parametrisasi DDPM suku KL itu menyederhanakan menjadi MSE pada prediksi noise dengan koefisien spesifik; Ho membuang koefisiennya (menyebutnya loss "simple") dan kualitas *membaik*.

```figure
diffusion-denoise
```

## Bangun Sendiri

`code/main.py` mengimplementasikan DDPM 1-D. Data adalah campuran dua-mode. "Net"-nya adalah MLP mungil yang mengambil `(x_t, t)` dan mengeluarkan noise prediksi. Training adalah loss satu-baris. Sampling mengiterasikan rantai reverse.

### Langkah 1: schedule forward (closed form)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Langkah 2: sampel `x_t` dalam sekali jalan

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Langkah 3: satu langkah training

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Langkah 4: sampling reverse

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

Untuk problem 1-D dengan 40 timestep dan MLP 24 unit, ini mempelajari campuran dua-mode dalam ~200 epoch.

## Conditioning waktu

Net perlu tahu timestep mana yang ia denoise. Dua opsi standar:

- **Embedding sinusoidal.** Seperti positional encoding Transformer. `embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`. Lewatkan melalui MLP, broadcast ke net.
- **Conditioning Film / group-norm.** Proyeksikan embedding ke scale/bias per-channel (FiLM) di setiap block.

Kode mainan kita memakai sinusoidal → concat. U-Net produksi memakai FiLM.

## Jebakan

- **Schedule sangat penting.** `β` linear adalah bawaan DDPM tapi schedule cosine (Nichol & Dhariwal, 2021) memberi FID lebih baik untuk komputasi yang sama. Ganti schedule kalau kualitas plateau.
- **Embedding timestep itu rapuh.** Melewatkan `t` mentah sebagai float bekerja untuk 1-D mainan tapi gagal untuk gambar; selalu pakai embedding yang benar.
- **V-prediction vs ε-prediction.** Untuk rezim sempit (t sangat kecil atau sangat besar), `ε` punya signal-to-noise buruk. V-prediction (`v = α·ε - σ·x`) lebih stabil; SDXL, SD3, dan Flux memakainya.
- **Classifier-free guidance.** Saat inference, hitung `ε` kondisional maupun tanpa-syarat, lalu `ε_cfg = (1 + w) · ε_cond - w · ε_uncond` dengan `w ≈ 3-7`. Dibahas di Pelajaran 08.
- **1000 langkah itu banyak.** Produksi memakai DDIM (20-50 langkah), DPM-Solver (10-20 langkah), atau distilasi (1-4 langkah). Lihat Pelajaran 12.

## Gunakan

| Role | Typical stack in 2026 |
|------|-----------------------|
| Image pixel-space diffusion (small, toy) | DDPM + U-Net |
| Image latent diffusion | VAE encoder + U-Net or DiT (Lesson 07) |
| Video latent diffusion | Spatiotemporal DiT (Sora, Veo, WAN) |
| Audio latent diffusion | Encodec + diffusion transformer |
| Science (molecules, proteins, physics) | Equivariant diffusion (EDM, RFdiffusion, AlphaFold3) |

Diffusion adalah backbone generatif universal. Flow matching (Pelajaran 13) adalah kompetitor 2024-2026 yang biasanya menang pada kecepatan inference untuk kualitas yang sama.

## Rilis

Simpan `outputs/skill-diffusion-trainer.md`. Skill menerima dataset + budget komputasi dan mengeluarkan: schedule (linear/cosine/sigmoid), target prediksi (ε/v/x), jumlah langkah, skala guidance, keluarga sampler, dan protokol eval.

## Latihan

1. **Mudah.** Ubah T dari 40 menjadi 10 di `code/main.py`. Bagaimana kualitas sampel (histogram visual output) terdegradasi? Pada T berapa struktur dua-mode runtuh?
2. **Sedang.** Beralih dari ε-prediction ke v-prediction. Derivasikan ulang langkah reverse. Bandingkan kualitas sampel final.
3. **Sulit.** Tambahkan classifier-free guidance. Kondisikan pada label kelas `c ∈ {0, 1}`, drop 10% selama training, dan saat sampling pakai `ε = (1+w)·ε_cond - w·ε_uncond`. Ukur conditional-mode-hit rate pada `w = 0, 1, 3, 7`.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Forward process | "Adding noise" | Fixed Markov chain `q(x_t \| x_{t-1})` that destroys the data. |
| Reverse process | "Denoising" | Learned chain `p_θ(x_{t-1} \| x_t)` that reconstructs the data. |
| β schedule | "The noise ladder" | Per-step variance; linear, cosine, or sigmoid. |
| α̅ | "Alpha bar" | Cumulative product `∏(1 - β)`; gives closed-form `x_t` from `x_0`. |
| Simple loss | "MSE on noise" | `\|\|ε - ε_θ(x_t, t)\|\|²`; all variational derivations collapse to this. |
| ε-prediction | "Predict noise" | Output is the noise added; standard DDPM. |
| V-prediction | "Predict velocity" | Output is `α·ε - σ·x`; better conditioning across t. |
| DDPM | "The paper" | Ho et al. 2020; linear β, 1000 steps, U-Net. |
| DDIM | "Deterministic sampler" | Non-Markov sampler, 20-50 steps, same training objective. |
| Classifier-free guidance | "CFG" | Mix conditional and unconditional noise predictions to amplify conditioning. |

## Catatan produksi: inference diffusion adalah problem jumlah-langkah

Makalah DDPM menjalankan T=1000 langkah reverse. Tidak ada yang mengirim itu di produksi. Setiap stack inference nyata memilih satu dari tiga strategi — dan masing-masing memetakan bersih ke pembingkaian produksi tentang "dari mana latensi berasal":

1. **Sampler lebih cepat, model sama.** DDIM (20-50 langkah), DPM-Solver++ (10-20), UniPC (8-16). Pengganti drop-in loop reverse; bobot `ε_θ` terlatih tak tersentuh. Memotong latensi 20-50×.
2. **Distilasi.** Latih murid agar cocok dengan guru dalam langkah lebih sedikit: Progressive Distillation (2 → 1), Consistency Model (sembarang → 1-4), LCM, SDXL-Turbo, SD3-Turbo. Memotong latensi 5-10× lagi, butuh retraining.
3. **Caching dan kompilasi.** `torch.compile(unet, mode="reduce-overhead")`, backend diffusion TensorRT-LLM, attention `xformers`/SDPA, bobot bf16. Memotong latensi per-langkah ~2×. Bertumpuk dengan (1) dan (2).

Untuk server diffusion produksi, percakapan budget sama seperti yang dijelaskan literatur produksi untuk LLM: latensi adalah `num_steps × step_cost + VAE_decode`, throughput adalah `batch_size × (num_steps × step_cost)^-1`. TTFT kecil (satu langkah); TPOT-ekuivalen adalah waktu respons penuh karena pembangkitan gambar bersifat "sekaligus" dari sudut pandang pengguna.

## Bacaan Lanjutan

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — makalah diffusion, jauh sebelum zamannya.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM.
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM, langkah lebih sedikit.
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — schedule cosine, variansi terpelajari.
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — classifier guidance.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — notasi terpadu, resep terbersih.
