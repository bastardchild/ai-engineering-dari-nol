# Sampling Methods

> Sampling adalah cara AI menjelajahi ruang kemungkinan.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**Time:** ~120 minutes

## Learning Objectives

- Mengimplementasikan inverse CDF, rejection, dan importance sampling dari nol hanya menggunakan bilangan acak uniform
- Membangun temperature, top-k, dan top-p (nucleus) sampling untuk generasi token language model
- Menjelaskan trik reparameterization dan mengapa ia memungkinkan backpropagation melalui sampling di VAE
- Menjalankan MCMC Metropolis-Hastings untuk sampling dari distribusi target yang tidak ternormalisasi

## Masalahnya

Sebuah language model selesai memproses prompt-mu dan menghasilkan vektor dari 50.000 logits. Satu untuk setiap token di kosakatanya. Sekarang ia harus memilih satu. Bagaimana?

Jika ia selalu memilih token dengan probabilitas tertinggi, setiap respons identik. Deterministik. Membosankan. Jika ia memilih secara acak seragam, keluarannya sampah. Jawabannya ada di suatu tempat di antara ekstrem-ekstrem ini, dan tempat itu dikendalikan oleh sampling.

Sampling tidak terbatas pada generasi teks. Reinforcement learning mengestimasi policy gradient dengan sampling trajektori. VAE mempelajari representasi laten dengan sampling dari distribusi yang dipelajari dan backprop melalui keacakan. Model diffusion menghasilkan gambar dengan sampling noise dan mendenoise secara iteratif. Metode Monte Carlo mengestimasi integral yang tidak punya solusi bentuk tertutup. Algoritma MCMC menjelajahi distribusi posterior dimensi tinggi yang mustahil untuk dienumerasi.

Setiap sistem AI generatif adalah sistem sampling. Strategi sampling menentukan kualitas, keragaman, dan dapat-dikontrolnya keluaran. Pelajaran ini membangun setiap metode sampling utama dari nol, dimulai dari bilangan acak uniform dan berakhir dengan teknik yang menopang LLM dan model generatif modern.

## Konsepnya

### Mengapa Sampling Penting

Sampling muncul dalam empat peran fundamental di seluruh AI dan machine learning:

**Generasi.** Language model, model diffusion, dan GAN semuanya menghasilkan keluaran dengan sampling. Algoritma sampling secara langsung mengendalikan kreativitas, koherensi, dan keragaman. Temperature, top-k, dan nucleus sampling adalah tombol yang diputar para engineer setiap hari.

**Training.** Stochastic gradient descent melakukan sampling mini-batch. Dropout menonaktifkan neuron secara acak. Data augmentation menerapkan transformasi acak. Importance sampling memberi bobot ulang sampel untuk mengurangi varians gradien di reinforcement learning (PPO, TRPO).

**Estimasi.** Banyak kuantitas dalam ML tidak punya solusi bentuk tertutup. Expected loss di atas distribusi data, partition function dari model berbasis energi, evidence dalam inferensi Bayesian. Estimasi Monte Carlo mengaproksimasi semua ini dengan mengrata-ratakan di atas sampel.

**Exploration.** Algoritma MCMC menjelajahi distribusi posterior dalam inferensi Bayesian. Evolutionary strategies melakukan sampling gangguan parameter. Thompson sampling menyeimbangkan exploration dan exploitation di bandit.

Tantangan intinya: kamu hanya bisa melakukan sampling langsung dari distribusi sederhana (uniform, normal). Untuk semua yang lain, kamu membutuhkan metode untuk mengubah sampel sederhana menjadi sampel dari distribusi target-mu.

### Sampling Acak Uniform

Setiap metode sampling dimulai di sini. Pembangkit bilangan acak uniform menghasilkan nilai di [0, 1) di mana setiap sub-interval dengan panjang sama punya probabilitas yang sama.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

Untuk melakukan sampling seragam dari set diskret berisi n item, hasilkan U dan kembalikan floor(n * U). Untuk melakukan sampling dari rentang kontinu [a, b], hitung a + (b - a) * U.

Wawasan kuncinya: satu bilangan acak uniform berisi persis jumlah keacakan yang tepat untuk menghasilkan satu sampel dari distribusi mana pun. Triknya adalah menemukan transformasi yang tepat.

### Metode Inverse CDF (Inverse Transform Sampling)

Cumulative distribution function (CDF) memetakan nilai ke probabilitas:

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

Inverse CDF memetakan kembali probabilitas ke nilai. Jika U ~ Uniform(0, 1), maka X = F_inverse(U) mengikuti distribusi target.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Contoh distribusi eksponensial:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

Ini bekerja dengan sempurna ketika kamu bisa menuliskan F_inverse dalam bentuk tertutup. Untuk distribusi normal, tidak ada inverse CDF bentuk tertutup, jadi kita memakai metode lain (Box-Muller, atau aproksimasi numerik).

**Versi diskret:** Untuk distribusi diskret, bangun CDF sebagai jumlah kumulatif, hasilkan U, dan temukan indeks pertama di mana jumlah kumulatif melampaui U. Inilah cara kerja `sample_categorical` di Lesson 06.

### Rejection Sampling

Saat kamu tidak bisa meng-invert CDF tapi bisa mengevaluasi PDF target hingga sebuah konstanta, rejection sampling bekerja.

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

Makin ketat batas M, makin tinggi tingkat penerimaannya. Di dimensi rendah (1-3), rejection sampling bekerja dengan baik. Di dimensi tinggi, tingkat penerimaan turun secara eksponensial karena sebagian besar volume proposal ditolak. Ini adalah kutukan dimensi untuk rejection sampling.

**Contoh: sampling dari normal terpotong.** Gunakan proposal uniform di atas rentang yang dipotong. Envelope M adalah maksimum dari PDF normal di rentang tersebut.

**Contoh: sampling dari setengah lingkaran.** Proposal secara seragam di dalam persegi panjang pembungkus. Terima jika titik jatuh di dalam setengah lingkaran. Inilah cara Monte Carlo menghitung pi: tingkat penerimaan sama dengan rasio luas pi/4.

### Importance Sampling

Terkadang kamu tidak membutuhkan sampel dari distribusi target p(x). Kamu perlu mengestimasi sebuah ekspektasi di bawah p(x), dan kamu punya sampel dari distribusi berbeda q(x).

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

Ini kritis di reinforcement learning. Di PPO (Proximal Policy Optimization), kamu mengumpulkan trajektori di bawah kebijakan lama pi_old tapi ingin mengoptimalkan kebijakan baru pi_new. Importance weight-nya adalah pi_new(a|s) / pi_old(a|s). PPO memotong bobot-bobot ini untuk mencegah kebijakan baru menyimpang terlalu jauh dari yang lama.

Variance dari estimator importance sampling bergantung pada seberapa mirip q dengan p. Jika q sangat berbeda dari p, beberapa sampel mendapat bobot sangat besar dan mendominasi estimator. Self-normalized importance sampling membagi dengan jumlah bobot untuk mengurangi masalah ini:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Estimasi Monte Carlo

Estimasi Monte Carlo mengaproksimasi integral dengan mengrata-ratakan sampel acak. Hukum bilangan besar menjamin konvergensi.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

Tingkat galatnya independen dari dimensi. Itulah sebabnya metode Monte Carlo mendominasi di dimensi tinggi di mana integrasi berbasis grid tidak mungkin.

**Mengestimasi pi:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**Mengestimasi ekspektasi:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC): Metropolis-Hastings

MCMC membangun rantai Markov yang distribusi stasionernya adalah distribusi target p(x). Setelah cukup banyak langkah, sampel dari rantai tersebut adalah (kira-kira) sampel dari p(x).

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

Untuk proposal simetris (q(x'|x) = q(x|x')), rasionya menyederhanakan menjadi p(x')/p(x). Ini adalah algoritma Metropolis asli.

**Mengapa ini bekerja.** Aturan penerimaan memastikan detailed balance: probabilitas berada di x dan bergerak ke x' sama dengan probabilitas berada di x' dan bergerak ke x. Detailed balance menyiratkan bahwa p(x) adalah distribusi stasioner dari rantai tersebut.

**Pertimbangan praktis:**
- Burn-in: buang sampel awal sebelum rantai mencapai kesetimbangan
- Thinning: simpan setiap k-th sampel untuk mengurangi autokorelasi
- Skala proposal: terlalu kecil dan rantai bergerak lambat (penerimaan tinggi, eksplorasi lambat); terlalu besar dan sebagian besar proposal ditolak (penerimaan rendah, macet di tempat)
- Tingkat penerimaan optimal untuk proposal Gaussian di dimensi tinggi kira-kira 0.234

### Gibbs Sampling

Gibbs sampling adalah kasus khusus MCMC untuk distribusi multivariat. Alih-alih mengusulkan gerakan di semua dimensi sekaligus, ia memperbarui satu variabel pada satu waktu dari distribusinya yang bersyarat.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs sampling mengharuskan kamu bisa melakukan sampling dari setiap distribusi bersyarat p(x_i | x_{-i}). Ini langsung untuk banyak model:
- Jaringan Bayesian: conditional mengikuti dari struktur graf
- Mixture Gaussian: conditional-nya Gaussian
- Model Ising: conditional setiap spin hanya bergantung pada tetangganya

Tingkat penerimaannya selalu 1 (setiap proposal diterima) karena sampling dari conditional yang tepat secara otomatis memenuhi detailed balance.

**Keterbatasan.** Ketika variabel sangat berkorelasi, Gibbs sampling tercampur lambat karena memperbarui satu variabel pada satu waktu tidak bisa membuat gerakan diagonal besar melalui distribusi.

### Temperature Sampling (Digunakan di LLM)

Language model mengeluarkan logits z_1, ..., z_V untuk setiap token di kosakata. Softmax mengubahnya menjadi probabilitas. Temperature men skalakan ulang logits sebelum softmax:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**Mengapa ini bekerja.** Membagi logits dengan T < 1 memperbesar perbedaan antar logits. Jika z_1 = 2 dan z_2 = 1, membagi dengan T = 0.5 memberi z_1/T = 4 dan z_2/T = 2, membuat celahnya lebih besar. Setelah softmax, token dengan logit tertinggi mendapat porsi jauh lebih besar.

**Dalam praktiknya:**
- T = 0.0: greedy decoding, terbaik untuk Q&A faktual
- T = 0.3-0.7: sedikit kreatif, baik untuk generasi kode
- T = 0.7-1.0: seimbang, baik untuk percakapan umum
- T = 1.0-1.5: penulisan kreatif, brainstorming
- T > 1.5: semakin acak, jarang berguna

Temperature tidak mengubah token mana yang mungkin. Ia mengubah massa probabilitas yang dialokasikan ke setiap token.

### Top-k Sampling

Top-k sampling membatasi himpunan kandidat ke k token dengan probabilitas tertinggi, lalu menormalisasi ulang dan melakukan sampling dari himpunan terbatas itu.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k mencegah model memilih token yang sangat tidak mungkin (salin ketik, omong kosong) yang ada di ekor panjang distribusi kosakata. Masalahnya: k tetap terlepas dari konteks. Ketika model yakin (satu token punya probabilitas 95%), k = 40 masih mengizinkan 39 alternatif. Ketika model tidak yakin (probabilitas tersebar di 1000 token), k = 40 memotong opsi yang masuk akal.

### Top-p (Nucleus) Sampling

Top-p sampling menyesuaikan ukuran himpunan kandidat secara dinamis. Alih-alih menyimpan jumlah token tetap, ia menyimpan himpunan token terkecil yang probabilitas kumulatifnya melampaui p.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

Ketika model yakin, nucleus sampling menyimpan sedikit token (mungkin 2-3). Ketika model tidak yakin, ia menyimpan banyak (mungkin 200). Perilaku adaptif inilah mengapa nucleus sampling umumnya menghasilkan teks yang lebih baik daripada top-k.

**Kombinasi umum:**
- Temperature 0.7 + top-p 0.9: pengaturan serbaguna yang baik
- Temperature 0.0 (greedy): terbaik untuk tugas deterministik
- Temperature 1.0 + top-k 50: pengaturan paper asli Fan et al. (2018)

Top-k dan top-p bisa dikombinasikan. Terapkan top-k terlebih dahulu, lalu top-p pada himpunan yang tersisa.

### Trik Reparameterization (Digunakan di VAE)

Variational autoencoder (VAE) belajar dengan meng-encode input menjadi distribusi di ruang laten, melakukan sampling dari distribusi itu, lalu mendecode sampelnya kembali. Masalahnya: kamu tidak bisa backprop melalui operasi sampling.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Trik reparameterization memisahkan keacakan dari parameter:

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

Ini bekerja karena N(mu, sigma^2) punya distribusi yang sama dengan mu + sigma * N(0, 1). Wawasan kuncinya: pindahkan keacakan ke sumber tanpa parameter (epsilon), lalu ekspresikan sampel sebagai transformasi yang dapat diturunkan dari parameter.

**Dalam training loop VAE:**
1. Encoder mengeluarkan mu dan log(sigma^2) untuk setiap input
2. Sampel epsilon ~ N(0, 1)
3. Hitung z = mu + sigma * epsilon
4. Decode z untuk merekonstruksi input
5. Backprop melalui langkah 4, 3, 2, 1 (mungkin karena langkah 3 dapat diturunkan)

Tanpa trik reparameterization, VAE tidak bisa dilatih dengan backpropagation standar. Wawasan tunggal inilah yang membuat VAE praktis.

### Gumbel-Softmax (Sampling Kategorikal yang Dapat Diturunkan)

Trik reparameterization bekerja untuk distribusi kontinu (Gaussian). Untuk distribusi kategorikal diskret, kita membutuhkan pendekatan berbeda. Gumbel-Softmax menyediakan aproksimasi yang dapat diturunkan untuk sampling kategorikal.

**Trik Gumbel-Max (tidak dapat diturunkan):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax (aproksimasi yang dapat diturunkan):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax menghasilkan relaksasi kontinu dari sampel diskret. Keluarannya adalah vektor probabilitas (soft one-hot) alih-alih hard one-hot. Gradien mengalir melalui softmax. Selama forward pass dalam training, kamu bisa menggunakan estimator "straight-through": gunakan argmax keras untuk forward pass tapi gradien Gumbel-Softmax yang lembut untuk backward pass.

**Aplikasi:**
- Variabel laten diskret di VAE
- Neural architecture search (memilih operasi diskret)
- Mekanisme hard attention
- Reinforcement learning dengan aksi diskret

### Stratified Sampling

Sampling Monte Carlo standar bisa meninggalkan celah di ruang sampel secara kebetulan. Stratified sampling memaksa cakupan merata dengan membagi ruang menjadi strata dan melakukan sampling dari setiap strata.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Stratified sampling selalu punya varians lebih rendah atau sama dengan Monte Carlo standar:

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**Aplikasi:**
- Integrasi numerik (quasi-Monte Carlo)
- Split data training (memastikan keseimbangan kelas di setiap fold)
- Importance sampling dengan stratifikasi (menggabungkan kedua teknik)
- NeRF (Neural Radiance Fields) menggunakan stratified sampling di sepanjang berkas kamera

### Kaitan dengan Model Diffusion

Model diffusion menghasilkan gambar melalui proses sampling. Proses maju menambahkan noise Gaussian ke gambar selama T langkah sampai menjadi noise murni. Proses mundur belajar mendenoise, memulihkan gambar asli langkah demi langkah.

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

Kaitannya dengan metode di pelajaran ini:
- Setiap langkah denoising menggunakan trik reparameterization (sampel noise, terapkan transformasi deterministik)
- Jadwal noise {alpha_t} mengendalikan bentuk annealing temperature
- Training menggunakan estimasi Monte Carlo untuk mengaproksimasi ELBO (evidence lower bound)
- Ancestral sampling dalam model diffusion adalah rantai Markov (setiap langkah hanya bergantung pada status saat ini)

Seluruh proses generasi gambar adalah sampling iteratif: mulai dari noise, dan di setiap langkah, sampling versi yang sedikit lebih bersih yang dikondisikan pada model denoising yang dipelajari.

```figure
monte-carlo-pi
```

## Bangun Sendiri

### Langkah 1: Sampling uniform dan inverse CDF

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

Hasilkan 10.000 sampel eksponensial dan verifikasi bahwa mean-nya 1/lambda.

### Langkah 2: Rejection sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

Gunakan rejection sampling untuk menggambar dari distribusi normal terpotong. Verifikasi bentuknya dengan membuat histogram dari sampel-sampelnya.

### Langkah 3: Importance sampling

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

Estimasi E[X^2] di bawah distribusi normal menggunakan proposal uniform. Bandingkan dengan jawaban yang diketahui (mu^2 + sigma^2).

### Langkah 4: Estimasi Monte Carlo dari pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Langkah 5: MCMC Metropolis-Hastings

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

Lakukan sampling dari distribusi bimodal (mixture dua Gaussian). Visualisasikan trajektori rantainya.

### Langkah 6: Gibbs sampling

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Langkah 7: Temperature sampling

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

Tunjukkan bagaimana temperature mengubah distribusi keluaran untuk sekumpulan logits token.

### Langkah 8: Sampling top-k dan top-p

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Langkah 9: Trik reparameterization

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

Demonstrasikan bahwa gradien mengalir melalui sampel yang di-reparameterisasi tapi tidak melalui sampling langsung.

### Langkah 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

Tunjukkan bagaimana menurunkan temperature membuat keluaran mendekati vektor one-hot.

Implementasi lengkap dengan semua visualisasi ada di `code/sampling.py`.

## Gunakan

Dengan NumPy dan SciPy, versi versi produksi:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

Untuk MCMC skala besar, gunakan pustaka khusus:
- PyMC: pemodelan Bayesian lengkap dengan NUTS (HMC adaptif)
- emcee: ensemble MCMC sampler
- NumPyro/JAX: MCMC dipercepat GPU

Kamu membangun semua ini dari nol. Sekarang kamu tahu apa yang dilakukan panggilan pustaka-pustaka itu.

## Latihan

1. Implementasikan sampling inverse CDF untuk distribusi Cauchy. CDF-nya adalah F(x) = 0.5 + arctan(x)/pi. Hasilkan 10.000 sampel dan plot histogram terhadap PDF yang sebenarnya. Perhatikan ekor-ekor beratnya (nilai ekstrem yang jauh dari pusat).

2. Gunakan rejection sampling untuk menghasilkan sampel dari distribusi Beta(2, 5) menggunakan proposal Uniform(0, 1). Plot sampel yang diterima terhadap Beta PDF yang sebenarnya. Berapa tingkat penerimaan teoretisnya?

3. Estimasi integral sin(x) dari 0 ke pi menggunakan Monte Carlo dengan 1.000, 10.000, dan 100.000 sampel. Bandingkan galat di setiap level. Verifikasi bahwa galat berskala O(1/sqrt(N)).

4. Implementasikan Metropolis-Hastings untuk melakukan sampling dari distribusi 2D p(x, y) yang proporsional dengan exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2). Plot sampel dan trajektori rantainya. bereksperimen dengan simpangan baku proposal yang berbeda.

5. Bangun demo generasi teks lengkap: diberikan kosakata berisi 10 kata dengan logits, hasilkan sekuens 20 token menggunakan (a) greedy, (b) temperature=0.7, (c) top-k=3, (d) top-p=0.9. Bandingkan keragaman keluaran di 5 kali percobaan.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sampling | "Drawing random values" | Generating values according to a probability distribution. The mechanism behind all generative AI |
| Uniform distribution | "All equally likely" | Every value in [a, b] has equal probability density 1/(b-a). The starting point for all sampling methods |
| Inverse CDF | "Probability transform" | F_inverse(U) converts a uniform sample into a sample from any distribution with known CDF. Exact and efficient |
| Rejection sampling | "Propose and accept/reject" | Generate from a simple proposal, accept with probability proportional to target/proposal ratio. Exact but wastes samples |
| Importance sampling | "Reweight samples" | Estimate expectations under p(x) using samples from q(x) by weighting each sample by p(x)/q(x). Core to PPO in RL |
| Monte Carlo | "Average random samples" | Approximate integrals as sample averages. Error O(1/sqrt(N)) regardless of dimension |
| MCMC | "Random walk that converges" | Construct a Markov chain whose stationary distribution is the target. Metropolis-Hastings is the foundational algorithm |
| Metropolis-Hastings | "Accept uphill, sometimes downhill" | Propose moves, accept based on density ratio. Detailed balance ensures convergence to target distribution |
| Gibbs sampling | "One variable at a time" | Update each variable from its conditional distribution holding others fixed. 100% acceptance rate |
| Temperature | "Confidence knob" | Divides logits by T before softmax. T<1 sharpens (more confident), T>1 flattens (more diverse) |
| Top-k sampling | "Keep the k best" | Zero out all but the k highest-probability tokens, renormalize, sample. Fixed candidate set size |
| Nucleus sampling (top-p) | "Keep the probable ones" | Keep the smallest set of tokens whose cumulative probability exceeds p. Adaptive candidate set size |
| Reparameterization trick | "Move randomness outside" | Write z = mu + sigma * epsilon where epsilon ~ N(0,1). Makes sampling differentiable. Essential for VAE training |
| Gumbel-Softmax | "Soft categorical sampling" | Differentiable approximation to categorical sampling using Gumbel noise + softmax with temperature |
| Stratified sampling | "Forced coverage" | Divide sample space into strata, sample from each. Always lower variance than naive Monte Carlo |
| Burn-in | "Warm-up period" | Initial MCMC samples discarded before the chain reaches its stationary distribution |
| Detailed balance | "Reversibility condition" | p(x) * T(x->y) = p(y) * T(y->x). Sufficient condition for p to be the stationary distribution of a Markov chain |
| Diffusion sampling | "Iterative denoising" | Generate data by starting from noise and applying learned denoising steps. Each step is a conditional sampling operation |

## Bacaan Lanjutan

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - tutorial detail tentang fondasi MCMC
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - paper asli Gumbel-Softmax
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - paper nucleus (top-p) sampling
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - paper VAE yang memperkenalkan trik reparameterization
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM menghubungkan sampling dengan generasi gambar
