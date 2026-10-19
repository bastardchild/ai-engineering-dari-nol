# Probabilitas dan Distribusi

> Probabilitas adalah bahasa yang dipakai AI untuk mengungkapkan ketidakpastian.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Implementasikan PMF dan PDF dari nol untuk distribusi Bernoulli, kategorikal, Poisson, uniform, dan normal
- Hitung nilai harapan, varians, dan pakai Teorema Limit Pusat untuk menjelaskan mengapa Gaussian mendominasi
- Bangun fungsi softmax dan log-softmax dengan trik stabilitas numerik (kurangi max logit)
- Hitung loss cross-entropy dari logit dan hubungkan ke negative log-likelihood

## Masalahnya

Pengklasifikasi mengeluarkan `[0.03, 0.91, 0.06]`. Language model memilih kata berikutnya dari 50.000 kandidat. Diffusion model menghasilkan gambar dengan sampling dari distribusi yang dipelajari. Semua ini adalah probabilitas dalam aksi.

Setiap prediksi yang dibuat model adalah distribusi probabilitas. Setiap fungsi loss mengukur seberapa jauh distribusi prediksi dari yang benar. Setiap langkah training menyesuaikan parameter agar satu distribusi makin mirip yang lain. Tanpa probabilitas, kamu tidak bisa membaca satu pun makalah ML, men-debug satu pun model, atau memahami mengapa loss training-mu NaN.

## Konsepnya

### Event, Ruang Sampel, dan Probabilitas

Ruang sampel S adalah himpunan semua hasil yang mungkin. Event adalah subset dari ruang sampel. Probabilitas memetakan event ke angka antara 0 dan 1.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Tiga aksioma mendefinisikan seluruh probabilitas:
1. P(A) >= 0 untuk event A apa pun
2. P(S) = 1 (sesuatu selalu terjadi)
3. P(A or B) = P(A) + P(B) ketika A dan B tidak bisa terjadi bersamaan

Segala hal lain (teorema Bayes, ekspektasi, distribusi) mengikuti ketiga aturan ini.

### Probabilitas Bersyarat dan Independensi

P(A|B) adalah probabilitas A jika B telah terjadi.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Dua event independen ketika mengetahui yang satu tidak memberi tahu apa pun tentang yang lain:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

Lemparan koin independen. Mengambil kartu tanpa pengembalian tidak.

### Fungsi Massa Probabilitas vs Fungsi Densitas Probabilitas

Variabel acak diskret memiliki probability mass function (PMF). Setiap hasil memiliki probabilitas spesifik yang bisa kamu baca langsung.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

Variabel acak kontinu memiliki probability density function (PDF). Densitas pada satu titik bukanlah probabilitas. Probabilitas berasal dari mengintegralkan densitas pada suatu interval.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

Perbedaan ini penting di ML. Output klasifikasi adalah PMF (pilihan diskret). Ruang laten VAE memakai PDF (kontinu).

### Distribusi Umum

**Bernoulli:** satu percobaan, dua hasil. Memodelkan klasifikasi biner.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Kategorikal:** satu percobaan, k hasil. Memodelkan klasifikasi multi-kelas (output softmax).

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:** semua hasil sama mungkinnya. Dipakai untuk inisialisasi acak.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):** kurva lonceng. Diparameterisasi oleh mean (mu) dan varians (sigma^2).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:** cacahan event langka dalam interval tetap. Memodelkan laju event.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### Nilai Harapan dan Varians

Nilai harapan adalah rata-rata hasil yang terbobot.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

Varians mengukur sebaran di sekitar mean.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

Di ML, nilai harapan muncul sebagai fungsi loss (rata-rata loss atas distribusi data). Varians memberi tahu tentang stabilitas model. Varians tinggi pada gradien berarti training yang berisik.

### Distribusi Joint dan Marginal

Distribusi joint P(X, Y) menggambarkan dua variabel acak secara bersamaan.

Contoh PMF joint (X = cuaca, Y = payung):

| | Y=0 (no umbrella) | Y=1 (umbrella) | Marginal P(X) |
|---|---|---|---|
| X=0 (sun) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (rain) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Marginal P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

Distribusi marginal menjumlahkan variabel lainnya:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

Total baris dan kolom pada tabel di atas adalah marginalnya.

### Mengapa Distribusi Normal Muncul di Mana-Mana

Teorema Limit Pusat: jumlah (atau rata-rata) dari banyak variabel acak independen konvergen ke distribusi normal, apa pun distribusi aslinya.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

Inilah alasannya:
- Error pengukuran kira-kira normal (banyak sumber kecil yang independen)
- Inisialisasi bobot dalam neural network memakai distribusi normal
- Noise gradien dalam SGD kira-kira normal (jumlah dari banyak gradien sampel)
- Distribusi normal adalah distribusi entropi maksimum untuk mean dan varians tertentu

### Log Probabilitas

Probabilitas mentah menyebabkan masalah numerik. Mengalikan banyak probabilitas kecil dengan cepat underflow menjadi nol.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

Log probabilitas memperbaikinya. Perkalian menjadi penjumlahan.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

Aturan:
- log(a * b) = log(a) + log(b)
- log probabilitas selalu <= 0 (karena 0 < P <= 1)
- Makin negatif = makin tidak mungkin
- Loss cross-entropy adalah negative log probability dari kelas yang benar

### Softmax sebagai Distribusi Probabilitas

Neural network mengeluarkan skor mentah (logit). Softmax mengubahnya menjadi distribusi probabilitas yang valid.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Trik softmax: kurangi logit maksimum sebelum mengeksponensiasi untuk mencegah overflow.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax menggabungkan softmax dan log untuk stabilitas numerik. PyTorch memakainya secara internal untuk loss cross-entropy.

### Sampling

Sampling berarti mengambil nilai acak dari sebuah distribusi. Di ML:
- Dropout mengambil sampel acak neuron mana yang di-nol-kan
- Augmentasi data mengambil sampel transformasi acak
- Language model mengambil sampel token berikutnya dari distribusi prediksi
- Diffusion model mengambil sampel noise dan melakukan denoise secara progresif

Sampling dari distribusi arbitrer membutuhkan teknik seperti inverse transform sampling, rejection sampling, atau trik reparameterisasi (dipakai dalam VAE).

```figure
gaussian-pdf
```

## Bangun Sendiri

### Langkah 1: Dasar-dasar probabilitas

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Langkah 2: PMF dan PDF dari nol

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Langkah 3: Nilai harapan dan varians

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Langkah 4: Sampling dari distribusi

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Langkah 5: Softmax dan log probabilitas

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Langkah 6: Demonstrasi Teorema Limit Pusat

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Langkah 7: Visualisasi

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Implementasi lengkap dengan semua visualisasi ada di `code/probability.py`.

## Gunakan

Dengan NumPy dan SciPy, semua hal di atas adalah one-liner:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

Kamu membangun ini dari nol. Sekarang kamu tahu apa yang dilakukan panggilan-panggilan pustaka itu.

## Latihan

1. Implementasikan inverse transform sampling untuk distribusi eksponensial. Verifikasi dengan mengambil sampel 10.000 nilai dan membandingkan histogram dengan PDF yang sebenarnya.

2. Bangun tabel distribusi joint untuk dua dadu yang dimuati. Hitung distribusi marginal dan cek apakah dadunya independen.

3. Hitung loss cross-entropy untuk pengklasifikasi 5 kelas yang mengeluarkan logit `[2.0, 0.5, -1.0, 3.0, 0.1]` ketika kelas yang benar adalah indeks 3. Lalu verifikasi jawabanmu dengan `nn.CrossEntropyLoss` milik PyTorch.

4. Tulis fungsi yang menerima daftar log probabilitas dan mengembalikan sekuens paling mungkin, total log probabilitas, dan probabilitas mentah yang ekuivalen. Uji dengan kalimat 50 kata di mana setiap kata berprobabilitas 0.01.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sample space | "All the possibilities" | The set S of every possible outcome of an experiment |
| PMF | "The probability function" | A function that gives the exact probability of each discrete outcome, summing to 1 |
| PDF | "The probability curve" | A density function for continuous variables. Integrate it over an interval to get probability |
| Conditional probability | "Probability given something" | P(A\|B) = P(A and B) / P(B). The foundation of Bayesian thinking and Bayes' theorem |
| Independence | "They don't affect each other" | P(A and B) = P(A) * P(B). Knowing one event tells you nothing about the other |
| Expected value | "The average" | The probability-weighted sum of all outcomes. The loss function is an expected value |
| Variance | "How spread out" | The expected squared deviation from the mean. High variance = noisy, unstable estimates |
| Normal distribution | "The bell curve" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). Appears everywhere due to the CLT |
| Central Limit Theorem | "Averages become normal" | The mean of many independent samples converges to a normal distribution regardless of the source |
| Joint distribution | "Two variables together" | P(X, Y) describes the probability of every combination of X and Y outcomes |
| Marginal distribution | "Sum out the other variable" | P(X) = sum_y P(X, Y). Recovers one variable's distribution from the joint |
| Log probability | "Log of the probability" | log P(x). Turns products into sums, preventing numerical underflow in long sequences |
| Softmax | "Turn scores into probabilities" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). Maps real-valued logits to a valid probability distribution |
| Cross-entropy | "The loss function" | -sum(p_true * log(p_predicted)). Measures how different two distributions are. Lower is better |
| Logits | "Raw model outputs" | Unnormalized scores before softmax. Named after the logistic function |
| Sampling | "Drawing random values" | Generating values according to a probability distribution. How models generate output |

## Bacaan Lanjutan

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - bukti visual mengapa rata-rata menjadi normal
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - referensi ringkas yang mencakup semua hal di sini dan lebih banyak lagi
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - mengapa stabilitas numerik penting dan cara mencapainya
