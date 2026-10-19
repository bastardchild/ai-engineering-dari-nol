# Numerical Stability

> Floating point adalah abstraksi yang bocor. Ia akan menggigitmu saat training, dan kamu tidak akan melihatnya datang.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## Learning Objectives

- Mengimplementasikan softmax dan log-sum-exp yang stabil secara numerik menggunakan trik pengurangan maksimum
- Mengidentifikasi overflow, underflow, dan catastrophic cancellation dalam komputasi floating-point
- Memverifikasi gradien analitis terhadap gradien numerik menggunakan finite difference terpusat
- Menjelaskan mengapa bfloat16 lebih disukai daripada float16 untuk training dan bagaimana loss scaling mencegah gradient underflow

## Masalahnya

Model-mu melatih selama tiga jam, lalu loss-nya menjadi NaN. Kamu menambahkan satu print. Logits-nya baik di step 9.000. Di step 9.001 nilainya `inf`. Di step 9.002 setiap gradien adalah `nan` dan training sudah mati.

Atau: model-mu melatih sampai selesai tapi akurasinya 2% lebih buruk daripada yang diklaim paper. Kamu memeriksa semuanya. Arsitektur cocok. Hyperparameter cocok. Data cocok. Masalahnya adalah paper memakai float32 dan kamu memakai float16 tanpa scaling yang tepat. Tiga puluh dua bit galat pembulatan yang terakumulasi diam-diam menghabisi akurasimu.

Atau: kamu mengimplementasikan cross-entropy loss dari nol. Ia bekerja pada logits kecil. Ketika logits melebihi 100, ia mengembalikan `inf`. Softmax-nya overflow karena `exp(100)` lebih besar dari yang bisa direpresentasikan float32. Setiap framework ML menangani ini dengan trik dua baris. Kamu tidak tahu trik itu ada.

Stabilitas numerik bukan kekhawatiran teoritis. Ini adalah perbedaan antara training run yang berhasil dan yang diam-diam gagal. Setiap bug ML serius yang kamu debug pada akhirnya berujung ke floating point.

## Konsepnya

### IEEE 754: Bagaimana Komputer Menyimpan Bilangan Real

Komputer menyimpan bilangan real sebagai nilai floating point mengikuti standar IEEE 754. Sebuah float punya tiga bagian: bit tanda, exponent, dan mantissa (significand).

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

Mantissa menentukan presisi (berapa banyak digit signifikan). Exponent menentukan rentang (seberapa besar atau kecil sebuah bilangan bisa menjadi).

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 memberimu sekitar 7 digit desimal presisi. Artinya ia bisa membedakan 1.0000001 dan 1.0000002, tapi tidak 1.00000001 dan 1.00000002. Setelah 7 digit, semuanya adalah noise pembulatan.

float16 memberimu sekitar 3 digit. Bilangan terbesar yang bisa ia representasikan adalah 65.504. Itu mengganggu kecil untuk ML di mana logits, gradien, dan aktivasi secara rutin melampaui ini.

bfloat16 adalah jawaban Google atas masalah rentang float16. Ia punya exponent 8-bit yang sama dengan float32 (rentang yang sama, sampai 3.4e38) tapi hanya 7 bit mantissa (presisi lebih rendah dari float16). Untuk melatih neural network, rentang lebih penting daripada presisi, jadi bfloat16 biasanya menang.

### Mengapa 0.1 + 0.2 != 0.3

Bilangan 0.1 tidak bisa direpresentasikan secara tepat dalam binary floating point. Dalam basis 2, ini adalah pecahan berulang:

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 memotong ini menjadi 23 bit mantissa. Nilai yang tersimpan kira-kira 0.100000001490116. Demikian pula, 0.2 disimpan sebagai kira-kira 0.200000002980232. Jumlahnya 0.300000004470348, bukan 0.3.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

Ini penting untuk ML karena:

1. Perbandingan loss seperti `if loss < threshold` bisa memberi jawaban salah
2. Mengakumulasi banyak nilai kecil (update gradien selama ribuan step) menyimpang dari jumlah yang sebenarnya
3. Checksum dan tes reproducibility gagal jika kamu membandingkan float dengan `==`

Perbaikannya: jangan pernah membandingkan float dengan `==`. Gunakan `abs(a - b) < epsilon` atau `math.isclose()`.

### Catastrophic Cancellation

Ketika kamu mengurangkan dua bilangan floating point yang hampir sama, digit signifikan saling menghapus dan yang tersisa adalah noise pembulatan yang dipromosikan ke digit terdepan.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

Itu adalah galat relatif 19% dari satu pengurangan. Di ML, ini terjadi setiap kali kamu:

- Menghitung varians data dengan mean besar: `E[x^2] - E[x]^2` saat E[x] besar
- Mengurangkan log-probability yang hampir sama
- Menghitung gradien finite-difference dengan epsilon terlalu kecil

Perbaikannya: susun ulang formula untuk menghindari pengurangan bilangan besar yang hampir sama. Untuk varians, gunakan algoritma Welford atau center data terlebih dahulu. Untuk log-probability, bekerja di log-space sepanjang jalan.

### Overflow dan Underflow

Overflow terjadi ketika hasil terlalu besar untuk direpresentasikan. Underflow terjadi ketika terlalu kecil (lebih dekat ke nol daripada bilangan positif terkecil yang bisa direpresentasikan).

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

Fungsi `exp()` adalah sumber utama overflow di ML:

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

Fungsi `log()` mengenai arah sebaliknya:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

Di ML, `exp()` muncul di softmax, sigmoid, dan komputasi probabilitas. `log()` muncul di cross-entropy, log-likelihoods, dan KL divergence. Kombinasi `log(exp(x))` adalah ladang ranjau tanpa trik yang tepat.

### Trik Log-Sum-Exp

Menghitung `log(sum(exp(x_i)))` secara langsung berbahaya secara numerik. Jika ada `x_i` yang besar, `exp(x_i)` overflow. Jika semua `x_i` sangat negatif, setiap `exp(x_i)` underflow ke nol dan `log(0)` adalah `-inf`.

Triknya: kurangi nilai maksimum sebelum mengeksponensialkan.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

Mengapa ini bekerja: setelah mengurangi `max(x)`, eksponen terbesar adalah `exp(0) = 1`. Overflow tidak mungkin. Setidaknya satu istilah dalam jumlah adalah 1, jadi jumlahnya setidaknya 1, dan `log(1) = 0`. Underflow ke `-inf` tidak mungkin.

Bukti:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

Atur `c = max(x)` dan overflow tereliminasi.

Trik ini muncul di mana-mana di ML:
- Normalisasi softmax
- Komputasi cross-entropy loss
- Penjumlahan log-probability di model sekuens
- Mixture of Gaussians
- Inferensi variational

### Mengapa Softmax Membutuh Trik Pengurangan Maksimum

Softmax mengubah logits menjadi probabilitas:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Tanpa triknya, logits [100, 101, 102] menyebabkan overflow:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

Dengan triknya, kurangi max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

Probabilitasnya identik. Komputasinya aman. Ini bukan optimasi. Ini adalah syarat kebenaran.

### NaN dan Inf: Deteksi dan Pencegahan

`nan` (Not a Number) dan `inf` (infinity) menyebar secara viral melalui komputasi. Satu `nan` dalam update gradien membuat bobot menjadi `nan`, yang membuat setiap keluaran berikutnya `nan`. Training mati dalam satu step.

Bagaimana `inf` muncul:
- `exp()` dari bilangan positif besar
- Pembagian dengan nol: `1.0 / 0.0`
- Overflow `float32` dalam akumulasi

Bagaimana `nan` muncul:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` dari bilangan negatif
- `log()` dari bilangan negatif
- Aritmetika apa pun yang melibatkan `nan` yang sudah ada

Deteksi:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

Strategi pencegahan:

1. Clamp masukan ke `exp()`: `exp(clamp(x, -80, 80))`
2. Tambahkan epsilon ke penyebut: `x / (y + 1e-8)`
3. Tambahkan epsilon di dalam `log()`: `log(x + 1e-8)`
4. Gunakan implementasi stabil (log-sum-exp, stable softmax)
5. Gradient clipping untuk mencegah ledakan bobot
6. Periksa `nan`/`inf` setelah setiap forward pass saat debugging

### Pemeriksaan Gradien Numerik

Gradien analitis (dari backpropagation) bisa punya bug. Pemeriksaan gradien numerik memverifikasinya dengan menghitung gradien menggunakan finite differences.

Formula difference terpusat:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

Ini akurat hingga O(h^2), jauh lebih baik daripada forward difference `(f(x+h) - f(x)) / h` yang hanya O(h).

Memilih h: terlalu besar dan aproksimasinya salah. Terlalu kecil dan catastrophic cancellation menghancurkan jawabannya. `h = 1e-5` sampai `1e-7` adalah yang umum.

Pemeriksaannya: hitung perbedaan relatif antara gradien analitis dan numerik.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

Aturan praktis:
- relative_error < 1e-7: sempurna, gradien benar
- relative_error < 1e-5: dapat diterima, mungkin benar
- relative_error > 1e-3: ada yang salah
- relative_error > 1: gradien sepenuhnya salah

Selalu periksa gradien saat mengimplementasikan layer atau fungsi loss baru. PyTorch menyediakan `torch.autograd.gradcheck()` untuk ini.

### Mixed Precision Training

GPU modern punya perangkat keras khusus (Tensor Cores) yang menghitung perkalian matriks float16 2-8x lebih cepat dari float32. Mixed precision training memanfaatkan ini:

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

Masalah dengan training float16 murni: gradien sering sangat kecil (1e-8 atau lebih kecil). Float16 meng-underflowkan apa pun di bawah ~6e-8 menjadi nol. Model-mu berhenti belajar karena semua update gradien nol.

Perbaikannya adalah loss scaling:

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Dynamic loss scaling menyesuaikan faktor skala secara otomatis. Mulai dengan nilai besar (65536). Jika gradien overflow ke `inf`, bagi dua. Jika N step berlalu tanpa overflow, gandakan.

### bfloat16 vs float16: Mengapa bfloat16 Menang untuk Training

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 punya presisi lebih banyak (10 bit mantissa vs 7) tapi rentang terbatas (maks ~65.504). bfloat16 punya presisi lebih sedikit tapi rentang yang sama dengan float32 (maks ~3.4e38).

Untuk melatih neural network:

- Aktivasi dan logits secara rutin melampaui 65.504 selama spike training. float16 overflow; bfloat16 menanganinya.
- Loss scaling diperlukan dengan float16 tapi biasanya tidak perlu dengan bfloat16 karena rentangnya mencakup spektrum magnitudo gradien.
- bfloat16 adalah pemotongan sederhana dari float32: buang 16 bit bawah dari mantissa. Konversi trivial dan tanpa kehilangan di exponent-nya.

float16 lebih disukai untuk inference di mana nilai terbatas dan presisi lebih penting. bfloat16 lebih disukai untuk training di mana rentang lebih penting. Itulah mengapa TPU dan GPU NVIDIA modern (A100, H100) punya dukungan native bfloat16.

### Gradient Clipping

Exploding gradients terjadi ketika gradien tumbuh secara eksponensial melalui banyak layer (umum di RNN, jaringan dalam, dan transformer). Satu gradien besar bisa merusak semua bobot dalam satu step.

Dua jenis clipping:

**Clip by value:** clamp setiap elemen gradien secara independen.

```
grad = clamp(grad, -max_val, max_val)
```

Sederhana tapi bisa mengubah arah vektor gradien.

**Clip by norm:** skalakan seluruh vektor gradien agar norm-nya tidak melebihi ambang.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Mempertahankan arah gradien. Inilah yang dilakukan `torch.nn.utils.clip_grad_norm_()`. Ini adalah pilihan standar.

Nilai tipikal: `max_norm=1.0` untuk transformer, `max_norm=0.5` untuk RL, `max_norm=5.0` untuk jaringan yang lebih sederhana.

Gradient clipping bukan hack. Ini adalah mekanisme keselamatan. Tanpanya, satu batch outlier bisa menghasilkan gradien yang cukup besar untuk merusak berminggu-minggu training.

### Layer Normalization sebagai Stabilizer Numerik

Batch normalization, layer normalization, dan RMS normalization biasanya disajikan sebagai regularizer yang membantu training konvergen. Mereka juga adalah stabilizer numerik.

Tanpa normalisasi, aktivasi bisa tumbuh atau menyusut secara eksponensial melalui layer:

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Normalisasi meng-center ulang dan menskalakan ulang aktivasi di setiap layer:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon` (biasanya 1e-5) mencegah pembagian dengan nol ketika semua aktivasi identik. Parameter yang dipelajari `gamma` dan `beta` memungkinkan jaringan memulihkan skala apa pun yang ia butuhkan.

Ini menjaga nilai tetap di rentang yang aman secara numerik di seluruh jaringan, mencegah overflow di forward pass dan ledakan gradien di backward pass.

### Bug Numerik ML yang Umum

**Bug: Loss menjadi NaN setelah beberapa epoch.**
Penyebab: logits tumbuh terlalu besar, softmax overflow. Atau learning rate terlalu tinggi dan bobot divergen.
Perbaikan: gunakan stable softmax (pengurangan maksimum), kurangi learning rate, tambahkan gradient clipping.

**Bug: Loss tertahan di log(num_classes).**
Penyebab: keluaran model adalah probabilitas yang hampir seragam. Sering berarti gradien vanishing atau model tidak belajar sama sekali.
Perbaikan: periksa bahwa label data benar, verifikasi fungsi loss, periksa ReLU yang mati.

**Bug: Akurasi validasi 1-3% lebih rendah dari yang diharapkan.**
Penyebab: mixed precision tanpa loss scaling yang tepat. Gradient underflow diam-diam mengosongkan update kecil.
Perbaikan: aktifkan dynamic loss scaling, atau beralih ke bfloat16.

**Bug: Norm gradien 0.0 untuk beberapa layer.**
Penyebab: neuron ReLU yang mati (semua masukan negatif), atau underflow float16.
Perbaikan: gunakan LeakyReLU atau GELU, gunakan gradient scaling, periksa inisialisasi bobot.

**Bug: Model bekerja di satu GPU tapi memberi hasil berbeda di GPU lain.**
Penyebab: urutan akumulasi floating point non-deterministik. Reduksi paralel GPU menjumlahkan dalam urutan berbeda di perangkat keras berbeda, dan penjumlahan floating point tidak asosiatif.
Perbaikan: terima perbedaan kecil (1e-6), atau atur `torch.use_deterministic_algorithms(True)` dan terima penalti kecepatan.

**Bug: `exp()` mengembalikan `inf` dalam komputasi loss.**
Penyebab: logits mentah dilewatkan ke `exp()` tanpa trik pengurangan maksimum.
Perbaikan: gunakan `torch.nn.functional.log_softmax()` yang mengimplementasikan log-sum-exp secara internal.

**Bug: Training divergen setelah beralih dari float32 ke float16.**
Penyebab: float16 tidak bisa merepresentasikan magnitudo gradien di bawah 6e-8 atau aktivasi di atas 65.504.
Perbaikan: gunakan mixed precision dengan loss scaling (AMP), atau gunakan bfloat16 sebagai gantinya.

```figure
logsumexp-stability
```

## Bangun Sendiri

### Langkah 1: Mendemonstrasikan batas presisi floating point

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Langkah 2: Implementasikan softmax naive vs stabil

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Langkah 3: Implementasikan log-sum-exp stabil

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Langkah 4: Implementasikan cross-entropy stabil

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Langkah 5: Pemeriksaan gradien

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Gunakan

### Simulasi mixed precision

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient clipping

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### Deteksi NaN/Inf

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

Lihat `code/numerical.py` untuk implementasi lengkap dengan semua edge case yang didemonstrasikan.

## Rilis

Pelajaran ini menghasilkan:
- `code/numerical.py` dengan softmax stabil, log-sum-exp, cross-entropy, pemeriksaan gradien, dan simulasi mixed precision
- `outputs/prompt-numerical-debugger` untuk mendiagnosis NaN/Inf dan masalah numerik dalam training

Implementasi stabil ini muncul kembali di Phase 3 saat membangun training loop dan di Phase 4 saat mengimplementasikan mekanisme attention.

## Latihan

1. **Catastrophic cancellation.** Hitung varians dari [1000000.0, 1000001.0, 1000002.0] menggunakan formula naive `E[x^2] - E[x]^2` dalam float32. Lalu hitung menggunakan algoritma online Welford. Bandingkan galatnya dengan varians yang sebenarnya (0.6667).

2. **Pemburuan presisi.** Temukan nilai float32 positif terkecil `x` sehingga `1.0 + x == 1.0` di Python. Ini adalah machine epsilon. Verifikasi bahwa nilainya cocok dengan `numpy.finfo(numpy.float32).eps`.

3. **Edge case log-sum-exp.** Uji fungsi `logsumexp_stable`-mu dengan: (a) semua nilai sama, (b) satu nilai jauh lebih besar dari yang lain, (c) semua nilai sangat negatif (-1000). Verifikasi bahwa ia memberi hasil benar di mana versi naive gagal.

4. **Pemeriksaan gradien pada layer neural network.** Implementasikan satu layer linear `y = Wx + b` dan backward pass analitisnya. Gunakan `numerical_gradient` untuk memverifikasi kebenaran untuk matriks bobot 3x2.

5. **Eksperimen loss scaling.** Simulasikan training dengan float16: buat gradien acak di rentang [1e-9, 1e-3], konversi ke float16, dan ukur berapa fraksi yang menjadi nol. Lalu terapkan loss scaling (kalikan dengan 1024), konversi ke float16, skalakan kembali, dan ukur fraksi nolnya lagi.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| IEEE 754 | "The float standard" | International standard defining binary floating point formats, rounding rules, and special values (inf, nan). Every modern CPU and GPU implements it. |
| Machine epsilon | "The precision limit" | The smallest value e such that 1.0 + e != 1.0 in a given float format. For float32, it is about 1.19e-7. |
| Catastrophic cancellation | "Precision loss from subtraction" | When subtracting nearly equal floating point numbers, significant digits cancel and rounding noise dominates the result. |
| Overflow | "Number too big" | A result exceeds the maximum representable value and becomes inf. exp(89) overflows float32. |
| Underflow | "Number too small" | A result is closer to zero than the smallest representable positive number and becomes 0.0. exp(-104) underflows float32. |
| Log-sum-exp trick | "Subtract the max first" | Computing log(sum(exp(x))) by factoring out exp(max(x)) to prevent overflow and underflow. Used in softmax, cross-entropy, and log-probability math. |
| Stable softmax | "Softmax that does not explode" | Subtracting max(logits) before exponentiating. Numerically identical result, no overflow possible. |
| Gradient checking | "Verify your backprop" | Comparing analytical gradients from backpropagation against numerical gradients from finite differences to catch implementation bugs. |
| Mixed precision | "Float16 forward, float32 backward" | Using lower-precision floats for speed-critical operations and higher-precision floats for numerically sensitive operations. Typical speedup is 2-3x. |
| Loss scaling | "Prevent gradient underflow" | Multiplying the loss by a large constant before backprop so gradients stay in float16's representable range, then dividing by the same constant before weight updates. |
| bfloat16 | "Brain floating point" | Google's 16-bit format with 8 exponent bits (same range as float32) and 7 mantissa bits (less precision than float16). Preferred for training. |
| Gradient clipping | "Cap the gradient norm" | Scaling the gradient vector so its norm does not exceed a threshold. Prevents exploding gradients from ruining weights. |
| NaN | "Not a Number" | Special float value from undefined operations (0/0, inf-inf, sqrt(-1)). Propagates through all subsequent arithmetic. |
| Inf | "Infinity" | Special float value from overflow or division by zero. Can combine to produce NaN (inf - inf, inf * 0). |
| Numerical gradient | "Brute force derivative" | Approximating a derivative by evaluating f(x+h) and f(x-h) and dividing by 2h. Slow but reliable for verification. |

## Bacaan Lanjutan

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- referensi definitif, padat tapi lengkap
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- paper NVIDIA yang memperkenalkan loss scaling untuk training float16
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- panduan praktis tentang mixed precision di PyTorch
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- mengapa Google memilih format ini untuk TPU
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- algoritma untuk mengurangi galat pembulatan dalam penjumlahan floating point
