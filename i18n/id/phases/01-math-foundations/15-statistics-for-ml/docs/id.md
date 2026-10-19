# Statistics for Machine Learning

> Statistik adalah cara kamu mengetahui apakah model-mu benar-benar bekerja atau hanya beruntung.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**Time:** ~120 minutes

## Learning Objectives

- Menghitung statistik deskriptif, korelasi Pearson/Spearman, dan matriks kovarians dari nol
- Melakukan uji hipotesis (t-test, chi-squared) dan menafsirkan p-value serta confidence interval dengan benar
- Menggunakan bootstrap resampling untuk membangun confidence interval untuk metrik apa pun tanpa asumsi distribusional
- Membedakan signifikansi statistik dari signifikansi praktis menggunakan ukuran effect size

## Masalahnya

Kamu melatih dua model. Model A skornya 0.87 di test set-mu. Model B skornya 0.89. Kamu deploy Model B. Tiga minggu kemudian, metrik produksi lebih buruk dari sebelumnya. Apa yang terjadi?

Model B sebenarnya tidak mengungguli Model A. Selisih 0.02 itu noise. Test set-mu terlalu kecil, atau variansnya terlalu tinggi, atau keduanya. Kamu merilis keacakan yang menyamar sebagai peningkatan.

Ini terus-menerus terjadi. Pengguncangan leaderboard Kaggle. Paper yang gagal direproduksi. A/B test yang menyatakan pemenang berdasarkan beberapa ratus sampel. Akar penyebabnya selalu sama: seseorang melewatkan statistiknya.

Statistik memberimu alat untuk membedakan sinyal dari noise. Ia memberitahu kapan sebuah perbedaan itu nyata, seberapa yakin kamu, dan berapa banyak data yang kamu butuhkan sebelum kamu bisa mempercayai sebuah hasil. Setiap pipeline ML, setiap perbandingan model, setiap eksperimen membutuhkan statistik. Tanpanya, kamu menebak.

## Konsepnya

### Statistik Deskriptif: Meringkas Data-mu

Sebelum kamu memodelkan apa pun, kamu perlu tahu seperti apa data-mu. Statistik deskriptif mengompres dataset menjadi beberapa angka yang menangkap bentuknya.

**Ukuran tendensi sentral** menjawab "di mana titik tengahnya?"

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

Mean adalah titik keseimbangan. Median adalah tanda pertengahan. Ketika keduanya menyimpang, distribusimu miring. Distribusi pendapatan punya mean >> median (right skew dari miliarder). Distribusi loss selama training sering punya mean << median (left skew dari sampel yang mudah).

**Ukuran sebaran** menjawab "seberapa tersebar datanya?"

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Persentil** membagi data terurut menjadi 100 bagian sama besar. Persentil ke-25 (Q1) berarti 25% nilai berada di bawah titik ini. Persentil ke-50 adalah median. Persentil ke-75 adalah Q3.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

Di ML, kamu peduli pada persentil untuk latency inference, distribusi keyakinan prediksi, dan pemahaman distribusi galat. Sebuah model dengan galat rata-rata rendah tapi galat P99 yang mengerikan mungkin tidak berguna untuk aplikasi safety-critical.

**Statistik sampel vs populasi.** Saat menghitung varians dari sampel, bagi dengan (n-1) alih-alih n. Ini adalah koreksi Bessel. Ia mengkompensasi fakta bahwa mean sampelmu bukanlah mean populasi yang sebenarnya. Dengan n di penyebut, kamu secara sistematis meremehkan varians yang sebenarnya. Dengan (n-1), estimasinya tidak bias.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

Dalam praktiknya: jika n besar (ribuan sampel), perbedaannya bisa diabaikan. Jika n kecil (puluhan sampel), perbedaannya berarti.

### Korelasi: Bagaimana Variabel Bergerak Bersama

Korelasi mengukur kekuatan dan arah hubungan linier antara dua variabel.

**Koefisien korelasi Pearson** mengukur asosiasi linier:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson mengasumsikan hubungannya linier dan kedua variabel berdistribusi kira-kira normal. Ia sensitif terhadap outlier. Satu titik ekstrem bisa menyeret r dari 0.1 ke 0.9.

**Korelasi rank Spearman** mengukur asosiasi monotonic:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**Kapan menggunakan masing-masing:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**Aturan emas:** korelasi tidak menyebabkan kausalitas. Penjualan es krim dan kematian akibat tenggelam berkorelasi karena keduanya meningkat di musim panas. Akurasi model-mu dan jumlah parameter berkorelasi, tapi menambahkan parameter tidak otomatis memperbaiki akurasi (lihat: overfitting).

### Matriks Kovarians

Kovarians antara dua variabel mengukur bagaimana keduanya bervariasi bersama:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

Untuk d fitur, matriks kovarians C adalah matriks d x d di mana C[i][j] = Cov(feature_i, feature_j). Entri diagonal C[i][i] adalah varians dari setiap fitur.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**Kaitan dengan PCA.** PCA melakukan eigendecomposition pada matriks kovarians. Eigenvector-nya adalah komponen utama (arah variansi maksimum). Eigenvalue memberitahu berapa banyak variansi yang ditangkap setiap komponen. Inilah persis yang dibahas di Lesson 10, tapi sekarang kamu melihat mengapa matriks kovarians adalah objek yang tepat untuk didekomposisi: ia mengkodekan semua hubungan linier berpasangan dalam data-mu.

**Kaitan dengan korelasi.** Matriks kovarians adalah matriks kovarians dari variabel yang distandardisasi (masing-masing dibagi dengan simpangan baku-nya). Korelasi menormalkan kovarians agar semua nilai jatuh di [-1, 1].

### Uji Hipotesis

Uji hipotesis adalah kerangka untuk mengambil keputusan di bawah ketidakpastian. Kamu mulai dengan sebuah klaim, mengumpulkan data, dan menentukan apakah data itu konsisten dengan klaimnya.

**Susunannya:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p-value** adalah probabilitas melihat data sama ekstremnya dengan yang kamu amati, dengan asumsi H0 benar. Ini BUKAN probabilitas bahwa H0 benar. Ini kesalahpahaman statistik yang paling umum.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Confidence interval** memberikan rentang nilai yang masuk akal untuk sebuah parameter:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

Lebar confidence interval memberitahu tentang presisi. Interval lebar berarti ketidakpastian tinggi. Interval sempit berarti estimasimu presisi (tapi tidak pasti akurat, jika data-mu bias).

### T-test

T-test membandingkan mean. Ada beberapa variasi.

**One-sample t-test:** apakah mean populasi berbeda dari nilai yang dihipotesiskan?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Two-sample t-test (independen):** apakah dua mean kelompok berbeda?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Paired t-test:** saat pengukuran datang dalam pasangan (model yang sama dievaluasi pada split data yang sama):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

Di ML, paired t-test umum: kamu menjalankan kedua model pada 10 fold cross-validation yang sama dan membandingkan skor mereka secara berpasangan.

### Uji Chi-squared

Uji chi-squared memeriksa apakah frekuensi yang diamati cocok dengan frekuensi yang diharapkan. Berguna untuk data kategorikal.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### A/B Testing untuk Model ML

A/B testing di ML tidak sama dengan A/B testing web. Perbandingan model punya tantangan spesifik:

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**Prosedurnya:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### Signifikansi Statistik vs Signifikansi Praktis

Sebuah hasil bisa secara statistik signifikan tapi secara praktis tidak berarti. Dengan data yang cukup, bahkan perbedaan sepele pun menjadi secara statistik signifikan.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Effect size** mengkuantifikasi seberapa besar perbedaannya, independen dari ukuran sampel:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

Selalu laporkan baik p-value maupun effect size. P-value memberitahu apakah perbedaannya nyata. Effect size memberitahu apakah itu penting.

### Masalah Perbandingan Ganda

Saat kamu menguji banyak hipotesis, beberapa akan "signifikan" secara kebetulan. Jika kamu menguji 20 hal pada alpha = 0.05, kamu mengharapkan 1 false positive bahkan ketika tidak ada yang nyata.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Koreksi Bonferroni:** bagi alpha dengan jumlah tes.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

Di ML, ini penting saat kamu membandingkan model di banyak metrik, menguji banyak konfigurasi hyperparameter, atau mengevaluasi di banyak dataset.

### Metode Bootstrap

Bootstrapping mengestimasi distribusi sampling dari sebuah statistik dengan me-resample data-mu dengan penggantian. Tidak diperlukan asumsi apa pun tentang distribusi dasarnya.

**Algoritmanya:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Confidence interval bootstrap (metode persentil):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**Mengapa bootstrap penting untuk ML:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**Bootstrap untuk perbandingan model:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

Ini lebih kuat daripada paired t-test karena tidak membuat asumsi distribusional apa pun.

### Uji Parametrik vs Non-parametrik

**Uji parametrik** mengasumsikan distribusi tertentu (biasanya normal):

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Uji non-parametrik** tidak membuat asumsi distribusional apa pun:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**Kapan menggunakan non-parametrik:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**Kapan menggunakan parametrik:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

Dalam eksperimen ML, kamu biasanya punya n kecil (5 atau 10 fold cross-validation), jadi uji non-parametrik seperti Wilcoxon signed-rank sering lebih tepat daripada t-test.

### Teorema Limit Pusat: Implikasi Praktis

CLT menyatakan bahwa distribusi mean sampel mendekati distribusi normal saat n tumbuh, terlepas dari distribusi populasi dasarnya.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**Mengapa ini penting untuk ML:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**Apa yang TIDAK dilakukan CLT:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### Kesalahan Statistik Umum di Paper ML

1. **Menguji pada training set.** Menjamin overfitting. Selalu sisihkan data yang tidak pernah dilihat model selama training.

2. **Tanpa confidence interval.** Melaporkan satu angka akurasi tanpa ketidakpastian membuat hasil tidak dapat direproduksi dan tidak dapat diverifikasi.

3. **Mengabaikan perbandingan ganda.** Menguji 50 konfigurasi dan melaporkan yang terbaik tanpa koreksi meningkatkan tingkat false positive.

4. **Membingungkan signifikansi statistik dan praktis.** p-value 0.001 pada peningkatan akurasi 0.01% tidak bermakna.

5. **Menggunakan akurasi pada data tidak seimbang.** Akurasi 99% pada dataset dengan kelas negatif 99% berarti model tidak belajar apa pun. Gunakan precision, recall, F1, atau AUC.

6. **Cherry-picking metrik.** Hanya melaporkan metrik di mana model-mu menang. Evaluasi yang jujur melaporkan semua metrik yang relevan.

7. **Kebocoran informasi antara split train/test.** Menormalisasi sebelum split, atau menggunakan data masa depan untuk memprediksi masa lalu.

8. **Test set kecil tanpa estimasi varians.** Mengevaluasi pada 100 sampel dan mengklaim peningkatan 2% adalah noise, bukan sinyal.

9. **Mengasumsikan ketergantungan padahal data tidak independen.** Gambar medis dari pasien yang sama, banyak kalimat dari dokumen yang sama. Observasi dalam satu grup berkorelasi.

10. **P-hacking.** Mencoba tes, subset, atau kriteria eksklusi berbeda sampai kamu mendapat p < 0.05. Hasilnya adalah artefak dari pencarian itu.

## Bangun Sendiri

Kamu akan mengimplementasikan:

1. **Statistik deskriptif dari nol** (mean, median, mode, simpangan baku, persentil, IQR)
2. **Fungsi korelasi** (Pearson dan Spearman, dengan matriks kovarians)
3. **Uji hipotesis** (one-sample t-test, two-sample t-test, uji chi-squared)
4. **Confidence interval bootstrap** (untuk statistik apa pun, tanpa asumsi)
5. **Simulator A/B test** (hasilkan data, uji, periksa Type I dan Type II error)
6. **Demo signifikansi statistik vs praktis** (menunjukkan bahwa n besar membuat semuanya "signifikan")

Semuanya dari nol, hanya menggunakan `math` dan `random`. Tanpa numpy, tanpa scipy.

```figure
f3-bootstrap-resample
```

## Istilah Kunci

| Term | Definition |
|---|---|
| Mean | Sum of values divided by count. Sensitive to outliers. |
| Median | Middle value of sorted data. Robust to outliers. |
| Standard deviation | Square root of variance. Measures spread in original units. |
| Percentile | Value below which a given percentage of data falls. |
| IQR | Interquartile range. Q3 minus Q1. The spread of the middle 50%. |
| Pearson correlation | Measures linear association between two variables. Range [-1, 1]. |
| Spearman correlation | Measures monotonic association using ranks. |
| Covariance matrix | Matrix of pairwise covariances between all features. |
| Null hypothesis | Default assumption of no effect or no difference. |
| p-value | Probability of data this extreme given the null hypothesis is true. |
| Confidence interval | Range of plausible values for a parameter at a given confidence level. |
| t-test | Tests whether means differ significantly. Uses the t-distribution. |
| Chi-squared test | Tests whether observed frequencies differ from expected frequencies. |
| Effect size | Magnitude of a difference, independent of sample size. Cohen's d is common. |
| Bonferroni correction | Divides significance threshold by number of tests to control false positives. |
| Bootstrap | Resampling with replacement to estimate sampling distributions. |
| Type I error | False positive. Rejecting H0 when it is true. |
| Type II error | False negative. Failing to reject H0 when it is false. |
| Statistical power | Probability of correctly rejecting a false H0. Power = 1 minus Type II error rate. |
| Central limit theorem | Sample means converge to a normal distribution as sample size grows. |
| Parametric test | Assumes a specific distribution for the data (usually normal). |
| Non-parametric test | Makes no distributional assumptions. Works on ranks or signs. |
