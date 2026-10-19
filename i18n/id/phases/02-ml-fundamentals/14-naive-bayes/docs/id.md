# Naive Bayes

> Asumsi "naive"-nya salah, tetapi tetap bekerja. Itulah keindahannya.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Implementasikan Multinomial Naive Bayes dari nol dengan Laplace smoothing untuk klasifikasi teks
- Jelaskan mengapa asumsi independensi naive secara matematis salah tetapi menghasilkan peringkat kelas yang benar dalam praktik
- Bandingkan varian Multinomial, Bernoulli, dan Gaussian Naive Bayes dan pilih yang tepat untuk tipe fitur yang diberikan
- Evaluasi Naive Bayes terhadap regresi logistik pada data sparse berdimensi tinggi dan jelaskan bias-variance tradeoff yang bekerja

## Masalahnya

Kamu perlu mengklasifikasi teks. Email menjadi spam atau bukan spam. Ulasan pelanggan menjadi positif atau negatif. Tiket dukungan menjadi kategori-kategori. Kamu punya ribuan fitur (satu per kata) dan data training terbatas.

Kebanyakan classifier tersedak di sini. Regresi logistik membutuhkan cukup sampel untuk mengestimasikan ribuan weight secara andal. Decision trees membelah pada satu kata pada satu waktu dan overfit secara liar. KNN di 10.000 dimensi tidak bermakna karena setiap titik sama jauhnya dari setiap titik lainnya.

Naive Bayes menangani ini. Ia membuat sebuah asumsi yang secara matematis salah (bahwa setiap fitur independen dari setiap fitur lainnya diberikan kelasnya), dan tetap mengungguli model yang "lebih pintar" pada klasifikasi teks, terutama dengan set training kecil. Ia melatih dalam satu kali lewat (pass) melalui data. Ia berskala hingga jutaan fitur. Ia menghasilkan estimasi probabilitas (meskipun sering kali poorly calibrated karena asumsi independensi).

Memahami mengapa sebuah asumsi yang salah menghasilkan prediksi yang baik mengajarkanmu sesuatu yang fundamental tentang machine learning: model terbaik bukan yang paling benar, tetapi yang punya bias-variance tradeoff terbaik untuk datamu.

## Konsepnya

### Teorema Bayes (Tinjauan Cepat)

Teorema Bayes membalik probabilitas bersyarat:

```
P(class | features) = P(features | class) * P(class) / P(features)
```

Kita menginginkan `P(class | features)` -- probabilitas bahwa sebuah dokumen termasuk sebuah kelas diberikan kata-kata di dalamnya. Kita dapat menghitung ini dari:
- `P(features | class)` -- likelihood melihat kata-kata ini dalam dokumen kelas ini
- `P(class)` -- probabilitas prior kelas (seberapa umum spam secara umum?)
- `P(features)` -- evidence, sama untuk semua kelas, jadi kita dapat mengabaikannya saat membandingkan

Kelas dengan `P(class | features)` tertinggi menang.

### Asumsi Independensi Naive

Menghitung `P(features | class)` secara tepat membutuhkan estimasi probabilitas gabungan (joint) dari semua fitur bersama-sama. Dengan kosakata 10.000 kata, kamu perlu mengestimasikan sebuah distribusi di atas 2^10.000 kombinasi yang mungkin. Mustahil.

Asumsi naive: setiap fitur bersyarat independen diberikan kelasnya.

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

Alih-alih satu distribusi gabungan yang mustahil, kamu mengestimasi n distribusi per-fitur sederhana. Masing-masing hanya membutuhkan sebuah hitungan (count).

Asumsi ini jelas salah. Kata-kata "machine" dan "learning" tidak independen dalam dokumen mana pun. Tetapi classifier tidak membutuhkan estimasi probabilitas yang benar. Ia membutuhkan peringkat yang benar -- kelas mana yang punya probabilitas tertinggi. Asumsi independensi memperkenalkan kesalahan sistematis, tetapi kesalahan itu memengaruhi semua kelas dengan cara yang serupa, sehingga peringkatnya tetap benar.

### Mengapa Ini Tetap Bekerja

Tiga alasan:

1. **Peringkat di atas kalibrasi.** Klasifikasi hanya membutuhkan kelas berperingkat teratas untuk benar. Bahkan jika P(spam) = 0,99999 ketika probabilitas yang sebenarnya 0,7, classifier tetap memilih spam dengan benar. Kita tidak membutuhkan probabilitas yang benar. Kita membutuhkan pemenang yang benar.

2. **Bias tinggi, variance rendah.** Asumsi independensi adalah sebuah prior yang kuat. Ia membatasi model secara berat, yang mencegah overfitting. Dengan data training terbatas, sebuah model yang sedikit salah tetapi stabil mengalahkan sebuah model yang secara teori benar tetapi tidak stabil secara liar. Ini adalah bias-variance tradeoff yang sedang bekerja.

3. **Redundansi fitur saling menghapus.** Fitur yang berkorelasi menyediakan evidence yang redundan. Classifier menghitung ganda evidence ini, tetapi menghitung gandanya juga untuk kelas yang benar. Jika "machine" dan "learning" selalu muncul bersama, keduanya menyediakan evidence untuk kelas "tech". NB menghitungnya dua kali, tetapi menghitungnya dua kali untuk kelas yang benar.

Alasan keempat, yang praktis: Naive Bayes sangat cepat. Training adalah satu kali lewat melalui data untuk menghitung frekuensi. Prediksi adalah sebuah perkalian matriks. Kamu dapat melatih pada sejuta dokumen dalam hitungan detik. Kecepatan ini berarti kamu dapat mengiterasi lebih cepat, mencoba lebih banyak set fitur, dan menjalankan lebih banyak eksperimen dibanding model yang lebih lambat.

### Matematikanya Langkah demi Langkah

Mari kita telusuri sebuah contoh konkret. Misalkan kita punya dua kelas: spam dan bukan spam. Kosakata kita punya tiga kata: "free", "money", "meeting".

Data training:
- Email spam menyebut "free" 80 kali, "money" 60 kali, "meeting" 10 kali (150 kata total)
- Email bukan spam menyebut "free" 5 kali, "money" 10 kali, "meeting" 100 kali (115 kata total)
- 40% email adalah spam, 60% bukan spam

Dengan Laplace smoothing (alpha=1):

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

Email baru berisi: "free" (2 kali), "money" (1 kali), "meeting" (0 kali).

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

Spam menang dengan selisih yang besar. Kata "free" yang muncul dua kali adalah evidence kuat untuk spam. Perhatikan bahwa "meeting" yang tidak muncul berkontribusi nol pada kedua jumlah log (0 * log(P)) -- dalam Multinomial NB, kata yang absen tidak berpengaruh. Bernoulli NB yang secara eksplisit memodelkan ketiadaan kata.

### Tiga Varian

Naive Bayes datang dalam tiga varian. Masing-masing memodelkan `P(feature | class)` secara berbeda.

#### Multinomial Naive Bayes

Memodelkan setiap fitur sebagai sebuah hitungan (count). Terbaik untuk data teks di mana fitur adalah frekuensi kata atau nilai TF-IDF.

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha` adalah Laplace smoothing (dijelaskan di bawah). Varian ini adalah andalan untuk klasifikasi teks.

#### Gaussian Naive Bayes

Memodelkan setiap fitur sebagai sebuah distribusi normal. Terbaik untuk fitur kontinu.

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

Setiap kelas mendapat mean dan variance-nya sendiri per fitur. Ini bekerja dengan baik ketika fitur benar-benar mengikuti kurva lonceng di dalam setiap kelas.

#### Bernoulli Naive Bayes

Memodelkan setiap fitur sebagai biner (present atau absent). Terbaik untuk teks pendek atau vektor fitur biner.

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

Berbeda dengan Multinomial, Bernoulli secara eksplisit memberi penalti pada ketiadaan sebuah kata. Jika "free" biasanya muncul di spam tetapi absen dari email ini, Bernoulli menghitungnya sebagai evidence melawan spam.

### Kapan Menggunakan Varian Mana

| Varian | Tipe Fitur | Terbaik untuk | Contoh |
|---------|-------------|----------|---------|
| Multinomial | Counts atau frekuensi | Klasifikasi teks, bag-of-words | Email spam, klasifikasi topik |
| Gaussian | Nilai kontinu | Data tabular dengan fitur yang agak normal | Klasifikasi iris, data sensor |
| Bernoulli | Biner (0/1) | Teks pendek, vektor fitur biner | SMS spam, fitur presence/absence |

### Laplace Smoothing

Apa yang terjadi ketika sebuah kata muncul di data test tetapi tidak pernah muncul di data training untuk sebuah kelas tertentu?

Tanpa smoothing: `P(word | class) = 0/N = 0`. Sebuah nol yang dikalikan ke seluruh produk membuat `P(class | features) = 0`, terlepas dari semua evidence lainnya. Sebuah kata yang belum pernah terlihat menghancurkan seluruh prediksi, seberapa pun banyak evidence lain yang mendukungnya.

Llaplace smoothing menambahkan hitungan kecil `alpha` (biasanya 1) ke setiap hitungan fitur:

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

Dengan alpha=1, setiap kata mendapat setidaknya probabilitas kecil. Kata "discombobulate" yang muncul di sebuah email test tidak lagi membunuh probabilitas spam. Smoothing punya interpretasi Bayesian: ini ekuivalent dengan menempatkan sebuah prior Dirichlet seragam (uniform) pada distribusi kata-kata.

Alpha lebih tinggi berarti smoothing lebih kuat (distribusi lebih seragam). Alpha lebih rendah berarti model mempercayai data lebih banyak. Alpha adalah sebuah hyperparameter yang kamu setel.

Efek alpha:

| Alpha | Efek | Kapan digunakan |
|-------|--------|-------------|
| 0,001 | Hampir tanpa smoothing, percayai data | Set training sangat besar, tidak mengharapkan fitur yang belum terlihat |
| 0,1 | Smoothing ringan | Set training besar |
| 1,0 | Laplace smoothing standar | Titik awal default |
| 10,0 | Smoothing berat, meratakan distribusi | Set training sangat kecil, mengharapkan banyak fitur yang belum terlihat |

### Komputasi Log-Space

Mengalikan ratusan probabilitas (masing-masing kurang dari 1) menyebabkan floating-point underflow. Produk menjadi nol dalam floating point meskipun nilai yang sebenarnya adalah sebuah bilangan positif yang sangat kecil.

Solusinya: bekerja dalam log space. Alih-alih mengalikan probabilitas, tambahkan logaritmanya:

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

Ini mengubah prediksi menjadi sebuah dot product:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

Perkalian matriks. Itulah mengapa prediksi Naive Bayes sangat cepat -- ini operasi yang sama seperti model linear lapisan tunggal (single-layer).

### Naive Bayes vs Regresi Logistik

Keduanya adalah classifier linear untuk teks. Perbedaannya terletak pada apa yang mereka modelkan.

| Aspek | Naive Bayes | Regresi Logistik |
|--------|------------|-------------------|
| Tipe | Generative (memodelkan P(X\|Y)) | Discriminative (memodelkan P(Y\|X)) |
| Training | Menghitung frekuensi | Mengoptimalkan fungsi loss |
| Data kecil | Lebih baik (prior kuat membantu) | Lebih buruk (tidak cukup untuk mengestimasikan weight) |
| Data besar | Lebih buruk (asumsi yang salah merugikan) | Lebih baik (batas yang fleksibel) |
| Fitur | Mengasumsikan independensi | Menangani korelasi |
| Kecepatan | Single pass, sangat cepat | Optimasi iteratif |
| Kalibrasi | Probabilitas buruk | Probabilitas lebih baik |

Rule of thumb: mulailah dengan Naive Bayes. Jika kamu punya cukup data dan NB mencapai plateau, beralih ke regresi logistik.

### Pipeline Klasifikasi

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

Dalam praktik, kita bekerja dalam log space untuk menghindari floating-point underflow. Alih-alih mengalikan banyak probabilitas kecil, kita menambahkan logaritmanya:

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## Bangun Sendiri

Kode di `code/naive_bayes.py` mengimplementasikan MultinomialNB dan GaussianNB dari nol.

### MultinomialNB

Implementasi dari-nol-nya:

1. **fit(X, y)**: Untuk setiap kelas, hitung frekuensi setiap fitur. Tambahkan Laplace smoothing. Hitung log probabilitas. Simpan prior kelas (log dari frekuensi kelas).

2. **predict_log_proba(X)**: Untuk setiap sampel, hitung log P(class) + jumlah dari log P(feature_i | class) untuk semua kelas. Ini adalah sebuah perkalian matriks: X @ log_probs.T + log_priors.

3. **predict(X)**: Kembalikan kelas dengan log probabilitas tertinggi.

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

Insight kuncinya: setelah fitting, prediksi hanyalah perkalian matriks plus sebuah bias. Itulah mengapa Naive Bayes sangat cepat.

### GaussianNB

Untuk fitur kontinu, kita mengestimasi mean dan variance per kelas per fitur:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

Prediksi menggunakan PDF Gaussian per fitur, dikalikan di seluruh fitur (ditambahkan dalam log space).

### Demo: Klasifikasi Teks

Kode menghasilkan data bag-of-words sintetis yang mensimulasikan dua kelas (artikel teknologi vs artikel olahraga). Setiap kelas punya distribusi frekuensi kata yang berbeda. MultinomialNB mengklasifikasikannya menggunakan hitungan kata.

Data sintetis bekerja seperti ini: kita membuat 200 "words" (kolom fitur). Kata 0-39 punya frekuensi tinggi di artikel teknologi dan rendah di olahraga. Kata 80-119 punya frekuensi tinggi di olahraga dan rendah di teknologi. Kata 40-79 frekuensi sedang di keduanya. Ini menciptakan skenario realistis di mana beberapa kata adalah indikator kelas yang kuat dan yang lain adalah noise.

### Demo: Fitur Kontinu

Kode menghasilkan data mirip Iris (3 kelas, 4 fitur, klaster Gaussian). GaussianNB mengklasifikasi menggunakan mean dan variance per kelas. Setiap kelas punya pusat yang berbeda (vektor mean) dan sebaran (variance) yang berbeda, meniru data dunia nyata di mana pengukuran berbeda secara sistematis antar kategori.

Kode juga mendemonstrasikan:
- **Perbandingan smoothing:** Melatih MultinomialNB dengan alpha berbeda untuk menunjukkan efek kekuatan smoothing pada akurasi.
- **Eksperimen ukuran training:** Bagaimana akurasi NB membaik seiring data training tumbuh dari 20 hingga 1600 sampel. NB mencapai akurasi yang layak bahkan dengan sangat sedikit sampel -- ini adalah keuntungan utamanya.
- **Confusion matrix:** Precision, recall, dan F1 score per kelas untuk menunjukkan di mana NB membuat kesalahan.

### Kecepatan Prediksi

Prediksi Naive Bayes adalah sebuah perkalian matriks. Untuk n sampel dengan d fitur dan k kelas:
- MultinomialNB: satu perkalian matriks (n x d) @ (d x k) = O(n * d * k)
- GaussianNB: n * k evaluasi PDF Gaussian, masing-masing di seluruh d fitur = O(n * d * k)

Keduanya linear di setiap dimensi. Bandingkan dengan KNN (yang membutuhkan komputasi jarak ke semua titik training) atau SVM dengan kernel RBF (yang membutuhkan evaluasi kernel terhadap semua support vector). NB lebih cepat beberapa orders of magnitude saat prediksi.

## Gunakan

Dengan sklearn, kedua varian hanyalah one-liner:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

Untuk klasifikasi teks dengan sklearn:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

Kode di `naive_bayes.py` membandingkan implementasi dari-nol dengan sklearn pada data yang sama untuk memverifikasi kebenaran.

### TF-IDF dengan Naive Bayes

Hitungan kata mentah memberikan setiap kata bobot yang sama per kemunculan. Tetapi kata-kata umum seperti "the" dan "is" sering muncul di setiap kelas -- mereka tidak membawa informasi. TF-IDF (Term Frequency - Inverse Document Frequency) menurunkan bobot kata-kata umum dan menaikkan bobot kata-kata langka yang diskriminatif.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

Nilai TF-IDF tidak negatif, jadi bekerja dengan MultinomialNB. Kombinasi TF-IDF + MultinomialNB adalah salah satu baseline terkuat untuk klasifikasi teks. Sering kali mengalahkan model yang lebih kompleks pada dataset dengan kurang dari 10.000 sampel training.

### BernoulliNB untuk Teks Pendek

Untuk teks pendek (tweet, SMS, pesan chat), BernoulliNB dapat mengungguli MultinomialNB. Teks pendek punya hitungan kata rendah, sehingga informasi frekuensi yang diandalkan MultinomialNB adalah noisy. Bernoulli hanya peduli pada presence atau absence, yang lebih andal dengan teks pendek.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

Flag `binary=True` di CountVectorizer mengubah semua hitungan menjadi 0/1. Tanpanya, BernoulliNB tetap bekerja tetapi melihat hitungan yang tidak ia rancang untuk itu.

### Kalibrasi Probabilitas NB

Probabilitas NB poorly calibrated. Ketika NB mengatakan P(spam) = 0,95, probabilitas yang sebenarnya mungkin 0,7. Jika kamu membutuhkan estimasi probabilitas yang andal (misalnya, untuk menetapkan sebuah threshold atau untuk menggabungkan dengan model lain), gunakan CalibratedClassifierCV dari sklearn:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

Ini meng-fit sebuah regresi logistik di atas skor mentah NB menggunakan cross-validation. Probabilitas hasilnya jauh lebih dekat ke frekuensi kelas yang sebenarnya.

### Jebakan Umum

1. **Nilai fitur negatif.** MultinomialNB membutuhkan fitur tidak negatif. Jika kamu punya nilai negatif (seperti TF-IDF dengan pengaturan tertentu atau fitur yang distandardisasi), gunakan GaussianNB, atau geser (shift) fitur agar menjadi positif.

2. **Fitur zero variance.** GaussianNB membagi dengan variance. Jika sebuah fitur punya variance nol untuk sebuah kelas (semua nilai identik), komputasi probabilitas rusak. Kode menambahkan sebuah istilah smoothing kecil (1e-9) ke semua variance untuk mencegah ini.

3. **Ketidakseimbangan kelas (class imbalance).** Jika 99% email bukan spam, prior P(not-spam) = 0,99 sangat kuat sehingga menenggelamkan evidence likelihood. Kamu dapat menetapkan prior kelas secara manual atau menggunakan parameter class_prior di sklearn.

4. **Feature scaling.** MultinomialNB tidak membutuhkan scaling (ia bekerja pada hitungan). GaussianNB juga tidak membutuhkan scaling (ia mengestimasi statistik per-fitur). Ini adalah keuntungan dibanding regresi logistik dan SVM, yang sensitif terhadap skala fitur.

## Rilis

Pelajaran ini menghasilkan:
- `outputs/skill-naive-bayes-chooser.md` -- sebuah skill keputusan untuk memilih varian NB yang tepat
- `code/naive_bayes.py` -- MultinomialNB dan GaussianNB dari nol, dengan perbandingan sklearn

### Kapan Naive Bayes Gagal

NB gagal ketika asumsi independensi menyebabkan peringkat yang salah (bukan hanya probabilitas yang salah). Ini terjadi ketika:

1. **Interaksi fitur yang kuat.** Jika kelas bergantung pada kombinasi dua fitur tetapi tidak pada salah satunya sendirian (pola mirip XOR), NB akan melewatkannya sepenuhnya. Setiap fitur sendirian tidak menyediakan evidence apa pun, dan NB tidak dapat menggabungkannya secara nonlinear.

2. **Fitur yang berkorelasi kuat dengan evidence yang bertentangan.** Jika fitur A mengatakan "spam" dan fitur B mengatakan "not-spam", tetapi A dan B berkorelasi sempurna (mereka selalu setuju dalam kenyataan), NB akan melihat evidence yang bertentangan padahal tidak ada yang bertentangan.

3. **Set training yang sangat besar.** Dengan cukup data, model discriminative seperti regresi logistik mempelajari batas keputusan yang sebenarnya dan mengungguli NB. Asumsi independensi yang membantu dengan data kecil kini menahan model.

Dalam praktik, mode-mode kegagalan ini jarang untuk klasifikasi teks. Fitur teks banyak, secara individu lemah, dan kesalahan asumsi independensi cenderung saling menghapus. Untuk data tabular dengan sedikit fitur yang berkorelasi kuat, pertimbangkan regresi logistik atau model berbasis pohon terlebih dahulu.

## Latihan

1. **Eksperimen smoothing.** Latih MultinomialNB pada data teks dengan alpha 0,01; 0,1; 1,0; 10,0; dan 100,0. Plot akurasi vs alpha. Di mana performanya mencapai puncak? Mengapa alpha yang sangat tinggi merugikan?

2. **Uji independensi fitur.** Ambil sebuah dataset teks nyata. Pilih dua kata yang jelas berkorelasi ("machine" dan "learning"). Hitung P(word1 | class) * P(word2 | class) dan bandingkan dengan P(word1 AND word2 | class). Seberapa salahnya asumsi independensi? Apakah ini memengaruhi akurasi klasifikasi?

3. **Implementasi Bernoulli.** Perluas kode dengan sebuah kelas BernoulliNB. Konversi bag-of-words menjadi biner (present/absent) dan bandingkan akurasinya dengan MultinomialNB pada data teks. Kapan Bernoulli menang?

4. **NB vs Regresi Logistik.** Latih keduanya pada data teks. Mulai dengan 100 sampel training dan tingkatkan hingga 10.000. Plot akurasi vs ukuran set training untuk keduanya. Pada titik mana Regresi Logistik melampaui Naive Bayes?

5. **Filter spam.** Bangun sebuah classifier spam lengkap: tokenisasi teks email mentah, bangun kosakata, buat fitur bag-of-words, latih MultinomialNB, evaluasi dengan precision dan recall (bukan hanya akurasi -- mengapa?).

## Istilah Kunci

| Istilah | Apa yang orang katakan | Apa artinya yang sebenarnya |
|------|----------------|----------------------|
| Naive Bayes | "Classifier probabilistik sederhana" | Sebuah classifier yang menerapkan teorema Bayes dengan asumsi bahwa fitur bersyarat independen diberikan kelasnya |
| Independensi bersyarat | "Fitur tidak saling memengaruhi" | P(A, B \| C) = P(A \| C) * P(B \| C) -- mengetahui B tidak memberi tahu apa pun yang baru tentang A setelah kamu mengetahui C |
| Laplace smoothing | "Add-one smoothing" | Menambahkan hitungan kecil ke setiap fitur untuk mencegah probabilitas nol mendominasi prediksi |
| Prior | "Apa yang kamu percaya sebelum melihat data" | P(class) -- probabilitas setiap kelas sebelum mengamati fitur apa pun |
| Likelihood | "Seberapa baik data cocok" | P(features \| class) -- probabilitas mengamati fitur-fitur ini jika kelasnya diketahui |
| Posterior | "Apa yang kamu percaya setelah melihat data" | P(class \| features) -- probabilitas kelas yang diperbarui setelah mengamati fitur |
| Model generative | "Memodelkan bagaimana data dihasilkan" | Sebuah model yang mempelajari P(X \| Y) dan P(Y), lalu menggunakan teorema Bayes untuk mendapatkan P(Y \| X) |
| Model discriminative | "Memodelkan batas keputusan" | Sebuah model yang secara langsung mempelajari P(Y \| X) tanpa memodelkan bagaimana X dihasilkan |
| Log probability | "Menghindari underflow" | Bekerja dengan log P alih-alih P untuk mencegah produk dari banyak bilangan kecil menjadi nol dalam floating point |

## Bacaan Lanjutan

- [dokumen Naive Bayes scikit-learn](https://scikit-learn.org/stable/modules/naive_bayes.html) -- semua tiga varian dengan detail matematis
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- perbandingan klasik Multinomial vs Bernoulli untuk teks
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- peningkatan NB untuk teks
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- membuktikan NB konvergen lebih cepat daripada LR dengan data lebih sedikit
