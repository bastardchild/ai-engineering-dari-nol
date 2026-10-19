# Analisis Sentimen

> Tugas NLP kanonis. Kebanyakan hal yang perlu kamu ketahui tentang klasifikasi teks klasik muncul di sini.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75 minutes

## Masalahnya

"The food was not great." Positif atau negatif?

Sentimen terdengar sederhana. Seorang pengulas bilang ia suka atau tidak suka sesuatu. Beri label pada kalimatnya. Alasan ia menjadi tugas NLP kanonis adalah setiap kasus yang tampak mudah menyembunyikan kasus sulit. Negasi membalik makna. Sarkasme membalikkannya. "Not bad at all" itu positif meski punya dua kata berkode negatif. Emoji membawa lebih banyak sinyal daripada teks sekitarnya. Kosakata domain penting (`tight` dalam ulasan musik versus `tight` dalam ulasan fashion).

Sentimen adalah lab kerja untuk NLP klasik. Jika kamu paham mengapa setiap baseline naif punya mode kegagalan spesifik, kamu paham mengapa setiap model yang lebih kaya diciptakan. Pelajaran ini membangun baseline Naive Bayes dari nol, menambahkan regresi logistik, dan menyebutkan jebakan yang membuat sentimen produksi menjadi masalah setingkat compliance.

## Konsepnya

Sentimen klasik adalah resep dua langkah.

1. **Representasikan.** Ubah teks menjadi vektor fitur. BoW, TF-IDF, atau n-gram.
2. **Klasifikasikan.** Fit model linear (Naive Bayes, regresi logistik, SVM) pada contoh berlabel.

Naive Bayes adalah model paling bodoh yang berhasil. Asumsikan setiap fitur independen diberi label. Estimasi `P(word | positive)` dan `P(word | negative)` dari hitungan. Saat inference, kalikan probabilitasnya. Asumsi independensi yang "naif" itu salah secara menyedihkan namun hasilnya mengejutkan kuatnya. Alasannya: dengan fitur teks sparse dan data sedang, pengklasifikasi lebih peduli ke sisi mana tiap kata condong daripada seberapa besar.

Regresi logistik memperbaiki asumsi independensi. Ia mempelajari bobot per fitur, termasuk bobot negatif. `not good` sebagai fitur bigram mendapat bobot negatif. Naive Bayes tidak bisa melakukan itu untuk bigram yang belum pernah dilabelinya.

```figure
sentiment-logits
```

## Bangun Sendiri

### Langkah 1: mini-dataset nyata

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

Sengaja kecil. Kerja nyata memakai puluhan ribu contoh (IMDb, SST-2, polaritas Yelp). Matematikanya identik.

### Langkah 2: Naive Bayes multinomial dari nol

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Smoothing aditif (alpha=1.0) adalah smoothing Laplace. Tanpanya, kata yang tak terlihat di suatu kelas berprobabilitas nol dan log-nya meledak. `alpha=0.01` umum dalam praktik. `alpha=1.0` adalah default pengajaran.

### Langkah 3: regresi logistik dari nol

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

Regularisasi L2 penting di sini. Fitur teks itu sparse; tanpa L2 model menghafal contoh training. Mulai dari `0.01` dan tune.

### Langkah 4: menangani negasi (mode kegagalan)

Perhatikan "not good" dan "not bad". Pengklasifikasi BoW melihat `{not, good}` dan `{not, bad}` lalu belajar dari mana pun yang lebih sering muncul saat training. Pengklasifikasi bigram melihat `not_good` dan `not_bad` lalu mempelajarinya sebagai fitur berbeda. Itu biasanya cukup.

Perbaikan yang lebih kasar tetapi berhasil ketika kamu tidak punya bigram: **negation scoping**. Awali token setelah kata negasi dengan `NOT_` sampai tanda baca berikutnya.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

Kini `good` dan `NOT_good` adalah fitur berbeda. Pengklasifikasi bisa membobotinya berlawanan. Tiga baris pra-pemrosesan, lompatan akurasi yang terukur pada benchmark sentimen.

### Langkah 5: metrik evaluasi yang penting

Akurasi saja menyesatkan jika kelas tak seimbang. Korpus sentimen nyata biasanya 70-80% positif atau 70-80% negatif; pengklasifikasi konstan-mayoritas mendapat akurasi 80% dan tidak berguna. Laporkan setiap hal berikut:

- **Presisi dan recall per kelas.** Satu pasangan per kelas. Rata-ratakan secara macro untuk mendapat satu angka yang menghormati keseimbangan kelas.
- **Macro-F1 (metrik utama untuk data tak seimbang).** Rata-rata skor F1 per kelas, dibobot setara. Pakai ini alih-alih akurasi ketika kelas tak seimbang.
- **Weighted-F1 (alternatif).** Sama seperti macro tetapi dibobot frekuensi kelas. Laporkan berdampingan macro-F1 ketika ketidakseimbangannya sendiri bermakna bisnis.
- **Confusion matrix.** Hitungan mentah. Selalu periksa sebelum memercayai metrik skalar apa pun; ia mengungkap pasangan kelas mana yang membingungkan model.
- **Sampel error per kelas.** Ambil 5 prediksi salah per kelas. Baca. Tak ada yang menggantikan membaca error aslinya.

Untuk data yang sangat tak seimbang (rasio > 95-5), laporkan **AUROC** dan **AUPRC** alih-alih akurasi. AUPRC lebih sensitif terhadap kelas minoritas, yang biasanya kamu pedulikan (spam, fraud, sentimen jarang).

**Bug umum yang harus dihindari.** Melaporkan micro-F1 alih-alih macro-F1 pada data tak seimbang menghasilkan angka yang tampak tinggi karena didominasi kelas mayoritas. Macro-F1 memaksamu melihat performa kelas minoritas.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## Gunakan

scikit-learn melakukannya dalam enam baris, dengan benar.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

Tiga hal yang perlu diperhatikan. `stop_words=None` mempertahankan negasi. `ngram_range=(1, 2)` menambahkan bigram sehingga `not_good` menjadi fitur. `sublinear_tf=True` meredam kata berulang. Tiga flag ini adalah pembeda antara baseline 75%-akurat dan baseline 85%-akurat pada SST-2.

### Kapan beralih ke transformer

- Deteksi sarkasme. Model klasik gagal di sini. Titik.
- Ulasan panjang di mana sentimen berubah di tengah dokumen.
- Sentimen berbasis aspek. "Camera was great but battery was terrible." Kamu perlu mengatribusikan sentimen ke aspek. Hanya transformer atau model output terstruktur.
- Bahasa non-Inggris yang rendah sumber daya. Multilingual BERT memberimu baseline zero-shot gratis.

Jika kamu butuh salah satu di atas, lompat ke fase 7 (deep dive transformer). Jika tidak, Naive Bayes atau regresi logistik pada TF-IDF plus bigram plus penanganan negasi adalah baseline produksimu di 2026.

### Jebakan reproduksibilitas (lagi)

Melatih ulang model sentimen itu rutin. Mengevaluasi ulang tidak. Angka akurasi di paper memakai split spesifik, pra-pemrosesan spesifik, tokenizer spesifik. Jika kamu membandingkan model barumu dengan baseline tanpa memakai pipeline yang identik, kamu akan mendapat delta yang menyesatkan. Selalu regenerasi baseline pada pipelinemmu, bukan angka paper-nya.

## Rilis

Simpan sebagai `outputs/prompt-sentiment-baseline.md`:

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## Latihan

1. **Mudah.** Tambahkan `apply_negation` sebagai langkah pra-pemrosesan dalam pipeline scikit-learn dan ukur delta F1 pada dataset sentimen kecil.
2. **Sedang.** Implementasikan regresi logistik berbobot kelas (teruskan `class_weight="balanced"` ke scikit-learn, atau turunkan gradiennya sendiri). Ukur efeknya pada ketidakseimbangan kelas 90-10 sintetis.
3. **Sulit.** Bangun detektor sarkasme dengan melatih pengklasifikasi kedua pada residual model sentimen. Dokumentasikan setup eksperimenmu. Peringatkan pembaca ketika akurasimu di bawah chance (chance pada sarkasme 2-kelas adalah ~50%, dan kebanyakan percobaan pertama mendarat di sana).

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Polarity | Positive or negative | Binary label; sometimes extended to neutral or fine-grained (5-star). |
| Aspect-based sentiment | Per-aspect polarity | Attribute sentiment to specific entities or attributes mentioned in text. |
| Negation scoping | Reversing nearby tokens | Prefix tokens after "not" with `NOT_` until punctuation. |
| Laplace smoothing | Adding 1 to counts | Prevents zero-probability features in Naive Bayes. |
| L2 regularization | Shrinking weights | Adds `lambda * sum(w^2)` to loss. Essential for sparse text features. |

## Bacaan Lanjutan

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — the foundational survey. Long, but the first four sections cover everything classical.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — the paper that showed bigrams + Naive Bayes is hard to beat on short text.
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — reference for `CountVectorizer`, `TfidfVectorizer`, and every knob you'll tune.
