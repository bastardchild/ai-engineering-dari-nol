# Word Embedding — Word2Vec dari Nol

> Sebuah kata adalah teman di sekitarnya. Latih shallow net pada ide itu dan geometri pun muncul.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## Masalahnya

TF-IDF tahu `dog` dan `puppy` adalah kata berbeda. Ia tidak tahu keduanya berarti hampir sama. Pengklasifikasi yang dilatih pada `dog` tidak bisa menggeneralisasi ke ulasan tentang `puppy`. Kamu bisa menambalnya dengan mendaftar sinonim, tetapi itu gagal pada term jarang, jargon domain, dan setiap bahasa yang tak kamu antisipasi.

Kamu menginginkan representasi di mana `dog` dan `puppy` mendarat berdekatan di ruang vektor. Di mana `king - man + woman` mendarat dekat `queen`. Di mana model yang dilatih pada `dog` mentransfer sebagian sinyal ke `puppy` secara gratis.

Word2Vec memberi kita ruang itu. Jaringan neural dua lapis, training run bertoken triliun, dipublikasikan tahun 2013. Arsitekturnya hampir memalukan sederhananya. Hasilnya membentuk ulang NLP selama satu dekade.

## Konsepnya

**Hipotesis distribusional** (Firth, 1957): "You shall know a word by the company it keeps." Jika dua kata muncul dalam konteks mirip, keduanya mungkin berarti hal mirip.

Word2Vec hadir dalam dua rasa, keduanya mengeksploitasi ide itu.

- **Skip-gram.** Diberi kata pusat, prediksi kata-kata sekitarnya. `cat -> (the, sat, on)` dengan ukuran window 2.
- **CBOW (continuous bag of words).** Diberi kata-kata sekitar, prediksi pusatnya. `(the, sat, on) -> cat`.

Skip-gram lebih lambat dilatih tetapi menangani kata jarang lebih baik. Ia menjadi default.

Jaringan ini punya satu hidden layer tanpa nonlinearitas. Input adalah vektor one-hot atas vocabulary. Output adalah softmax atas vocabulary. Setelah training, kamu membuang output layer. Bobot hidden layer adalah embedding-nya.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

Triknya: softmax atas 100k kata terlalu mahal. Word2Vec memakai **negative sampling** untuk mengubahnya menjadi tugas klasifikasi biner. Prediksi "apakah kata konteks ini muncul di dekat kata pusat ini, ya atau tidak". Sampel segenggam kata negatif (yang tidak berko-okurensi) per pasangan training alih-alih menghitung softmax atas seluruh vocabulary.

```figure
word-vector-arithmetic
```

## Bangun Sendiri

### Langkah 1: pasangan training dari korpus

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Setiap pasangan (pusat, konteks) dalam window adalah contoh training positif.

### Langkah 2: tabel embedding

Dua matriks. `W` adalah tabel embedding kata-pusat (yang kamu simpan). `W'` adalah tabel kata-konteks (sering dibuang, kadang dirata-ratakan dengan `W`).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

Init acak kecil. Ukuran vocab 10k dan dim 100 itu realistis; untuk pengajaran, 50 vocab x 16 dim cukup untuk melihat geometrinya.

### Langkah 3: objektif negative sampling

Untuk setiap pasangan positif `(center, context)`, sampel `k` kata acak dari vocabulary sebagai negatif. Latih model supaya dot product `W[center] · W'[context]` tinggi untuk positif dan rendah untuk negatif.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

Rumus ajaibnya: logistic loss pada pasangan positif (ingin sigmoid dekat 1) plus logistic loss pada pasangan negatif (ingin sigmoid dekat 0). Gradien mengalir ke kedua tabel. Derivasi penuh ada di paper aslinya; telusuri sekali dengan pensil dan kertas kalau kamu ingin benar-benar menancap.

### Langkah 4: latih pada korpus mainan

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

Setelah epoch cukup pada korpus besar, kata-kata yang berbagi konteks punya embedding pusat yang mirip. Pada korpus mainan, kamu melihat efeknya samar-samar. Pada miliaran token, kamu melihatnya secara dramatis.

### Langkah 5: trik analogi

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

Pada vektor Google News 300d pra-terlatih:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. Bukan karena model tahu apa itu kerajaan. Karena vektor `(king - man)` menangkap sesuatu seperti "royal", dan menambahkannya ke `woman` mendarat dekat wilayah royal-female.

## Gunakan

Menulis Word2Vec dari nol itu untuk belajar. NLP produksi memakai `gensim`.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

Untuk kerja nyata, kamu hampir tak pernah melatih Word2Vec sendiri. Kamu mengunduh vektor pra-terlatih.

- **GloVe** — pendekatan faktorisasi matriks ko-okurensi dari Stanford. Checkpoint 50d, 100d, 200d, 300d. Cakupan umum yang bagus. Pelajaran 04 membahas GloVe secara khusus.
- **fastText** — ekstensi Word2Vec dari Facebook yang meng-embedding n-gram karakter. Menangani kata out-of-vocabulary dengan menyusun subword. Pelajaran 04.
- **Word2Vec pra-terlatih pada Google News** — 300d, vocabulary 3M kata, dipublikasikan 2013. Masih diunduh setiap hari.

### Kapan Word2Vec masih menang di 2026

- Retrieval ringan yang spesifik domain. Latih pada abstrak medis dalam satu jam di laptop, dapatkan vektor terspesialisasi yang tak ditangkap model umum mana pun.
- Rekayasa fitur gaya analogi. `gender_vector = mean(man - woman pairs)`. Kurangkan dari kata lain untuk mendapat sumbu netral-gender. Masih dipakai dalam riset fairness.
- Interpretabilitas. 100d cukup kecil untuk diplot via PCA atau t-SNE dan benar-benar melihat klaster terbentuk.
- Di mana pun inference harus berjalan on-device tanpa GPU. Lookup Word2Vec hanyalah satu pengambilan baris.

### Di mana Word2Vec gagal

Tembok polisemi. `bank` punya satu vektor. `river bank` dan `financial bank` berbagi vektor itu. `table` (spreadsheet vs. furnitur) berbagi vektor itu. Pengklasifikasi downstream tidak bisa membedakan makna dari vektornya.

Embedding kontekstual (ELMo, BERT, setiap transformer sejak itu) memecahkannya dengan menghasilkan vektor berbeda untuk tiap kemunculan kata berdasarkan konteks sekitarnya. Itulah lompatan dari Word2Vec ke BERT: dari statis ke kontekstual. Fase 7 membahas separuh transformer-nya.

Masalah out-of-vocabulary adalah kegagalan lainnya. Word2Vec tak pernah melihat `Zoomer-approved` jika itu tidak ada di data training. Tanpa fallback. fastText memperbaikinya dengan komposisi subword (pelajaran 04).

## Rilis

Simpan sebagai `outputs/skill-embedding-probe.md`:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Latihan

1. **Mudah.** Jalankan loop training pada korpus mungil (20 kalimat tentang kucing dan anjing). Setelah 200 epoch, verifikasi `nearest(vocab, W, W[vocab["cat"]])` mengembalikan `dog` di 3 teratasnya. Jika tidak, tambah epoch atau vocabulary.
2. **Sedang.** Tambahkan subsampling kata sering. Kata dengan frekuensi di atas `10^-5` dibuang dari pasangan training dengan probabilitas proporsional terhadap frekuensinya. Ukur efeknya pada kemiripan kata jarang.
3. **Sulit.** Latih model pada korpus 20 Newsgroups. Hitung dua sumbu bias: `he - she` dan `doctor - nurse`. Proyeksikan kata-kata pekerjaan ke kedua sumbu. Laporkan pekerjaan mana yang punya gap bias terbesar. Ini jenis probe yang dipakai peneliti fairness.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Word embedding | Word as a vector | A dense, low-dim (typically 100-300) representation learned from context. |
| Skip-gram | Word2Vec trick | Predict context words from center word. Slower than CBOW, better for rare words. |
| Negative sampling | Training shortcut | Replace softmax over full vocab with binary classification against `k` random words. |
| Static embedding | One vector per word | Same vector regardless of context. Fails on polysemy. |
| Contextual embedding | Context-sensitive vector | Different vector for each occurrence based on surrounding words. What transformers produce. |
| OOV | Out of vocabulary | Word not seen in training. Word2Vec cannot produce a vector for these. |

## Bacaan Lanjutan

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — the negative-sampling paper. Short and readable.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — the clearest derivation of the gradients, if the original paper's math feels dense.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — production training settings that actually work.
