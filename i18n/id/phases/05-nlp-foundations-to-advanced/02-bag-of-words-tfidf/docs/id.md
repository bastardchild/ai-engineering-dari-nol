# Bag of Words, TF-IDF, dan Representasi Teks

> Hitung dulu, pikir belakangan. TF-IDF masih mengalahkan embedding pada tugas yang terdefinisi jelas di tahun 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time:** ~75 minutes

## Masalahnya

Model butuh angka. Kamu punya string.

Setiap pipeline NLP harus menjawab pertanyaan yang sama. Bagaimana mengubah aliran token bervariabel panjang menjadi vektor berukuran tetap yang bisa dikonsumsi pengklasifikasi. Jawaban pertama yang ditemukan bidang ini adalah yang paling bodoh yang berhasil. Hitung kata-katanya. Buat vektor.

Vektor itu telah mengangkut NLP produksi lebih banyak daripada model embedding mana pun. Filter spam, pengklasifikasi topik, deteksi anomali log, pemeringkatan pencarian (sebelum BM25), gelombang pertama analisis sentimen, dekade pertama benchmark NLP akademik. Praktisi 2026 masih meraihnya duluan pada tugas klasifikasi sempit. Ia cepat, dapat diinterpretasi, dan sering tak terbedakan dari model embedding 400M-parameter pada tugas di mana kehadiran kata adalah yang penting.

Pelajaran ini membangun bag of words, lalu TF-IDF, dari nol. Lalu menunjukkan scikit-learn melakukan hal yang sama dalam tiga baris. Lalu menyebutkan mode kegagalan yang membuatmu beralih ke embedding.

## Konsepnya

**Bag of Words (BoW)** membuang urutan. Untuk setiap dokumen, hitung berapa kali setiap kata vocabulary muncul. Panjang vektor adalah ukuran vocabulary. Posisi `i` adalah hitungan kata `i`.

**TF-IDF** membobot ulang BoW. Kata yang muncul di setiap dokumen tidak informatif, jadi kecilkan skalanya. Kata yang jarang di seluruh korpus tetapi sering dalam satu dokumen adalah sinyal, jadi besarkan skalanya.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

Di mana `TF` adalah frekuensi term dalam dokumen, `df` adalah frekuensi dokumen (berapa dokumen mengandung kata itu), `N` adalah total dokumen. `log` menjaga bobot tetap terbatas untuk kata yang ada di mana-mana.

Sifat kunci: keduanya menghasilkan vektor sparse dengan sumbu yang dapat diinterpretasi. Kamu bisa melihat bobot pengklasifikasi terlatih dan membaca kata mana yang mendorong dokumen ke tiap kelas. Kamu tidak bisa melakukan ini dengan embedding BERT 768-dimensi.

```figure
bow-tfidf
```

## Bangun Sendiri

### Langkah 1: bangun vocabulary

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

Input: daftar dokumen yang sudah di-tokenisasi (tokenizer level kata mana pun boleh; `code/main.py` di pelajaran ini memakai varian lowercase yang disederhanakan). Output: dict `{word: index}`. Urutan penyisipan yang stabil berarti indeks kata 0 adalah kata pertama yang terlihat di dokumen pertama. Konvensi bervariasi; scikit-learn mengurutkan secara alfabetis.

### Langkah 2: bag of words

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

Baris adalah dokumen. Kolom adalah indeks vocabulary. Entri `[i][j]` berarti "berapa kali kata `j` muncul di dokumen `i`." Dok 1 punya `cat` dua kali karena memang begitu. Dok 0 punya `ran` nol kali karena tidak ada.

### Langkah 3: frekuensi term dan frekuensi dokumen

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

Dua trik smoothing yang patut disebut. `(n+1)/(d+1)` menghindari `log(x/0)`. `+1` di akhir memastikan kata di setiap dokumen tetap punya IDF 1 (bukan 0), sesuai default scikit-learn. Implementasi lain memakai `log(N/df)` mentah. Keduanya berhasil; versi smoothed lebih bersahabat.

### Langkah 4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

Tiga dokumen, lima kata vocab (`the`, `cat`, `sat`, `dog`, `ran`). `the` muncul di ketiganya, jadi IDF-nya rendah. `dog` muncul di satu dokumen, jadi IDF-nya tinggi. Vektornya sparse (kebanyakan entri kecil) dan kata-kata diskriminatif menonjol.

### Langkah 5: normalisasi L2 per baris

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

Tanpa normalisasi, dokumen yang lebih panjang mendapat vektor lebih besar dan mendominasi skor kemiripan. Normalisasi L2 menaruh setiap dokumen pada hypersphere satuan. Kemiripan kosinus antar baris kini hanyalah dot product.

## Gunakan

scikit-learn menyediakan versi produksinya.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` melakukan tokenisasi, vocabulary, dan BoW dalam satu panggilan. `TfidfVectorizer` menambahkan pembobotan IDF dan normalisasi L2. Keduanya mengembalikan matriks sparse. Untuk 100k dokumen, versi dense tidak muat di memori; tetap sparse sampai pengklasifikasi menuntut dense.

Kenop yang mengubah segalanya:

| Arg | Effect |
|-----|--------|
| `ngram_range=(1, 2)` | Include bigrams. Usually boosts classification. |
| `min_df=2` | Drop words in fewer than 2 docs. Trims vocabulary on noisy data. |
| `max_df=0.95` | Drop words in more than 95% of docs. Approximates stopword removal without a hardcoded list. |
| `stop_words="english"` | scikit-learn's builtin stopword list. Task-dependent — sentiment analysis should *not* drop negations. |
| `sublinear_tf=True` | Use `1 + log(tf)` instead of raw `tf`. Helps when a term repeats many times in one doc. |

### Kapan TF-IDF masih menang (per 2026)

- Deteksi spam, pelabelan topik, penandaan anomali log. Kehadiran kata adalah yang penting; nuansa semantik tidak.
- Rezim data kecil (ratusan contoh berlabel). TF-IDF plus regresi logistik tanpa biaya pretraining.
- Di mana pun latensi penting. TF-IDF plus model linear menjawab dalam mikrodetik. Meng-embedding dokumen lewat transformer butuh 10-100ms.
- Sistem yang harus menjelaskan prediksinya. Periksa koefisien pengklasifikasi. Kata-kata positif teratas adalah alasannya.

### Kapan TF-IDF gagal

Kegagalan kebutaan semantik. Perhatikan dua dokumen ini:

- "The movie was not good at all."
- "The movie was excellent."

Satu adalah ulasan negatif. Satu positif. Overlap TF-IDF mereka tepat `{the, movie, was}`. Pengklasifikasi bag-of-words harus menghafal bahwa kata `not` di dekat `good` membalik label. Ia bisa mempelajarinya dengan data cukup, tetapi tidak pernah seelegan model yang memahami sintaksis.

Kegagalan lain: kata out-of-vocabulary saat inference. Model BoW yang dilatih pada ulasan IMDb tidak tahu harus berbuat apa dengan `Zoomer-approved` jika token itu tak pernah muncul saat training. Embedding subword (pelajaran 04) menanganinya. TF-IDF tidak bisa.

### Hibrida: embedding berbobot TF-IDF

Default pragmatis 2026 untuk klasifikasi data menengah: pakai bobot TF-IDF sebagai attention atas word embedding.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

Kamu mendapat kapasitas semantik dari embedding, dan penekanan kata jarang dari TF-IDF. Pengklasifikasi dilatih pada vektor gabungan. Ini mengungguli keduanya sendirian untuk klasifikasi sentimen, topik, dan intent di bawah sekitar 50k contoh berlabel.

## Rilis

Simpan sebagai `outputs/prompt-vectorization-picker.md`:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## Latihan

1. **Mudah.** Implementasikan `cosine_similarity(doc_vec_a, doc_vec_b)` pada output TF-IDF yang sudah dinormalisasi L2. Verifikasi bahwa dokumen identik mendapat skor 1.0 dan dokumen dengan vocabulary disjoint mendapat skor 0.0.
2. **Sedang.** Tambahkan dukungan `n-gram` ke `bag_of_words`. Parameter `n` menghasilkan hitungan atas `n`-gram. Uji bahwa `n=2` pada `["the", "cat", "sat"]` menghasilkan hitungan bigram untuk `["the cat", "cat sat"]`.
3. **Sulit.** Bangun hibrida embedding-berbobot-TF-IDF di atas memakai vektor GloVe 100d (unduh sekali, cache). Bandingkan akurasi klasifikasi melawan TF-IDF polos dan mean-pooled embedding polos pada dataset 20 Newsgroups. Laporkan mana yang menang di mana.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BoW | Word frequency vector | Counts of vocabulary words in one document. Throws away order. |
| TF | Term frequency | Count of a word in a document, optionally normalized by document length. |
| DF | Document frequency | Count of documents containing the word at least once. |
| IDF | Inverse document frequency | `log(N / df)` smoothed. Downweights words that appear everywhere. |
| Sparse vector | Mostly zeros | Vocabulary is typically 10k-100k words; most are absent from any given document. |
| Cosine similarity | Vector angle | Dot product of L2-normalized vectors. 1 is identical, 0 is orthogonal. |

## Bacaan Lanjutan

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — the canonical API reference, plus notes on every knob.
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — the paper that made TF-IDF the default for a decade.
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 take on when the old method wins and why.
