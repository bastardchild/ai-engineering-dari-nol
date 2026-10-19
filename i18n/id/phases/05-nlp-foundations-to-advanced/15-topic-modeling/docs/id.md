# Pemodelan Topik — LDA dan BERTopic

> LDA: dokumen adalah campuran topik, topik adalah distribusi atas kata. BERTopic: dokumen mengelompok dalam ruang embedding, klaster adalah topik. Tujuan sama, dekomposisi berbeda.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45 minutes

## Masalahnya

Kamu punya 10.000 tiket dukungan pelanggan, 50.000 artikel berita, atau 200.000 tweet. Kamu perlu tahu koleksi itu tentang apa tanpa membacanya. Kamu tidak punya kategori berlabel. Kamu bahkan tidak tahu berapa kategori yang ada.

Pemodelan topik menjawabnya tanpa supervisi. Beri korpus, dapatkan kembali himpunan kecil topik koheren dan, untuk tiap dokumen, distribusi atas topik itu.

Dua keluarga algoritme mendominasi. LDA (2003) memperlakukan tiap dokumen sebagai campuran topik laten dan tiap topik sebagai distribusi atas kata. Inferensinya Bayesian. Ia masih dikirim di produksi di mana kamu butuh penugasan topik keanggotaan-campuran dan distribusi probabilitas level-kata yang dapat dijelaskan.

BERTopic (2020) meng-encode dokumen dengan BERT, mereduksi dimensionalitas dengan UMAP, mengelompokkan dengan HDBSCAN, dan mengekstrak kata topik via TF-IDF berbasis kelas. Ia menang pada teks pendek, media sosial, dan apa pun di mana kemiripan semantik lebih penting daripada overlap kata. Satu dokumen mendapat satu topik, yang merupakan keterbatasan untuk konten long-form.

Pelajaran ini membangun intuisi untuk keduanya dan menyebutkan mana yang dipilih untuk korpus tertentu.

## Konsepnya

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**Cerita generatif LDA.** Setiap topik adalah distribusi atas kata. Setiap dokumen adalah campuran topik. Untuk menghasilkan kata dalam dokumen, sampel topik dari campuran dokumen, lalu sampel kata dari distribusi topik itu. Inferensi membaliknya: diberi kata teramati, inferensikan distribusi topik per dokumen dan distribusi kata per topik. Collapsed Gibbs sampling atau variational Bayes mengerjakan matematikanya.

Output kunci LDA:

- `doc_topic`: matriks `(n_docs, n_topics)`, tiap baris berjumlah 1 (campuran topik dokumen).
- `topic_word`: matriks `(n_topics, vocab_size)`, tiap baris berjumlah 1 (distribusi kata topik).

**Pipeline BERTopic.**

1. Encode tiap dokumen dengan sentence transformer (mis. `all-MiniLM-L6-v2`). Vektor 384-dim.
2. Reduksi dimensionalitas dengan UMAP ke ~5 dimensi. Embedding BERT terlalu tinggi-dim untuk clustering.
3. Kelompokkan dengan HDBSCAN. Berbasis densitas, menghasilkan klaster berukuran variabel dan label "outlier".
4. Untuk tiap klaster, hitung TF-IDF berbasis kelas atas dokumen klaster untuk mengekstrak kata teratas.

Output adalah satu topik per dokumen (plus label outlier -1). Opsional, keanggotaan lunak via vektor probabilitas HDBSCAN.

```figure
topic-drift
```

## Bangun Sendiri

### Langkah 1: LDA via scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

Perhatikan: stopword dibuang, min_df dan max_df memfilter term jarang dan ada-di-mana-mana, CountVectorizer (bukan TfidfVectorizer) karena LDA mengharapkan hitungan mentah.

### Langkah 2: BERTopic (produksi)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

Filter pada `Topic != -1` membuang bucket outlier BERTopic (dokumen yang tak bisa dikelompokkan HDBSCAN). `min_topic_size` mengontrol ukuran klaster minimum HDBSCAN; default pustaka BERTopic adalah 10. Contoh ini menyetelnya ke 15 secara eksplisit untuk skala pelajaran. Untuk korpus di atas 10.000 dokumen, naikkan ke 50 atau 100.

### Langkah 3: evaluasi

Kedua metode mengeluarkan kata topik. Pertanyaannya apakah kata itu koheren.

- **Koherensi topik (c_v).** Menggabungkan NPMI (normalized pointwise mutual information) pasangan kata-teratas atas konteks sliding-window, mengagregasi skor menjadi vektor topik, dan membandingkan vektor itu via kemiripan kosinus. Lebih tinggi lebih baik. Pakai `gensim.models.CoherenceModel` dengan `coherence="c_v"`.
- **Diversitas topik.** Fraksi kata unik lintas kata-teratas semua topik. Lebih tinggi lebih baik (topik tidak overlap).
- **Inspeksi kualitatif.** Baca kata teratas tiap topik. Apakah mereka menamai hal nyata? Penilaian manusia masih garis pertahanan terakhir.

## Kapan memilih yang mana

| Situation | Pick |
|-----------|------|
| Short text (tweets, reviews, headlines) | BERTopic |
| Long documents with topic mixtures | LDA |
| No GPU / limited compute | LDA or NMF |
| Need document-level multi-topic distributions | LDA |
| LLM integration for topic labeling | BERTopic (direct support) |
| Resource-constrained edge deployment | LDA |
| Max semantic coherence | BERTopic |

Pertimbangan praktis terbesar adalah panjang dokumen. Embedding BERT ter-truncate; hitungan LDA bekerja pada panjang berapa pun. Untuk dokumen lebih panjang dari konteks model embedding, chunk + agregat atau pakai LDA.

## Gunakan

Stack 2026:

- **BERTopic.** Default untuk teks pendek dan apa pun di mana semantik penting.
- **`gensim.models.LdaModel`.** LDA klasik untuk produksi, matang, teruji tempur.
- **`sklearn.decomposition.LatentDirichletAllocation`.** LDA mudah untuk eksperimen.
- **NMF.** Faktorisasi matriks non-negatif. Alternatif cepat untuk LDA, kualitas sebanding pada teks pendek.
- **Top2Vec.** Desain mirip BERTopic. Komunitas lebih kecil tetapi bagus pada sebagian benchmark.
- **FASTopic.** Lebih baru, lebih cepat daripada BERTopic pada korpus sangat besar.
- **Pelabelan berbasis LLM.** Jalankan clustering apa pun, lalu prompt model untuk menamai tiap klaster.

## Rilis

Simpan sebagai `outputs/skill-topic-picker.md`:

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## Latihan

1. **Mudah.** Fit LDA dengan 5 topik pada dataset 20 Newsgroups. Cetak 10 kata teratas per topik. Labeli tiap topik dengan tangan. Apakah algoritme menemukan kategori aslinya?
2. **Sedang.** Fit BERTopic pada subset 20 Newsgroups yang sama. Bandingkan jumlah topik yang ditemukan, kata teratas, dan koherensi kualitatif melawan LDA. Mana yang memunculkan kategori asli lebih bersih?
3. **Sulit.** Hitung koherensi c_v untuk LDA dan BERTopic pada korpusmu. Jalankan masing-masing dengan 5, 10, 20, 50 topik. Plot koherensi vs jumlah topik. Laporkan metode mana yang lebih stabil lintas jumlah topik.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Topic | A thing the corpus is about | A probability distribution over words (LDA) or a cluster of similar documents (BERTopic). |
| Mixed membership | Doc is multiple topics | LDA assigns each document a distribution over all topics. |
| UMAP | Dimensionality reduction | Manifold learning that preserves local structure; used in BERTopic. |
| HDBSCAN | Density clustering | Finds variable-size clusters; produces "noise" label (-1) for outliers. |
| c_v coherence | Topic quality metric | Average pointwise mutual information of top topic words within sliding windows. |

## Bacaan Lanjutan

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — the LDA paper.
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — the BERTopic paper.
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — the paper that introduced c_v and friends.
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — the production reference. Excellent examples.
