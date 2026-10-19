# GloVe, FastText, dan Embedding Subword

> Word2Vec melatih satu embedding per kata. GloVe memfaktorkan matriks ko-okurensi. FastText meng-embedding potongannya. BPE menjembatani ke transformer.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 minutes

## Masalahnya

Word2Vec menyisakan dua pertanyaan terbuka.

Pertama, ada lini riset paralel yang memfaktorkan matriks ko-okurensi secara langsung (LSA, HAL) alih-alih melakukan update skip-gram online. Apakah pendekatan iteratif Word2Vec secara fundamental lebih baik, atau perbedaannya sekadar artefak cara kedua metode menangani hitungan? **GloVe** menjawabnya: faktorisasi matriks dengan loss yang dipilih secara cermat menyamai atau mengalahkan Word2Vec, dan lebih murah dilatih.

Kedua, tak ada metode yang punya cerita untuk kata yang belum pernah dilihat. `Zoomer-approved`, `dogecoin`, proper noun apa pun yang diciptakan minggu lalu, setiap bentuk infleksi dari akar yang jarang. **FastText** memperbaikinya dengan meng-embedding n-gram karakter: sebuah kata adalah jumlah bagian-bagiannya, termasuk morfem, sehingga bahkan kata out-of-vocabulary mendapat vektor yang masuk akal.

Ketiga, begitu transformer tiba, pertanyaannya bergeser lagi. Vocabulary level kata mentok sekitar sejuta entri; bahasa nyata lebih terbuka dari itu. **Byte-pair encoding (BPE)** dan kerabatnya memecahkannya dengan mempelajari vocabulary unit subword yang sering muncul sehingga mencakup segalanya. Setiap tokenizer modern untuk setiap LLM modern adalah tokenizer subword.

Pelajaran ini membahas ketiganya, lalu menjelaskan mana yang dipakai kapan.

## Konsepnya

**GloVe (Global Vectors).** Bangun matriks ko-okurensi kata-kata `X` di mana `X[i][j]` adalah seberapa sering kata `j` muncul dalam konteks kata `i`. Latih vektor sehingga `v_i · v_j + b_i + b_j ≈ log(X[i][j])`. Boboti loss supaya pasangan sering tidak mendominasi. Selesai.

**FastText.** Sebuah kata adalah jumlah n-gram karakternya plus kata itu sendiri. `where` menjadi `<wh, whe, her, ere, re>, <where>`. Vektor kata adalah jumlah vektor-vektor komponen itu. Latih seperti Word2Vec. Manfaatnya: kata tak terlihat (`whereupon`) tersusun dari n-gram yang dikenal.

**BPE (Byte-Pair Encoding).** Mulai dengan vocabulary berisi byte individual (atau karakter). Hitung setiap pasangan bersebelahan di korpus. Gabungkan pasangan tersering menjadi token baru. Ulangi selama `k` iterasi. Hasil: vocabulary berisi `k + 256` token di mana sekuens sering (`ing`, `tion`, `the`) menjadi token tunggal dan kata jarang pecah menjadi potongan familiar. Setiap kalimat ter-tokenisasi menjadi sesuatu.

```figure
n5-subword-merge
```

## Bangun Sendiri

### GloVe: faktorkan matriks ko-okurensi

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

Dua bagian bergerak yang patut disebut. Fungsi pembobot `f(x) = (x/x_max)^alpha` mengecilkan pasangan sangat sering (seperti `(the, and)`) supaya tidak mendominasi loss. Embedding final adalah jumlah tabel `W` (pusat) dan `W_tilde` (konteks). Menjumlahkan keduanya adalah trik yang dipublikasikan dan cenderung mengungguli pemakaian satu tabel saja.

### FastText: embedding sadar-subword

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

Setiap kata direpresentasikan oleh himpunan n-gram-nya (umumnya 3 sampai 6 karakter). Embedding kata adalah jumlah embedding n-gram-nya. Untuk training skip-gram, pasang ini di tempat Word2Vec memakai satu vektor.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

Untuk kata tak terlihat, kamu tetap mendapat vektor selama sebagian n-gram-nya dikenal. `whereupon` berbagi `<wh`, `her`, `ere`, dan `<where` dengan `where`, sehingga keduanya mendarat berdekatan.

### BPE: vocabulary subword terpelajari

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

Iterasi pertama menggabungkan pasangan bersebelahan yang paling umum. Setelah iterasi cukup, substring sering (`low`, `est`, `tion`) menjadi token tunggal dan kata jarang pecah dengan bersih.

Tokenizer GPT / BERT / T5 asli mempelajari 30k-100k merge. Hasilnya: teks apa pun ter-tokenisasi menjadi sekuens terbatas dari ID yang dikenal, tanpa OOV selamanya.

## Gunakan

Dalam praktiknya, kamu jarang melatih semua ini sendiri. Kamu memuat checkpoint pra-terlatih.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Untuk tokenisasi subword gaya BPE di era transformer:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

Prefiks `Ġ` menandai batas kata (konvensi GPT-2). Setiap tokenizer modern adalah varian BPE, WordPiece (BERT), atau SentencePiece (T5, LLaMA).

### Kapan memilih yang mana

| Situation | Pick |
|-----------|------|
| Pretrained general-purpose word vectors, no OOV tolerance needed | GloVe 300d |
| Pretrained general-purpose word vectors, must handle misspellings / neologisms / morphologically rich languages | FastText |
| Anything going into a transformer (training or inference) | Whatever tokenizer the model shipped with. Never swap. |
| Training your own language model from scratch | Train a BPE or SentencePiece tokenizer on your corpus first |
| Production text classification with a linear model | Still TF-IDF. Lesson 02. |

## Rilis

Simpan sebagai `outputs/skill-embeddings-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## Latihan

1. **Mudah.** Jalankan `char_ngrams("playing")` dan `char_ngrams("played")`. Hitung overlap Jaccard dari kedua himpunan n-gram. Kamu akan melihat banyak bagian bersama (`pla`, `lay`, `play`), itulah alasan FastText mentransfer dengan baik antar varian morfologis.
2. **Sedang.** Perluas `learn_bpe` untuk melacak pertumbuhan vocabulary. Plot token-per-karakter-korpus sebagai fungsi jumlah merge. Kamu akan melihat kompresi cepat di awal, mendatar mendekati ~2-3 karakter per token.
3. **Sulit.** Latih BPE 1k-merge pada karya lengkap Shakespeare. Bandingkan tokenisasi kata umum vs. proper noun jarang. Ukur rata-rata token per kata sebelum dan sesudah. Tulis apa yang mengejutkanmu.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Co-occurrence matrix | Word-word frequency table | `X[i][j]` = how often word `j` appears in a window around word `i`. |
| Subword | Piece of a word | A character n-gram (FastText) or learned token (BPE/WordPiece/SentencePiece). |
| BPE | Byte-pair encoding | Iterative merging of most-frequent adjacent pairs until vocabulary hits target size. |
| OOV | Out of vocabulary | Word the model has never seen. Word2Vec/GloVe fail. FastText and BPE handle it. |
| Byte-level BPE | BPE on raw bytes | GPT-2's scheme. Vocabulary starts with 256 bytes, so nothing is ever OOV. |

## Bacaan Lanjutan

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — the GloVe paper, seven pages, still the best derivation of the loss.
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — the paper that introduced BPE to modern NLP.
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — how BPE, WordPiece, and SentencePiece actually differ in practice.
