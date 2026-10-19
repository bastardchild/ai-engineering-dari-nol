# Generasi Teks Sebelum Transformer — Model Bahasa N-gram

> Jika sebuah kata mengejutkan, modelnya buruk. Perplexity menjadikan kejutan sebuah angka. Smoothing menjaganya tetap berhingga.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## Masalahnya

Sebelum transformer, sebelum RNN, sebelum word embedding, model bahasa memprediksi kata berikutnya dengan menghitung seberapa sering ia mengikuti `n-1` kata sebelumnya. Hitung "the cat" → "sat" 47 kali, "the cat" → "jumped" 12 kali, "the cat" → "refrigerator" 0 kali. Normalisasi untuk mendapat distribusi probabilitas.

Itulah model bahasa n-gram. Ia menjalankan setiap speech recognizer, setiap spell checker, dan setiap sistem terjemahan mesin berbasis frasa dari 1980 sampai 2015. Ia masih berjalan ketika kamu butuh pemodelan bahasa on-device yang murah.

Masalah menariknya adalah apa yang dilakukan terhadap n-gram tak terlihat. Model mentah berbasis hitungan memberi probabilitas nol pada apa pun yang belum dilihat, yang katastropik karena kalimat itu panjang dan hampir setiap kalimat panjang memuat setidaknya satu sekuens tak terlihat. Lima puluh tahun riset smoothing memperbaikinya. Kneser-Ney smoothing adalah hasilnya, dan deep learning modern mewarisi tradisi empirisnya.

## Konsepnya

![N-gram model: count, smooth, generate](../assets/ngram.svg)

### Permainan prediksi

Sebelum semua mesin ini ada, satu eksperimen mendefinisikan apa itu model bahasa. Tutupi huruf berikutnya dari kalimat Inggris. Minta seseorang menebaknya, satu tebakan per satu, sampai benar. Catat jumlah tebakannya. Ulangi untuk beberapa ratus huruf.

Jumlah tebakan bukan trivia. Mereka adalah re-encoding lossless dari teks: berikan sekuens hitungan ke penebak kedua yang identik dan mereka bisa merekonstruksi setiap huruf, karena pada tiap posisi mereka tahu persis tebakan mana yang lebih dulu. Pesan yang bisa di-encode ulang dalam simbol lebih sedikit membawa informasi lebih sedikit per simbol, sehingga statistik jumlah tebakan menaruh langit-langit pada entropi bahasa Inggris.

Shannon menjalankannya tahun 1951 dan mendapat angka yang masih mengatur bidang ini. Alfabet 27-simbol (26 huruf plus spasi) bisa membawa `log2(27) ≈ 4.75` bit per huruf. Penebak manusia dengan 100 huruf konteks mendarat antara 0.6 dan 1.3 bit per huruf. Bahasa Inggris kira-kira tiga perempatnya langkah paksa. Struktur yang harus dipelajari model telah diukur sebelum model mana pun bisa mempelajarinya.

Setiap model bahasa sejak itu adalah pemain mekanis permainan ini, dan setiap angka evaluasi di pelajaran ini adalah permainan yang diskorkan:

- **Loss cross-entropy** adalah rata-rata jumlah bit yang dibutuhkan model per simbol. Melatih LM secara harfiah meminimalkan skornya pada permainan menebak.
- **Perplexity** adalah `2^bits` (atau `e^nats`): faktor percabangan yang masih dihadapi model setelah tebakannya. Tebakan seragam atas 27 simbol adalah perplexity 27; pemain 1-bit-per-huruf punya perplexity 2.
- **Panjang konteks adalah memori pemain.** Model trigram bermain dengan memori dua token. Transformer memainkan permainan sama dengan 100K token. Aturannya tak pernah berubah; pemainnya membaik.

Satu peralihan satuan yang perlu dilacak: permainan menilai per huruf dalam bit (`log2`), sementara rumus n-gram di bawah menilai per token kata dalam nat (log natural) — dan karena perplexity `e^H` dalam nat sama dengan `2^H` dalam bit, kedua pandangan adalah pengukuran sama dalam satuan berbeda.

```figure
prediction-game
```

**Probabilitas N-gram:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. Tetapkan `n` (umumnya 3 untuk trigram, 4 untuk 4-gram). Hitung dari hitungan:

```text
P(w | context) = count(context, w) / count(context)
```

**Masalah hitungan-nol.** N-gram apa pun yang tak terlihat saat training mendapat probabilitas nol. Studi 2007 pada korpus Brown menemukan bahkan model 4-gram punya 30% 4-gram held-out yang tak terlihat saat training. Kamu tidak bisa mengevaluasi pada teks nyata mana pun tanpa smoothing.

**Pendekatan smoothing, berurutan kecanggihan:**

1. **Laplace (add-one).** Tambah 1 ke setiap hitungan. Sederhana, buruk pada kejadian jarang.
2. **Good-Turing.** Realokasi massa probabilitas dari kejadian frekuensi-tinggi ke yang tak terlihat berdasarkan frekuensi-dari-frekuensi.
3. **Interpolasi.** Gabungkan estimasi n-gram, (n-1)-gram, dst., dengan bobot yang bisa di-tune.
4. **Backoff.** Jika hitungan n-gram nol, mundur ke (n-1)-gram. Backoff Katz menormalkannya.
5. **Absolute discounting.** Kurangi diskon tetap `D` dari semua hitungan, distribusikan ulang ke yang tak terlihat.
6. **Kneser-Ney.** Absolute discounting plus pilihan cerdas untuk model orde-rendah: pakai *probabilitas kontinuitas* (di berapa konteks sebuah kata muncul) alih-alih frekuensi mentah.

Wawasan Kneser-Ney itu dalam. "San Francisco" adalah bigram umum. Unigram "Francisco" muncul mostly setelah "San." Discounting absolut naif memberi "Francisco" probabilitas unigram tinggi (karena hitungannya tinggi). Kneser-Ney memperhatikan bahwa "Francisco" muncul hanya dalam satu konteks dan menurunkan probabilitas kontinuitasnya. Hasilnya: bigram baru yang berakhiran "Francisco" mendapat probabilitas rendah yang semestinya.

**Evaluasi: perplexity.** Eksponen dari rata-rata negative log-likelihood per kata pada test set held-out. Lebih rendah lebih baik. Perplexity 100 berarti model sebingung memilih seragam di antara 100 kata.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## Bangun Sendiri

### Langkah 1: hitungan trigram

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

Input adalah daftar kalimat yang sudah di-tokenisasi. Output adalah hitungan n-gram dan hitungan konteks. `<s>` dan `</s>` adalah batas kalimat.

### Langkah 2: smoothing Laplace

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

Tambah 1 ke setiap hitungan. Menghaluskan tetapi mengalokasi massa berlebih ke kejadian tak terlihat, merugikan kejadian-jarang-dikenal juga.

### Langkah 3: Kneser-Ney (bigram, terinterpolasi)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

Tiga bagian bergerak. `continuation_prob` menangkap "di berapa konteks berbeda kata ini muncul?" (inovasi Kneser-Ney). `lambda_prev` adalah massa yang dibebaskan diskon, dipakai untuk memboboti backoff. Probabilitas final adalah term utama terdiskon plus term kontinuitas terbobot.

### Langkah 4: menghasilkan teks dengan sampling

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

Sampling proporsional terhadap probabilitas. Selalu memberi output berbeda per seed. Untuk output ala beam-search, pilih argmax pada tiap langkah (greedy) dan tambahkan kenop keacakan kecil (temperature).

### Langkah 5: perplexity

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

Lebih rendah lebih baik. Untuk korpus Brown, model KN 4-gram yang di-tune baik mencapai perplexity sekitar 140. LM transformer mencapai 15-30 pada test set sama. Gap-nya sekitar 10x. Gap itulah alasan bidang ini move on.

## Gunakan

- **Pengajaran NLP klasik.** Paparan paling jelas terhadap smoothing, MLE, dan perplexity yang bisa kamu dapat.
- **KenLM.** Pustaka n-gram produksi. Dipakai sebagai rescorer dalam sistem speech dan MT di mana latensi rendah penting.
- **Autocomplete on-device.** Model trigram di keyboard. Masih.
- **Baseline.** Selalu hitung perplexity LM n-gram sebelum menyatakan LM neuralmu bagus. Jika transformermu tidak mengalahkan KN dengan margin lebar, ada yang salah.

## Rilis

Simpan sebagai `outputs/prompt-lm-baseline.md`:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## Latihan

1. **Mudah.** Latih LM trigram pada korpus Shakespeare 1.000-kalimat. Hasilkan 20 kalimat. Mereka akan plausible secara lokal tetapi inkoheren secara global. Ini demo kanonis.
2. **Sedang.** Implementasikan perplexity untuk model KN-mu pada split Shakespeare held-out. Bandingkan dengan Laplace. Kamu akan melihat KN menurunkan perplexity 30-50%.
3. **Sulit.** Bangun korektor ejaan trigram: diberi kata salah eja dan konteksnya, hasilkan koreksi dan rangking berdasarkan probabilitas konteks di bawah LM. Evaluasi pada korpus ejaan Birkbeck (publik).

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| N-gram | Word sequence | Sequence of `n` consecutive tokens. |
| Smoothing | Avoiding zeros | Reallocating probability mass so unseen events get non-zero probability. |
| Perplexity | LM quality metric | `exp(-average log-prob)` on held-out data. Lower is better. |
| Backoff | Fallback to shorter context | If trigram count is zero, use bigram. Katz backoff formalizes this. |
| Kneser-Ney | Best smoothing for n-grams | Absolute discounting + continuation probability for the lower-order model. |
| Continuation probability | KN-specific | `P(w)` weighted by number of contexts `w` appears in, not by raw count. |
| Entropy of text | Information per symbol | Average bits needed to encode the next symbol given the context. Shannon's 1951 estimate for printed English with up to 100 letters of context: 0.6-1.3 bits/letter, measured before any model existed. |

## Bacaan Lanjutan

- [Shannon (1951). Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) — the guessing-game experiment that defined the target every language model still optimizes.
- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — the canonical treatment of n-gram LMs and smoothing.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — the paper that settled Kneser-Ney as the best n-gram smoother.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — the original KN paper.
- [KenLM](https://kheafield.com/code/kenlm/) — fast production n-gram LM, still used in 2026 for latency-sensitive applications.
