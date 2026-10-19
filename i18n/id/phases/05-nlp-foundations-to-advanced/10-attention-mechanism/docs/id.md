# Mekanisme Attention — Terobosan

> Decoder berhenti menyipitkan mata pada ringkasan terkompresi dan mulai melihat seluruh sumber. Semua setelah ini adalah attention plus engineering.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 09 (Sequence-to-Sequence Models)
**Time:** ~45 minutes

## Masalahnya

Pelajaran 09 berakhir pada kegagalan terukur. Encoder-decoder GRU yang dilatih pada tugas salin mainan turun dari akurasi 89% pada panjang 5 ke nyaris-chance pada panjang 80. Alasannya struktural, bukan bug training: setiap bit informasi yang dipungut encoder harus muat dalam satu hidden state berukuran tetap, dan decoder tak pernah melihat apa pun selain itu.

Bahdanau, Cho, dan Bengio menerbitkan perbaikan tiga baris pada 2014. Alih-alih memberi decoder hanya state akhir encoder, simpan setiap state encoder. Pada tiap langkah decoder, hitung rata-rata berbobot state encoder di mana bobotnya berkata "seberapa perlu decoder melihat posisi encoder `i` sekarang?" Rata-rata berbobot itu adalah konteks, dan ia berubah setiap langkah decoder.

Itulah seluruh idenya. Transformer memperluasnya. Self-attention menerapkannya pada satu sekuens. Multi-head attention menjalankannya paralel. Tetapi versi 2014 sudah mendobrak bottleneck, dan begitu kamu memahaminya, pivot ke transformer adalah engineering, bukan konseptual.

## Konsepnya

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

Pada tiap langkah decoder `t`:

1. Pakai hidden state decoder sebelumnya `s_{t-1}` sebagai **query**.
2. Skorkan terhadap setiap hidden state encoder `h_1, ..., h_T`. Satu skalar per posisi encoder.
3. Softmax skornya untuk mendapat bobot attention `α_{t,1}, ..., α_{t,T}` yang berjumlah 1.
4. Vektor konteks `c_t = Σ α_{t,i} * h_i`. Rata-rata berbobot state encoder.
5. Decoder mengambil `c_t` plus token output sebelumnya, menghasilkan token berikutnya.

Rata-rata berbobot adalah intinya. Ketika decoder perlu menerjemahkan "Je" menjadi "I", ia memboboti state encoder atas "Je" tinggi dan lainnya rendah. Ketika butuh "not", ia memboboti "pas" tinggi. Vektor konteks berubah bentuk tiap langkah.

## Bentuk (hal yang menggigit semua orang)

Di sinilah setiap implementasi attention salah pada percobaan pertama. Baca perlahan.

| Thing | Shape | Notes |
|-------|-------|-------|
| Encoder hidden states `H` | `(T_enc, d_h)` | If BiLSTM, `d_h = 2 * d_hidden` |
| Decoder hidden state `s_{t-1}` | `(d_s,)` | One vector |
| Attention score `e_{t,i}` | scalar | One per encoder position |
| Attention weight `α_{t,i}` | scalar | After softmax over all `i` |
| Context vector `c_t` | `(d_h,)` | Same shape as an encoder state |

**Skor Bahdanau (aditif).** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}` berbentuk `(d_s,)`, `h_i` berbentuk `(d_h,)`.
- `W_a` berbentuk `(d_attn, d_s)`. `U_a` berbentuk `(d_attn, d_h)`.
- Jumlahnya di dalam tanh berbentuk `(d_attn,)`.
- `v_α` berbentuk `(d_attn,)`. Inner product dengan `v_α` runtuh menjadi skalar. **Inilah tugas `v_α`.** Bukan sihir. Ia adalah proyeksi yang mengubah vektor berdimensi-attention menjadi skor skalar.

**Skor Luong (multiplikatif).** Tiga varian:

- `dot`: `e_{t,i} = s_t^T * h_i`. Mensyaratkan `d_s == d_h`. Konstrain keras. Lewati jika encodermu bidirectional.
- `general`: `e_{t,i} = s_t^T * W * h_i` dengan `W` berbentuk `(d_s, d_h)`. Menghapus konstrain dimensi setara.
- `concat`: pada dasarnya bentuk Bahdanau. Jarang dipakai karena dua yang pertama lebih murah.

**Satu gotcha Bahdanau / Luong yang patut disebut.** Bahdanau memakai `s_{t-1}` (state decoder *sebelum* menghasilkan kata kini). Luong memakai `s_t` (state *sesudahnya*). Mencampurnya menghasilkan gradien yang halus salahnya dan sangat sulit di-debug. Pilih satu paper dan patuhi konvensinya.

```figure
attention-heatmap
```

## Bangun Sendiri

### Langkah 1: attention aditif (Bahdanau)

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

Cek shape-mu terhadap tabel di atas. `encoder_states` berbentuk `(T_enc, d_h)`. `projected_enc` berbentuk `(T_enc, d_attn)`. `projected_dec` berbentuk `(d_attn,)` dan ter-broadcast. `combined` berbentuk `(T_enc, d_attn)`. `scores` berbentuk `(T_enc,)`. `weights` berbentuk `(T_enc,)`. `context` berbentuk `(d_h,)`. Rilis.

### Langkah 2: Luong dot dan general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

Masing-masing tiga baris. Inilah alasan paper Luong mendarat. Akurasi sama pada kebanyakan tugas, kode jauh lebih sedikit.

### Langkah 3: contoh numerik yang dikerjakan

Diberi tiga state encoder (kira-kira "cat", "sat", "mat") dan state decoder yang paling selaras dengan yang pertama, distribusi attention terkonsentrasi pada posisi 0. Jika state decoder bergeser selaras dengan yang terakhir, attention pindah ke posisi 2. Vektor konteks mengikuti.

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

Baris pertama menang. Lalu geser state decoder mendekati state encoder ketiga dan saksikan bobot bergeser. Itu saja. Attention adalah alignment eksplisit.

### Langkah 4: mengapa ini jembatan ke transformer

Terjemahkan bahasa di atas ke Q/K/V:

- **Query** = state decoder `s_{t-1}`
- **Key** = state encoder (yang kita skorkan)
- **Value** = state encoder (yang kita boboti dan jumlahkan)

Dalam attention klasik, key dan value adalah hal yang sama. Self-attention memisahkannya: kamu bisa meng-query sekuens terhadap dirinya sendiri, dengan proyeksi terpelajari berbeda untuk K dan V. Multi-head attention menjalankannya paralel dengan proyeksi terpelajari berbeda. Transformer menumpuk seluruh panggung berkali-kali dan membuang RNN.

Matematikanya sama. Shape-nya sama. Lompatan pedagogis dari attention Bahdanau ke scaled dot-product attention sebagian besar notasi.

## Gunakan

PyTorch dan TensorFlow menyediakan attention langsung.

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

Itu adalah layer attention transformer. Batch query 5 posisi, batch key/value 10 posisi, masing-masing 128-dim, 8 head. `output` adalah query baru yang diperkaya konteks. `weights` adalah matriks alignment 5x10 yang bisa kamu visualisasikan.

### Kapan attention klasik masih penting

- Pedagogi. Versi single-head, single-layer, berbasis RNN membuat setiap konsep terlihat.
- Tugas sekuens on-device di mana transformer tidak muat.
- Paper apa pun dari 2014-2017. Kamu akan salah baca tanpa tahu konvensi Bahdanau.
- Analisis alignment rinci dalam MT. Bobot attention mentah adalah alat interpretabilitas bahkan pada model transformer, dan membacanya menuntut tahu apa itu.

### Jebakan bobot-attention-sebagai-penjelasan

Bobot attention tampak dapat diinterpretasi. Mereka adalah bobot berjumlah satu lintas posisi; bisa kamu plot; tinggi berarti "melihat ini." Reviewer menyukainya.

Mereka tidak se-dapat-diinterpretasi kelihatannya. Jain and Wallace (2019) menunjukkan distribusi attention bisa dipermutasi dan diganti alternatif sembarang tanpa mengubah prediksi model untuk sebagian tugas. Jangan pernah melaporkan bobot attention sebagai bukti penalaran tanpa cek ablasi atau counterfactual.

## Rilis

Simpan sebagai `outputs/prompt-attention-shapes.md`:

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## Latihan

1. **Mudah.** Implementasikan masking `softmax` supaya token padding di encoder mendapat bobot attention nol. Uji pada batch dengan sekuens bervariabel panjang.
2. **Sedang.** Tambahkan multi-head attention ke bentuk `general` Luong. Pecah `d_h` menjadi `n_heads` grup, jalankan attention per head, gabungkan. Verifikasi kasus single-head cocok dengan implementasimu sebelumnya.
3. **Sulit.** Latih encoder-decoder GRU dengan attention Bahdanau pada tugas salin mainan dari pelajaran 09. Plot akurasi vs panjang sekuens. Bandingkan dengan baseline tanpa attention. Kamu akan melihat gap melebar seiring bertambahnya panjang, mengonfirmasi attention mengangkat bottleneck.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Attention | Looking at things | Weighted average of a value sequence, weights computed from a query-key similarity. |
| Query, Key, Value | QKV | Three projections: Q asks, K is what to match, V is what to return. |
| Additive attention | Bahdanau | Feed-forward score: `v^T tanh(W q + U k)`. |
| Multiplicative attention | Luong dot / general | Score is `q^T k` or `q^T W k`. Cheaper, same accuracy on most tasks. |
| Alignment matrix | The pretty picture | Attention weights as a `(T_dec, T_enc)` grid. Read it to see what the model attended to. |

## Bacaan Lanjutan

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — the paper.
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — the three score variants and their comparison.
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — the interpretability caveat.
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — runnable walkthrough with PyTorch.
