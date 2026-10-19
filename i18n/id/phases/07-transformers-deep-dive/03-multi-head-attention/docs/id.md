# Multi-Head Attention

> Satu head attention mempelajari satu relasi dalam satu waktu. Delapan head mempelajari delapan. Head itu gratis. Ambil lebih banyak.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## Masalahnya

Satu head self-attention menghitung satu matriks attention. Matriks itu menangkap satu jenis relasi — biasanya yang meminimalkan loss pada apa pun sinyal training-nya. Kalau datamu punya kesesuaian subjek-verba, ko-referensi, wacana jarak jauh, dan chunking sintaktis yang saling bertautan, satu head melebur semuanya menjadi satu distribusi soft-max dan kehilangan separuh sinyal.

Solusi dari makalah Vaswani 2017: jalankan beberapa fungsi attention secara paralel, masing-masing dengan proyeksi Q, K, V sendiri, dan gabungkan output-nya. Setiap head beroperasi dalam subruang berdimensi lebih kecil `d_model / n_heads`. Total parameter tetap sama. Daya ekspresif naik.

Multi-head attention adalah bawaan yang dibawa setiap transformer pada 2026. Satu-satunya perdebatan adalah soal *berapa banyak* head dan apakah key dan value berbagi proyeksi (Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention).

## Konsepnya

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split.** Ambil `X` ber-shape `(N, d_model)`. Proyeksikan ke Q, K, V masing-masing ber-shape `(N, d_model)`. Reshape menjadi `(N, n_heads, d_head)` dengan `d_head = d_model / n_heads`. Transpose menjadi `(n_heads, N, d_head)`.

**Attend secara paralel.** Jalankan scaled dot-product attention di dalam setiap head. Setiap head menghasilkan `(N, d_head)`. Head-head itu beroperasi pada subruang embedding yang berbeda dan tidak pernah berkomunikasi selama komputasi attention itu sendiri.

**Gabungkan dan proyeksikan.** Tumpuk kembali head menjadi `(N, d_model)` dan kalikan dengan matriks output terpelajari `W_o` ber-shape `(d_model, d_model)`. `W_o` adalah tempat head-head bercampur.

**Mengapa ini bekerja.** Setiap head bisa berspesialisasi tanpa berebut budget representasional dengan head lain. Studi probing 2019–2024 menunjukkan peran head yang berbeda: positional head, head yang attend ke token sebelumnya, copy head, named-entity head, induction head (yang mendasari in-context learning).

**Lini varian 2026:**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA adalah bawaan modern karena ia memangkas memori KV-cache dengan faktor `N/G` sambil menjaga kualitas nyaris penuh. MLA melangkah lebih jauh dengan mengompresi K/V ke ruang laten, lalu memproyeksikannya kembali saat komputasi — butuh FLOPs, menghemat jauh lebih banyak memori.

```figure
multihead-split
```

## Bangun Sendiri

### Langkah 1: pisahkan head dari attention satu-head yang sudah kita punya

Ambil `SelfAttention` dari Pelajaran 02 dan bungkus dengan pasangan split/concat. Lihat `code/main.py` untuk implementasi numpy; logikanya:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Satu reshape dan satu transpose. Tanpa loop. Inilah tepatnya yang dilakukan PyTorch di balik `nn.MultiheadAttention`.

### Langkah 2: jalankan scaled-dot-product attention per head

Setiap head mendapat irisannya sendiri dari Q, K, V. Attention menjadi batched matmul:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

Pada hardware nyata `Qh @ Kh.transpose(...)` adalah satu `bmm`. GPU melihat satu batched matmul ber-shape `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`. Menambah head itu gratis.

### Langkah 3: varian Grouped-Query Attention

Hanya proyeksi key dan value yang berubah. Q mendapat `n_heads` grup; K dan V mendapat `n_kv_heads < n_heads` grup dan diulang agar cocok:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

Saat inference ini menghemat memori karena hanya `n_kv_heads` salinan yang hidup di KV cache, bukan `n_heads`. Llama 3 70B memakai 64 query head dengan 8 KV head — penyusutan cache 8×.

### Langkah 4: selidiki apa yang dipelajari setiap head

Jalankan MHA pada kalimat pendek dengan 4 head. Untuk setiap head, cetak matriks attention `(N, N)`. Kamu akan melihat head berbeda menangkap struktur berbeda meski dengan inisialisasi acak — itu sebagian sinyal, sebagian simetri rotasional dalam subruang.

## Gunakan

Dalam PyTorch, versi satu barisnya:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA pada PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**Berapa banyak head?** Aturan praktis dari model produksi pada 2026:

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` hampir selalu jatuh pada 64 atau 128. Ia adalah satuan seberapa banyak yang bisa "dilihat" satu head. Turun di bawah 32 dan head mulai bergumul dengan faktor scaling `sqrt(d_head)`; naik di atas 256 dan kamu kehilangan manfaat "banyak spesialis kecil".

## Rilis

Lihat `outputs/skill-mha-configurator.md`. Skill ini merekomendasikan jumlah head, jumlah kv-head, dan strategi proyeksi untuk transformer baru berdasarkan budget parameter, panjang sequence, dan target deployment.

## Latihan

1. **Mudah.** Ambil MHA dari `code/main.py` dan ubah `n_heads` dari 1 menjadi 16 dengan `d_model=64` tetap. Plot loss model satu lapis mungil pada synthetic copy task. Apakah head yang lebih banyak membantu, plateau, atau merugikan?
2. **Sedang.** Implementasikan MQA (satu KV head dipakai bersama semua query head). Ukur seberapa besar jumlah parameter turun vs MHA penuh. Hitung seberapa besar ukuran KV-cache menyusut saat inference untuk N=2048.
3. **Sulit.** Implementasikan versi mungil Multi-head Latent Attention: kompres K,V ke laten ber-rank-`r`, simpan laten di KV cache, dekompresi saat waktu attention. Pada `r` berapa memori cache melewati di bawah 1/8 MHA penuh sementara kualitas tetap dalam 1 bit validation ppl?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Head | "A single attention circuit" | One Q/K/V projection of dimension `d_head = d_model / n_heads` with its own attention matrix. |
| d_head | "Head dimension" | Per-head hidden width; almost always 64 or 128 in production. |
| Split / combine | "Reshape tricks" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose around attention. |
| W_o | "Output projection" | `(d_model, d_model)` matrix applied after concatenating heads; where heads mix. |
| MQA | "One KV head" | Multi-Query Attention: single shared K/V projection. Smallest KV cache, some quality loss. |
| GQA | "The default since Llama 2" | Grouped-Query Attention with `n_kv_heads < n_heads`; repeats to match Q. |
| MLA | "DeepSeek's trick" | Multi-head Latent Attention: K,V compressed to low-rank latent, decompressed at attend time. |
| Induction head | "The circuit behind in-context learning" | A pair of heads that detect previous occurrences and copy what followed them. |

## Bacaan Lanjutan

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — spek multi-head orisinal.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — makalah MQA.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — cara mengonversi MHA ke GQA setelah training.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA dan mengapa ia mengalahkan MHA/GQA dalam memori cache.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — telaah mekanistis tentang apa yang sebenarnya dilakukan head.
