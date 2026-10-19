# Varian Attention — Sliding Window, Sparse, Differential

> Attention penuh adalah lingkaran. Setiap token melihat setiap token, dan memori membayar harganya. Empat varian menekuk bentuk lingkaran dan memulihkan separuh biaya.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**Time:** ~60 minutes

## Masalahnya

Attention penuh butuh memori `O(N²)` dan komputasi `O(N²)` dalam panjang sequence. Untuk Llama 3 70B konteks-128K itu berarti 16 miliar entri attention per lapis, kali 80 lapis. Flash Attention (Pelajaran 12) menyembunyikan memori aktivasi `O(N²)` tapi tidak mengubah biaya aritmetika — setiap token tetap attend ke setiap token lain.

Tiga kelas varian mengubah topologi matriks attention itu sendiri:

1. **Sliding window attention (SWA).** Setiap token attend ke window tetangga yang tetap, bukan seluruh prefix. Memori dan komputasi turun menjadi `O(N · W)` dengan `W` adalah window. Gemma 2/3, lapis pertama Mistral 7B, Phi-3-Long.
2. **Sparse / block attention.** Hanya pasangan terpilih `(i, j)` yang diberi skor; sisanya dipaksa berbobot nol. Longformer, BigBird, sparse transformer OpenAI.
3. **Differential attention.** Hitung dua peta attention dengan proyeksi Q/K terpisah, kurangkan satu dari yang lain. Membunuh "attention sink" yang membocorkan bobot ke beberapa token pertama. DIFF Transformer Microsoft (2024).

Ini hidup berdampingan. Model frontier 2026 sering mencampurnya: sebagian besar lapis adalah SWA-1024, setiap kelima adalah attention penuh global, dan segelintir adalah differential head yang membersihkan retrieval. Rasio SWA-ke-global 5:1 milik Gemma 3 adalah bawaan textbook saat ini.

## Konsepnya

### Sliding Window Attention (SWA)

Setiap query pada posisi `i` hanya attend ke posisi dalam `[i - W, i]` (SWA kausal) atau `[i - W/2, i + W/2]` (dua arah). Token di luar window mendapat `-inf` dalam matriks skor.

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

Untuk `N = 8192` dan `W = 1024`, matriks skor punya 1024 × 8192 baris non-nol dalam ekspektasi — reduksi 8×.

**KV cache menyusut dengan SWA.** Hanya `W` token terakhir dari K dan V yang perlu disimpan per lapis. Untuk konfigurasi ala-Gemma-3 (window 1024, konteks 128K), KV cache turun 128×.

**Biaya kualitas.** Transformer SWA-saja kesulitan dengan retrieval jarak jauh. Solusinya: selingi lapis SWA dengan lapis attention-penuh. Gemma 3 memakai 5:1 SWA:global. Mistral 7B memakai stack SWA-kausal di mana informasi "mengalir maju" melalui window yang bertumpuk — setiap lapis memperpanjang receptive field efektif sebesar `W`, dan setelah `L` lapis model bisa attend `L × W` token ke belakang.

### Sparse / Block Attention

Pilih pola sparsity `N × N` di muka. Tiga bentuk kanonis:

- **Lokal + strided (sparse transformer OpenAI).** Attend ke `W` token terakhir plus setiap token ke-`stride` sebelumnya. Menangkap lokal sekaligus jarak jauh pada komputasi `O(N · sqrt(N))`.
- **Longformer / BigBird.** Window lokal + segelintir token global (mis. `[CLS]`) yang attend ke semua dan di-attend semua + tautan random-sparse. Konteks 2× empiris pada kualitas sebanding.
- **Native Sparse Attention (DeepSeek, 2025).** Pelajari block `(Q, K)` mana yang penting; lewati block nol di level kernel. Kompatibel-FlashAttention.

Sparse attention adalah cerita kernel-engineering. Matematikanya sederhana (mask matriks skor); kemenangannya datang dari tidak pernah memuat entri nol ke SRAM. FlashAttention-3 dan API FlexAttention 2026 membuat pola sparse kustom first-class di PyTorch.

### Differential Attention (DIFF Transformer, 2024)

Attention reguler punya problem "attention sink": softmax memaksa setiap baris berjumlah 1, jadi token yang tidak ingin attend ke apa pun membuang bobot ke token pertama (atau beberapa pertama). Ini mencuri kapasitas yang seharusnya ke konten nyata.

Differential attention memperbaikinya dengan menghitung **dua** peta attention dan mengurangkan:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

dengan `λ` adalah skalar terpelajari (tipikalnya 0.5–0.8). A1 menangkap bobot konten nyata; A2 menangkap sink. Pengurangan membatalkan sink, merealokasi bobot ke token relevan.

Hasil yang dilaporkan (Microsoft 2024): perplexity 5–10% lebih rendah, konteks efektif 1.5–2× lebih panjang pada panjang terlatih yang sama, retrieval needle-in-haystack lebih tajam.

### Perbandingan Varian

| Variant | Compute | KV cache | Quality vs full | Production use |
|---------|---------|----------|-----------------|----------------|
| Full attention | O(N²) | O(N) per layer | baseline | every model's default layer |
| SWA (window 1024) | O(N·W) | O(W) per layer | -0.1 ppl, good with global layers | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | O(N·√N) | mixed | similar to SWA | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | O(N) approx | mixed | matches full at 2× context | early long-context BERT |
| Native Sparse (DeepSeek-V3.2) | O(N · active fraction) | O(N) | within 0.05 ppl | DeepSeek-V3.2, 2025 |
| Differential | O(2·N²) | O(2N) | -5 to -10% ppl | DIFF Transformer, early 2026 models |

```figure
gqa-kv-sharing
```

## Bangun Sendiri

Lihat `code/main.py`. Kita mengimplementasikan pembanding causal mask yang menunjukkan attention penuh, SWA, local+strided, dan differential berdampingan pada sequence mainan.

### Langkah 1: mask kausal penuh (baseline)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Baseline dari Pelajaran 07. Segitiga-bawah; bobot nol di atas diagonal.

### Langkah 2: mask kausal sliding window

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

Satu parameter — `window`. Untuk `window >= n`, kamu memulihkan attention kausal penuh. Untuk `window = 1`, setiap token hanya attend ke dirinya sendiri.

### Langkah 3: mask sparse lokal + strided

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Window lokal padat plus setiap token ke-`stride` kembali ke awal sequence. Receptive field tumbuh dalam langkah log dengan lapis tambahan.

### Langkah 4: differential attention

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

Dua pass attention, kurangkan dengan koefisien pencampuran terpelajari. Dalam kode kita membandingkan heatmap attention-sink milik tunggal vs differential dan menyaksikan sink runtuh.

### Langkah 5: ukuran KV cache

Cetak ukuran cache per lapis pada `N = 131072` untuk tiap varian. Varian SWA dan sparse turun 10–100×. Differential berlipat dua. Bayar tagihan memorimu dengan sadar.

## Gunakan

Pola produksi 2026:

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

FlexAttention di PyTorch 2.5+ menerima fungsi mask:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

Ini mengompilasi menjadi kernel Triton kustom. Dalam 10% kecepatan FlashAttention-3 untuk pola umum, dan fungsi mask-nya adalah callable Python.

**Kapan memilih masing-masing:**

- **Attention penuh murni** — setiap lapis sampai konteks ~16K, atau ketika kualitas retrieval terpenting.
- **Campuran SWA + global** — konteks panjang (>32K), memory-bound saat training dan inference. Bawaan 2026 di atas 32K.
- **Block attention sparse** — kernel kustom, pola kustom. Dicadangkan untuk workload khusus (retrieval, audio).
- **Differential attention** — workload apa pun di mana kontaminasi attention-sink merugikan (RAG konteks panjang, needle-in-haystack).

## Rilis

Lihat `outputs/skill-attention-variant-picker.md`. Skill ini memilih topologi attention untuk model baru berdasarkan panjang konteks target, tuntutan retrieval, dan profil komputasi training/inference.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Verifikasi SWA pada `window=4` menolkan semua di luar 4 token terakhir per baris. Verifikasi `window=n` mereproduksi attention kausal penuh secara bit-identik.
2. **Sedang.** Implementasikan SWA kausal dengan `window=1024` di atas capstone Pelajaran 07. Latih 1.000 langkah pada tinyshakespeare. Seberapa besar val loss regresi vs attention penuh? Seberapa besar memori puncak turun?
3. **Sulit.** Implementasikan campuran lapis 5:1 ala-Gemma-3 (5 SWA, 1 global) dalam model capstone. Bandingkan loss, memori, dan kualitas pembangkitan terhadap baseline SWA-murni dan global-murni pada parameter sebanding.
4. **Sulit.** Implementasikan differential attention dengan `λ` terpelajari per head. Latih pada task retrieval sintetis (satu needle, 2.000 distractor). Ukur akurasi retrieval vs baseline attention-tunggal pada parameter sebanding.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Sliding window attention (SWA) | "Local attention" | Each query attends to its last `W` tokens; KV cache shrinks to `O(W)`. |
| Effective receptive field | "How far back the model sees" | In an `L`-layer SWA stack with window `W`, up to `L × W` tokens. |
| Longformer / BigBird | "Local + global + random" | Sparse patterns with a few always-attending global tokens; early long-context approach. |
| Native Sparse Attention | "DeepSeek's kernel trick" | Learn block-level sparsity; skip zero blocks at the kernel level while keeping quality. |
| Differential attention | "Two maps, one subtracts" | DIFF Transformer: subtract a learned `λ` times a second attention map from the first to cancel attention sinks. |
| Attention sink | "Weight bleeds to token 0" | Softmax normalization forces rows to sum to 1; uninformative queries dump weight on position 0. |
| FlexAttention | "Mask-as-Python" | PyTorch 2.5+ API that compiles arbitrary mask functions into FlashAttention-shape kernels. |
| Layer type mix | "5:1 SWA-to-global" | Interleave sparse and full attention layers in a stack to keep quality at lower memory. |

## Bacaan Lanjutan

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — makalah sliding-window + token-global kanonis.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — lokal + global + acak.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — pola lokal+strided OpenAI.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — campuran 1:1 SWA:global.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — campuran 5:1 dengan window=1024 yang kini menjadi bawaan textbook.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — makalah DIFF Transformer.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — attention sparsity-terpelajari milik DeepSeek-V3.2.
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — referensi API untuk pola mask-sebagai-callable di Gunakan.
