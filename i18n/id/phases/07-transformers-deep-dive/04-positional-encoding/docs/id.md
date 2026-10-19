# Positional Encoding — Sinusoidal, RoPE, ALiBi

> Attention itu permutation-invariant. "The cat sat on the mat" dan "mat the on sat cat the" menghasilkan output yang sama tanpa sinyal posisional. Tiga algoritma memperbaikinya — masing-masing dengan taruhan berbeda tentang arti "posisi".

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## Masalahnya

Scaled dot-product attention itu buta urutan. Matriks attention `softmax(Q K^T / √d) V` dihitung dari kemiripan berpasangan. Acak baris-baris `X`, dapatkan baris-baris output teracak dengan cara yang sama. Tidak ada apa pun di dalam attention yang peduli posisi.

Itu bukan bug pada model bag-of-words. Untuk bahasa, kode, audio, video — apa pun di mana urutan membawa makna — itu fatal.

Solusinya adalah menyuntikkan posisi ke embedding entah bagaimana. Tiga era jawaban:

1. **Absolute sinusoidal** (Vaswani 2017). Tambahkan `sin/cos` posisi ke embedding. Sederhana, tanpa pembelajaran, ekstrapolasinya buruk melampaui panjang terlatih.
2. **RoPE — Rotary Position Embeddings** (Su 2021). Putar vektor Q dan K dengan sudut proporsional terhadap posisi. Mengenkode posisi *relatif* langsung dalam dot product. Dominan pada 2026.
3. **ALiBi — Attention with Linear Biases** (Press 2022). Lewati embedding sepenuhnya; tambahkan penalti linear per-head ke skor attention berdasarkan jarak. Ekstrapolasi panjang yang luar biasa.

Pada 2026, pada dasarnya setiap model open terdepan memakai RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. Segelintir model konteks panjang memakai ALiBi atau varian modernnya. Absolute sinusoidal tinggal sejarah.

## Konsepnya

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### Sinusoidal absolut

Pre-compute matriks tetap `PE` ber-shape `(max_len, d_model)`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

Lalu `X' = X + PE[:N]` sebelum attention. Setiap dimensi adalah sinusoid pada frekuensi berbeda. Model belajar membaca posisi dari pola fasenya. Gagal melampaui `max_len`: tidak ada yang memberi tahu model apa yang terjadi pada posisi 2048 ketika ia hanya melihat posisi 0–2047.

### RoPE

Putar vektor Q dan K (bukan embedding). Untuk sepasang dimensi `(2i, 2i+1)`:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

Terapkan rotasi yang sama ke key dengan posisi `pos_k`. Dot product `q'_m · k'_n` menjadi fungsi dari `(m - n)` semata. Artinya: **skor attention hanya bergantung pada jarak relatif**, meski rotasinya dikunci pada posisi absolut. Trik yang indah.

Memperpanjang RoPE: `base` bisa diskalakan (NTK-aware, YaRN, LongRoPE) untuk mengekstrapolasi ke konteks lebih panjang tanpa retraining. Llama 3 diperpanjang dari konteks 8K ke 128K dengan cara ini.

### ALiBi

Lewati trik embedding. Bias skor attention secara langsung:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

Dengan `m_h` adalah slope khusus-head (mis. `1 / 2^(8·h/H)`). Token yang lebih dekat di-boost; token jauh dipenalti. Tanpa biaya waktu training. Makalahnya menunjukkan ekstrapolasi panjang mengalahkan sinusoidal dan menyamai RoPE pada panjang terlatih orisinalnya.

### Apa yang dipilih pada 2026

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

RoPE menang karena ia terselip ke attention tanpa mengubah arsitektur, mengenkode posisi relatif, dan hiperparameter `base`-nya memberi knob yang bersih untuk fine-tuning konteks panjang.

```figure
rope-explorer
```

## Bangun Sendiri

### Langkah 1: encoding sinusoidal

Lihat `code/main.py`. Komputasi 4 baris:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

Tambahkan ini ke matriks embedding sebelum lapisan attention pertama.

### Langkah 2: RoPE diterapkan ke Q, K

RoPE beroperasi in-place pada Q dan K. Untuk setiap pasang dim:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Penting: terapkan fungsi yang sama ke Q pada posisi `m` dan K pada posisi `n`. Dot product mereka menangkap faktor `cos((m-n)·θ_i)` pada setiap pasang koordinat. Attention mempelajari posisi relatif secara gratis.

### Langkah 3: slope dan bias ALiBi

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

Tambahkan `bias[h]` ke matriks skor attention `(seq_len, seq_len)` milik head `h`, lalu softmax.

### Langkah 4: verifikasi properti jarak-relatif RoPE

Pilih dua vektor acak `a, b`. Putar dengan `(pos_a, pos_b)`. Lalu dengan `(pos_a + k, pos_b + k)`. Kedua dot product harus cocok dalam galat floating-point. Properti itu adalah seluruh inti RoPE — ia invarian terhadap offset absolut, hanya gap relatif yang penting.

## Gunakan

PyTorch 2.5+ menyertakan utilitas RoPE di `torch.nn.functional`. Sebagian besar kode produksi memakai `flash_attn` atau `xformers` di mana RoPE diterapkan di dalam kernel attention.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Trik konteks panjang pada 2026:**

- **NTK-aware interpolation.** Skalakan ulang `base` menjadi `base * (scale_factor)^(d/(d-2))` saat memperpanjang dari 4K ke 16K+.
- **YaRN.** Interpolasi lebih cerdas yang menjaga entropi attention pada konteks panjang. Llama 3.1 128K memakainya.
- **LongRoPE.** Metode Microsoft 2024 yang memakai evolutionary search untuk memilih faktor skala per-dimensi. Phi-3-Long memakainya.
- **Interpolasi posisi + fine-tuning.** Cukup susutkan posisi dengan faktor perpanjangan dan fine-tune selama 1–5B token. Mengejutkan efektif.

## Rilis

Lihat `outputs/skill-positional-encoding-picker.md`. Skill ini memilih strategi encoding untuk model baru berdasarkan panjang konteks target, kebutuhan ekstrapolasi, dan budget training.

## Latihan

1. **Mudah.** Plot matriks `PE` sinusoidal sebagai heatmap untuk `max_len=512, d=128`. Konfirmasi pola "garis-garis melebar seiring indeks dimensi bertambah".
2. **Sedang.** Implementasikan scaling RoPE NTK-aware. Latih LM mungil pada sequence sepanjang 256, lalu uji pada panjang 1024 dengan dan tanpa scaling. Ukur perplexity.
3. **Sulit.** Implementasikan ALiBi dan RoPE dalam modul attention yang sama. Latih transformer 4 lapis pada copy task dengan sequence sepanjang 512. Ekstrapolasikan ke 2048 saat testing. Bandingkan degradasinya.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Positional encoding | "Tells attention about order" | Any signal added to embeddings or attention that encodes position. |
| Sinusoidal | "The original one" | `sin/cos` at geometric frequencies added to embeddings; doesn't extrapolate. |
| RoPE | "Rotary embeddings" | Rotate Q, K by position-dependent angle; dot product encodes relative distance. |
| ALiBi | "Linear bias trick" | Add `-m·\|i-j\|` to attention scores; no embedding needed, great extrapolation. |
| base | "RoPE's knob" | The frequency scaler in RoPE; increase to extend context at inference. |
| NTK-aware | "A RoPE scaling trick" | Rescale `base` so high-frequency dims aren't squeezed when context expands. |
| YaRN | "The fancy one" | Per-dimension interpolation+extrapolation that preserves attention entropy. |
| Extrapolation | "Works beyond trained length" | Can the position scheme serve correct output past `max_len` seen in training? |

## Bacaan Lanjutan

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — sinusoidal orisinal.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — makalah RoPE.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — scaling RoPE state of the art.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — makalah konteks panjang Llama 2 dari Meta.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — metode Microsoft yang dipakai Phi-3-Long dan dikutip di bagian Gunakan.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — implementasi production-grade untuk setiap skema scaling RoPE (default, linear, dynamic, YaRN, LongRoPE, Llama-3).
