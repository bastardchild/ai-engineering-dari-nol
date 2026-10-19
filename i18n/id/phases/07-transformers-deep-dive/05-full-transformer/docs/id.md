# Transformer Penuh — Encoder + Decoder

> Attention adalah bintangnya. Sisanya — residual, normalisasi, feed-forward, cross-attention — adalah perancah yang membuatmu bisa menumpuknya dalam-dalam.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## Masalahnya

Satu lapisan attention adalah ekstraktor fitur, bukan model. Satu matmul per lapisan tidak cukup kapasitas untuk bahasa. Kamu butuh depth — dan depth rusak tanpa pemipaan yang tepat.

Makalah Vaswani 2017 mengemas enam keputusan desain yang mengubah satu lapisan attention menjadi block yang bisa ditumpuk. Setiap transformer sejak itu — encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5) — mewarisi kerangka yang sama. Pada 2026 block-blocknya telah disempurnakan (RMSNorm, SwiGLU, pre-norm, RoPE) tapi kerangkanya identik.

Pelajaran ini adalah kerangkanya. Pelajaran berikutnya mengkhususkannya — 06 untuk encoder, 07 untuk decoder, 08 untuk encoder-decoder.

## Konsepnya

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### Enam bagiannya

1. **Embedding + sinyal posisional.** Token → vektor. Posisi disuntikkan via RoPE (modern) atau sinusoidal (klasik).
2. **Self-attention.** Setiap posisi attend ke posisi lain. Di-mask pada decoder.
3. **Feed-forward network (FFN).** MLP dua lapis per posisi: `W_2 · activation(W_1 · x)`. Rasio ekspansi 4× secara default.
4. **Residual connection.** `x + sublayer(x)`. Tanpa ini, gradien menghilang melewati ~6 lapis.
5. **Layer normalization.** `LayerNorm` atau `RMSNorm` (modern). Menstabilkan residual stream.
6. **Cross-attention (hanya decoder).** Query datang dari decoder, key dan value dari output encoder.

Perhatikan sebuah vektor mengalir melalui satu block: attention mencampur lintas posisi, residual membawanya maju, FFN mentransformasinya, dan norm menjaga stream tetap stabil.

```figure
transformer-block
```

### Block encoder (dipakai BERT, encoder T5)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Encoder itu dua arah. Tanpa masking. Semua posisi melihat semua posisi.

### Block decoder (dipakai GPT, decoder T5)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Decoder punya tiga sublapisan per block. Yang tengah — cross-attention — adalah satu-satunya tempat informasi mengalir dari encoder ke decoder. Pada arsitektur decoder-only murni (GPT), cross-attention dihilangkan dan kamu hanya punya masked self-attention + FFN.

### Pre-norm vs post-norm

Makalah orisinal: `x + sublayer(LN(x))` vs `LN(x + sublayer(x))`. Post-norm kehilangan pamor sekitar 2019 — ia lebih sulit dilatih dalam-dalam tanpa warmup yang hati-hati. Pre-norm (`LN` *sebelum* sublapisan) adalah bawaan 2026: Llama, Qwen, GPT-3+, Mistral semuanya memakainya.

### Block modern 2026

Vaswani 2017 membawa LayerNorm + ReLU. Stack modern mengganti keduanya. Beginilah wujud block produksi sebenarnya:

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm membuang pemusatan-mean milik LayerNorm (satu pengurangan lebih sedikit), yang menghemat komputasi dan secara empiris setidaknya sama stabil. SwiGLU (`Swish(W1 x) ⊙ W3 x`) konsisten mengungguli FFN ReLU/GELU sekitar ~0.5 poin ppl dalam makalah Llama, PaLM, dan Qwen.

### Jumlah parameter

Untuk satu block dengan `d_model = d` dan ekspansi FFN `r`:

- MHA: `4 · d²` (proyeksi Q, K, V, O)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- Norm: negligible

Pada `d = 4096, r = 2.6, layers = 32` (kira-kira Llama 3 8B), total: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B` (plus embedding dan head). Cocok dengan jumlah yang dipublikasikan.

## Bangun Sendiri

### Langkah 1: block-block pembangun

Memakai kelas `Matrix` mungil dari Pelajaran 03 (disalin ke file ini agar mandiri):

- `layer_norm(x, eps=1e-5)` — kurangi mean, bagi dengan std.
- `rms_norm(x, eps=1e-6)` — bagi dengan RMS. Tanpa pengurangan mean.
- `gelu(x)` dan `silu(x) * W3 x` (SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`.
- `encoder_block(x, params)` dan `decoder_block(x, enc_out, params)`.

Lihat `code/main.py` untuk pengkabelan penuhnya.

### Langkah 2: sambungkan encoder 2 lapis dan decoder 2 lapis

Tumpuk. Teruskan output encoder ke setiap cross-attention decoder. Tambahkan LN final sebelum proyeksi output.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Langkah 3: jalankan forward pada contoh mainan

Umpankan source 6 token dan target 5 token. Verifikasi shape output adalah `(5, vocab)`. Tanpa training — pelajaran ini soal arsitektur, bukan loss.

### Langkah 4: tukarkan RMSNorm + SwiGLU

Ganti LayerNorm dan ReLU-FFN dengan RMSNorm dan SwiGLU. Konfirmasi shape masih cocok. Inilah modernisasi 2026 dengan satu substitusi fungsi.

## Gunakan

Implementasi referensi PyTorch/TF: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`. Tapi sebagian besar kode produksi 2026 meracik block sendiri karena:

- Flash Attention dipanggil di dalam attention, bukan via `nn.MultiheadAttention`.
- GQA / MLA tidak ada di referensi stdlib.
- RoPE, RMSNorm, SwiGLU bukan bawaan PyTorch.

HF `transformers` punya block referensi bersih yang sebaiknya kamu baca: `modeling_llama.py` adalah block decoder-only kanonis 2026. Panjangnya ~500 baris dan layak ditelusuri sekali.

**Encoder vs decoder vs encoder-decoder — kapan memilih:**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

Decoder-only memenangkan bahasa karena ia berskala paling bersih dan menangani pemahaman sekaligus pembangkitan. Encoder-decoder masih terbaik ketika input punya identitas "source sequence" yang jelas (terjemahan, pengenalan speech, task terstruktur).

## Rilis

Lihat `outputs/skill-transformer-block-reviewer.md`. Skill ini mereviu implementasi block transformer baru terhadap bawaan 2026 dan menandai bagian yang hilang (pre-norm, RoPE, RMSNorm, GQA, rasio ekspansi FFN).

## Latihan

1. **Mudah.** Hitung parameter di encoder_block-mu pada `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Validasi dengan mengimplementasikan block-nya dan memakai `sum(p.numel() for p in block.parameters())`.
2. **Sedang.** Beralih dari post-norm ke pre-norm. Inisialisasi keduanya dan ukur norm aktivasi setelah 12 lapis bertumpuk pada input acak. Aktivasi post-norm seharusnya meledak; aktivasi pre-norm seharusnya tetap terbatas.
3. **Sulit.** Implementasikan encoder-decoder 4 lapis pada copy task mainan (salin `x` secara terbalik). Latih 100 langkah. Laporkan loss. Tukarkan RMSNorm + SwiGLU + RoPE — apakah loss turun?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Block | "One transformer layer" | Stack of norm + attention + norm + FFN, wrapped in residual connections. |
| Residual | "Skip connection" | `x + f(x)` output; enables gradient flow through deep stacks. |
| Pre-norm | "Normalize before, not after" | Modern: `x + sublayer(LN(x))`. Trains deeper without warmup gymnastics. |
| RMSNorm | "LayerNorm without the mean" | Divide by RMS; one less op, same empirical stability. |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`. Beats ReLU/GELU on LM ppl. |
| Cross-attention | "How the decoder sees the encoder" | MHA with Q from decoder, K/V from encoder outputs. |
| FFN expansion | "How wide the middle MLP is" | Ratio of hidden-size to d_model, usually 4 (LayerNorm) or 2.6 (SwiGLU). |
| Bias-free | "Drop the +b terms" | Modern stacks omit biases in linear layers; slight ppl improvement, smaller model. |

## Bacaan Lanjutan

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — spek block orisinal.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — mengapa pre-norm mengalahkan post-norm pada depth.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — makalah SwiGLU.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — block decoder-only kanonis 2026.
