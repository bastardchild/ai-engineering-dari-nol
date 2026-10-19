# Membangun Transformer dari Nol — Capstone

> Tiga belas pelajaran. Satu model. Tanpa jalan pintas.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## Masalahnya

Kamu sudah membaca setiap makalah. Kamu sudah mengimplementasikan attention, split multi-head, positional encoding, block encoder dan decoder, loss BERT dan GPT, MoE, KV cache. Kini buat semuanya bekerja bersama pada task nyata.

Capstone: latih transformer decoder-only kecil end-to-end pada task pemodelan bahasa level-karakter. Ia membaca Shakespeare. Ia membangkitkan Shakespeare baru. Ia cukup kecil untuk dilatih di laptop dalam di bawah 10 menit. Ia cukup benar sehingga menukar dataset yang lebih besar dan training lebih lama memberimu LM sungguhan.

Ini adalah "nanoGPT" kursus ini. Ia tidak orisinal — tutorial nanoGPT Karpathy 2023 adalah implementasi referensi yang ditulis setiap murid setidaknya sekali. Kita mengangkat bentuknya dan menata ulang di sekitar apa yang telah kita bahas.

## Konsepnya

![Transformer-from-scratch block diagram](../assets/capstone.svg)

Arsitekturnya, beranotasi:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### Apa yang kita kirim

- `GPTConfig` — satu tempat untuk mengonfigurasi semua hiperparameter.
- `MultiHeadAttention` — kausal, batched, dengan jalur ala-Flash opsional (`scaled_dot_product_attention` milik PyTorch).
- `SwiGLUFFN` — FFN modern.
- `Block` — attention + FFN ber-pre-norm dan berbalut-residual.
- `GPT` — embedding, block bertumpuk, head LM, generate().
- Loop training dengan AdamW, LR cosine, gradient clipping.
- Tokenizer level-char pada teks Shakespeare.

### Apa yang tidak kita kirim

- RoPE — diimplementasikan secara konseptual di Pelajaran 04. Di sini kita memakai embedding posisional terpelajari demi kesederhanaan. Latihan memintamu menukar RoPE.
- KV cache saat pembangkitan — setiap langkah pembangkitan menghitung ulang attention atas seluruh prefix. Lebih lambat tapi lebih sederhana. Latihan memintamu menambahkan KV cache.
- Flash Attention — PyTorch 2.0+ auto-dispatch kalau input cocok; kita memakai `F.scaled_dot_product_attention`.
- MoE — satu FFN per block. Kamu melihat MoE di Pelajaran 11.

### Metrik target

Pada laptop Mac M2, GPT 4 lapis, 4 head, d_model=128 yang dilatih 2.000 langkah pada `tinyshakespeare.txt`:

- Loss training konvergen dari ~4.2 (acak) menjadi ~1.5 dalam sekitar 6 menit.
- Output sampel tampak berbentuk-Shakespeare: kata arkais, jeda baris, nama diri seperti "ROMEO:" muncul.
- Loss val (10% final teks yang ditahan) mengikuti loss training dengan dekat; tanpa overfitting pada ukuran/budget ini.

```figure
n5-block-stack
```

## Bangun Sendiri

Pelajaran ini memakai PyTorch. Instal `torch` (build CPU tidak apa-apa). Lihat `code/main.py`. Skrip menangani:

- Mengunduh `tinyshakespeare.txt` kalau hilang (atau membaca salinan lokal).
- Tokenizer char level-byte.
- Split train/val pada 90/10.
- Loop training dengan autocast bf16 pada hardware yang didukung.
- Sampling setelah training selesai.

### Langkah 1: data

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 karakter unik. Vocabulary mungil. Muat dalam vocab_size 4-byte. Tanpa BPE, tanpa drama tokenizer.

### Langkah 2: model

Lihat `code/main.py`. Block-nya textbook dari Pelajaran 05 — pre-norm, RMSNorm, SwiGLU, MHA kausal. Jumlah parameter untuk 4/4/128: ~800K.

### Langkah 3: loop training

Ambil batch acak window token sepanjang-256. Forward. Cross-entropy shift-by-one. Backward. Langkah AdamW. Log. Ulangi.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Langkah 4: sampel

Diberikan prompt, forward berulang, sampel dari logit top-p, tempelkan, dan lanjutkan. Berhenti setelah 500 token.

### Langkah 5: baca output-nya

Setelah 2.000 langkah:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

Bukan Shakespeare. Tapi berbentuk-Shakespeare. Kemenangan jelas untuk ~800K parameter dan 6 menit di laptop.

## Gunakan

Capstone ini adalah arsitektur referensi. Tiga ekstensi untuk mengirimnya menjadi sesuatu yang nyata:

1. **Tukar tokenizer.** Pakai BPE (mis. `tiktoken.get_encoding("cl100k_base")`). Ukuran vocab melonjak dari 65 menjadi ~50.000. Kapasitas model perlu diskalakan untuk mengimbangi.
2. **Latih pada korpus lebih besar.** Pakai `OpenWebText` atau `fineweb-edu` (HuggingFace). 10B token pada satu A100 memakan ~24 jam untuk GPT 125M-param.
3. **Tambahkan RoPE + KV cache + Flash Attention.** Latihan di bawah memandumu melalui masing-masing.

Ini berakhir sebagai GPT 125M-parameter yang membangkitkan bahasa Inggris fasih. Bukan model frontier. Tapi jalur kode yang sama — hanya lebih besar — dipakai Karpathy, EleutherAI, dan Allen Institute untuk melatih checkpoint riset pada 2026.

## Rilis

Lihat `outputs/skill-transformer-review.md`. Skill ini mereviu implementasi transformer-dari-nol untuk kebenaran lintas seluruh 13 pelajaran sebelumnya.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Verifikasi validation loss langkah-final model terlatihmu di bawah 2.0. Ubah `max_steps` dari 2.000 menjadi 5.000 — apakah val loss terus membaik?
2. **Sedang.** Ganti embedding posisional terpelajari dengan RoPE. Terapkan rotasi ke Q dan K di dalam `MultiHeadAttention`. Latih dan verifikasi val loss setidaknya sama rendah.
3. **Sedang.** Implementasikan KV cache dalam loop sampling. Bangkitkan 500 token dengan dan tanpa cache. Wall-clock seharusnya membaik 5–20× di laptop.
4. **Sulit.** Tambahkan head kedua ke model yang memprediksi token next-plus-one (MTP — Multi-Token Prediction dari DeepSeek-V3). Latih bersama. Apakah membantu?
5. **Sulit.** Ganti FFN tunggal per block dengan MoE 4-expert. Router + routing top-2. Lihat bagaimana val loss berubah pada parameter aktif yang sebanding.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy's tutorial repo" | Minimal decoder-only transformer training code, ~300 LOC; the canonical reference. |
| tinyshakespeare | "The standard toy corpus" | ~1.1 MB of text; every character-LM tutorial since 2015 uses it. |
| Tied embeddings | "Share input/output matrix" | LM head weight = transpose of token embedding matrix; saves parameters, improves quality. |
| bf16 autocast | "Training precision trick" | Run forward/back in bf16, keep optimizer state in fp32; standard since 2021. |
| Gradient clipping | "Stops spikes" | Cap global grad norm at 1.0; prevents training blowups. |
| Cosine LR schedule | "The 2020+ default" | LR ramps up linearly (warmup) then decays cosine-shaped to 10% of peak. |
| MFU | "Model FLOP Utilization" | Achieved FLOPs / theoretical peak; 40% dense, 30% MoE is strong in 2026. |
| Val loss | "Held-out loss" | Cross-entropy on data the model never saw; overfit detector. |

## Bacaan Lanjutan

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — implementasi beranotasi klasik.
