# GPT — Causal Language Modeling

> BERT melihat kedua sisi. GPT hanya melihat masa lalu. Mask segitiga adalah satu baris kode paling berpengaruh dalam AI modern.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time:** ~75 minutes

## Masalahnya

Model bahasa menjawab satu pertanyaan: diberikan `t-1` token pertama, apa distribusi probabilitas atas token `t`? Latih pada sinyal itu — prediksi token berikutnya — dan kamu dapat model yang bisa membangkitkan teks sembarang satu token per langkah.

Untuk melatihnya end-to-end pada seluruh sequence secara paralel, kamu butuh prediksi setiap posisi hanya bergantung pada posisi-posisi sebelumnya. Kalau tidak, model mencontek dengan melihat jawabannya.

Causal mask melakukan ini. Ia adalah satu matriks segitiga-atas berisi nilai `-inf` yang ditambahkan ke skor attention sebelum softmax. Setelah softmax, posisi-posisi itu menjadi 0. Setiap posisi hanya bisa attend ke dirinya sendiri dan posisi-posisi sebelumnya. Dan karena kamu menerapkannya sekali ke seluruh sequence, kamu dapat N prediksi token-berikutnya yang paralel dalam satu forward pass.

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023), GPT-5 (2025), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi — semuanya adalah transformer kausal decoder-only dengan loop inti yang sama. Yang membedakan mereka adalah kualitas data, skala dan penyempurnaan arsitektural, serta post-training (SFT, RLHF, DPO, dan para penerusnya).

## Konsepnya

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### Mask-nya

Diberikan sequence sepanjang `N`, bangun matriks `N × N`:

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

Tambahkan `M` ke skor attention mentah sebelum softmax. `exp(-inf) = 0`, jadi posisi ter-mask berkontribusi bobot nol. Setiap baris matriks attention adalah distribusi probabilitas hanya atas posisi-posisi sebelumnya.

Biaya implementasi: satu panggilan `torch.tril()`. Waktu komputasi: nanodetik. Dampak pada bidang ini: segalanya.

### Dari mana segitiga itu berasal

Mask biasanya disajikan sebagai tempelan yang dibautkan ke attention. Jalankan derivasinya ke arah sebaliknya dan ia berhenti menjadi misterius: attention adalah penyempurnaan ketiga dari rata-rata prefix, dan segitiga itu adalah batas loop dari rata-rata tersebut, ditulis sebagai matriks.

**Tahap 1 — rata-rata prefix.** Ringkasan kausal paling bodoh dari sebuah sequence: posisi `i` menjadi rata-rata posisi `0…i`. Sebagai loop, itu adalah `out[i] = X[:i+1].mean(0)`. Komputasi yang sama adalah satu perkalian matriks. Ambil matriks segitiga-bawah berisi satu, bagi setiap baris dengan cacahannya, kalikan:

```python
import numpy as np

A = np.tril(np.ones((n, n)))
A = A / A.sum(axis=1, keepdims=True)
out = A @ X
```

Baris `i` dari `A` adalah `[1/(i+1), …, 1/(i+1), 0, …, 0]`. Nol di atas diagonal adalah kausalitasnya. Tidak ada soal masa depan yang di-mask; masa depan memang tidak pernah ada dalam penjumlahan.

**Tahap 2 — bobot terpelajari.** Rata-rata seragam memperlakukan setiap token masa lalu sama relevannya. Ganti angka satu dengan matriks skor terpelajari `S`. Kini baris tidak lagi berjumlah satu secara konstruksi, jadi normalkan setiap baris dengan softmax alih-alih membagi dengan cacahan. Softmax tidak pernah mengeluarkan nol persis, yang merusak kausalitas — kecuali skor masa depan masuk sebagai `-inf`, karena `exp(-inf) = 0`:

```python
def softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = S + np.triu(np.full((n, n), -np.inf), k=1)
A = softmax(S, axis=1)
out = A @ X
```

Segitiga yang sama, matriks row-stochastic yang sama, satu matmul yang sama. Mask `-inf` bukan mesin baru. Ia adalah entri nol tahap 1, diterjemahkan ke domain input softmax.

**Tahap 3 — bobot yang bergantung konten.** Pada tahap 2, `S` tetap setelah training: posisi 7 selalu membobot posisi 3 dengan sama, apa pun kata tokennya. Biarkan skor bergantung pada token itu sendiri: `S = Q @ K.T / sqrt(d_k)`. Tidak ada lagi yang berubah. Mask, softmax, matmul — identik.

Tiga tahap, satu invarian: matriks row-stochastic segitiga-bawah dikali sequence. Rata-rata seragam, bobot statis terpelajari, bobot yang bergantung konten. Mask tidak pernah ditambahkan ke attention. Ia bertahan dari rata-rata.

```figure
mask-derivation
```

### Training paralel, inference serial

Training: forward-pass seluruh sequence `(N, d_model)` sekali, hitung N loss cross-entropy (satu per posisi), jumlahkan, backprop. Paralel sepanjang sequence. Inilah mengapa training GPT berskala — kamu memproses 1M token dalam satu batch dalam satu pass GPU.

Inference: kamu membangkitkan token per token. Umpankan `[t1, t2, t3]`, dapatkan `t4`. Umpankan `[t1, t2, t3, t4]`, dapatkan `t5`. Umpankan `[t1, t2, t3, t4, t5]`, dapatkan `t6`. KV cache (Pelajaran 12) menyimpan hidden state `t1…tn` supaya kamu tidak menghitung ulang setiap langkah. Tapi depth serial saat inference = panjang output. Itulah pajak autoregresif dan mengapa decoding menjadi bottleneck latensi setiap LLM.

### Loss-nya — shift-by-one

Diberikan token `[t1, t2, t3, t4]`:

- Input: `[t1, t2, t3]`
- Target: `[t2, t3, t4]`

Untuk setiap posisi `i`, hitung `-log P(target_i | inputs[:i+1])`. Jumlahkan. Inilah cross-entropy untuk seluruh sequence.

Setiap LM transformer yang pernah kamu dengar berlatih pada loss ini. Pre-training, fine-tuning, SFT — loss sama, data berbeda.

### Strategi decoding

Setelah training, pilihan sampling lebih penting dari yang orang kira.

| Method | What it does | When to use |
|--------|--------------|-------------|
| Greedy | Argmax every step | Deterministic tasks, code completion |
| Temperature | Divide logits by T, sample | Creative tasks, higher T = more diversity |
| Top-k | Sample from top-k tokens only | Kills low-probability tails |
| Top-p (nucleus) | Sample from smallest set with cumulative prob ≥ p | 2020+ default; adapts to distribution shape |
| Min-p | Keep tokens with `p > min_p * max_p` | 2024+; better at rejecting long tails than top-p |
| Speculative decoding | Draft model proposes N tokens, big model verifies | 2–3× latency reduction at same quality |

Pada 2026, min-p + temperature 0.7 adalah bawaan yang masuk akal untuk model open-weight. Speculative decoding adalah syarat meja untuk setiap stack inference produksi.

### Apa yang membuat "resep GPT" berhasil

1. **Decoder-only.** Tanpa overhead encoder. Satu pass attention + FFN per lapis.
2. **Scaling.** 124M → 1.5B → 175B → triliunan. Scaling law Chinchilla (Pelajaran 13) memberi tahu cara membelanjakan komputasi.
3. **In-context learning.** Muncul sekitar 6B–13B. Model bisa mengikuti contoh few-shot tanpa fine-tuning.
4. **RLHF.** Post-training pada preferensi manusia mengubah teks pretraining mentah menjadi asisten chat.
5. **Pre-norm + RoPE + SwiGLU.** Training stabil pada skala besar.

Arsitektur intinya tidak banyak berubah sejak GPT-2. Semua hal menarik terjadi pada data, skala, dan post-training.

```figure
causal-mask
```

## Bangun Sendiri

### Langkah 1: causal mask

Lihat `code/main.py`. Satu baris:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Tambahkan ke skor attention sebelum softmax. Itulah seluruh mekanismenya.

### Langkah 2: model ala GPT 2 lapis

Tumpuk dua block decoder (masked self-attention + FFN, tanpa cross-attention). Tambahkan embedding token, positional encoding, dan unembedding (diikat ke matriks embedding token — trik standar sejak GPT-2).

### Langkah 3: prediksi token-berikutnya, end-to-end

Pada vocab mainan 20 token, hasilkan logit di setiap posisi. Hitung loss cross-entropy terhadap target shift-by-one. Tanpa gradien — ini sanity check forward-pass.

### Langkah 4: sampling

Implementasikan greedy, temperature, top-k, top-p, min-p. Jalankan masing-masing pada prompt tetap dan bandingkan output-nya. Fungsi sampling hanya 10 baris.

## Gunakan

PyTorch, idiom 2026:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

Di balik layar, `generate()` menjalankan forward pass, mengambil logit posisi final, menyampel token berikutnya, menempelkannya, dan mengulang. Setiap stack inference LLM produksi (vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX) mengimplementasikan loop yang sama dengan optimasi berat — batched prefill, continuous batching, paging KV cache, speculative decoding.

**GPT vs BERT, satu baris masing-masing:** GPT memprediksi `P(x_t | x_{<t})`. BERT memprediksi `P(x_masked | x_unmasked)`. Loss menentukan apakah model bisa membangkitkan.

## Rilis

Lihat `outputs/skill-sampling-tuner.md`. Skill ini memilih parameter sampling untuk task pembangkitan baru dan menandai kapan decoding deterministik diwajibkan.

## Latihan

1. **Mudah.** Jalankan `code/main.py` dan verifikasi matriks attention kausal segitiga-bawah setelah softmax. Cek spot: baris 3 seharusnya punya bobot hanya di kolom 0–3.
2. **Sedang.** Implementasikan beam search untuk lebar 4. Bandingkan perplexity beam-4 vs greedy pada 10 prompt pendek. Apakah beam selalu menang? (Petunjuk: biasanya untuk terjemahan, bukan untuk chat terbuka.)
3. **Sulit.** Implementasikan speculative decoding: pakai model mungil 2 lapis sebagai draft dan model 6 lapis sebagai verifier. Ukur speedup wall-clock pada 100 completion sepanjang 64. Konfirmasi output cocok dengan greedy milik verifier.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Causal mask | "The triangle" | Upper-triangular `-inf` matrix added to attention scores so position `i` only sees positions `≤ i`. |
| Next-token prediction | "The loss" | Cross-entropy of the model's distribution against the true next token at every position. |
| Autoregressive | "Generate one at a time" | Feed output back as input; parallelism only during training, not during generation. |
| Logits | "Pre-softmax scores" | Raw output of the LM head before softmax; sampling happens on these. |
| Temperature | "Creativity knob" | Divide logits by T; T→0 = greedy, T→∞ = uniform. |
| Top-p | "Nucleus sampling" | Truncate distribution to smallest set summing to ≥p; sample from what remains. |
| Min-p | "Better than top-p" | Keep tokens where `p ≥ min_p × max_p`; adapts cutoff to sharpness of distribution. |
| Speculative decoding | "Draft + verify" | Cheap model proposes N tokens; big model verifies in parallel. |
| Teacher forcing | "Training trick" | During training, feed the true previous token, not the model's prediction. Standard for every seq2seq LM. |

## Bacaan Lanjutan

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1.
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 dan in-context learning.
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — makalah spec decoding.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — kode referensi causal-LM kanonis.
