# KV Cache, Flash Attention & Optimasi Inference

> Training itu paralel dan FLOP-bound. Inference itu serial dan memory-bound. Bottleneck berbeda, trik berbeda.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~75 minutes

## Masalahnya

Decoder autoregresif yang naif melakukan kerja `O(N²)` untuk membangkitkan `N` token: di setiap langkah ia menghitung ulang attention atas seluruh prefix. Untuk respons 4K token itu berarti 16M operasi attention, sebagian besar redundan. Setiap hidden state token prefix bersifat deterministik begitu dihitung — kamu hanya perlu menjalankan query token baru terhadap key dan value yang di-cache dari semua sebelumnya.

Di atas itu, attention sendiri memindahkan banyak data. Attention standar mematerialisasi matriks skor N×N, output softmax N×d, output final N×d — terlalu banyak baca dan tulis ke HBM. Untuk N≥2K, attention menjadi memory-bound sebelum menjadi FLOP-bound. Kernel attention klasik mendayagunakan GPU modern 4–10× di bawah kapasitas.

Dua optimasi, keduanya dari Dao et al., mendorong inference frontier dari "lambat" menjadi "cepat":

1. **KV cache.** Simpan vektor K dan V setiap token prefix. Attention token baru hanyalah satu query terhadap key yang di-cache. Inference turun dari `O(N²)` menjadi `O(N)` per langkah pembangkitan.
2. **Flash Attention.** Tile komputasi attention supaya matriks N×N penuh tidak pernah menyentuh HBM. Semua softmax + matmul terjadi di SRAM. Speedup wall-clock 2–4× pada A100; 5–10× pada H100 dengan FP8.

Pada 2026 keduanya universal. Setiap stack inference produksi (vLLM, TensorRT-LLM, SGLang, llama.cpp) mengasumsikannya. Setiap model frontier dikirim dengan Flash Attention menyala.

## Konsepnya

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### Matematika KV cache

Per lapis decoder, per token, per head:

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

Untuk model 7B dengan 32 lapis, 32 head, d_head=128, fp16:

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

Untuk Llama 3 70B (80 lapis, d_head=128, GQA dengan 8 KV head):

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

10 GB itu mengapa Llama 3 70B pada konteks 128K butuh sebagian besar A100 40 GB hanya untuk KV cache pada batch size 1.

**GQA adalah kemenangan KV-cache.** MHA dengan 64 head akan menjadi 32 GB. MLA mengompresi lebih jauh lagi.

Seret dimensinya dan perhatikan ukuran cache bergerak. Dorong panjang sequence atau batch naik dan lihat seberapa cepat ia melampaui satu GPU:

```figure
kv-cache-sizer
```

### Flash Attention — trik tiling

Attention standar:

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

Tiga round trip HBM. Pada H100, bandwidth HBM 3 TB/s; SRAM 30 TB/s. Setiap trip HBM adalah perlambatan faktor-10 vs menjaga semuanya on-chip.

Flash Attention:

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

Satu trip HBM per tile. Jejak memori total turun dari `O(N²)` menjadi `O(N)`. Backward pass menghitung ulang sebagian nilai dari forward pass alih-alih menyimpannya — kemenangan memori lagi.

**Trik numerik.** Softmax berjalan mempertahankan `(max, sum)` lintas tile sehingga normalisasi final eksak. Bukan aproksimasi — Flash Attention menghitung output bit-identik dengan attention standar (modulo non-asosiativitas fp16).

**Evolusi versi:**

| Version | Year | Key change | Speedup on reference hardware |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | Tiled SRAM kernel | 2× on A100 |
| Flash 2 | 2023 | Better parallelism, causal-first ordering | 3× on A100 |
| Flash 3 | 2024 | Hopper asynchrony, FP8 | 1.5–2× on H100 (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5-stage pipeline, software exp2 | Inference-first (forward only initially) |

Flash 4 hanya forward-pass saat peluncuran. Training masih memakai Flash 3. Dukungan GQA dan varlen untuk Flash 4 tertunda (pertengahan 2026).

### Speculative decoding — kemenangan latensi lainnya

Model murah mengusulkan N token. Model besar memverifikasi semua N secara paralel. Kalau verifikasi menerima k token, kamu membayar 1 forward pass model-besar untuk k pembangkitan. Tipikal k=3–5 pada kode dan prosa.

Bawaan 2026:
- **EAGLE 2 / Medusa.** Head draft terintegrasi yang berbagi hidden state verifier. Speedup 2–3× tanpa kehilangan kualitas.
- **Speculative decoding dengan model draft.** Speedup 2–4× pada hardware konsumen.
- **Lookahead decoding.** Iterasi Jacobi; tanpa model draft. Niche tapi gratis.

### Continuous batching

Inference batch klasik: tunggu sequence terlambat selesai, lalu mulai batch baru. Menyia-nyiakan GPU ketika respons pendek selesai duluan.

Continuous batching (pertama dikirim dalam Orca, kini di vLLM, TensorRT-LLM, SGLang): tukar request baru ke batch begitu yang lama selesai. Kenaikan throughput 5–10× untuk workload chat tipikal.

### PagedAttention — KV cache sebagai memori virtual

Fitur utama vLLM. KV cache dialokasikan dalam block 16 token; page table memetakan posisi logis ke block fisik. Membuatmu berbagi KV lintas sampel paralel (beam search, sampling paralel), hot-swap prefix untuk prompt caching, dan mendefrag memori. Peningkatan throughput 4× atas alokasi contiguous naif.

```figure
flash-attention-memory
```

## Bangun Sendiri

Lihat `code/main.py`. Kita mengimplementasikan:

1. Decoder inkremental `O(N²)` yang naif.
2. Decoder `O(N)` ber-KV-cache.
3. Softmax ber-tile yang mensimulasikan algoritma running-max Flash Attention.

### Langkah 1: KV cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

Sederhana: terus menumbuhkan vektor K, V per-token dalam daftar per-lapis, per-head.

### Langkah 2: softmax ber-tile

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

Output bit-identik dengan `softmax(qK) V` sekali jalan, tapi kapan pun working set-nya adalah block `tile × d_head`, bukan `N × d_head` penuh.

### Langkah 3: bandingkan decoding naif vs cached pada pembangkitan 100 token

Hitung operasi attention. Naif: `O(N²)` = 5050. Cached: `O(N)` = 100. Kode mencetak keduanya.

## Gunakan

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM produksi:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

Prefix caching lintas request adalah kemenangan besar 2026 — system prompt yang sama, contoh few-shot, atau dokumen konteks panjang memakai ulang KV lintas panggilan. Untuk workload agen dengan prompt tool berulang, prefix caching rutin memberi kenaikan throughput 5×.

## Rilis

Lihat `outputs/skill-inference-optimizer.md`. Skill ini memilih implementasi attention, strategi KV cache, kuantisasi, dan speculative decoding untuk deployment inference baru.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Konfirmasi decoder naif dan cached menghasilkan output yang sama; perhatikan perbedaan jumlah op.
2. **Sedang.** Implementasikan prefix caching: diberikan prompt P dan beberapa completion, jalankan satu forward pass atas P untuk mengisi KV cache, lalu cabangkan per-completion. Ukur speedup vs mengenkode ulang P untuk masing-masing.
3. **Sulit.** Implementasikan PagedAttention mainan: KV cache dalam block 16 token tetap dengan free-list. Ketika sequence selesai, kembalikan block-nya ke pool. Simulasikan 1.000 completion chat dengan panjang bervariasi. Bandingkan fragmentasi memori vs alokasi contiguous.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| KV cache | "The trick that makes decoding fast" | Stored K and V from every prefix token; new queries attend to them instead of recomputing. |
| HBM | "GPU main memory" | High Bandwidth Memory; 80 GB on H100, 192 GB on B200. ~3 TB/s bandwidth. |
| SRAM | "On-chip memory" | Per-SM fast memory, ~256 KB per SM on H100. ~30 TB/s bandwidth. |
| Flash Attention | "Tiled attention kernel" | Computes attention without materializing N×N in HBM. |
| Continuous batching | "No-wait batching" | Swap finished sequences out, new ones in, without draining the batch. |
| PagedAttention | "vLLM's headline" | KV cache allocated in fixed blocks with a page table; eliminates fragmentation. |
| Prefix caching | "Reuse long prompts" | Cache KV for a shared prefix across requests; major cost cut for agents. |
| Speculative decoding | "Draft + verify" | Cheap draft model proposes tokens; big model verifies k in one pass. |

## Bacaan Lanjutan

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — pipeline 5 tahap Blackwell dan trik software-exp2; baca README repo untuk caveat peluncuran forward-only yang disebut pelajaran ini.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — makalah vLLM.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — spec decoding.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — makalah EAGLE-1/2 untuk pendekatan draft-terintegrasi yang dikutip pelajaran ini.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — pendekatan Medusa yang dirujuk berdampingan dengan EAGLE.
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — telaah mendalam kanonis tentang block 16 token dan desain page-table.
