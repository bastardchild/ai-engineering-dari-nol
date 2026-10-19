# Speculative Decoding — Draft, Verifikasi, Ulangi

> Decoding autoregresif itu serial. Setiap token menunggu token sebelumnya. Speculative decoding memutus rantai: model murah mendraft N token, model mahal memverifikasi semua N dalam satu forward pass. Ketika draft benar kamu membayar satu forward besar untuk N pembangkitan.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time:** ~60 minutes

## Masalahnya

LLM 70B menyampel satu token memakan ~30 ms pada H100. Model draft 3B memakan ~3 ms. Kalau kita biarkan draft 3B 5 token ke depan, lalu jalankan 70B *sekali* untuk memverifikasi kelimanya, totalnya `5×3 + 30 = 45 ms` untuk sampai 5 token yang diterima — versus `5×30 = 150 ms` untuk pembangkitan garis-lurus. Itulah seluruh pitch speculative-decoding: tukar sedikit memori GPU ekstra (model draft) dengan latensi decode 2–4× lebih rendah.

Triknya harus menjaga distribusi. Speculative sampling, diperkenalkan Leviathan et al. (2023) dan Chen et al. secara bersamaan, menjamin sequence output **terdistribusi identik** dengan yang akan dihasilkan model besar sendirian. Tanpa trade-off kualitas. Hanya lebih cepat.

Empat keluarga pasangan draft-verifier mendominasi inference 2026:

1. **Spekulatif vanilla (Leviathan 2023).** Model draft terpisah (mis. Llama 3 1B) + verifier (mis. Llama 3 70B).
2. **Medusa (Cai 2024).** Banyak decoding head pada verifier memprediksi posisi `t+1..t+k` secara paralel. Tanpa model draft terpisah.
3. **Keluarga EAGLE (Li 2024, 2025).** Draft ringan yang memakai ulang hidden state verifier; acceptance rate lebih dekat daripada vanilla; tipikal 3–4×.
4. **Lookahead decoding (Fu 2024).** Iterasi Jacobi; tanpa model draft sama sekali. Self-speculation. Niche tapi bebas-dependensi.

Setiap stack inference produksi pada 2026 menyertakan speculative decoding secara default. vLLM, TensorRT-LLM, SGLang, dan llama.cpp semuanya mendukung setidaknya vanilla + EAGLE-2.

## Konsepnya

### Algoritma inti

Diberikan verifier `M_q` dan draft lebih murah `M_p`:

1. Biarkan `x_1..x_k` menjadi prefix yang sudah di-decode.
2. **Draft**: pakai `M_p` untuk mengusulkan `d_{k+1}, d_{k+2}, ..., d_{k+N}` secara autoregresif dengan probabilitas draft `p_1..p_N`.
3. **Verifikasi paralel**: jalankan `M_q` sekali pada `x_1..x_k, d_{k+1}, ..., d_{k+N}`, dapatkan probabilitas verifier `q_1..q_{N+1}` untuk posisi `k+1..k+N+1`.
4. **Terima/tolak setiap token draft kiri ke kanan**: untuk setiap `i`, terima dengan probabilitas `min(1, q_i(d_i) / p_i(d_i))`.
5. Pada penolakan pertama di posisi `j`: sampel `t_j` dari distribusi "residual" `(q_j - p_j)_+` yang dinormalisasi. Semua draft setelah `j` dibuang.
6. Pada penerimaan semua `N`: sampel satu token ekstra `t_{N+1}` dari `q_{N+1}` (token bonus gratis).

Trik distribusi residual adalah wawasan matematis yang menjaga output terdistribusi persis seolah `M_q` menyampel dari nol.

### Apa yang menentukan speedup

Biarkan `α` = acceptance rate harapan per token draft. Biarkan `c` = rasio biaya draft-ke-verifier. Per langkah:

- Pembangkitan naif melakukan 1 panggilan model-besar per token.
- Spekulatif melakukan 1 panggilan model-besar per `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` token ketika `α` tinggi.

Aturan praktis tipikal pada `α = 0.75` dan `N = 5`: 3× lebih sedikit panggilan model-besar. Biaya draft 5× murah. Wall-clock total turun ~2.5×.

**α bergantung pada:**

- Seberapa baik draft mengaproksimasi verifier. Keluarga sama / data training sama mendongkrak α signifikan.
- Strategi decoding. Draft greedy melawan verifier greedy: α tinggi. Sampling temperature: lebih sulit cocok; acceptance turun.
- Tipe task. Kode dan output terstruktur lebih banyak diterima (terprediksi); tulisan kreatif bentuk-bebas lebih sedikit diterima.

### Medusa — draft tanpa model draft

Medusa mengganti model draft dengan head output ekstra pada verifier. Pada posisi `t`:

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

Setiap head mengeluarkan logitnya sendiri. Saat inference kamu menyampel dari setiap head untuk mendapat sequence kandidat, lalu verifikasi dengan satu forward pass memakai skema tree-attention yang mempertimbangkan semua kelanjutan kandidat sekaligus.

Pro: tanpa model kedua. Kontra: menambah parameter terlatih; butuh tahap fine-tuning supervisi (~1B token); acceptance rate sedikit lebih rendah daripada spekulatif vanilla dengan draft yang bagus.

### EAGLE — draft lebih baik dengan memakai ulang hidden state

EAGLE-1/2/3 (Li et al., 2024–2025) menjadikan model draft transformer mungil (tipikalnya 1 lapis) yang menelan hidden state lapis-terakhir verifier. Karena draft melihat representasi fitur verifier, prediksinya berkorelasi kuat dengan distribusi output verifier. Acceptance rate naik dari ~0.6 (vanilla) menjadi 0.85+.

EAGLE-3 (2025) menambahkan tree search atas kelanjutan kandidat. vLLM dan SGLang menyertakan EAGLE-2/3 sebagai jalur spec bawaan untuk Llama 3/4 dan Qwen 3.

### Tarian KV cache

Verifikasi mengumpankan `N` token draft ke verifier dalam satu forward pass. Ini memperpanjang KV cache verifier sebesar `N` entri. Kalau sebagian draft ditolak, kamu harus mengembalikan cache ke panjang prefix yang diterima.

Implementasi produksi (`--speculative-config` milik vLLM, LookaheadDecoder milik TensorRT-LLM) menangani ini dengan buffer KV scratch. Tulis dulu, commit saat diterima. Tidak sulit secara konseptual, tapi rewel.

```figure
draft-verify-tokens
```

## Bangun Sendiri

Lihat `code/main.py`. Kita mengimplementasikan algoritma speculative-sampling inti (langkah penolakan + distribusi residual) dengan:

- Model "besar" yang merupakan deterministic-softmax atas distribusi hand-coded (supaya kita bisa memverifikasi matematika acceptance secara analitis).
- Model "draft" yang merupakan perturbasi model besar.
- Loop terima / tolak yang menghasilkan distribusi marginal yang sama dengan sampling langsung.

### Langkah 1: langkah penolakan

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` adalah bilangan acak seragam. `q_prob` adalah probabilitas verifier untuk token draft. `p_prob` adalah probabilitas model draft. Teorema Leviathan menyatakan keputusan Bernoulli ini, diikuti sampling dari residual saat penolakan, menjaga distribusi verifier secara eksak.

### Langkah 2: distribusi residual

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

Kurangkan `p` dari `q` per elemen, clamp nilai negatif ke nol, renormalisasi. Sampel dari sini pada penolakan apa pun.

### Langkah 3: satu langkah spekulatif

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

Lima diterima → satu bonus → enam token dihasilkan dalam satu pass verifier.

### Langkah 4: ukur acceptance rate

Jalankan 10.000 langkah spekulatif pada berbagai level kualitas draft. Plot acceptance rate vs divergensi KL antara distribusi draft dan verifier. Kamu seharusnya melihat relasi monoton yang bersih.

### Langkah 5: verifikasi kesetaraan distribusi

Secara empiris: histogram token yang dihasilkan loop spekulatif seharusnya cocok dengan histogram yang dihasilkan sampling langsung dari verifier. Inilah teorema Leviathan dalam praktik. Uji chi-square mengonfirmasi dalam galat sampling.

## Gunakan

Produksi:

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-config '{"method": "eagle", "model": "/models/llama-3.1-eagle-70b", "draft_tensor_parallel_size": 1, "num_speculative_tokens": 5}'

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-config '{"method": "draft_model", "model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 5}'
```

TensorRT-LLM punya jalur Medusa tercepat pada pertengahan 2026. `faster-whisper` membungkus speculative decoding untuk Whisper-large dengan draft kecil.

**Memilih draft:**

| Strategy | When to pick | Speedup |
|----------|--------------|---------|
| Vanilla draft (1B/3B Llama family) | Fast prototype, no training | 1.8–2.3× |
| Medusa heads | You can fine-tune the verifier | 2–3× |
| EAGLE-2 / 3 | Production, max speed | 3–4× |
| Lookahead | No draft, no training, no extra params | 1.3–1.6× |

**Kapan TIDAK spec-decode:**

- Pembangkitan sequence-tunggal sepanjang 1–5 token. Overhead mendominasi.
- Sampling sangat kreatif / temperature tinggi (α turun).
- Deployment yang terkendala-memori (model draft menambah VRAM).

## Rilis

Lihat `outputs/skill-spec-decode-picker.md`. Skill ini memilih strategi speculative decoding (vanilla / Medusa / EAGLE / lookahead) dan parameter tuning (N, temperature draft) untuk workload inference baru.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Konfirmasi distribusi token spekulatif cocok dengan distribusi sampel-langsung verifier pada 50.000 token dalam chi-square p > 0,05.
2. **Sedang.** Plot speedup (token per forward model-besar) sebagai fungsi `N` untuk `α = 0.5, 0.7, 0.85`. Identifikasi `N` optimal untuk setiap α. (Petunjuk: token harapan per panggilan verify = `(1 - α^{N+1}) / (1 - α)`.)
3. **Sulit.** Implementasikan Medusa mungil: ambil GPT capstone dari Pelajaran 14, tambahkan 3 head LM ekstra yang memprediksi posisi t+2, t+3, t+4. Latih pada tinyshakespeare dengan loss multi-head bersama. Bandingkan acceptance rate vs draft vanilla yang dibuat dengan memotong model yang sama.
4. **Sulit.** Implementasikan rollback: mulai dengan KV cache prefix 10 token, umpankan 5 token draft, simulasikan penolakan pada posisi 3. Verifikasi bacaan cache-mu cocok dengan "prefix + 2 draft pertama yang diterima" pada iterasi berikutnya.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Draft model | "The cheap one" | A smaller model that proposes candidate tokens; usually 10–50× cheaper than the verifier. |
| Verifier | "The big one" | The target model whose distribution we preserve; runs once per speculative step. |
| Acceptance rate (α) | "How often the draft is right" | Per-token probability that the verifier accepts the draft. 0.7–0.9 typical. |
| Residual distribution | "The rejection fallback" | `(q - p)_+` normalized; sampling from this on rejection preserves the verifier's distribution. |
| Bonus token | "The free one" | When all N drafts accepted, sample one more from the verifier's next-step distribution. |
| Medusa | "Draft-less speculative" | Multiple LM heads on the verifier predict positions t+1..t+k in parallel. |
| EAGLE | "Hidden-state draft" | Tiny transformer draft conditioned on the verifier's last-layer hidden states. |
| Lookahead decoding | "Jacobi iteration" | Self-speculation using a fixed-point iteration; no draft model. |
| Tree attention | "Verify many candidates at once" | Branching verification that considers several draft continuations simultaneously. |
| KV rollback | "Undo rejected drafts" | Scratch KV buffer; commit on acceptance, discard on reject. |

## Bacaan Lanjutan

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — algoritma inti dan teorema kesetaraan.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — pengantar bersamaan; bukti Bernoulli-rejection yang bersih.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — makalah Medusa; verifikasi tree-attention.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1; draft berkondisi-hidden-state.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2; depth tree dinamis.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead, pendekatan tanpa-draft.
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — referensi produksi kanonis dengan keempat strategi tersambung.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — kode referensi untuk EAGLE-1/2/3.
