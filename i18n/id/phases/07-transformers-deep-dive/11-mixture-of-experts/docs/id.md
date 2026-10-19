# Mixture of Experts (MoE)

> Transformer dense 70B mengaktifkan setiap parameter untuk setiap token. MoE 671B hanya mengaktifkan 37B per token dan mengalahkannya di setiap benchmark. Sparsity adalah ide scaling terpenting dekade ini.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## Masalahnya

FLOPs transformer dense saat inference sama dengan jumlah parameternya (kali 2 untuk forward pass). Perbesar model dense dan setiap token membayar tagihan penuh. Pada 2024 frontier menabrak tembok komputasi: untuk menjadi jauh lebih pintar, kamu butuh FLOPs per token yang jauh lebih besar secara eksponensial.

Mixture of Experts memutus tautan ini. Ganti setiap FFN dengan `E` expert independen + router yang memilih `k` expert per token. Total parameter = `E × FFN_size`. Parameter aktif per token = `k × FFN_size`. Konfigurasi tipikal 2026: `E=256`, `k=8`. Penyimpanan berskala dengan `E`, komputasi berskala dengan `k`.

Frontier 2026 hampir seluruhnya MoE: DeepSeek-V3 (671B total / 37B aktif), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss. Pada leaderboard independen Artificial Analysis, 10 model open-source teratas semuanya MoE.

## Konsepnya

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### Pertukaran FFN

Block transformer dense:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

Block MoE:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

Setiap expert adalah FFN independen (tipikalnya SwiGLU). Router adalah satu lapis linear. Setiap token memilih `k` expert-nya sendiri dan mendapat campuran gated dari output mereka.

### Problem load-balancing

Kalau router menaruh 90% token lewat expert 3, expert lain kelaparan. Tiga perbaikan telah dicoba:

1. **Auxiliary load-balancing loss** (Switch Transformer, Mixtral). Tambahkan penalti proporsional terhadap variansi pemakaian expert. Bekerja, tapi menambah hiperparameter dan sinyal gradien kedua.
2. **Kapasitas expert + token dropping** (Switch awal). Setiap expert memproses paling banyak `C × N/E` token; token overflow melewatkan lapisan. Merugikan kualitas.
3. **Balancing tanpa auxiliary-loss** (DeepSeek-V3). Tambahkan bias terpelajari per-expert yang menggeser seleksi top-k router. Bias diupdate di luar loss training. Tanpa penalti pada objektif utama. Unlock besar 2024.

Pendekatan DeepSeek-V3: setelah setiap langkah training, untuk setiap expert, cek apakah pemakaiannya di atas atau di bawah target. Dorong bias sebesar `±γ`. Seleksi memakai `scores + bias`. Probabilitas expert yang dipakai untuk gating adalah `scores` mentah tanpa perubahan. Memisahkan routing dari ekspresi.

### Shared expert

DeepSeek-V2/V3 juga membagi expert menjadi *shared* dan *routed*. Setiap token melewati semua shared expert. Routed expert dipilih via top-k. Shared expert menangkap pengetahuan umum; routed expert berspesialisasi. V3 menjalankan 1 shared expert plus top-8 dari 256 routed.

### Expert fine-grained

MoE klasik (GShard, Switch): setiap expert selebar FFN penuh. `E` kecil (8–64), `k` kecil (1–2).

MoE fine-grained modern (DeepSeek-V3, Qwen-MoE): setiap expert lebih sempit (1/8 ukuran FFN). `E` besar (256+), `k` lebih besar (8+). Total parameter sama, tapi kombinasi berskala jauh lebih cepat. `C(256, 8) = 400 triliun` "expert" yang mungkin per token. Kualitas naik, latensi tetap datar.

### Profil biaya

Per token, per lapis:

| Config | Active params / token | Total params |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 mengalahkan Llama 3 70B (dense) pada hampir setiap benchmark sambil melakukan **FLOPs aktif per token yang lebih sedikit**. Lebih banyak parameter = lebih banyak pengetahuan. Lebih banyak FLOPs aktif = lebih banyak komputasi per token. MoE memisahkan keduanya.

### Harganya: memori

Semua expert hidup di GPU terlepas dari mana yang menyala. Model 671B butuh ~1.3 TB VRAM untuk bobot fp16. Deployment MoE frontier membutuhkan expert parallelism — shard expert lintas GPU, route token lintas network. Latensi didominasi komunikasi all-to-all, bukan matmul.

```figure
expert-routing
```

## Bangun Sendiri

Lihat `code/main.py`. Lapisan MoE ringkas dalam stdlib murni dengan:

- `n_experts=8` expert ala-SwiGLU (satu linear masing-masing, untuk ilustrasi)
- routing top-k=2
- bobot gating yang dinormalisasi-softmax
- balancing tanpa-auxiliary-loss via bias per-expert

### Langkah 1: router

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias memengaruhi seleksi, bukan bobot gate. Itulah trik DeepSeek-V3 — bias mengoreksi ketidakseimbangan load tanpa menyetir prediksi model.

### Langkah 2: jalankan 100 token melalui router

Lacak expert mana menyala seberapa sering. Tanpa bias, pemakaian miring. Dengan loop update bias (`-γ` untuk expert yang dipakai-berlebih, `+γ` untuk yang kurang-dipakai), pemakaian konvergen ke distribusi seragam dalam beberapa iterasi.

### Langkah 3: perbandingan jumlah param

Cetak "dense equivalent" dari konfigurasi MoE. Berbentuk-DeepSeek-V3: 256 routed + 1 shared, 8 aktif, d_model=7168. Jumlah parameter total mencengangkan. Jumlah aktif seperujuh dense Llama 3 70B.

## Gunakan

Loading HuggingFace:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

Inference produksi 2026: vLLM mendukung routing MoE secara native. SGLang punya jalur expert-parallel tercepat. Keduanya otomatis menangani seleksi top-k dan expert parallelism.

**Kapan memilih MoE:**
- Kamu mau kualitas frontier dengan biaya inference per token lebih rendah.
- Kamu punya infrastruktur VRAM / expert-parallel.
- Workload-mu token-heavy (chat, kode) bukan konteks-heavy (dokumen panjang).

**Kapan TIDAK memilih MoE:**
- Deployment edge — kamu membayar penyimpanan penuh untuk FLOPs aktif apa pun.
- Serving single-user yang kritis-latensi — routing expert menambah overhead.
- Model kecil (<7B) — keunggulan kualitas MoE hanya muncul di atas ambang komputasi (~6B param aktif).

## Rilis

Lihat `outputs/skill-moe-configurator.md`. Skill ini memilih E, k, dan layout shared-expert untuk MoE baru berdasarkan budget parameter, token training, dan target deployment.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Perhatikan bagaimana update bias tanpa-auxiliary-loss meratakan pemakaian expert selama 50 iterasi.
2. **Sedang.** Ganti router terpelajari dengan router berbasis-hash (deterministik, tanpa pembelajaran). Bandingkan kualitas dan keseimbangan. Mengapa router terpelajari lebih baik?
3. **Sulit.** Implementasikan "rollout-matched routing" ala-GRPO (trik DeepSeek-V3.2): catat expert mana menyala saat inference, paksa routing yang sama saat komputasi gradien. Ukur efeknya pada setup policy-gradient mainan.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Expert | "One FFN among many" | An independent feed-forward network; parameters dedicated to a sparse slice of the FFN computation. |
| Router | "The gate" | A tiny linear layer that scores each token against each expert; top-k selection. |
| Top-k routing | "k active experts per token" | Each token's FFN computation goes through exactly k experts, weighted by gate. |
| Auxiliary loss | "Load-balance penalty" | Extra loss term that penalizes skewed expert usage. |
| Auxiliary-loss-free | "DeepSeek-V3's trick" | Balance via per-expert bias on the router's selection only; no extra gradient. |
| Shared expert | "Always on" | Extra expert through which every token passes; captures common knowledge. |
| Expert parallelism | "Shard by expert" | Distribute different experts to different GPUs; route tokens across the network. |
| Sparsity | "Active params < total params" | The ratio `k × expert_size / (E × expert_size)`; 37/671 ≈ 5.5% for DeepSeek-V3. |

## Bacaan Lanjutan

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — idenya.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch, MoE klasik.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + MoE tanpa-auxiliary-loss + MTP.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — makalah balancing berbasis-bias.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — pemisahan fine-grained + shared-expert yang dipakai router pelajaran ini.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — makalah shared-expert orisinal.
