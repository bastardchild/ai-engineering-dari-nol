# Capstone 14 — Server Inference Speculative Decoding

> Speculative decoding — sebuah draft murah mengusulkan token, model target memverifikasinya dalam satu pass — kini sudah menjadi optimisasi siap produksi, bukan trik riset. EAGLE-3 di vLLM 0.7 mengirimkan throughput 2,5-3x pada traffic nyata. P-EAGLE (AWS 2026) mendorong spekulasi paralel lebih jauh lagi. SpecForge milik SGLang melatih draft head dalam skala besar. Hub Speculators milik Red Hat menerbitkan draft yang selaras untuk model open umum. TensorRT-LLM menjadikan speculative decoding kelas satu di NVIDIA. Stack serving produksi 2026 adalah vLLM atau SGLang dengan draft keluarga EAGLE, kuantisasi FP8 atau INT4, dan HPA berbasis queue-wait. Capstone ini adalah menyajikan dua model open dengan throughput 2,5x+ baseline beserta laporan tail-latency lengkap.

**Type:** Capstone
**Languages:** Python (serving), C++ / CUDA (kernel inspection), YAML (configs)
**Prerequisites:** Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 17 (infrastructure)
**Phases exercised:** P3 · P7 · P10 · P17
**Time:** 30 hours

## Masalahnya

Speculative decoding menjadi komoditas pada 2026. Draft head EAGLE-3 dilatih pada hidden state model target dan memprediksi N token ke depan; model target memverifikasi dalam satu pass tunggal. Laju akseptasi 60-80% berarti throughput end-to-end 2-3x. vLLM 0.7 mengintegrasikan ini secara native. SGLang + SpecForge memberimu pipeline training-nya. Speculators milik Red Hat menerbitkan draft yang selaras untuk Llama 3.3 70B, Qwen3-Coder-30B MoE, dan GPT-OSS-120B.

Keahliannya ada di operasi serving, bukan di modelnya. Laju akseptasi berubah mengikuti distribusi traffic (ShareGPT vs kode vs data domain). Tail latency saat penolakan lebih buruk daripada tanpa spekulasi — kamu harus melaporkan p99 pada beberapa ukuran batch, bukan hanya tokens/sec steady-state. Biaya per 1M token dibanding API Anthropic / OpenAI adalah tuas kredibilitasnya.

## Konsepnya

Speculative decoding punya dua lapisan. Sebuah model **draft** (kepala EAGLE-3, ngram, atau model lebih kecil yang selaras dengan target) mengusulkan k kandidat token per langkah. Model **target** memverifikasi semua k dalam satu pass; prefix yang diterima apa adanya menggantikan jalur greedy. Laju akseptasi bergantung pada kesejajaran draft-target dan distribusi masukan.

EAGLE-3 mengalahkan draft ngram pada sebagian besar traffic. P-EAGLE menjalankan spekulasi paralel untuk pohon draft yang lebih dalam. Trade-off-nya: latensi P99 saat penolakan lebih tinggi karena pass verifikasinya lebih besar. Konfigurasi serving harus melaporkan latensi per-bucket-ukuran-batch untuk menampilkannya.

Deployment-nya Kubernetes. vLLM 0.7 menjalankan satu replica per GPU atau shard tensor-parallel. HPA menskala-otomatis berdasarkan queue-wait, bukan CPU. Kuantisasi FP8 (Marlin) dan INT4 (AWQ) menjaga memori GPU tetap di dalam amplop H100 / H200. Laporan end-to-end-nya adalah throughput, laju akseptasi, p50/p99 pada batch 1/8/32, dan $/1M token.

## Arsitektur

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## Stack

- Serving: vLLM 0.7 atau SGLang 0.4
- Metode speculative: draft head EAGLE-3, spekulasi paralel P-EAGLE, fallback ngram
- Training draft: SpecForge (SGLang) atau Red Hat Speculators
- Model target: Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B
- Kuantisasi: FP8 (Marlin), INT4 AWQ
- Deployment: Kubernetes + NVIDIA device plugin; HPA pada metrik queue-wait
- Eval: ShareGPT, MT-Bench-v2, GSM8K, HumanEval untuk pengukuran akseptasi sebaran-domain
- Referensi: speculative decoding TensorRT-LLM sebagai baseline vendor

```figure
cf-spec-decode
```

## Bangun Sendiri

1. **Persiapan model target.** Pilih Llama 3.3 70B. Kuantisasi ke FP8 via Marlin. Deploy di bawah vLLM 0.7 pada 1xH100 (atau 2x tensor-parallel).

2. **Sumber draft.** Tarik sebuah draft head EAGLE-3 yang selaras dari Red Hat Speculators (atau latih satu via SpecForge). Muat ke konfigurasi speculative-decoding vLLM.

3. **Angka baseline.** Sebelum spekulasi: tokens/s pada batch 1/8/32, latensi p50/p99, utilisasi GPU. Publikasikan.

4. **Aktifkan EAGLE-3.** Balik konfigurasinya; jalankan ulang benchmark yang sama. Laporkan speedup, laju akseptasi, delta tail-latency p99.

5. **P-EAGLE.** Aktifkan spekulasi paralel; ukur pohon draft yang lebih dalam vs EAGLE-3 serial. Laporkan titik belok di mana P-EAGLE membantu vs merugikan.

6. **Traffic domain.** Jalankan ShareGPT vs HumanEval vs traffic spesifik-domain melalui server yang sama. Ukur laju akseptasi per distribusi. Identifikasi kapan draft mulai drift.

7. **Model target kedua.** Jalankan pipeline yang sama pada Qwen3-Coder-30B MoE. Draft-nya lebih sulit (noise routing MoE). Laporkan.

8. **HPA K8s.** Deploy di bawah K8s dengan HPA yang melacak `queue_wait_ms`. Demonstrasikan scale-out ketika beban tiga kali lipat.

9. **Perbandingan biaya.** Hitung $/1M token dibanding Anthropic Claude Sonnet 4.7 dan OpenAI GPT-5.4 pada eval yang sama. Publikasikan.

## Gunakan

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## Rilis

`outputs/skill-inference-server.md` menjelaskan deliverable-nya. Stack serving yang terukur dengan speculative decoding, laporan benchmark lengkap, dan sebuah deployment K8s.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Measured speedup vs baseline | 2.5x+ throughput at matched quality on two models |
| 20 | Acceptance rate on realistic traffic | Per-distribution acceptance-rate report |
| 20 | P99 tail-latency discipline | p99 at batch 1/8/32 with and without speculation |
| 20 | Ops | K8s deploy, HPA on queue-wait, rollout smooth |
| 15 | Write-up and methodology | Clear explanation of what changed and why |
| **100** | | |

## Latihan

1. Ukur degradasi laju akseptasi ketika draft tertinggal satu versi dari target (mis., drift Llama 3.3 -> 3.4). Bangun sebuah alert monitoring.

2. Implementasikan fallback ngram: jika akseptasi EAGLE-3 turun di bawah ambang, beralih ke draft ngram. Laporkan peningkatan keandalan.

3. Jalankan eksperimen MoE terkontrol: Qwen3-Coder-30B yang sama dengan noise routing disuntikkan vs tanpa. Ukur sensitivitas akseptasi draft.

4. Perluas ke H200 (141 GB). Laporkan headroom ukuran-model-per-replica yang diperoleh dan apakah kamu dapat menyajikan Llama 3.3 70B tanpa kuantisasi.

5. Benchmark speculative decoding TensorRT-LLM pada perangkat keras H100 yang sama. Laporkan di mana ia menang vs vLLM.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Draft model | "Speculator" | Small model that proposes N tokens for the target to verify |
| EAGLE-3 | "2026 draft architecture" | Draft head trained on target hidden states; ~75% acceptance |
| P-EAGLE | "Parallel speculation" | Tree of draft branches verified in one target pass |
| Acceptance rate | "Hit rate" | Fraction of drafted tokens accepted without resampling |
| Quantization | "FP8 / INT4" | Lower-precision weights to fit more model in GPU memory |
| Queue wait | "HPA metric" | Time a request waits in the pending queue before inference starts |
| Speculators hub | "Aligned drafts" | Red Hat Neural Magic hub of EAGLE drafts for common open models |

## Bacaan Lanjutan

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — the reference serving stack
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — parallel speculative decoding paper + integration
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — draft-head training pipeline
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — aligned draft hub
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — vendor alternative
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — commercial reference
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — the method paper
- [vLLM repository](https://github.com/vllm-project/vllm) — code and benchmarks
