# Metrik Inferensi — TTFT, TPOT, ITL, Goodput, P99

> Empat metrik menentukan apakah deployment inferensi bekerja. TTFT adalah prefill plus antrean plus jaringan. TPOT (sama dengan ITL) adalah biaya decode memory-bound per token. Latensi end-to-end adalah TTFT plus TPOT kali panjang output. Throughput adalah token per detik yang diagregasi lintas fleet. Tetapi yang penting bagi produk adalah goodput — fraksi request yang memenuhi setiap SLO secara simultan. Throughput tinggi pada goodput rendah berarti kamu memproses token yang tidak pernah sampai ke pengguna tepat waktu. Angka referensi untuk Llama-3.1-8B-Instruct pada TRT-LLM pada 2026: rata-rata TTFT 162 ms, rata-rata TPOT 7,33 ms, rata-rata E2E 1.093 ms. Selalu laporkan P50, P90, P99 — jangan pernah hanya rata-rata. Dan awasi jebakan pengukuran: GenAI-Perf mengeluarkan TTFT dari perhitungan ITL, LLMPerf memasukkannya; dua tool tidak sepakat soal TPOT untuk run yang sama.

**Type:** Learn
**Languages:** Python (stdlib, toy percentile calculator and goodput reporter)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Definisikan TTFT, TPOT, ITL, E2E, throughput, dan goodput secara presisi dan sebutkan komponen yang diukur masing-masing.
- Jelaskan mengapa rata-rata adalah statistik yang salah untuk serving LLM dan cara membaca P50/P90/P99.
- Susun multi-constraint SLO (mis. TTFT<500 ms DAN TPOT<15 ms DAN E2E<2 detik) dan hitung goodput terhadapnya.
- Sebutkan dua tool benchmark yang tidak sepakat soal TPOT untuk run sama dan jelaskan mengapa.

## Masalahnya

"Throughput kami 15.000 token per detik." Lalu apa? Kalau 40% request melewati 2 detik end-to-end, pengguna meninggalkan sesi. Throughput saja tidak memberi tahu apakah produk bekerja.

Inferensi punya banyak sumbu latensi dan masing-masing gagal secara berbeda. Prefill compute-bound dan berskala dengan panjang prompt. Decode memory-bound dan berskala dengan ukuran batch. Delay antrean adalah masalah operasional. Jaringan adalah masalah jarak fisik. Kamu butuh metrik berbeda untuk tiap sumbu, kamu butuh persentil, dan kamu butuh satu komposit yang menyatakan "apakah pengguna mendapat yang diharapkan" — itulah goodput.

## Konsepnya

### TTFT — time to first token

`TTFT = queue_time + network_request + prefill_time`

Prefill mendominasi saat prompt panjang. Pada Llama-3.3-70B FP8 di H100, prompt 32k butuh ~800 ms prefill murni. Waktu antrean adalah perilaku scheduler pada beban. Request jaringan adalah waktu wire termasuk TLS. TTFT adalah latensi yang dilihat pengguna sebelum sesuatu ter-stream balik.

### TPOT / ITL — inter-token latency

Banyak nama untuk satu kuantitas. `TPOT` (time per output token), `ITL` (inter-token latency), `decode latency per token` — semuanya sama. Ia waktu antara token berurutan yang di-stream setelah token pertama.

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

Pada stack Llama-3.3-70B H100 yang sama dengan chunked prefill, rata-rata TPOT ~7 ms. Tanpa chunked prefill, selama prefill panjang pada sekuens tetangga, TPOT bisa melonjak ke 50 ms. Awasi P99, bukan rata-rata.

### Latensi E2E

`E2E = TTFT + TPOT * output_tokens + network_response`

Untuk output panjang (>500 token), E2E didominasi TPOT. Untuk output pendek dengan prompt panjang, E2E didominasi TTFT. Laporkan E2E yang dikondisikan panjang-output.

### Throughput

`throughput = total_output_tokens / elapsed_time`

Metrik agregat. Memberitahumu efisiensi fleet. Tidak memberitahumu kesehatan request individual.

### Goodput — metrik yang sebenarnya kamu pedulikan

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO adalah multi-constraint. Request "bagus" hanya jika setiap constraint terpenuhi. Goodput adalah porsinya. Throughput tinggi pada goodput 60% adalah kegagalan. Throughput lebih rendah pada goodput 99% adalah target.

Pada 2026, goodput adalah metrik yang dipakai dalam submisi MLPerf Inference v6.0 dan pelacakan SLA internal di provider platform AI.

### Mengapa rata-rata adalah statistik yang salah

Distribusi latensi LLM miring-ke-kanan. Batch decode dengan satu tetangga prefill-panjang bisa mengirim 500 token dengan TPOT ~7 ms dan 20 token dengan TPOT ~60 ms. Rata-rata TPOT adalah 9 ms. P99 TPOT adalah 65 ms. Pengguna rutin terkena P99 — itulah mengapa mereka pergi.

Selalu laporkan triplet (P50, P90, P99). Untuk pengalaman pengguna, P99 adalah yang kamu optimalkan.

### Angka referensi — Llama-3.1-8B-Instruct pada TRT-LLM, 2026

- rata-rata TTFT: 162 ms
- rata-rata TPOT: 7,33 ms
- rata-rata E2E: 1.093 ms
- P99 TPOT: bervariasi 10-25 ms tergantung konfigurasi chunked-prefill.

Ini titik referensi NVIDIA yang dipublikasikan. Mereka berubah menurut ukuran model (70B akan menunjukkan 3-5x), hardware (H100 vs B200 ~3x), dan load.

### Jebakan pengukuran

Dua tool benchmark 2026 yang paling dipakai tidak sepakat soal TPOT untuk run sama:

- **NVIDIA GenAI-Perf**: mengeluarkan TTFT dari perhitungan ITL. ITL mulai dari token 2.
- **LLMPerf**: memasukkan TTFT. ITL mulai dari token 1.

Untuk request dengan TTFT 500 ms dan 100 token output dalam 700 ms total decode, GenAI-Perf melaporkan `ITL = 700/99 = 7.07 ms`, LLMPerf melaporkan `ITL = 1200/100 = 12.00 ms`. Pilihan tool mengubah angkanya.

Selalu nyatakan tool mana. Selalu publikasikan definisinya.

### Menyusun SLO

SLO consumer-facing yang wajar untuk model chat 70B pada 2026:

- TTFT P99 <= 800 ms.
- TPOT P99 <= 25 ms.
- E2E P99 <= 3 detik untuk output <300-token.
- Target goodput >= 99%.

SLO enterprise mengetatkan TTFT (200-400 ms) dan melonggarkan E2E. Intinya tuliskan, ukur ketiganya, dan lacak goodput sebagai satu komposit.

### Cara mengukur

- Jalankan trafik nyata atau sintetis realistis (LLMPerf dengan `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`).
- Targetkan 2x konkurensi puncak untuk run benchmark.
- Jalankan 30-50 iterasi, ambil persentil dari sampel gabungan.
- Publikasikan dengan nama tool, versi tool, model, hardware, konkurensi, distribusi prompt.

```figure
throughput-latency
```

## Gunakan

`code/main.py` adalah kalkulator goodput mainan. Hasilkan distribusi latensi sintetis, terapkan SLO, dan hitung goodput. Juga menunjukkan perbedaan TPOT GenAI-Perf vs LLMPerf pada trace sama.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-slo-goodput-gate.md`. Dengan workload dan SLO, ia menghasilkan resep benchmark siap-CI/CD yang menggerbang deploy pada goodput, bukan throughput.

## Latihan

1. Jalankan `code/main.py`. Hasilkan distribusi dengan spike ekor 1%. Bagaimana goodput berubah saat kamu mengetatkan P99 TPOT dari 30 ms menjadi 15 ms?
2. Vendor mengutip "15.000 tok/detik pada Llama 3.3 70B H100". Sebutkan tiga pertanyaan sebelum mempercayainya.
3. Mengapa chunked prefill melindungi P99 TPOT tetapi bukan rata-rata TPOT?
4. Susun SLO konsumen untuk asisten suara (token pertama didengar, bukan dibaca). Metrik mana yang paling terlihat pengguna?
5. Baca README LLMPerf dan dokumen GenAI-Perf. Identifikasi tiga metrik lain di mana kedua tool tidak sepakat.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| TTFT | "time to first token" | Queue + network + prefill; dominated by prefill at long prompts |
| TPOT | "time per output token" | Memory-bound decode cost per token after first |
| ITL | "inter-token latency" | Same as TPOT in most tools (not all — see GenAI-Perf) |
| E2E | "end to end" | TTFT + TPOT * output_len; response-side network on top |
| Throughput | "tok/s" | Fleet efficiency; useless without latency percentiles |
| Goodput | "SLO-met rate" | Fraction of requests meeting every SLO constraint simultaneously |
| P99 | "tail" | 1-in-100 worst-case latency; the user experience metric |
| SLO multi-constraint | "the joint" | AND of all three latency bounds; a request fails if any one is violated |
| GenAI-Perf vs LLMPerf | "the tool trap" | Tools disagree on whether ITL includes TTFT |

## Further Reading

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — canonical definition of TTFT, ITL, TPOT.
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — alternative definitions and measurement recipe.
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — applied measurement on real deployments.
- [LLMPerf](https://github.com/ray-project/llmperf) — Ray-based open-source benchmark.
- [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md) — NVIDIA's benchmark tool.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — the industry-accepted goodput-based benchmark.
