# Load Testing API LLM — Mengapa k6 dan Locust Berbohong

> Load tester tradisional tidak dirancang untuk respons streaming, panjang output variabel, metrik level-token, atau saturasi GPU. Dua jebakan menggigit kebanyakan tim. Jebakan GIL: pengukuran level-token Locust menjalankan tokenisasi di bawah GIL Python, yang bersaing dengan generasi request pada konkurensi berat; backlog tokenisasi lalu menggelembungkan inter-token latency yang dilaporkan — klienmulah yang menjadi bottleneck, bukan server. Jebakan keseragaman-prompt: prompt identik dalam loop menguji satu titik pada distribusi token; trafik nyata punya panjang variabel dan kecocokan prefix beragam. LLMPerf memperbaikinya dengan `--mean-input-tokens` + `--stddev-input-tokens`. Pemetaan tool pada 2026: khusus-LLM (GenAI-Perf, LLMPerf, LLM-Locust, guidellm) untuk akurasi level-token; **k6 v2026.1.0** + **k6 Operator 1.0 GA (Sep 2025)** — sadar-streaming, Kubernetes-native terdistribusi via CRD TestRun/PrivateLoadZone, terbaik untuk gerbang CI/CD; Vegeta untuk saturasi constant-rate Go; Locust 2.43.3 hanya dengan ekstensi LLM-Locust untuk streaming. Pola load: steady-state, ramp, spike (uji autoscaling), soak (kebocoran memori).

**Type:** Build
**Languages:** Python (stdlib, toy realistic-prompt generator + latency collector)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan dua antipola (jebakan GIL, jebakan keseragaman-prompt) yang membuat load tester generik berbohong untuk API LLM.
- Pilih tool untuk tujuan tertentu: LLMPerf (run benchmark), k6 + ekstensi streaming (gerbang CI), guidellm (sintetis skala-besar), GenAI-Perf (referensi NVIDIA).
- Rancang empat pola load (steady, ramp, spike, soak) dan sebutkan mode kegagalan yang ditangkap masing-masing.
- Bangun distribusi prompt realistis memakai rata-rata + stddev token input, bukan panjang tetap.

## Masalahnya

Kamu menguji-k6 endpoint LLM-mu pada 500 pengguna konkuren. Ia bertahan. Kamu merilis. Di produksi pada 200 pengguna aktual layanan tumbang — TTFT P99 meledak, GPU terjepit.

Dua hal terjadi. Pertama, k6 mengirim 500 prompt identik — request-coalescing dan prefix caching-mu membuatnya tampak menangani 500 decode konkuren padahal sebenarnya menangani satu. Kedua, k6 tidak melacak inter-token latency pada respons streaming seperti yang dialami mata; ia melihat satu koneksi HTTP, bukan 500 token yang tiba pada interval bervariasi.

Load testing untuk LLM adalah disiplin tersendiri.

## Konsepnya

### Jebakan GIL (Locust)

Locust memakai Python dan menjalankan tokenisasi sisi-klien di bawah GIL. Pada konkurensi tinggi tokenizer antre di belakang generasi request. Inter-token latency yang dilaporkan mencakup backlog tokenisasi sisi-klien. Kamu mengira server lambat; sebenarnya harness ujinya.

Perbaikan: ekstensi LLM-Locust memindahkan tokenisasi ke proses terpisah, atau pakai harness bahasa-terkompilasi (k6, LLMPerf memakai tokenizers.rs).

### Jebakan keseragaman-prompt

Semua load tester yang dikenal memungkinkanmu mengonfigurasi satu prompt. Dalam uji loop 10.000 iterasi prompt yang persis sama terkirim tiap kali. Server melihat prefix sama tiap kali — cache hit prefix mendekati 100%, throughput tampak hebat.

Perbaikan: sampel dari distribusi prompt. LLMPerf memakai `--mean-input-tokens 500 --stddev-input-tokens 150` — panjang beragam, konten beragam.

### Empat pola load

1. **Steady-state** — RPS konstan selama 30-60 menit. Menangkap: regresi performa baseline.
2. **Ramp** — naikkan RPS linear dari 0 ke target selama 15 menit. Menangkap: breakpoint kapasitas, anomali pemanasan.
3. **Spike** — lonjakan RPS 3-10x mendadak selama 2 menit lalu kembali. Menangkap: latensi autoscaling, saturasi antrean, dampak cold-start.
4. **Soak** — steady-state selama 4-8 jam. Menangkap: kebocoran memori, drift connection-pool, overflow observabilitas.

### Pemetaan tool 2026

**LLMPerf** (Anyscale) — Python tetapi tokenisasi didukung-Rust. Prompt rata-rata/stddev. Sadar-streaming. Default terbaik untuk run performa.

**NVIDIA GenAI-Perf** — referensi NVIDIA. Memakai klien Triton; cakupan metrik komprehensif. Perhatikan ITL-nya mengeluarkan TTFT; LLMPerf memasukkannya. Dua tool menghasilkan TPOT berbeda untuk server sama.

**LLM-Locust** (TrueFoundry) — ekstensi Locust yang memperbaiki jebakan GIL. DSL Locust familiar + metrik streaming.

**guidellm** — benchmarking sintetis skala-besar.

**k6 v2026.1.0** + **k6 Operator 1.0 GA (Sep 2025)**:
- k6 sendiri (Go, terkompilasi, tanpa GIL) menambah metrik sadar-streaming.
- k6 Operator memakai CRD TestRun / PrivateLoadZone untuk testing terdistribusi Kubernetes-native.
- Terbaik untuk gerbang CI/CD dan testing SLA.

**Vegeta** — Go, lebih sederhana daripada k6. Saturasi HTTP constant-rate. Tidak sadar-LLM tetapi bagus untuk testing gateway / rate-limit.

**Locust 2.43.3 stock** — punya jebakan GIL untuk LLM. Hanya dengan ekstensi LLM-Locust.

### Gerbang SLA di CI

Jalankan k6 pada PR dengan:

- 30-50 iterasi masing-masing pada RPS baseline.
- Gerbang: TTFT P50/P95, 5xx < 5%, TPOT di bawah ambang.
- Gagalkan build saat pelanggaran.

### Distribusi prompt realistis

Bangun dari sampel trafik nyata (kalau kamu punya) atau dari distribusi yang dipublikasikan (mis. prompt ShareGPT untuk chat, HumanEval untuk kode). Umpankan rata-rata + stddev ke LLMPerf. Hindari loop-dengan-satu-prompt dengan segala cara.

### Angka yang perlu kamu ingat

- k6 Operator 1.0 GA: September 2025.
- k6 v2026.1.0: metrik sadar-streaming.
- Run LLMPerf tipikal: 100-1000 request pada konkurensi X.
- Gerbang CI tipikal: 30-50 iterasi per PR.
- Empat pola: steady, ramp, spike, soak.

```figure
load-pattern-waves
```

## Gunakan

`code/main.py` mensimulasikan load test dengan distribusi prompt realistis, mengukur TPOT efektif, dan mendemonstrasikan jebakan prompt-seragam.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-load-test-plan.md`. Dengan workload dan SLA, ia memilih tool dan merancang empat pola load.

## Latihan

1. Jalankan `code/main.py`. Bandingkan distribusi seragam vs realistis — di mana selisihnya?
2. Tulis skrip k6 untuk gerbang CI: TTFT P95 < 800 ms pada 100 konkuren, runtime 5 menit.
3. Soak test-mu menunjukkan memori tumbuh 50 MB/jam. Sebutkan tiga penyebab dan instrumentasi untuk memilih di antaranya.
4. Uji spike dari 10 RPS ke 100 RPS. Berapa waktu pemulihan ekspektasi kalau Karpenter + vLLM production-stack terpasang (Fase 17 · 03 + 18)?
5. GenAI-Perf melaporkan TPOT=6ms; LLMPerf melaporkan TPOT=11ms pada server sama. Jelaskan.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| LLMPerf | "the LLM harness" | Anyscale benchmark tool, streaming-aware |
| GenAI-Perf | "NVIDIA tool" | NVIDIA reference harness |
| LLM-Locust | "Locust for LLMs" | Locust extension fixing GIL trap |
| guidellm | "synthetic benchmark" | Large-scale synthetic tool |
| k6 Operator | "K8s k6" | CRD-based distributed k6 |
| GIL trap | "Python client overhead" | Tokenization backlog inflates reported latency |
| Prompt-uniformity trap | "single-prompt lie" | Loop with same prompt hits cache, inflates throughput |
| Steady-state | "constant load" | Flat RPS for N minutes |
| Ramp | "linear up" | 0 to target over duration |
| Spike | "burst test" | Sudden multiplier then revert |
| Soak | "long test" | Hours for leak detection |

## Further Reading

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
