# Prefill/Decode Terdisagregasi — NVIDIA Dynamo dan llm-d

> Prefill compute-bound; decode memory-bound. Menjalankan keduanya pada GPU sama membuang satu resource. Disagregasi memisahkannya ke pool terpisah dan mentransfer KV cache di antaranya lewat NIXL (RDMA/InfiniBand atau fallback TCP). NVIDIA Dynamo (diumumkan GTC 2025, 1.0 GA) berada di atas vLLM/SGLang/TRT-LLM — Planner Profiler + SLA Planner-nya otomatis mencocokkan rasio prefill:decode untuk memenuhi SLO. NVIDIA mempublikasikan gain throughput pada kisaran ini — developer.nvidia.com (2025-06) menunjukkan peningkatan ~6x untuk DeepSeek-R1 MoE pada GB200 NVL72 + Dynamo dalam rezim latensi-menengah, dan halaman produk Dynamo (developer.nvidia.com, tanpa tanggal) mengiklankan hingga 50x throughput MoE pada GB300 NVL72 + Dynamo vs Hopper. Angka "30x" adalah agregat komunitas lintas laporan full-stack Blackwell + Dynamo + DeepSeek-R1; kami tidak menemukan satu sumber primer yang menyatakan persis 30x, jadi perlakukan sebagai klaim direksional. llm-d (Red Hat + AWS) bersifat Kubernetes-native: prefill / decode / router sebagai Service independen dengan HPA per-peran. llm-d 0.5 menambah offloading KV hierarkis, routing LoRA sadar-cache, networking UCCL, scale-to-zero. Ekonomi: rollup internal atas banyak disclosure pelanggan menunjukkan hemat 30–40% pada spend inferensi kelas-$2M (yakni, $600-800K/tahun) saat beralih dari serving kolokasi ke terdisagregasi dengan Dynamo pada SLA konstan; angka spesifik $2M→$600-800K adalah komposit internal, bukan satu studi kasus yang dipublikasikan — pakai sebagai jangkar orde-besaran, bukan kutipan referensi. Prompt pendek (<512 token, output pendek) tidak membenarkan biaya transfer.

**Type:** Learn
**Languages:** Python (stdlib, toy disaggregated-vs-colocated simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 08 (Inference Metrics)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan mengapa prefill dan decode punya alokasi GPU optimal berbeda dan kuantifikasi pemborosan pada kolokasi.
- Gambarkan arsitektur terdisagregasi: pool prefill, pool decode, transfer KV via NIXL, router.
- Sebutkan kondisi saat disagregasi TIDAK menguntungkan (prompt pendek, output pendek).
- Bedakan NVIDIA Dynamo (stack-di-atas) dari llm-d (Kubernetes-native) dan pasangkan masing-masing dengan konteks operasional.

## Masalahnya

Kamu menjalankan Llama 3.3 70B pada 8 H100. Pada workload campuran (prompt panjang + output pendek), GPU idle selama decode karena sebagian besar komputasi habis pada prefill. Pada workload berbeda (prompt pendek + output panjang), sebaliknya terjadi. Kolokasi prefill + decode berarti kamu over-provision keduanya.

Dampak budget: 20-40% waktu GPU terbuang pada resource yang salah. Kamu membeli komputasi H100 untuk menjalankan decode memory-bound, atau membeli bandwidth HBM H100 untuk menjalankan prefill compute-bound. Keduanya pemborosan mahal.

Disagregasi memisahkan prefill dan decode ke pool terpisah yang diukur untuk bottleneck masing-masing. KV cache ditransfer dari pool prefill ke pool decode via interkonek bandwidth-tinggi.

## Konsepnya

### Mengapa bottleneck berbeda

**Prefill** — jalankan transformer atas seluruh prompt input dalam satu forward. Perkalian matriks mendominasi; compute-bound. H100 FP8 memberi ~2000 TFLOPS throughput berguna. Efisiensi batch bagus — satu forward memproses banyak token.

**Decode** — hasilkan satu token dalam satu waktu, membaca seluruh bobot tiap iterasi. Memory-bandwidth-bound. HBM3 memberi ~3 TB/detik. Efisiensi batch bagus hanya pada konkurensi tinggi — pembacaan bobot teramortisasi lintas batch.

Kolokasi keduanya: kamu membeli GPU yang dioptimalkan untuk keduanya. H100 bagus pada keduanya tetapi harganya sama. Pada skala, kamu ingin pool prefill pada H100 / berat-komputasi; pool decode pada H200 / berat-memori, atau dengan kuantisasi agresif.

### Arsitekturnya

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL adalah transport antar-node NVIDIA. Memakai RDMA/InfiniBand bila tersedia, fallback TCP bila tidak. Latensi transfer nyata — biasanya 20-80 ms untuk KV cache prompt 4K-token pada 70B FP8. Inilah mengapa prompt pendek tidak membenarkan disagregasi: pajak transfer melebihi hematnya.

### Dynamo vs llm-d

**NVIDIA Dynamo** (diumumkan GTC 2025, 1.0 GA):
- Berada di atas vLLM, SGLang, TRT-LLM sebagai orkestrator.
- Planner Profiler mengukur workload, SLA Planner otomatis mengonfigurasi rasio prefill:decode.
- Inti Rust, ekstensibilitas Python.
- Gain throughput: NVIDIA melaporkan 6x untuk DeepSeek-R1 MoE pada GB200 NVL72 + Dynamo dalam rezim latensi-menengah (developer.nvidia.com, 2025-06); laporan komunitas "hingga 30x" pada stack penuh Blackwell + Dynamo + DeepSeek-R1 tidak punya satu sumber primer dan sebaiknya diperlakukan direksional.
- GB300 NVL72 + Dynamo: throughput MoE hingga 50x vs Hopper menurut halaman produk Dynamo (developer.nvidia.com, tanpa tanggal).

**llm-d** (Red Hat + AWS, Kubernetes-native):
- Prefill / decode / router sebagai Service Kubernetes independen.
- HPA per-peran dengan sinyal kedalaman antrean (prefill) / utilisasi KV (decode).
- `topologyConstraint packDomain: rack` mengepak klik prefill+decode pada rack sama untuk transfer KV bandwidth-tinggi.
- llm-d 0.5 (2026): offloading KV hierarkis, routing LoRA sadar-cache, networking UCCL, scale-to-zero.

Pakai Dynamo kalau kamu menginginkan orkestrator stack-di-atas terkelola. Pakai llm-d kalau kamu menginginkan primitif Kubernetes-native dan berkomitmen pada ekosistem CNCF.

### Ekonomi

Komposit internal (bukan satu studi kasus yang dipublikasikan — jangkar orde-besaran):

- Spend inferensi $2M/tahun pada serving kolokasi.
- Beralih ke terdisagregasi dengan Dynamo.
- Volume request sama, SLA latensi P99 sama.
- Hemat yang dilaporkan: $600K–$800K/tahun (reduksi 30–40%).
- Tanpa hardware baru.

Kami mensintesis angka ini dari banyak disclosure pelanggan, bukan satu studi kasus yang bisa dikutip; titik data terbitan terdekat adalah TTFT 2x lebih cepat / throughput 61% lebih tinggi ala-Baseten dengan routing KV Dynamo (baseten.co, 2025-10), dan proyeksi VAST + CoreWeave sebesar 60–130% lebih banyak token/$ pada hit rate KV 40–60% (vastdata.com, 2025-12). Hemat berasal dari right-sizing tiap pool; workload berat-prefill (RAG dengan prefix 8K+) diuntungkan lebih daripada yang seimbang.

### Kapan TIDAK mendisagregasi

- Prompt < 512 token dan output < 200 token: pajak transfer mendominasi gain.
- Klaster kecil (< 4 GPU): diversitas pool tidak cukup.
- Tim tidak bisa mengoperasikan dua pool GPU dengan scaling per-peran: Dynamo membantu tetapi tidak sepele.
- Tanpa fabric RDMA: pajak transfer TCP lebih berat.

### Router terintegrasi dengan Fase 17 · 11

Router terdisagregasi sadar-KV-cache (Fase 17 · 11). Request mendarat pada pool decode pemegang prefix-nya — kalau tidak ada kecocokan, ia mengalir prefill → decode. Hit rate dan disagregasi bertumpuk — router sadar-cache menentukan apakah prefill baru bahkan dibutuhkan.

### MoE pada Blackwell adalah tempat angka sesungguhnya

GB300 NVL72 + Dynamo menunjukkan throughput MoE 50x di atas baseline Hopper. Routing expert MoE berat-komputasi pada prefill tetapi berat-memori pada decode (cache expert), sehingga disagregasi menang ganda. Serving model frontier 2026 didominasi-MoE (DeepSeek-V3, varian GPT-5 mendatang).

### Angka yang perlu kamu ingat

Angka benchmark bergeser — NVIDIA dan stack inferensi memposting hasil terbaru tiap kuartal. Cek ulang sebelum mengutip.

- DeepSeek-R1 pada GB200 NVL72 + Dynamo: throughput ~6x vs baseline dalam rezim latensi-menengah (developer.nvidia.com, 2025-06); klaim komunitas "hingga 30x" pada stack penuh Blackwell + Dynamo bersifat agregat direksional tanpa satu sumber primer.
- GB300 NVL72 + Dynamo: throughput MoE hingga 50x vs Hopper (developer.nvidia.com, tanpa tanggal).
- Jangkar hemat (komposit internal, bukan satu studi kasus): $600-800K/tahun dari spend tahunan $2M pada SLA konstan.
- Ambang disagregasi: prompt >512 token + output >200 token.
- Transfer KV via NIXL: 20-80 ms untuk KV prompt-4K pada 70B FP8.

```figure
prefill-decode-split
```

## Gunakan

`code/main.py` mensimulasikan serving kolokasi vs terdisagregasi. Melaporkan throughput, biaya per request, dan crossover panjang-prompt.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-disaggregation-decider.md`. Dengan workload dan klaster, ia memutuskan apakah mendisagregasi.

## Latihan

1. Jalankan `code/main.py`. Pada panjang prompt berapa disagregasi mengalahkan kolokasi?
2. Rancang pool prefill dan pool decode untuk layanan RAG dengan panjang prefix P99 8K, output 300.
3. Dynamo vs llm-d: pilih satu untuk shop murni-Kubernetes tanpa preferensi runtime Python.
4. Hitung biaya transfer KV: prefill 4K pada 70B FP8 = ~500 MB KV. Pada RDMA 100 GB/detik, transfer = 5 ms. Pada TCP 10 GB/detik = 50 ms. Mana yang penting bagi SLA-mu?
5. Routing expert MoE mengubah pola akses KV. Bagaimana perilaku disagregasi dengan MoE yang mengaktifkan expert berbeda per token?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Disaggregated serving | "split prefill/decode" | Separate GPU pools for each phase |
| NIXL | "NVIDIA transport" | Dynamo's inter-node KV transfer (RDMA/TCP) |
| NVIDIA Dynamo | "the orchestrator" | Stack-above coordinator for vLLM/SGLang/TRT-LLM |
| llm-d | "Kubernetes native" | Red Hat + AWS K8s disaggregated stack |
| Planner Profiler | "Dynamo auto-config" | Measures workload, configures pool ratios |
| SLA Planner | "Dynamo policy" | Auto-rate-matches prefill:decode to meet SLOs |
| `packDomain: rack` | "llm-d topology" | Pack prefill+decode on same rack for fast KV |
| UCCL | "unified collective" | llm-d 0.5 networking layer for scale-to-zero |
| MoE expert routing | "expert per token" | DeepSeek-V3 pattern; disaggregation helps |

## Further Reading

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
