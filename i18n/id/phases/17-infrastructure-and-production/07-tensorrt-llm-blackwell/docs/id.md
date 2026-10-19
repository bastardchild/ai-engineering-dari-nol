# Kompilasi Inferensi Spesialis-Hardware — FP8 dan NVFP4 pada Blackwell

> Kompilasi inferensi spesialis-hardware menukar portabilitas dengan throughput, dan TensorRT-LLM — khusus NVIDIA, di-tune untuk Blackwell — adalah contoh paling jelas bahwa pertukaran itu menguntungkan. Pada GB200 NVL72 dengan orkestrasi Dynamo, SemiAnalysis InferenceX mengukur $0,012 per juta token pada model 120B pada Q1-Q2 2026, berbanding $0,09/M pada H100 + vLLM — selisih ekonomi 7x. Stack-nya adalah tiga rezim floating-point yang bertumpuk: FP8 tetap kritis untuk KV cache dan kernel attention karena punya dynamic range yang mereka butuhkan; NVFP4 (microscaling 4-bit) menangani bobot dan aktivasi; prediksi multi-token (MTP) dan prefill/decode terdisagregasi menambah 2-3x lagi di atasnya. Dukungan model hari-nol memuat bobot FP4 langsung tanpa konversi pasca-training. Jebakan bagi tim engineering 2026: TRT-LLM open-source tetapi spesifik-NVIDIA — terspesialisasi CUDA dan Blackwell — sehingga mengadopsinya berarti menukar portabilitas dengan throughput. Hitung matematikamu pada campuran model dan hardware sebelum berkomitmen.

**Type:** Learn
**Languages:** Python (stdlib, toy FP8/NVFP4 memory and cost calculator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 10 · 13 (Quantization)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan mengapa FP8 tetap kritis untuk KV cache dan attention bahkan ketika bobot sudah NVFP4.
- Hitung footprint HBM model frontier pada BF16, FP8, dan NVFP4 dan nalar dari mana hematnya berasal.
- Sebutkan fitur spesifik-Blackwell yang dieksploitasi TRT-LLM (FP4 hari-nol, MTP, serving terdisagregasi, primitif all-to-all).
- Putuskan kapan kunci-NVIDIA TRT-LLM layak dibanding selisih biaya 7x vs vLLM pada Hopper.

## Masalahnya

Frontier ekonomi inferensi pada 2026 adalah "berapa token per dolar". Jawabannya bergantung pada empat pilihan bertumpuk: generasi hardware (Hopper H100/H200 vs Blackwell B200/GB200), presisi (BF16 → FP8 → NVFP4), serving engine (vLLM vs SGLang vs TRT-LLM), dan orkestrasi (polos vs terdisagregasi vs Dynamo).

Pada Hopper dengan vLLM, MoE 120B berjalan sekitar $0,09 per juta token. Pada Blackwell dengan TRT-LLM + Dynamo, model sama berjalan $0,012 — 7x lebih murah. Sebagian selisih adalah hardware (Blackwell 11-15x throughput LLM per-GPU vs Hopper). Sebagian adalah stack: bobot FP4, draft MTP, prefill/decode terdisagregasi, dan all-to-all NVLink 5 untuk komunikasi expert MoE.

Kamu tidak bisa mereplikasi ini di luar stack NVIDIA. Itulah tradeoff-nya — portabilitas demi ekonomi. Memahami pilihan stack mana menyumbang berapa bagian selisih adalah inti pelajaran ini.

## Konsepnya

### Mengapa FP8 tetap batas bawah KV cache

Kesalahan umum pada 2026: menganggap NVFP4 berlaku di mana-mana. Tidak. KV cache butuh FP8 (8-bit floating point) karena menyimpan key dan value attention yang merentang dynamic range lebar. Menguantisasi KV ke FP4 menyebabkan kehilangan akurasi katastropik — ekor distribusi terpotong dan skor attention runtuh. Bit eksponen FP8 memberi KV cache rentang yang dibutuhkan.

NVFP4 (2025-2026) berlaku untuk bobot dan aktivasi. Microscaling: tiap blok bobot punya faktor skala sendiri sehingga blok-blok kecil bisa merentang dynamic range berbeda tanpa kehilangan skala per-tensor. Untuk aktivasi, FP4 bertahan karena aktivasi berjangkauan kecil dalam satu lapisan.

Konfigurasi Blackwell tipikal:

- Bobot: NVFP4 (microscaling 4-bit).
- Aktivasi: NVFP4.
- KV cache: FP8.
- Akumulator attention: FP32 (stabilitas softmax).

### Primitif spesifik-Blackwell yang dipakai TRT-LLM

- **Bobot FP4 hari-nol**: provider model merilis bobot FP4 langsung; TRT-LLM memuat tanpa konversi pasca-training. Tanpa langkah AWQ / GPTQ untuk FP4.
- **Prediksi multi-token (MTP)**: ide sama seperti EAGLE (Fase 17 · 05) tetapi terintegrasi ke build TRT-LLM.
- **Serving terdisagregasi**: prefill dan decode pada pool GPU terpisah, KV cache ditransfer lewat NVLink atau InfiniBand. Ide sama seperti Dynamo (Fase 17 · 20).
- **Primitif komunikasi all-to-all**: NVLink 5 memangkas latensi komunikasi expert MoE 3x vs Hopper. Kernel MoE TRT-LLM di-tune untuk ini.
- **Microscaling NVFP4 + MXFP8**: penanganan faktor-skala yang diakselerasi hardware pada Tensor Core Blackwell.

### Angka yang perlu kamu hafal

- HGX B200 pada $0,02/M token pada GPT-OSS-120B via TRT-LLM.
- GB200 NVL72 pada $0,012/M token via Dynamo (mengorkestrasi TRT-LLM).
- H100 + vLLM ≈ $0,09/M token pada workload sebanding.
- Gain throughput 2,8x dalam tiga bulan update TRT-LLM (2026).
- Throughput LLM per-GPU 11-15x, Blackwell vs Hopper.
- MLPerf Inference v6.0 (April 2026): Blackwell mendominasi setiap task yang disubmit.

### Berapa biaya kualitas FP4 sesungguhnya

NVFP4 itu agresif. Pada workload yang berat-penalaran (chain-of-thought, matematika, generasi-kode dengan konteks panjang), bobot FP4 terdegradasi secara terlihat. Kalibrasi per-blok memitigasi tetapi tidak menghilangkan. Tim yang merilis model penalaran sering memakai bobot FP8 + aktivasi FP4 sebagai kompromi, atau bertahan di H200 dengan FP8 penuh.

Aturannya: selalu validasi kualitas task pada eval set-mu sebelum berkomitmen ke bobot NVFP4.

### Mengapa ini keputusan kunci-NVIDIA

TRT-LLM adalah C++ + CUDA + kernel closed-source. Model perlu dikompilasi untuk SKU GPU spesifik. Tanpa AMD, tanpa Intel, tanpa ARM. Kalau strategi inframu multi-vendor, TRT-LLM non-starter untuk tier yang dilayani-TRT-LLM — kamu tetap bisa menyajikan dari vLLM pada hardware campuran. Kalau kamu khusus-NVIDIA, selisih 7x membayar kuncinya.

### Resep praktis 2026

Untuk tagihan inferensi tahunan $100M+, berjalan pada Hopper + vLLM menyisakan 7-10x di meja. Migrasikan workload dominan-biaya ke Blackwell + TRT-LLM + Dynamo. Pertahankan tier eksperimen pada H100 + vLLM untuk kecepatan iterasi model. Validasi kualitas pada setiap model hasil konversi NVFP4 sebelum produksi.

### Bonus disagregasi

Serving terdisagregasi TRT-LLM (pool prefill dan decode terpisah) dibahas mendalam di Fase 17 · 20. Pada Blackwell, pengalinya bertumpuk: bobot FP4 × speedup MTP × penempatan terdisagregasi × routing sadar-cache. Angka 7x mengasumsikan stack penuh ini.

```figure
pipeline-parallel
```

## Gunakan

`code/main.py` menghitung footprint HBM, throughput decode (rezim memory-bound), dan $/M-token untuk model pada tiga stack: H100 + BF16 + vLLM, H100 + FP8 + vLLM, B200 + NVFP4/FP8 + TRT-LLM. Jalankan untuk melihat efek bertumpuk dan porsi selisih yang disumbang tiap perubahan.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-trtllm-blackwell-advisor.md`. Dengan workload, ukuran model, dan volume token tahunan, ia memutuskan apakah stack Blackwell + TRT-LLM layak dengan kunci-NVIDIA-nya.

## Latihan

1. Jalankan `code/main.py`. Pada MoE 120B dengan 30% parameter aktif, hitung throughput decode yang dibatasi bandwidth-memori pada H100 BF16, H100 FP8, dan B200 NVFP4/FP8. Dari mana lompatan terbesar berasal?
2. Pelanggan menghabiskan $2M/tahun pada H100 + vLLM. Berapa titik-impas jumlah GPU Blackwell yang perlu dibeli untuk mengamortisasi migrasi ke TRT-LLM dalam 12 bulan, dengan selisih ekonomi 7x?
3. Kamu melihat akurasi turun 3 poin pada MATH setelah konversi bobot NVFP4. Sebutkan dua path pemulihan: satu mengutamakan kualitas (pertahankan bobot FP8), satu mengutamakan biaya (kalibrasi dengan data in-domain).
4. Baca hasil inferensi MLPerf v6.0. Task mana yang selisih Blackwell-atas-Hopper-nya terkecil, dan mengapa?
5. Hitung HBM yang dibutuhkan untuk model 405B pada bobot NVFP4 + KV cache FP8 pada konteks 128k. Apakah muat pada satu node GB200 NVL72?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| FP8 | "eight-bit float" | 8-bit floating point; used for KV cache and attention due to dynamic range |
| NVFP4 | "four-bit micro" | NVIDIA's 4-bit microscaling FP format; weights and activations on Blackwell |
| MXFP8 | "MX eight" | Microscaling FP8 variant; hardware-accelerated on Blackwell Tensor Cores |
| Day-0 FP4 | "ship FP4 weights" | Model providers release weights already in FP4; no post-train conversion step |
| MTP | "multi-token prediction" | TRT-LLM's integrated speculative-decoding draft (Phase 17 · 05) |
| Disaggregated serving | "split prefill/decode" | Prefill and decode on separate GPU pools; KV transferred over NVLink/IB |
| All-to-all | "MoE expert comm" | Communication pattern routing tokens to expert GPUs; NVLink 5 cuts 3x |
| InferenceX | "SemiAnalysis inference bench" | The 2026 industry-accepted cost-per-token benchmark |

## Further Reading

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — April 2026 MLPerf results.
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all and MoE kernels.
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — official engine documentation.
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — disaggregated orchestration above TRT-LLM.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — the benchmark suite that publishes Blackwell numbers.
