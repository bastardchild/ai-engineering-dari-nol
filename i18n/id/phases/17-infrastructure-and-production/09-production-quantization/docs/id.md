# Kuantisasi Produksi — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

> Format kuantisasi bukan pilihan universal — ia fungsi dari hardware, serving engine, dan workload. GGUF Q4_K_M atau Q5_K_M menguasai CPU dan edge, disalurkan lewat llama.cpp dan Ollama. GPTQ menang di dalam vLLM saat kamu butuh multi-LoRA pada base yang sama. AWQ dengan kernel Marlin-AWQ menghasilkan ~741 tok/detik pada model kelas 7B dengan Pass@1 terbaik pada INT4 — default 2026 untuk produksi datacenter. FP8 tetap jalan tengah pada Hopper, Ada, dan Blackwell — nyaris-lossless dan didukung luas. NVFP4 dan MXFP4 (microscaling Blackwell) agresif dan butuh validasi per-blok. Dua jebakan menggigit tim: dataset kalibrasi harus cocok dengan domain deployment, dan KV cache terpisah dari kuantisasi bobot — pelajaran AWQ "modelku kini 4 GB" melupakan KV cache 10-30 GB pada batch size produksi.

**Type:** Learn
**Languages:** Python (stdlib, toy memory and throughput comparison across formats)
**Prerequisites:** Phase 10 · 13 (Quantization foundations), Phase 17 · 04 (Serving Engine Internals)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Sebutkan enam format kuantisasi produksi dan sweet spot masing-masing pada 2026.
- Pilih format dengan hardware (CPU vs GPU, Hopper vs Blackwell), engine (vLLM, TRT-LLM, llama.cpp), dan workload (chat rutin, penalaran, multi-LoRA) tertentu.
- Hitung memori bobot yang dihemat dan KV cache yang tak tersentuh untuk format pilihan.
- Sebutkan jebakan dataset-kalibrasi yang mendegradasi model terkuantisasi pada trafik domain.

## Masalahnya

Kuantisasi mengurangi memori dan bandwidth HBM, persis yang dibutuhkan decode. Model FP16 70B adalah 140 GB bobot. Kuantisasi bobot ke INT4 (AWQ atau GPTQ) dan model menjadi 35 GB — muat dalam satu H100 dengan ruang untuk KV cache, yang penting karena pada 128 sekuens konkuren dengan konteks 2k, KV cache saja 20-30 GB.

Tetapi kuantisasi tidak gratis. Kuantisasi agresif mendegradasi kualitas, terutama pada task berat-penalaran. Format berbeda bekerja dengan engine berbeda. Hardware berbeda mendukung presisi berbeda secara native. Kebun binatang format 2026 itu nyata dan kamu tidak bisa menyalin pilihan orang lain — kamu harus memilih berdasarkan stack-mu.

## Konsepnya

### Enam format

| Format | Bits | Sweet spot | Engines |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU, edge, laptops | llama.cpp, Ollama |
| GPTQ | 4-8 | Multi-LoRA on vLLM | vLLM, TGI |
| AWQ | 4 | Datacenter GPU production | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell datacenter | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell multi-user | TRT-LLM |
| NVFP4 | 4 | Blackwell multi-user | TRT-LLM |

### GGUF — default CPU/edge

GGUF adalah format file, bukan skema kuantisasi per se — ia membundel varian K-quant (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0) dalam satu kontainer. Q4_K_M dan Q5_K_M adalah default produksi — kualitas nyaris-BF16 pada 4-5 bit. Pilihan terbaik untuk serving CPU atau edge karena llama.cpp sejauh ini engine inferensi CPU tercepat.

Penalti throughput di vLLM: ~93 tok/detik pada 7B — format ini tidak dioptimalkan untuk kernel GPU. Pakai GGUF saat target deployment adalah CPU/edge. Selain itu jangan.

### GPTQ — multi-LoRA di vLLM

GPTQ adalah algoritma kuantisasi pasca-training dengan pass kalibrasi. Kernel Marlin membuatnya cepat pada GPU (speedup 2,6x vs GPTQ non-Marlin). ~712 tok/detik pada 7B.

Kemenangan unik: GPTQ-Int4 mendukung adapter LoRA di vLLM. Kalau kamu menyajikan model dasar plus 10-50 varian fine-tuned (masing-masing sebagai LoRA), GPTQ adalah jalanmu. NVFP4 belum mendukung LoRA per awal 2026.

### AWQ — default GPU datacenter

Activation-aware Weight Quantization. Melindungi ~1% bobot paling-salient selama kuantisasi. Kernel Marlin-AWQ: speedup 10,9x vs naif. ~741 tok/detik pada 7B, Pass@1 terbaik di antara format INT4.

Pilih AWQ untuk serving GPU baru kecuali kamu butuh multi-LoRA (GPTQ) atau FP4 Blackwell agresif (NVFP4).

### FP8 — jalan tengah yang andal

8-bit floating point. Nyaris-lossless. Didukung luas. Tensor Core Hopper mengakselerasi FP8 secara native. Blackwell mewarisi. FP8 adalah default 2026 yang aman saat kualitas non-negotiable (penalaran, medis, generasi-kode). Hemat memori separuh dari INT4 tetapi risiko kualitas jauh lebih rendah.

### MXFP4 / NVFP4 — agresif Blackwell

FP4 microscaling. Tiap blok bobot punya faktor skala sendiri. Agresif tetapi diakselerasi hardware pada Tensor Core Blackwell. Separuh byte per token dibanding FP8 — kemenangan ekonomi di Fase 17 · 07.

Caveat:
- Belum ada dukungan LoRA (awal 2026).
- Penurunan kualitas terlihat pada workload berat-penalaran.
- Validasi pada eval set-mu per model.

### Jebakan kalibrasi

AWQ dan GPTQ butuh dataset kalibrasi — biasanya C4 atau WikiText. Untuk model domain (kode, medis, legal), kalibrasi pada teks web generik membuat algoritma mengambil keputusan salah tentang bobot mana yang dilindungi. Pass@1 pada HumanEval bisa turun beberapa poin.

Perbaikannya: kalibrasi pada data in-domain. Ratusan sampel domain biasanya cukup. Uji pada eval set sebelum merilis.

### Jebakan KV cache

AWQ menyusutkan bobot ke 4 bit. KV cache terpisah dan tetap pada FP16/FP8. Untuk model 70B dengan AWQ:

- Bobot: ~35 GB (INT4 dari 140 GB).
- KV cache pada 128 konkuren × konteks 2k: ~20 GB.
- Aktivasi: ~5 GB.
- Total: ~60 GB — muat pada H100 80GB.

"Modelku kukuantisasi menjadi 4 GB" yang naif melupakan 30-50 GB lainnya. Budget HBM secara holistis.

Terpisah, kuantisasi KV cache (KV FP8 atau KV INT8) adalah pilihan berbeda dengan tradeoff sendiri — ia memengaruhi akurasi attention secara langsung dan bukan kemenangan gratis.

### AWQ INT4 berbahaya untuk penalaran

Chain-of-thought, matematika, generasi-kode dengan konteks panjang — ini menderita secara terlihat dari kuantisasi agresif. AWQ INT4 kehilangan ~3-5 poin pada MATH. Untuk workload berat-penalaran, rilis FP8 atau BF16; terima biaya memorinya.

### Panduan memilih 2026

- Serve CPU/edge: GGUF Q4_K_M. Selesai.
- Serve GPU, chat rutin, tanpa LoRA: AWQ.
- Serve GPU, multi-LoRA: GPTQ dengan Marlin.
- Workload penalaran: FP8.
- Datacenter Blackwell, kualitas tervalidasi: NVFP4 + KV FP8.
- Ambigu: jalankan eval 1.000-sampel pada tiap format kandidat.

```figure
gpu-memory-breakdown
```

## Gunakan

`code/main.py` menghitung footprint memori (bobot + KV + aktivasi) dan throughput relatif lintas enam format untuk rentang ukuran model. Menunjukkan di mana KV cache mendominasi, di mana kompresi bobot menguntungkan, dan di mana FP8 adalah pilihan aman.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-quantization-picker.md`. Dengan hardware, ukuran model, tipe workload, dan toleransi kualitas, ia memilih format dan menghasilkan rencana kalibrasi/validasi.

## Latihan

1. Jalankan `code/main.py`. Untuk model 70B pada 128 konkuren dengan konteks 2k, hitung total HBM untuk tiap format. Format mana yang membuatmu muat pada satu H100 80GB?
2. Kamu punya model coding 7B. Pilih format dan justifikasi. Kalau kamu salah soal toleransi kualitas, apa path pemulihannya?
3. Hitung ukuran dataset-kalibrasi yang dibutuhkan untuk mengkalibrasi AWQ bagi model domain medis. Mengapa lebih banyak data tidak selalu lebih baik?
4. Baca paper kernel Marlin-AWQ atau catatan rilisnya. Jelaskan dalam tiga kalimat mengapa AWQ mencapai 741 tok/detik pada 7B sementara GPTQ mentah mencapai ~712.
5. Kapan masuk akal menggabungkan bobot AWQ dengan KV cache FP8 vs mempertahankan KV pada BF16?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| GGUF | "llama.cpp format" | File format bundling K-quant variants; CPU/edge default |
| Q4_K_M | "Q4 K M" | 4-bit K-quant medium; the production GGUF default |
| GPTQ | "gee pee tee q" | Post-train INT4 with calibration; supports LoRA in vLLM |
| AWQ | "a w q" | Activation-aware INT4; Marlin kernels; best Pass@1 at INT4 |
| Marlin kernels | "fast INT4 kernels" | Custom CUDA kernels for INT4 on Hopper; 10x speedup |
| FP8 | "eight-bit float" | Safe precision default on Hopper/Ada/Blackwell |
| MXFP4 / NVFP4 | "microscaling four" | Blackwell 4-bit FP with per-block scale factors |
| Calibration dataset | "cal data" | Input text used to pick quantization parameters; must match domain |
| KV cache quantization | "KV INT8" | Separate choice from weights; affects attention accuracy |

## Further Reading

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — comparative benchmarks.
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — throughput numbers by format.
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — format-by-format picking.
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — supported formats and flags.
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — original AWQ formulation.
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — original GPTQ formulation.
