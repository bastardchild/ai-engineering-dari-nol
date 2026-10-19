# Inferensi Edge — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> Constraint inti edge adalah bandwidth memori, bukan komputasi. DRAM mobile berada pada 50-90 GB/detik; HBM3 datacenter mencapai 2-3 TB/detik — selisih 30-50x. Decode bersifat memory-bound sehingga selisihnya menentukan. Pada 2026 lanskap terbagi empat. Apple M4/A18 Neural Engine memuncak pada 38 TOPS dengan unified memory (tanpa salinan CPU↔NPU). Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon mencapai 45 TOPS. WebGPU + WebLLM menjalankan Llama 3.1 8B (Q4) pada ~41 tok/detik di M3 Max (sekitar 70-80% dari native); 17,6k bintang GitHub, API kompatibel-OpenAI, cakupan mobile ~70-75%. NVIDIA Jetson Orin Nano Super (8GB) memuat Llama 3.2 3B / Phi-3; AGX Orin menjalankan gpt-oss-20b via vLLM pada ~40 tok/detik; Jetson T4000 (JetPack 7.1) 2x AGX Orin. TensorRT Edge-LLM mendukung EAGLE-3, NVFP4, chunked prefill — didemokan di CES 2026 oleh Bosch, ThunderSoft, MediaTek.

**Type:** Learn
**Languages:** Python (stdlib, toy bandwidth-bound decode simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 09 (Production Quantization)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan mengapa inferensi LLM mobile memory-bandwidth-bound dan komputasi sekunder.
- Enumerasikan empat target edge (Apple ANE, Qualcomm Hexagon, WebGPU/WebLLM, NVIDIA Jetson) dan pasangkan masing-masing dengan kasus penggunaan.
- Sebutkan selisih cakupan WebGPU 2026 (Firefox Android menyusul) dan landing Safari iOS 26.
- Pilih format kuantisasi per target (Core ML INT4 + FP16 untuk ANE, QNN INT8/INT4 untuk Hexagon, Q4 WebGPU untuk browser, NVFP4 untuk Jetson Thor).

## Masalahnya

Pelanggan menginginkan chatbot on-device: voice-first, privat-by-default, bekerja offline. Pada MacBook Pro M3 Max, Llama 3.1 8B Q4 berjalan ~55 tok/detik — baik. Pada iPhone 16 Pro, model sama berjalan 3 tok/detik — tidak baik. Pada Android menengah dengan Snapdragon 8 Gen 3, 7 tok/detik. Di browser via WebGPU pada Chrome Android v121+, 4-8 tok/detik tergantung perangkat.

Varians throughput bukan masalah porting. Ia adalah selisih bandwidth kali format kuantisasi kali apakah NPU bisa diakses dari user-space. Inferensi edge pada 2026 adalah empat masalah berbeda dengan empat solusi berbeda.

## Konsepnya

### Bandwidth adalah plafon sesungguhnya

Decode membaca seluruh himpunan bobot untuk setiap token. Satu model 7B dalam Q4 adalah 3,5 GB. Membaca 3,5 GB pada 50 GB/detik butuh 70 ms — plafon teoretis ~14 tok/detik. Pada 90 GB/detik (DRAM mobile kelas atas) plafon bergerak ke ~25 tok/detik. Komputasi sebanyak apa pun tidak membantu di bawah angka ini.

HBM3 datacenter pada 3 TB/detik membersihkan 3,5 GB yang sama dalam 1,2 ms — plafonnya 830 tok/detik. Model sama, bobot sama. Subsistem memori berbeda.

### Apple Neural Engine (M4 / A18)

- Hingga 38 TOPS. Unified memory (CPU dan ANE berbagi pool sama) — tanpa overhead salinan.
- Akses lewat Core ML + model `.mlmodel` terkompilasi, atau lewat Metal Performance Shaders (MPS) melalui PyTorch.
- Backend Metal llama.cpp memakai MPS, bukan ANE langsung; ANE native butuh konversi Core ML.
- Path praktis terbaik untuk aplikasi iOS pada 2026: Core ML dengan bobot INT4 + aktivasi FP16.

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- Hingga 45 TOPS. Terintegrasi dengan CPU dan GPU dalam SoC tetapi domain memori terpisah.
- SDK QNN (Qualcomm Neural Network) dan AI Hub menyediakan konversi dari PyTorch/ONNX.
- Chat template, Llama 3.2, Phi-3 semuanya tersedia sebagai artefak kelas-satu di AI Hub.

### NPU Intel / AMD (Lunar Lake, Ryzen AI 300)

- 40-50 TOPS. Software tertinggal di belakang Apple/Qualcomm; OpenVINO membaik tetapi niche.
- Terbaik untuk aplikasi copilot Windows ARM; native pada desktop AMD/Intel untuk local-first.

### WebGPU + WebLLM

- Jalankan model di browser via compute shader WebGPU; tanpa instal.
- Llama 3.1 8B Q4 pada ~41 tok/detik di M3 Max — sekitar 70-80% dari native lewat backend sama.
- 17,6k bintang GitHub pada WebLLM; API JS kompatibel-OpenAI; Apache 2.0.
- Cakupan 2026: Chrome Android v121+, Safari iOS 26 GA, Firefox Android masih menyusul. Keseluruhan cakupan mobile ~70-75%.

### Keluarga NVIDIA Jetson

- Orin Nano Super (8GB): memuat Llama 3.2 3B, Phi-3 pada tok/detik yang bagus.
- AGX Orin: menjalankan gpt-oss-20b via vLLM pada ~40 tok/detik.
- Thor / T4000 (JetPack 7.1): performa 2x AGX Orin, EAGLE-3 dan NVFP4 didukung.
- TensorRT Edge-LLM (2026) mendukung speculative decoding EAGLE-3, bobot NVFP4, chunked prefill — optimasi datacenter yang di-port ke edge.

### Pilihan kuantisasi per target

| Target | Format | Notes |
|--------|--------|-------|
| Apple ANE | INT4 weights + FP16 activations | Core ML conversion path |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub converters |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | Use `mlc_llm convert_weight` + compiled `.wasm`; GGUF is not supported |
| Jetson Orin Nano | Q4 GGUF or TRT-LLM INT4 | Memory-bound |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM path |

### Jebakan long-context di edge

Konteks 128K Llama 3.1 adalah fitur datacenter. Pada ponsel dengan RAM 8 GB, model 4 GB + KV cache 2 GB untuk 32K token + overhead OS = OOM. Deployment edge menjaga konteks pada 4K-8K kecuali kuantisasi KV agresif (KV Q4) diterima.

### Suara adalah killer app

Agen suara sensitif-latensi (token pertama < 500 ms). Inferensi lokal menghilangkan latensi jaringan sepenuhnya. Gabungkan dengan speech-to-text (varian Whisper Turbo berjalan di edge) dan inferensi edge menjadi loop suara berkualitas-produksi.

### Angka yang perlu kamu ingat

- ANE Apple M4 / A18: 38 TOPS.
- Hexagon Qualcomm SD X Elite: 45 TOPS.
- WebLLM M3 Max: ~41 tok/detik pada Llama 3.1 8B Q4.
- AGX Orin: ~40 tok/detik pada gpt-oss-20b via vLLM.
- Selisih bandwidth datacenter-edge: 30-50x.
- Cakupan mobile WebGPU: ~70-75% (Firefox Android tertinggal).

```figure
edge-bandwidth-pipe
```

## Gunakan

`code/main.py` menghitung plafon throughput decode teoretis dari matematika bandwidth-bound lintas target edge. Membandingkan dengan benchmark terobservasi dan menyoroti di mana bandwidth, bukan komputasi, yang menjadi bottleneck.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-edge-target-picker.md`. Dengan platform (iOS/Android/browser/Jetson), model, dan budget latensi/memori, ia memilih format kuantisasi dan pipeline konversi.

## Latihan

1. Jalankan `code/main.py`. Untuk model 7B dalam Q4 pada Snapdragon 8 Gen 3 (~77 GB/detik bandwidth), hitung plafon decode. Bandingkan dengan 6-8 tok/detik terobservasi — apakah runtime efisien?
2. WebGPU pada Android butuh Chrome v121+. Rancang fallback untuk browser lama — sisi-server lewat API kompatibel-OpenAI yang sama.
3. Aplikasi iOS-mu butuh streaming konteks-4K. Kombinasi model/format mana yang membuatmu tetap di bawah 4 GB memori aktif pada iPhone 16?
4. Jetson AGX Orin menjalankan gpt-oss-20b pada 40 tok/detik. Jetson Nano hanya memuat 3B. Kalau produkmu menargetkan keduanya, bagaimana kamu menyatukan stack inferensi?
5. Argumenkan apakah "WebLLM production-ready pada 2026." Kutip cakupan, performa, dan selisih Firefox Android.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| ANE | "Apple neural engine" | On-device NPU in M-series and A-series; unified memory |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU; QNN SDK for access |
| WebGPU | "browser GPU" | W3C-standardized browser GPU API; Chrome/Safari 2026 |
| WebLLM | "browser LLM runtime" | MLC-LLM project; Apache 2.0; OpenAI-compatible JS |
| Jetson | "NVIDIA edge" | Orin Nano / AGX / Thor / T4000 family |
| TRT Edge-LLM | "edge TensorRT" | 2026 edge port of TensorRT-LLM; EAGLE-3 + NVFP4 |
| Unified memory | "shared pool" | CPU and NPU see same RAM; no copy overhead |
| Bandwidth-bound | "memory limited" | Decode gated by bytes/sec reading weights |
| Core ML | "Apple conversion" | Apple framework for ANE-native models |
| QNN | "Qualcomm stack" | Qualcomm Neural Network SDK |

## Further Reading

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — landscape and benchmarks.
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor.
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 edge port announcement.
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — design and benchmarks.
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE-native conversion.
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — pre-converted models for Hexagon.
