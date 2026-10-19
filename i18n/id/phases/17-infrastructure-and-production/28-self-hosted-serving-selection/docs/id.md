# Pemilihan Serving Self-Hosted — Mencocokkan Engine dengan Hardware dan Skala

> Pemilihan engine adalah fungsi hardware, skala, dan ekosistem — bukan bacaan leaderboard. Empat engine mendominasi inferensi self-hosted pada 2026: llama.cpp, Ollama, vLLM, SGLang, dengan TGI tertinggal dalam mode pemeliharaan. **llama.cpp** tercepat pada CPU — dukungan model terluas, kontrol penuh atas kuantisasi dan threading. **Ollama** adalah instal satu-perintah laptop-dev, ~15-30% lebih lambat daripada llama.cpp (Go + CGo + serialisasi HTTP), selisih throughput 3x pada load ala-prod. **TGI memasuki mode pemeliharaan 11 Desember 2025** — hanya perbaikan bug, throughput mentah ~10% lebih lambat daripada vLLM tetapi historis observabilitas teratas dan integrasi ekosistem-HF. Status pemeliharaan itu menjadikannya taruhan jangka-panjang berisiko — SGLang atau vLLM adalah default lebih aman untuk proyek baru. **vLLM** adalah default produksi general-purpose — v0.15.1 (Februari 2026) menambah PyTorch 2.10, RTX Blackwell SM120, optimasi H200. **SGLang** adalah spesialis agenik multi-turn / kaya-prefix — 400.000+ GPU dalam produksi (xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS). Constraint hardware: CPU-dulu → llama.cpp. AMD / non-NVIDIA → vLLM adalah path berdukungan-terkuat (TRT-LLM terkunci-NVIDIA). Pola pipeline 2026: dev = Ollama, staging = llama.cpp, prod = vLLM atau SGLang. Engine memakai format bobot berbeda — GGUF untuk keluarga llama.cpp, safetensors HF untuk engine GPU — sehingga konversi format bisa berada di antara tahap.

**Type:** Learn
**Languages:** Python (stdlib, engine-decision tree walker)
**Prerequisites:** All Phase 17 lessons covering engines (04, 06, 07, 09, 18)
**Time:** ~45 minutes

## Tujuan Pembelajaran

- Pilih engine dengan hardware (CPU / AMD / NVIDIA Hopper / Blackwell), skala (1 pengguna / 100 / 10.000), dan workload (chat umum / agen / long-context) tertentu.
- Sebutkan status mode-pemeliharaan TGI 2026 (11 Desember 2025) dan mengapa ia membiaskan proyek baru ke vLLM atau SGLang.
- Jelaskan pipeline dev/staging/prod, termasuk di mana konversi format GGUF-ke-safetensors berada di antara tahap.
- Jelaskan mengapa "CPU-dulu" menunjuk ke llama.cpp dan "AMD" menyingkirkan TRT-LLM.

## Masalahnya

Timmu memulai proyek LLM self-hosted baru. Satu engineer berkata Ollama, satu lagi berkata vLLM, yang ketiga berkata "bukankah TGI langsung bekerja out of the box?" Ketiganya benar untuk konteks berbeda. Tak ada yang benar untuk semua.

Pada 2026 pohon pilihan penting: hardware dulu, skala kedua, workload ketiga. Dan satu peristiwa spesifik 2025 — TGI memasuki mode pemeliharaan Desember 11 — mengubah default untuk proyek baru.

## Konsepnya

### Lima engine

| Engine | Best for | Notes |
|--------|----------|-------|
| **llama.cpp** | CPU / edge / minimal deps / widest model support | Fastest on CPU, full control |
| **Ollama** | Dev laptops, single user, one-command install | 15-30% slower than llama.cpp; 3x prod throughput gap |
| **TGI** | HF ecosystem, regulated industries | **Maintenance mode Dec 11, 2025** |
| **vLLM** | General-purpose production, 100+ users | Broad production default; v0.15.1 Feb 2026 |
| **SGLang** | Agentic multi-turn, prefix-heavy workloads | 400,000+ GPUs in production |

### Keputusan hardware-dulu

**CPU-dulu** → llama.cpp. Ollama juga bekerja tetapi lebih lambat. Tak ada engine lain yang kompetitif pada CPU.

**GPU AMD** → vLLM adalah path berdukungan-terkuat (dukungan ROCm AMD). SGLang juga bekerja. TRT-LLM terkunci-NVIDIA, sehingga gugur.

**NVIDIA Hopper (H100 / H200)** → vLLM atau SGLang atau TRT-LLM. Ketiganya tier-teratas.

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM adalah pemimpin throughput (Fase 17 · 07). vLLM dan SGLang mengikuti dekat.

**Apple Silicon (M-series)** → llama.cpp (Metal). Ollama membungkus ini.

### Keputusan skala-kedua

**1 pengguna / dev lokal** → Ollama. Satu perintah, token-pertama dalam hitungan detik.

**10-100 pengguna / tim kecil** → vLLM GPU-tunggal.

**100-10k pengguna / produksi** → vLLM production-stack (Fase 17 · 18) atau SGLang.

**10k+ pengguna / enterprise** → vLLM production-stack + terdisagregasi (Fase 17 · 17) + LMCache (Fase 17 · 18).

### Keputusan workload-ketiga

**Chat umum / Q&A** → vLLM menang pada default luas.

**Agenik multi-turn (tool, perencanaan, memori)** → RadixAttention SGLang (Fase 17 · 06) mendominasi.

**RAG dengan reuse prefix berat** → SGLang.

**Generasi kode** → vLLM baik; SGLang sedikit lebih baik pada cache.

**Konteks panjang (128K+)** → vLLM + chunked prefill; SGLang + KV bertier.

### Jebakan pemeliharaan TGI

Hugging Face TGI memasuki mode pemeliharaan 11 Desember 2025 — hanya perbaikan bug ke depan. Historis: observabilitas tier-teratas, integrasi ekosistem-HF terbaik di kelasnya (model card, tool keamanan), sedikit di belakang vLLM pada throughput mentah.

Untuk proyek baru pada 2026: default menjauhi TGI. Deployment TGI yang ada bisa berlanjut tetapi sebaiknya bermigrasi pada akhirnya. SGLang dan vLLM adalah default lebih aman.

### Pola pipeline

Dev (Ollama) → staging (llama.cpp) → prod (vLLM). Engine memakai format bobot berbeda — GGUF untuk keluarga llama.cpp, safetensors HF untuk engine GPU — sehingga konversi format bisa berada di antara tahap. Engineer beriterasi cepat pada laptop; staging mencerminkan kuantisasi produksi; prod adalah target serving.

### Caveat Ollama

Ollama hebat untuk dev. Ia tidak hebat untuk produksi bersama: serialisasi HTTP Go menambah overhead, manajemen konkurensi lebih sederhana daripada vLLM, dukungan OpenTelemetry tertinggal. Pakai Ollama di mana ia bersinar — satu pengguna, satu perintah — dan beralih ke vLLM untuk bersama.

### Self-hosted vs managed adalah keputusan terpisah

Fase 17 · 01 (hyperscaler managed), · 02 (platform inferensi) membahas managed. Pelajaran ini mengasumsikan kamu sudah memutuskan self-host. Alasan self-host: residensi data, fine-tune kustom, total cost ownership pada skala, model domain tak tersedia pada hosted.

### Angka yang perlu kamu ingat

- Mode pemeliharaan TGI: 11 Desember 2025.
- vLLM v0.15.1: Februari 2026; PyTorch 2.10; dukungan Blackwell SM120.
- Footprint produksi SGLang: 400.000+ GPU.
- Selisih throughput Ollama vs llama.cpp: 15-30% lebih lambat; 3x pada load prod.

```figure
data-parallel
```

## Gunakan

`code/main.py` adalah walker pohon-keputusan: dengan hardware + skala + workload, ia memilih engine dan menjelaskan mengapa.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-engine-picker.md`. Dengan constraint, ia memilih engine dan menulis rencana migrasi.

## Latihan

1. Jalankan `code/main.py` dengan hardware / skala / workload-mu. Apakah output cocok dengan intuisimu?
2. Inframu 12 H100 dan 8 MI300X AMD. Engine apa? Mengapa TRT-LLM gugur?
3. Tim ingin memakai TGI pada 2026 karena "sudah kami kenal." Argumenkan kasus migrasinya.
4. Ollama dev ke vLLM prod: apa yang berubah pada kuantisasi, konfigurasi, dan observabilitas?
5. Produk RAG dengan panjang prefix P99 8K dan reuse tinggi lintas tenant. Pilih engine dan tumpuk dengan Fase 17 · 11 + 18.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| llama.cpp | "the CPU one" | Widest model support, fastest on CPU |
| Ollama | "the laptop one" | One-command install, dev-grade throughput |
| TGI | "HF's serving" | Maintenance mode since Dec 2025 |
| vLLM | "the default" | Broad production baseline 2026 |
| SGLang | "the agentic one" | Prefix-heavy, RadixAttention |
| TRT-LLM | "NVIDIA-locked" | Blackwell throughput leader, NVIDIA only |
| GGUF | "llama.cpp format" | Bundled K-quant variants |
| Production-stack | "vLLM K8s" | Phase 17 · 18 reference deployment |
| Pipeline pattern | "dev→stage→prod" | Ollama → llama.cpp → vLLM; weight formats differ per engine |

## Further Reading

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — release notes.
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
