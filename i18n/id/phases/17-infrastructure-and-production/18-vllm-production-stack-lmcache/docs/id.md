# Stack Serving Produksi — Offloading KV dan Routing Sadar-Cache

> Stack serving produksi merangkai router, engine, dan observabilitas menjadi satu deployment Kubernetes — dan memperlakukan KV cache sebagai resource yang bisa keluar dari GPU. Offloading KV mengekstrak KV cache keluar dari memori GPU dan memakainya ulang lintas query dan engine (DRAM CPU, lalu disk/Ceph). Production-stack vLLM adalah deployment referensi; LMCache adalah lapisan offloading. Konektor KV Offloading vLLM 0.11.0 (Januari 2026) membuat ini asinkron dan pluggable via Connector API (v0.9.0+). Path offload biasanya tersembunyi dari path request, meski cache miss dan promosi bisa menambah latensi end-to-end. LMCache bernilai bahkan tanpa prefix bersama — saat GPU kehabisan slot KV, request ter-preempt bisa dipulihkan dari CPU alih-alih menghitung ulang prefill. Benchmark yang dipublikasikan pada 16x H100 (80GB HBM) lintas 4 a3-highgpu-4g: saat KV cache melebihi HBM, offload CPU native maupun LMCache sama-sama meningkatkan throughput secara substansial; pada footprint KV rendah, semua konfigurasi menyamai baseline dengan overhead kecil.

**Type:** Learn
**Languages:** Python (stdlib, toy KV-spill simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang/RadixAttention)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Gambarkan lapisan production-stack vLLM: router, engine, offload KV, observabilitas.
- Jelaskan Connector API KV Offloading (v0.9.0+) dan bagaimana path asinkron 0.11.0 menyembunyikan latensi offload.
- Kuantifikasi kapan DRAM-CPU LMCache membantu (KV > HBM) vs menambah overhead (KV cukup kecil untuk muat di HBM).
- Pilih antara offload CPU native vLLM dan konektor LMCache dengan constraint deployment tertentu.

## Masalahnya

Serving vLLM-mu menunjukkan GPU pada 100% HBM dengan event preemption setiap kali konkurensi naik. Request di-evict, diantrekan ulang, dan kamu melakukan prefill ulang atas prompt 2K-token yang sama empat kali dalam semenit. Komputasi GPU habis untuk prefill redundan; goodput jauh di bawah throughput mentah.

Menambah GPU berharga linear. Menambah HBM tidak mungkin. Tetapi DRAM CPU murah — satu soket punya 512 GB+ pada latensi berorde besaran lebih buruk daripada HBM tetapi baik-baik saja untuk KV cache "hangat-sementara".

LMCache mengekstrak KV cache ke DRAM CPU agar request ter-preempt pulih cepat, dan prefix berulang lintas engine berbagi cache tanpa tiap engine melakukan prefill ulang.

## Konsepnya

### Production-stack vLLM

`github.com/vllm-project/production-stack` adalah deployment Kubernetes referensi:

- **Router** — sadar-cache (Fase 17 · 11). Mengonsumsi event KV.
- **Engine** — worker vLLM. Satu per GPU atau per grup TP/PP.
- **Offload KV cache** — deployment LMCache atau konektor native.
- **Observabilitas** — scrape Prometheus, dasbor Grafana, trace OTel.
- **Control plane** — service discovery, konfigurasi, rolling update.

Dirilis sebagai Helm chart + operator.

### Connector API KV Offloading (v0.9.0+)

vLLM 0.9.0 memperkenalkan Connector API untuk backend KV cache pluggable. Engine-mu meng-offload blok ke konektor; konektor menyimpannya (RAM, disk, object storage, LMCache). Request butuh blok, konektor memuatnya kembali.

vLLM 0.11.0 (Januari 2026) menambah path offload asinkron — offload bisa terjadi di latar sehingga engine tidak terblokir karenanya pada kasus umum. Latensi end-to-end dan throughput tetap bergantung pada bentuk workload, hit rate KV cache, dan tekanan sistem; catatan vLLM sendiri menyebut offload kernel-kustom bisa mendegradasi throughput pada hit rate rendah dan scheduling async punya isu interaksi yang diketahui dengan speculative decoding.

### Offload CPU native vs LMCache

**Offload CPU native vLLM**: lokal-engine. Menyimpan blok KV di RAM host. Cepat diimplementasikan, nol network hop. Tidak lintas engine.

**Konektor LMCache**: skala-klaster. Menyimpan blok di server LMCache bersama (DRAM CPU + tier Ceph/S3). Blok bisa diakses engine mana pun. Benchmark 16x H100 dipublikasikan.

Pilih native saat satu engine punya tekanan HBM. Pilih LMCache saat banyak engine berbagi prefix (RAG dengan system prompt umum, multi-tenant dengan template bersama).

### Perilaku benchmark

Uji 16x H100 (80 GB HBM) tersebar pada 4 a3-highgpu-4g:

- Footprint KV rendah (prompt pendek, konkurensi rendah): semua konfigurasi menyamai baseline, LMCache menambah overhead ~3-5%.
- Footprint sedang: LMCache mulai membantu pada reuse prefix lintas engine.
- KV melebihi HBM: offload CPU native dan LMCache sama-sama meningkatkan throughput substansial; gain LMCache lebih besar karena berbagi lintas-engine.

### Saat LMCache menentukan

- Serving multi-tenant di mana system prompt dibagi lintas tenant.
- RAG di mana chunk dokumen berulang lintas query.
- Varian fine-tuned (LoRA) pada base sama di mana reuse KV base-model memangkas kerja redundan.
- Workload berat-preemption: pulihkan dari CPU lebih murah daripada prefill ulang.

### Saat TIDAK menyalakan

- Tekanan HBM kecil — kamu membayar overhead tanpa manfaat.
- Konteks pendek (<1K token) — waktu transfer > prefill ulang.
- Workload single-tenant single-prompt — tidak ada reuse yang bisa ditangkap.

### Integrasi dengan serving terdisagregasi

Serving terdisagregasi Fase 17 · 17 + LMCache bertumpuk: transfer KV dari pool prefill ke pool decode mendarat di LMCache bila tak dipakai; query berikutnya menarik dari LMCache. Router sadar-cache Fase 17 · 11 bisa merutekan ke engine yang cache lokal ATAU cache-bersama-LMCache-nya cocok.

### Angka yang perlu kamu ingat

- vLLM 0.9.0: Connector API dirilis.
- vLLM 0.11.0 (Jan 2026): path offload asinkron; dampak latensi end-to-end bergantung workload, hit rate KV, dan tekanan sistem (bukan garansi absolut).
- Benchmark 16x H100: LMCache membantu saat footprint KV melebihi HBM.
- Tekanan HBM kecil: overhead 3-5% tanpa manfaat.

```figure
zero-sharding
```

## Gunakan

`code/main.py` mensimulasikan workload berat-preemption dengan dan tanpa LMCache. Melaporkan prefill-ulang yang dihindari, gain throughput, dan utilisasi HBM titik-impas.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-vllm-stack-decider.md`. Dengan bentuk workload dan deployment vLLM, ia memutuskan native vs LMCache vs tidak keduanya.

## Latihan

1. Jalankan `code/main.py`. Pada utilisasi HBM berapa LMCache mulai menguntungkan?
2. Tenant berbagi system prompt 6K-token lintas 200 query/jam. Hitung hemat LMCache ekspektasi per tenant.
3. Server LMCache adalah titik kegagalan tunggal. Rancang strategi HA (replika, fallback ke native).
4. LMCache menyimpan ke Ceph pada spinning disk. Untuk KV 4K-token pada 70B FP8 (500 MB), berapa waktu bacanya vs prefill ulang?
5. Argumenkan apakah path asinkron vLLM 0.11.0 itu "gratis" — di mana overhead bersembunyi?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Production-stack | "the reference deployment" | vLLM's Kubernetes Helm chart + operator |
| Connector API | "KV backend interface" | vLLM 0.9.0+ pluggable KV store interface |
| Native CPU offload | "engine-local spill" | Store KV in host RAM of same engine |
| LMCache | "cluster KV cache" | Cross-engine KV cache server on CPU DRAM + disk |
| 0.11.0 async | "non-blocking offload" | Offload hidden behind engine stream |
| Preemption | "evict to make room" | KV cache shuffle when HBM full |
| Prefix reuse | "same system prompt" | Multiple queries share beginning; cache hit |
| Ceph tier | "disk tier" | Durable storage below DRAM in the cache hierarchy |

## Further Reading

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator.
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector implementation.
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — asynchronous path details.
