# Serving LLM Multi-Region dan Lokalitas KV Cache

> Load balancing round-robin aktif merugikan inferensi LLM yang di-cache. Request yang tidak mendarat pada node pemegang prefix-nya membayar biaya prefill penuh — sekitar 800 ms pada P50 untuk prompt panjang versus ~80 ms dengan cache hit. Pada 2026 pola produksinya adalah router yang sadar-cache (vLLM Router dalam Rust, router llm-d) yang mengonsumsi event KV-cache dan merutekan berdasarkan kecocokan prefix-hash. Riset terbaru (GORGO) menjadikan latensi jaringan lintas-region sebagai term eksplisit dalam objektif routing. Penawaran "inferensi lintas-region" komersial (Bedrock cross-region inference, gateway multi-klaster GKE) memperlakukan inferensi sebagai opaque — mereka menangani ketersediaan, bukan TTFT. JPMorgan dan Mayo Clinic menjalankan failover us-east-1 pada Nov 2024 dalam ~22 menit. Realita DR: 32% kegagalan DR LLM terjadi karena tim mem-backup bobot tetapi lupa file tokenizer atau konfigurasi kuantisasi.

**Type:** Learn
**Languages:** Python (stdlib, toy prefix-cache-aware router simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan mengapa load balancing round-robin merusak inferensi ter-cache dan kuantifikasi penalti TTFT-nya.
- Gambarkan router yang sadar-cache: input (event KV-cache), algoritma (kecocokan prefix-hash), tie-breaker (utilisasi GPU).
- Sebutkan pendorong 32% kegagalan DR untuk LLM (file tokenizer / konfigurasi kuantisasi hilang) dan nyatakan checklist DR tiga-file.
- Bedakan penawaran lintas-region komersial (Bedrock CRI, GKE Multi-Cluster Gateway) dari routing yang sadar-KV.

## Masalahnya

Layananmu berjalan di us-east-1, us-west-2, dan eu-west-1. Kamu menaruh ALB di depan dengan round-robin. Hit rate prefix cache produksi turun ke 8%. TTFT P50 menjadi tiga kali lipat. Log vLLM-mu menunjukkan setiap request membayar biaya prefill penuh.

Round-robin optimal untuk layanan stateless. Inferensi LLM stateful by design — KV cache mengenkode semua yang pernah dilihat model. Routing buta adalah routing ke cache yang salah.

Terpisah, timmu punya rencana DR. Kamu mem-backup bobot model ke S3 lintas-region. Outage regional terjadi; kamu mencoba failover; replika menolak start. Kamu lupa tokenizer.json, konfigurasi kuantisasi, dan konfigurasi scaling RoPE ada di bucket terpisah yang tidak kamu sinkronkan.

Serving LLM multi-region adalah masalah cache, masalah routing, dan masalah higienitas DR — bukan masalah load-balancer.

## Konsepnya

### Routing yang sadar-cache

Request tiba dengan prompt. Router meng-hash prefix (misal, 512 token pertama); ia bertanya ke tiap replika "apakah kamu punya prefix ini di cache?" Replika mempublikasikan event KV-cache pada kanal pub/sub saat mengalokasikan dan mengusir blok. Router memilih replika yang cocok, jatuh ke tie-breaker berbasis utilisasi-GPU kalau tidak ada yang cocok.

**vLLM Router** (Rust, production-stack 2026): berlangganan event `kv.cache.block_added`, memelihara indeks prefix-hash → replika, merutekan dengan lookup O(1). Jatuh ke least-queue-depth saat tidak ada kecocokan.

**Router llm-d**: pola sama, Kubernetes-native. Mempublikasikan event lewat ControlPlane API.

**SGLang RadixAttention** (Fase 17 · 06) adalah padanan intra-replika. Routing lintas-replika ketat berada di hulunya.

### Angka

TTFT P50 pada prompt 2K-token, Llama 3.3 70B FP8, H100:
- Cache hit (replika sama, prefix residen): ~80 ms.
- Cache miss (prefill dingin): ~800 ms.

Selisih 10x. Kalau routermu mengenai 60-80% prefix cache lintas replika, kamu mendekati performa replika-tunggal pada kapasitas N-replika. Kalau hanya mengenai 10%, kamu mendekati scaling naif.

### Lintas-region punya constraint baru — latensi jaringan

RTT antar-region:
- us-east-1 ↔ us-west-2: ~65 ms.
- us-east-1 ↔ eu-west-1: ~75 ms.
- us-east-1 ↔ ap-southeast-1: ~220 ms.

Kalau routing membawa request dari us-east-1 ke prefix panas di ap-southeast-1, hemat prefill (800 → 80 ms) kalah oleh round-trip 440 ms. GORGO (riset 2026) membuat ini eksplisit — minimalkan `prefill_time + network_latency` secara bersama, bukan prefill saja. Sering jawabannya adalah tetap routing regional kecuali pada prefix multi-MB masif di mana prefill mendominasi.

### "Inferensi lintas-region" komersial tidak membantu di sini

Bedrock cross-region inference otomatis merutekan request ke region lain saat ada tekanan kapasitas. Ia mengoptimalkan ketersediaan, bukan TTFT, dan memperlakukan inferensi sebagai opaque. GKE Multi-Cluster Gateway sama — failover level-layanan, tanpa kesadaran KV cache.

Kamu tetap butuh router sadar-cache lapisan-aplikasi bahkan saat memakai ini. Mereka menangani kasus "us-east-1 terbakar". Routing sadar-cache menangani kasus TTFT.

### Higienitas DR — masalah 32% file-hilang

Statistik 2026 yang banyak dikutip: 32% kegagalan DR LLM terjadi karena tim mem-backup bobot tetapi lupa:

- `tokenizer.json` atau `tokenizer.model`
- Konfigurasi kuantisasi (`quantize_config.json`, skala AWQ, zero-point GPTQ)
- Konfigurasi spesifik-model (scaling RoPE, attention mask, chat template)
- Konfigurasi engine (`vllm_config.yaml`, default sampling, manifest adapter LoRA)

Perbaikannya adalah manifest DR minimum tiga-file:

1. Semua file pada repo model HF (bobot + konfigurasi + tokenizer).
2. Konfigurasi serving spesifik-engine.
3. Manifest deployment (YAML K8s, Dockerfile, dependency lock).

Plus: jalankan drill DR tiap kuartal. Drill us-east-1 JPMorgan mencapai pemulihan 22 menit pada Nov 2024 hanya karena playbook-nya dilatih.

### Residensi data bersifat ortogonal

PHI pelanggan UE tidak boleh keluar UE. Kalau router sadar-cache-mu mengirim request asal-Paris ke us-east-1 demi kecocokan prefix, kamu melanggar GDPR terlepas dari gain TTFT. Partisikan router menurut batas residensi sebelum mengoptimasi cache.

### Angka yang perlu kamu ingat

- Selisih TTFT cache hit vs miss: ~10x (80 ms vs 800 ms pada prompt 2K).
- RTT antar-region US-EU: ~75 ms.
- Kegagalan DR: 32% kehilangan konfigurasi tokenizer/kuantisasi.
- Failover us-east-1 JPMorgan Nov 2024: 22 menit (SLA 30 menit).

```figure
cache-aware-router
```

## Gunakan

`code/main.py` mensimulasikan tiga strategi routing (round-robin, sadar-cache regional, sadar-cache global) pada workload multi-region. Melaporkan hit rate cache, TTFT P50/P99, dan tagihan lintas-region.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-multi-region-router.md`. Dengan region, constraint residensi, dan SLA, ia merancang rencana routing.

## Latihan

1. Jalankan `code/main.py`. Pada panjang prompt berapa routing lintas-region mengalahkan routing lokal-saja, dengan RTT 75 ms?
2. Hit rate cache-mu turun dari 70% menjadi 12%. Diagnosis tiga kemungkinan penyebab dan observable yang mengonfirmasi masing-masing.
3. Rancang manifest DR untuk model 70B terkuantisasi-AWQ yang disajikan di vLLM dengan 5 adapter LoRA. Daftarkan setiap file dan konfigurasi.
4. Argumenkan apakah Bedrock cross-region inference "cukup" bagi fintech dengan SLO TTFT ketat. Kutip perilaku spesifik.
5. Request asal-Paris cocok dengan prefix di us-east-1. Apakah kamu merutekannya? Tulis kebijakannya.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Cache-aware routing | "smart LB" | Route on prefix-hash match to KV-cache-holding replica |
| KV-cache events | "cache pub-sub" | Replicas publish block add/evict; router indexes |
| Prefix hash | "cache key" | Hash of first N tokens used as router lookup |
| GORGO | "cross-region routing research" | arXiv 2602.11688; network latency as explicit term |
| Cross-region inference | "Bedrock CRI" | AWS product; availability failover, not TTFT awareness |
| DR manifest | "the backup list" | Every file needed to restore — not just weights |
| Data residency | "GDPR boundary" | Legal constraint on which region sees user data |
| RTT | "round-trip time" | Network latency; 75 ms US-EU, 220 ms US-APAC |
| LLM-aware LB | "cache-hit LB" | Cache-aware router as a product category |

## Further Reading

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — cross-region KV-cache reuse with network latency term.
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — availability failover documentation.
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — cache-aware router source.
