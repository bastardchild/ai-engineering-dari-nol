# Mitigasi Cold Start untuk LLM Serverless

> Image model 20 GB butuh 5-10 menit (7B) hingga 20+ menit (70B) untuk beralih dari dingin ke serving. Di dunia serverless sejati, itu bukan pemanasan — melainkan outage. Mitigasi beroperasi pada lima lapisan: image node pra-seed (Bottlerocket di AWS, arsitektur dual-volume), streaming model (NVIDIA Run:ai Model Streamer, native di vLLM), snapshot memori GPU (checkpoint Modal, restart hingga 10x lebih cepat), warm pool (`min_workers=1`), pemuatan bertier (pipeline NVMe→DRAM→HBM ServerlessLLM, reduksi latensi 10-200x), dan migrasi live yang memindahkan token input (KB) alih-alih KV cache (GB). Modal mempublikasikan cold start 2-4 detik sebagai batas bawah; Baseten 5-10 detik default, sub-detik dengan pre-warming. Pelajaran ini mengajarimu mengukur, menganggarkan, dan menumpuk lima lapisan.

**Type:** Learn
**Languages:** Python (stdlib, toy cold-start path simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Enumerasikan lima lapisan mitigasi cold-start dan sebutkan satu tool atau pola pada tiap lapisan.
- Hitung total waktu cold-start sebagai jumlah (provision node) + (unduh bobot) + (load bobot ke HBM) + (init engine) untuk model 70B.
- Jelaskan mengapa migrasi live mentransfer token input (KB) bukan KV cache (GB) dan apa penalti-nya (komputasi ulang).
- Sebutkan tradeoff warm-pool (bayar GPU idle atau terima ekor cold-start) dan ambang SLA di mana `min_workers > 0` menjadi wajib.

## Masalahnya

Endpoint LLM serverless-mu melakukan scale-to-zero semalaman. Jam 8 pagi trafik melonjak. Request pertama menunggu sementara:

1. Karpenter memprovision node GPU: 45-60 detik.
2. Kontainer menarik image 30 GB berisi bobot: 120-300 detik.
3. Engine memuat bobot ke HBM: 45-120 detik tergantung ukuran model dan kecepatan storage.
4. vLLM atau TRT-LLM menginisialisasi CUDA graph, pool KV cache, tokenizer: 10-30 detik.

Total: 220-510 detik (sekitar 3-8 menit) sebelum satu token kembali. SLA-mu 2 detik. Kamu merilis warm-pool (`min_workers=1`) dan masalah seolah hilang — tetapi kini kamu membayar satu GPU idle 24x7. Kalau layananmu punya 5 produk masing-masing dengan satu replika hangat, itu 5 × 24 × 30 = 3.600 GPU-jam/bulan terlepas ada pengguna yang memanggil atau tidak.

Mitigasi cold-start adalah cara mempertahankan ekonomi serverless sembari mendekati latensi always-on.

## Konsepnya

### Lapisan 1 — image node pra-seed (Bottlerocket)

Di AWS, arsitektur dual-volume Bottlerocket memisahkan OS dari data. Snapshot volume data dengan image kontainermu yang sudah pre-pull; referensikan ID snapshot di `EC2NodeClass`-mu. Node baru boot dengan bobot sudah ada di NVMe lokal — langkah 2 dan sebagian langkah 3 hilang. Bekerja dengan Karpenter secara native. Hemat tipikal: 2-4 menit per cold start untuk model besar.

Padanan di GCP: image VM kustom dengan lapisan kontainer pre-baked. Di Azure: snapshot managed disk dengan pola sama.

### Lapisan 2 — streaming model (Run:ai Model Streamer)

Alih-alih memuat seluruh file sebelum menjawab request pertama, stream bobot ke memori GPU lapis-demi-lapis dan mulai memproses begitu blok transformer pertama residen. NVIDIA Run:ai Model Streamer tersedia native di vLLM 2026. Bekerja dengan S3, GCS, dan NVMe lokal. Memangkas waktu load-bobot sekitar separuh untuk model besar dengan menumpuk I/O dan setup komputasi.

### Lapisan 3 — snapshot memori GPU (Modal)

Modal mengambil checkpoint state GPU (bobot, CUDA graph, region KV cache) setelah load pertama. Restart berikutnya mendeserialisasi langsung ke HBM — 10x lebih cepat daripada inisialisasi ulang. Ini hal terdekat dengan "boot GPU hangat dalam 2 detik." Tradeoff: snapshot bersifat per-topologi-GPU, sehingga kalau Karpenter memigrasimu ke SKU berbeda, kamu me-checkpoint ulang.

### Lapisan 4 — warm pool (min_workers=1)

Mitigasi paling sederhana: pertahankan satu replika selalu siap. Biayanya tarif per-jam satu GPU 24x7. Aritmetikanya brutal pada model kecil (kamu membayar $0,85-$1,50/jam untuk menghindari cold start 30 detik) dan ramah pada yang besar (bayar $4/jam untuk menghindari cold start 5 menit). Ambang SLA di mana warm pool menjadi wajib: biasanya TTFT P99 < 60 detik pada model 70B+.

### Lapisan 5 — pemuatan bertier (ServerlessLLM)

ServerlessLLM memperlakukan storage sebagai hierarki: NVMe (cepat tetapi besar), DRAM (menengah tetapi bertier), HBM (mungil tetapi instan). Bobot di-pre-load ke DRAM; load-on-demand ke HBM. Paper melaporkan reduksi latensi 10-200x pada cold load dibanding disk-ke-HBM naif. Adopsi produksi masih awal tetapi integrasi dengan vLLM ada.

### Lapisan 6 — migrasi live (pola bonus)

Saat node menjadi tak tersedia (eviksi spot, drain node), pola tradisional adalah cold-start replika lain dan menguras antrean request. Migrasi live memindahkan token input (kilobyte) ke destinasi yang sudah memuat model dan menghitung ulang KV cache di destinasi. Komputasi ulang lebih murah daripada mentransfer GB KV cache lewat jaringan. Berlaku untuk deployment terdisagregasi.

### Matematika warm-pool

Untuk layanan dengan SLA TTFT P99 2 detik, pertanyaannya bukan "warm pool ya/tidak" melainkan "berapa replika hangat, dan path mana yang mendapatkannya."

- Path interaktif bernilai-tinggi (chat live, agen suara): `min_workers=1-2`.
- Path batch latar (klasifikasi malam hari): scale-to-zero diterima, cold start 5-10 menit bisa ditoleransi.
- Tier premium: `min_workers` per tenant dengan kapasitas khusus.

### Ukur sebelum mengoptimasi

Anatomi cold-start untuk model 70B pada node segar (ilustratif):

| Phase | Time | Mitigation |
|-------|------|-----------|
| Node provision | 50s | Bottlerocket + pre-seeded image, warm pool |
| Image pull | 180s | Pre-seeded data volume (eliminate) |
| Weights to HBM | 75s | Model streamer (halve); GPU snapshot (eliminate) |
| Engine init | 20s | Persistent CUDA graph cache |
| First forward | 3s | Min inherent latency |
| **Total cold** | **328s** | |
| **Total with mitigations** | **~15s** | 22x reduction |

### Angka yang perlu kamu ingat

- Cold start Modal: 2-4 detik (dengan snapshot GPU).
- Cold start default Baseten: 5-10 detik; sub-detik dengan pre-warming.
- Cold start mentah 70B: 3-8 menit.
- Run:ai Model Streamer: speedup load-bobot ~2x.
- Pemuatan bertier ServerlessLLM: reduksi latensi 10-200x (angka paper).

```figure
cold-start-pipeline
```

## Gunakan

`code/main.py` memodelkan path cold-start dengan dan tanpa tiap mitigasi. Melaporkan total waktu cold-start, biaya warm-pool, dan request rate titik-impas di atas mana warm pool menguntungkan.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-cold-start-planner.md`. Dengan SLA, ukuran model, dan bentuk trafik, ia memilih mitigasi mana yang ditumpuk.

## Latihan

1. Jalankan `code/main.py`. Hitung request rate titik-impas di atas mana replika hangat lebih murah daripada membayar pajak cold-start lewat request drop tambahan pada SLO.
2. Kamu men-deploy model 13B dengan SLA TTFT P99 3 detik. Pilih stack mitigasi minimum (lapisan paling sedikit) yang mencapainya.
3. Pra-seed Bottlerocket menghilangkan image pull tetapi bobot tetap dimuat dari snapshot ke HBM. Hitung wall-clock untuk model 70B kalau NVMe berbasis-snapshot membaca pada 7 GB/detik.
4. Provider serverless-mu menawarkan snapshot GPU (Modal) dan timmu menolak karena "snapshot membocorkan PII." Argumenkan kedua sisi — apa risiko realistisnya, dan apa mitigasinya (snapshot ephemeral, enkripsi, isolasi namespace)?
5. Rancang kebijakan warm-pool bertier: berapa replika hangat untuk pengguna berbayar, pengguna trial, dan workload batch? Tunjukkan matematikanya.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Cold start | "the big pause" | Time from request to first token on a fresh replica |
| Warm pool | "always-on minimum" | `min_workers >= 1` to keep at least one replica ready |
| Pre-seeded image | "baked AMI" | Node image with container weights pre-resident |
| Bottlerocket | "AWS node OS" | AWS container-optimized OS with dual-volume snapshot support |
| Model streamer | "streaming load" | Overlap weights I/O with compute setup |
| GPU snapshot | "checkpoint to HBM" | Serialize post-load GPU state; deserialize on restart |
| Tiered loading | "NVMe + DRAM + HBM" | Hierarchy of storage tiers; load on demand |
| Live migration | "move tokens" | Transfer input (KB), recompute KV on destination |
| `min_workers` | "warm replicas" | Serverless minimum keep-alive count |
| Scale-to-zero | "full serverless" | No cost when idle; accept full cold-start tax |

## Further Reading

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Modal's published benchmarks and checkpoint architecture.
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — pre-seeded data volume snapshot pattern.
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — overlap weights load with compute setup.
- [Baseten — Cold starts](https://docs.baseten.co/deployment/autoscaling/cold-starts) — pre-warming playbook.
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — tiered loading design.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — live migration for disaggregated deployments.
