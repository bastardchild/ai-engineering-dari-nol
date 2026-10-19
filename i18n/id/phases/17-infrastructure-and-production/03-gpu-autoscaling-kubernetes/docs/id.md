# Autoscaling GPU di Kubernetes — Karpenter, KAI Scheduler, Gang Scheduling

> Tiga lapisan, bukan satu. Karpenter memprovision node secara dinamis (di bawah satu menit, 40% lebih cepat daripada Cluster Autoscaler). KAI Scheduler menangani gang scheduling, kesadaran topologi, dan antrean hierarkis — ia mencegah jebakan alokasi parsial 7-dari-8 di mana tujuh node menunggu dan membakar biaya karena satu GPU hilang. Autoscaler level aplikasi (NVIDIA Dynamo Planner, llm-d Workload Variant Autoscaler) melakukan scaling berdasarkan sinyal spesifik inferensi — kedalaman antrean, utilisasi KV cache — bukan duty cycle CPU/DCGM. Jebakan klasik HPA adalah `DCGM_FI_DEV_GPU_UTIL` merupakan pengukuran duty-cycle: 100% bisa berarti 10 request atau 100. vLLM mengalokasikan memori KV cache di muka, sehingga memori tidak pernah memicu scale-down. Pelajaran ini mengajarimu menyusun tiga lapisan dan menghindari kebijakan default Karpenter `WhenEmptyOrUnderutilized` yang menghentikan job GPU yang sedang berjalan di tengah inferensi.

**Type:** Learn
**Languages:** Python (stdlib, toy queue-depth autoscaler simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (Serving Engine Internals)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Gambarkan tiga lapisan autoscaling (provision node, gang scheduling, level aplikasi) dan sebutkan tool yang dipakai di tiap lapisan.
- Jelaskan mengapa `DCGM_FI_DEV_GPU_UTIL` adalah sinyal HPA yang salah untuk vLLM dan sebutkan dua penggantinya (kedalaman antrean, utilisasi KV cache).
- Jelaskan gang scheduling dan mode kegagalan alokasi-parsial yang dicegah KAI Scheduler (7 dari 8 GPU idle).
- Sebutkan kebijakan konsolidasi Karpenter (`WhenEmptyOrUnderutilized`) yang menghentikan job GPU yang berjalan dan nyatakan alternatif aman 2026.

## Masalahnya

Timmu merilis layanan serving LLM di Kubernetes. Kamu menyiapkan HPA dengan `DCGM_FI_DEV_GPU_UTIL` sebagai sinyal. Layanan menempel di 100% utilisasi selama jam kerja. HPA tidak pernah scale up — ia sudah menganggapmu penuh. Kamu menambah replika manual; TTFT turun. HPA tetap tidak scaling. Sinyalnya membohongimu.

Terpisah, kamu memakai Cluster Autoscaler untuk node. Prompt 1M-token datang jam 2 pagi; klaster menghabiskan 3 menit memprovision node, dan request keburu timeout.

Terpisah lagi, kamu men-deploy model 70B yang butuh 8 GPU di 2 node. Klaster punya 7 GPU kosong dan 1 tersebar di 3 node. Cluster Autoscaler memprovision node untuk 1 GPU yang hilang. Tujuh node menunggu 4 menit membakar uang sementara Kubernetes menyiapkan GPU terakhir.

Tiga lapisan, tiga mode kegagalan berbeda. Autoscaling yang sadar-GPU pada 2026 bukan "nyalakan HPA." Melainkan menyusun provision node, gang scheduling, dan autoscaling sinyal-aplikasi.

## Konsepnya

### Lapisan 1 — provision node (Karpenter)

Karpenter mengawasi pod yang pending dan memprovision node dalam ~45-60 detik (Cluster Autoscaler biasanya butuh 90-120 detik untuk node GPU). Ia memilih tipe instance secara dinamis sesuai constraint `NodePool` — kalau pod-mu butuh 8 H100 dan klaster tidak punya node yang cocok, Karpenter langsung memprovision-nya alih-alih men-scaling grup yang ada.

**Jebakan konsolidasi**: `consolidationPolicy: WhenEmptyOrUnderutilized` default Karpenter berbahaya untuk pool GPU. Ia akan menghentikan node GPU yang sedang berjalan untuk memigrasi pod ke instance right-sized yang lebih murah. Untuk workload inferensi artinya mengusir request yang berjalan dan memuat ulang model 70B di node baru. Kerugiannya adalah hilangnya kapasitas bermenit-menit plus request yang gagal.

Pengaturan aman untuk pool GPU:

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

Membiarkan Karpenter mengonsolidasi node yang benar-benar kosong setelah satu jam tetapi tidak pernah mengusir job yang berjalan.

### Lapisan 2 — gang scheduling (KAI Scheduler)

KAI Scheduler (proyek "Karp" lalu diganti nama) menangani apa yang tidak ditangani kube-scheduler default:

**Gang scheduling** — jadwalkan all-or-nothing. Pod inferensi terdistribusi yang butuh 8 GPU semuanya mulai bersamaan atau tidak sama sekali. Tanpa ini, kamu terkena jebakan alokasi-parsial: 7 dari 8 pod mulai, menunggu tanpa batas, membakar uang.

**Kesadaran topologi** — tahu GPU mana berbagi NVLink, mana yang satu rack, mana yang punya InfiniBand di antaranya. Tempatkan pod sesuai informasi itu. Workload tensor-parallel DeepSeek-V3 67B harus tetap dalam satu domain NVLink; KAI Scheduler menghormatinya.

**Antrean hierarkis** — banyak tim bersaing untuk pool GPU yang sama dengan prioritas dan kuota. Jepitan produksi Tim A hanya di-preempt oleh job training Tim B jika aturan prioritas mengizinkan.

KAI di-deploy berdampingan kube-scheduler sebagai scheduler sekunder; kamu menganotasi workload untuk memakainya. Ray dan vLLM production-stack keduanya terintegrasi.

### Lapisan 3 — sinyal level aplikasi

**Jebakan HPA**: `DCGM_FI_DEV_GPU_UTIL` adalah metrik duty-cycle — ia mengukur apakah GPU sedang bekerja pada tiap interval sampling. Utilisasi 100% bisa berarti 10 request konkuren atau 100; GPU sibuk dalam kedua kasus. Scaling berdasarkan duty cycle adalah scaling secara buta.

Lebih buruk lagi, vLLM dan engine sejenis mengalokasikan memori KV cache di muka (hingga `--gpu-memory-utilization`). Penggunaan memori tetap mendekati 90% bahkan pada satu request. HPA berbasis memori tidak pernah scale down.

**Sinyal pengganti 2026**:

- Kedalaman antrean (jumlah request menunggu prefill).
- Utilisasi KV cache (fraksi blok yang dialokasikan ke sekuens aktif).
- P99 TTFT per-replika (sinyal SLA-mu).
- Goodput (request yang memenuhi semua SLO per detik).

NVIDIA Dynamo Planner dan llm-d Workload Variant Autoscaler mengonsumsi sinyal ini dan men-scaling replika. Mereka menggantikan HPA sepenuhnya untuk serving LLM.

### Kapan memakai apa

| Scale decision | Tool |
|----------------|------|
| Add/remove nodes | Karpenter |
| Schedule multi-GPU jobs | KAI Scheduler |
| Add/remove replicas | Dynamo Planner / llm-d WVA (or custom HPA on queue depth) |
| Choose GPU type | Karpenter NodePool |
| Preempt low-priority | KAI Scheduler queues |

### Prefill/decode terdisagregasi mempersulit segalanya

Kalau kamu menjalankan prefill/decode terdisagregasi (Fase 17 · 17), kamu punya dua kelas pod dengan pemicu scaling berbeda: pod prefill melakukan scaling berdasarkan kedalaman antrean, pod decode berdasarkan tekanan KV cache. llm-d mengekspos ini sebagai `Services` terpisah dengan HPA per-peran. Jangan mencoba menaruh satu HPA di depan keduanya.

### Cold start juga penting di sini

Mitigasi cold-start (Fase 17 · 10) adalah titik di mana waktu provision node menjadi terlihat pengguna. Warm-up Karpenter 45-60 detik plus load model 20GB plus init engine berarti request dari-nol butuh 2-5 menit. Pertahankan warm pool (`min_workers=1`) untuk path kritis-SLO, atau pakai checkpointing ala-Modal di lapisan aplikasi.

### Angka yang perlu kamu ingat

- Provision node Karpenter: ~45-60 detik vs Cluster Autoscaler ~90-120 detik (node GPU).
- KAI Scheduler mencegah pemborosan alokasi-parsial — jebakan 7-dari-8.
- `DCGM_FI_DEV_GPU_UTIL` sebagai sinyal HPA: rusak; pakai kedalaman antrean atau utilisasi KV.
- Karpenter `WhenEmptyOrUnderutilized`: menghentikan job GPU yang berjalan. Pakai `WhenEmpty + consolidateAfter: 1h` untuk inferensi.

```figure
autoscaling
```

## Gunakan

`code/main.py` mensimulasikan autoscaler tiga lapis pada workload GPU yang bursty. Membandingkan HPA naif (duty cycle), HPA kedalaman-antrean, dan scaling terjadwal-gang KAI. Melaporkan request tak terpenuhi, menit-GPU idle, dan skor komposit.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-gpu-autoscaler-plan.md`. Dengan topologi klaster, bentuk workload, dan SLO, ia merancang rencana autoscaling tiga lapis.

## Latihan

1. Jalankan `code/main.py`. Pada workload bursty, berapa request yang di-drop HPA duty-cycle naif tetapi tertangkap HPA kedalaman-antrean? Dari mana perbedaannya berasal?
2. Rancang Karpenter NodePool untuk klaster yang menyajikan Llama 3.3 70B FP8 pada H100 SXM5. Spesifikasikan `capacity-type`, `disruption.consolidationPolicy`, `consolidateAfter`, dan taint yang menjauhkan workload non-GPU dari node ini.
3. Timmu melaporkan deployment macet Pending karena "GPU tersedia tetapi pod tidak mau schedule." Diagnosis — apakah ini Karpenter, kube-scheduler, atau KAI Scheduler? Metrik mana yang mengonfirmasi?
4. Pilih sinyal untuk autoscale pod prefill terdisagregasi dan sinyal berbeda untuk pod decode. Justifikasi keduanya.
5. Hitung biaya jebakan konsolidasi `WhenEmptyOrUnderutilized` pada layanan produksi 24x7 yang rata-rata mengalami 60 kejadian request-drop/hari pada P99 TTFT > 10 detik.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Karpenter | "the node provisioner" | Kubernetes node autoscaler; sub-minute provisioning |
| Cluster Autoscaler | "the old scaler" | Kubernetes node autoscaler predecessor; slower, group-based |
| KAI Scheduler | "the GPU scheduler" | Secondary scheduler for gang + topology + queues |
| Gang scheduling | "all or nothing" | Schedule N pods atomically or defer all of them |
| Topology awareness | "rack-aware" | Place pods based on NVLink/IB/rack placement |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU utilization" | Duty-cycle metric; NOT a scaling signal for LLMs |
| Queue depth | "waiting requests" | Correct HPA signal for prefill-bound scaling |
| KV cache utilization | "memory pressure" | Correct HPA signal for decode-bound scaling |
| Consolidation | "Karpenter consolidation" | Node termination to cheaper instance type |
| `WhenEmpty + 1h` | "safe consolidation" | Policy that doesn't evict running GPU jobs |

## Further Reading

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — design docs and configuration examples.
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — consolidation policy semantics and GPU-safe defaults.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner scaling signals.
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray integration pattern.
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — managed-Kubernetes-specific guidance.
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler design.
