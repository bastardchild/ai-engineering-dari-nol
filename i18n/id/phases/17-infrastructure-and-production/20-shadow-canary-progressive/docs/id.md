# Shadow Traffic, Canary Rollout, dan Deployment Progresif untuk LLM

> Rollout LLM menggabung bagian tersulit deployment software: tanpa unit test, mode kegagalan difus, sinyal tertunda. Urutannya adalah (1) mode shadow — duplikasi request prod ke model kandidat, catat, bandingkan dengan nol dampak pengguna; menangkap isu distribusi yang obvious tetapi bukan garansi kualitas; (2) rollout canary — pergeseran trafik progresif 10% → 25% → 50% → 75% → 100% dengan gerbang pada tiap langkah; lacak persentil latensi, biaya/request, error/refusal rate, distribusi panjang output, rate umpan-balik pengguna; (3) pengujian A/B untuk alternatif berbeda setelah stabilitas terkonfirmasi. Non-determinisme tak tereduksi — variasi akurasi hingga 15% lintas run dengan input identik akibat non-asosiativitas FP GPU plus varians ukuran-batch. Biaya adalah variabel, bukan konstanta — model 20% lebih baik bisa 3x lebih mahal per panggilan. Kecepatan rollback menentukan: kalau rollback butuh redeploy, kamu terlalu lambat. Kebijakan tinggal di config/flag; model tinggal di registry dengan digest ter-pin; rollback = balikkan kebijakan + kembalikan ambang + pin model lama dalam hitungan detik.

**Type:** Learn
**Languages:** Python (stdlib, toy canary-progression simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 21 (A/B Testing)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Bedakan mode shadow (bandingkan tanpa-dampak), canary (trafik live progresif), dan A/B (perbandingan setelah-stabilitas-terkonfirmasi).
- Enumerasikan lima metrik canary spesifik-LLM (latensi, biaya/request, error/refusal, distribusi panjang-output, umpan balik pengguna).
- Jelaskan mengapa non-determinisme LLM (hingga 15%) mengubah arti "stabil" dalam rollout.
- Rancang path rollback yang makan waktu detik (balikan kebijakan) bukan jam (redeploy).

## Masalahnya

Kamu merilis model baru. Eval offline menunjukkan gain akurasi 3%. Kamu menyalakannya di produksi. Dalam 24 jam, biaya naik 40%, thumbs-down pengguna naik 8%, tiga tiket pelanggan melaporkan "jawaban aneh." Kamu rollback. Redeploy makan 3 jam. Akhir pekanmu hancur.

Setiap bagian itu bisa dihindari. Mode shadow akan menangkap lonjakan biaya 40% sebelum pengguna mana pun melihatnya. Canary akan berhenti pada 10% saat thumbs-down bergerak. Rollback flag-kebijakan akan makan 30 detik. Disiplin inilah yang mengisi selisih antara "eval offline tampak bagus" dan "pengguna nyata senang."

## Konsepnya

### Mode shadow

Kandidat menerima request sama seperti produksi; output dicatat, tidak dikembalikan ke pengguna. Nol dampak pengguna. Catat:

- Konten output (diff terhadap produksi).
- Jumlah token (delta biaya).
- Latensi.
- Refusal dan error.

Menangkap: ledakan biaya, regresi panjang, perubahan refusal yang obvious, error keras. TIDAK menangkap: delta kualitas yang akan dirasakan pengguna. Shadow adalah smoke test, bukan quality test.

### Rollout canary

Pergeseran trafik progresif dengan gerbang. Progresi tipikal: 1% → 10% → 25% → 50% → 75% → 100%. Gerbangkan pada 5 metrik di tiap langkah:

1. **Persentil latensi** — P50, P95, P99. Pelanggaran: canary punya P99 > 1,5x baseline.
2. **Biaya per request** — $ campuran. Pelanggaran: >20% di atas baseline.
3. **Error / refusal rate** — 5xx plus refusal eksplisit. Pelanggaran: 2x baseline.
4. **Distribusi panjang output** — rata-rata + P99. Pelanggaran: pergeseran distribusional.
5. **Rate umpan-balik pengguna** — thumbs-down / pengajuan tiket. Pelanggaran: 1,5x baseline.

### Non-determinisme adalah varians baru

Input identik menghasilkan output tak identik. Alasan:

- Non-asosiativitas FP GPU (urutan reduksi floating-point bervariasi menurut batch).
- Varians ukuran-batch (prompt sama dalam batch 128 vs batch 16).
- Sampling (temperature > 0).

Terukur: variasi akurasi hingga 15% run-to-run pada eval set identik. "Stabil" dalam rollout berarti metrik dalam varians ekspektasi, bukan identik dengan baseline. Tetapkan gerbang di atas lantai noise.

### Biaya adalah variabel

Model 20% lebih baik bisa 3x lebih mahal per panggilan. Biaya/request adalah salah satu dari lima gerbang. Merilis model "lebih baik" yang merusak ekonomi unit adalah kasus rollback.

### Rollback adalah senjata

- Flag kebijakan (sistem feature flag): balikkan persentase di config; makan waktu detik.
- Pin model (digest registry): model ter-pin tidak auto-upgrade.
- Rollback = kembalikan flag + tetapkan digest ter-pin ke sebelumnya. Detik, bukan jam.

Kalau stack-mu butuh redeploy untuk rollback, perbaiki itu sebelum rolling.

### Tooling

**Argo Rollouts** / **Flagger** — controller delivery progresif Kubernetes. Terintegrasi dengan routing berbobot Istio/Linkerd.

**Routing berbobot Istio** — split trafik level service-mesh.

**KServe / Seldon Core** — serving model dengan canary bawaan.

**Feature flag** — LaunchDarkly, Flagsmith, Unleash. Balikan level-kebijakan, tanpa redeploy.

### Kadensi metrik

Gerbang canary mengecek tiap 5-15 menit tergantung volume trafik. Trafik 1% dengan 10 req/menit memberi 50-150 titik data per window — cukup untuk latensi tetapi berisik untuk umpan balik pengguna. 10% memberi ~10x lebih banyak. Progresi harus jeda cukup lama untuk mengakumulasi sampel yang cukup pada tiap langkah.

### Langkah A/B opsional

Kalau model baru jelas berbeda (perilaku berbeda, kurva biaya berbeda, tone berbeda), uji A/B pada 50% setelah canary lolos. Kalau hanya versi yang ditingkatkan, lompat ke 100% saat gerbang canary lolos.

### Angka yang perlu kamu ingat

- Progresi canary: 1% → 10% → 25% → 50% → 75% → 100%.
- Plafon non-determinisme: varians run-to-run hingga 15% pada input identik.
- Lima metrik canary: latensi, biaya, error/refusal, panjang output, umpan balik pengguna.
- Gerbang biaya: >20% di atas baseline adalah pelanggaran.
- Rollback: detik, bukan jam.

```figure
i4-canary-ramp
```

## Gunakan

`code/main.py` mensimulasikan rollout canary dengan regresi terinjeksi. Melaporkan pada tahap mana rollout berhenti dan gerbang mana yang terpicu.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-rollout-runbook.md`. Dengan model kandidat, baseline, dan toleransi risiko, ia merancang rencana shadow→canary→100%.

## Latihan

1. Jalankan `code/main.py`. Injeksikan regresi biaya 25%. Pada tahap mana canary berhenti?
2. Model barumu punya gain akurasi 3% offline tetapi biaya/request +18%. Apakah itu layak rilis? Tergantung kebijakan — tulis kedua path.
3. Rancang rollback yang makan di bawah 60 detik end-to-end. Daftarkan infrastruktur yang dibutuhkan.
4. Non-determinisme menunjukkan ±7% pada eval-mu. Tetapkan gerbang canary agar tidak false-alarm. Pengali berapa yang kamu pakai?
5. Mode shadow menangkap lonjakan biaya 40% sebelum canary. Tulis aturan alert yang menyala dalam shadow.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Shadow mode | "duplicate to new" | Zero-impact send-to-candidate for logging |
| Canary | "progressive traffic" | Gradual user-exposed rollout with gates |
| Gates | "rollout checks" | Metric thresholds that block progression |
| Non-determinism | "LLM variance" | Irreducible run-to-run differences |
| Policy flag | "flag flip rollback" | Config-level rollback, seconds not hours |
| Model pin | "registry digest" | Immutable reference to a model version |
| Argo Rollouts | "K8s progressive" | Kubernetes-native canary/rollback controller |
| KServe | "inference K8s" | Model serving with canary primitives |
| Istio weighted | "mesh split" | Service-mesh traffic splitter |

## Further Reading

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)
