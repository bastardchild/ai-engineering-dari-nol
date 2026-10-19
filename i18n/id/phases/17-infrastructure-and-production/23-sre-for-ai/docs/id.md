# SRE untuk AI — Respons Insiden Multi-Agen, Runbook, Deteksi Prediktif

> SRE AI memakai LLM yang di-grounding pada data infrastruktur (log, runbook, topologi layanan) via RAG untuk mengotomatisasi fase investigasi, dokumentasi, dan koordinasi. Pola arsitektur 2026 adalah orkestrasi multi-agen — agen terspesialisasi (log, metrik, runbook) dikoordinasikan supervisor; AI mengusulkan hipotesis dan query, manusia menyetujui judgment call. Datadog Bits AI dan Azure SRE Agent tersedia sebagai produk managed. Runbook berevolusi: NeuBird Hawkeye memakai evaluasi adversarial (dua model menganalisis insiden sama; sepakat = confidence, tak sepakat = ketidakpastian); memori operasional bertahan lintas pergantian tim. Auto-remediasi tetap hati-hati: AI menyarankan, manusia menyetujui. Aksi otonom-penuh sempit (restart pod, rollback deploy spesifik) dengan guardrail ketat — siapa pun yang menjual "setel lalu lupakan" melebih-lebihkan. Frontier yang muncul: prediksi pra-insiden. Riset MIT melaporkan LLM yang dilatih pada log historis + suhu GPU + pola error API memprediksi 89% outage 10-15 menit lebih awal. Proyeksi: 95% LLM enterprise punya failover otomatis pada akhir-2026.

**Type:** Learn
**Languages:** Python (stdlib, toy multi-agent incident triage simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 24 (Chaos Engineering)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Gambarkan arsitektur SRE AI multi-agen: supervisor + agen terspesialisasi (log, metrik, runbook) + gerbang persetujuan manusia.
- Jelaskan mengapa auto-remediasi sempit (restart pod, revert deploy) alih-alih luas (arsitektur-ulang layanan).
- Sebutkan pola evaluasi adversarial (NeuBird Hawkeye): dua model sepakat = confidence; tak sepakat = eskalasi.
- Kutip hasil deteksi-dini MIT 89% dan constraint operasional: prediksi tanpa aktuasi hanyalah dasbor.

## Masalahnya

Engineer on-call di-page jam 3 pagi. "Error rate tinggi di checkout." Mereka mengecek Datadog, Loki, tiga runbook, log deploy. 30 menit kemudian mereka sadar root cause adalah OOM vLLM akibat spike KV cache. Mereka me-restart pod; error hilang.

Pada 2026 dua puluh menit pertama investigasi itu bisa diotomatisasi. Mengelompokkan log per layanan, mengorelasikan ke deploy terbaru, mencocokkan dengan runbook — semuanya RAG + tool-use. Agen tersupervisi bisa melakukan triase pass-pertama dan menyajikan hipotesis sebelum manusia membuka Datadog.

Remediasi otonom-penuh adalah masalah berbeda. Restart pod: aman. Scale pool GPU: aman kalau kebijakan mengizinkan. Arsitektur-ulang layanan: jelas tidak. Disiplinnya adalah menggambar garis sempit itu.

## Konsepnya

### Arsitektur multi-agen

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Supervisor memecah insiden menjadi sub-query. Agen terspesialisasi punya akses tool (pencarian log, PromQL, retrieval dokumen). Supervisor mensintesis, menyajikan hipotesis + bukti ke manusia. Manusia menyetujui atau mengarahkan ulang.

### Scope auto-remediasi

**Aman (sempit)**: restart pod, revert deploy spesifik, scale pool dalam batas pra-disetujui, nyalakan feature flag pra-disetujui.

**Tidak aman (luas)**: ubah topologi layanan, ubah limit resource, deploy kode baru, ubah IAM, ubah database.

Siapa pun yang menjual "setel lalu lupakan" melebih-lebihkan. Himpunan aman tumbuh seiring matangnya SRE AI, tetapi batasnya nyata.

### Evaluasi adversarial (NeuBird Hawkeye)

Dua model independen menganalisis insiden sama. Kalau mereka sepakat soal root cause, confidence tinggi. Kalau tak sepakat, eskalasikan ke manusia dengan kedua hipotesis terlihat. Pola sederhana, filter efektif terhadap root cause halusinasi.

### Memori operasional

Turnover tim adalah pembunuh diam-diam SRE tradisional — pengetahuan tribal pergi. SRE AI menyimpan runbook + post-mortem di vector DB; agen me-retrieve pada setiap insiden baru. Saat engineer baru bergabung, AI punya histori penuh.

### Prediksi pra-insiden

Riset MIT 2025: LLM yang dilatih pada log historis, suhu GPU, pola error API memprediksi 89% outage 10-15 menit sebelum terjadi pada test set.

Reality check: prediksi tanpa aktuasi adalah dasbor. Pertanyaan operasionalnya adalah "saat kami memprediksi, apa yang kami lakukan?" Drain preemptif? Pager? Auto-scale? Jawabannya spesifik-kebijakan.

### Produk pada 2026

- **Datadog Bits AI** — kopilot SRE managed di dalam Datadog.
- **Azure SRE Agent** — Azure-native.
- **NeuBird Hawkeye** — eval adversarial + memori operasional.
- **PagerDuty AIOps** — triase + deduplikasi.
- **Incident.io Autopilot** — komandan insiden + koordinasi.

### Runbook sebagai kode

Runbook berevolusi dari halaman Confluence menjadi markdown terversi dengan seksi terstruktur (gejala, hipotesis, verifikasi, aksi). Runbook terstruktur memberi retrieval RAG lebih baik. Mulai rollout SRE-AI apa pun dengan mengubah runbook tak terstruktur menjadi terstruktur.

### Angka yang perlu kamu ingat

- Deteksi-dini MIT: 89% outage, lead time 10-15 menit.
- Triase multi-agen: supervisor + (log, metrik, runbook) + manusia.
- Himpunan auto-remediasi aman: restart pod, revert deploy, scale dalam batas.
- Eval adversarial: dua model independen; sepakat = confidence.

```figure
i4-incident-agents
```

## Gunakan

`code/main.py` mensimulasikan triase multi-agen: agen log menemukan error, agen metrik menemukan spike CPU, agen runbook mencocokkan ke isu yang dikenal. Supervisor merangking hipotesis.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-ai-sre-plan.md`. Dengan on-call kini, volume insiden, kematangan tim, ia merancang rollout SRE AI.

## Latihan

1. Jalankan `code/main.py`. Bagaimana kalau agen log dan metrik tak sepakat? Bagaimana supervisor menyelesaikan?
2. Definisikan tiga aksi auto-remediasi "aman" untuk layananmu. Justifikasi masing-masing.
3. Tulis template runbook terstruktur: seksi, field wajib, perintah verifikasi.
4. Deteksi prediktif menyala pada lead 12 menit. Apa kebijakanmu — pager, pre-drain, atau keduanya?
5. Argumenkan apakah tim 3-orang harus mengadopsi SRE AI pada 2026 atau menunggu. Pertimbangkan kematangan, volume, risiko.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| AI SRE | "agent for on-call" | LLM-backed incident investigation + coordination |
| Supervisor agent | "the orchestrator" | Top-level agent breaking incidents into sub-queries |
| Specialized agent | "domain agent" | Sub-agent with tool access (logs, metrics, runbooks) |
| Auto-remediation | "AI fixes it" | Narrow pre-approved action; NOT broad re-architecture |
| Operational memory | "vector runbooks" | Post-mortems + runbooks in vector DB for RAG |
| Adversarial eval | "two-model check" | Independent analyses; agreement = confidence |
| NeuBird Hawkeye | "the adversarial one" | Product with adversarial-eval + memory pattern |
| Bits AI | "Datadog's SRE agent" | Datadog-managed AI SRE |
| Pre-incident prediction | "early detection" | 10-15 min lead time on outage prediction |

## Further Reading

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
