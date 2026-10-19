# Routing Model sebagai Primitif Reduksi-Biaya

> Broker dinamis mengevaluasi setiap request (tipe task, panjang token, kemiripan embedding, confidence) dan mengirim query sederhana ke model murah, menaikkan yang kompleks ke model frontier. Juga disebut model cascading. Studi kasus produksi menunjukkan reduksi biaya 20-60% pada kualitas-iso lintas deployment US/UK/UE; peningkatan efisiensi routing 30% pada SaaS bervolume-tinggi berubah menjadi hemat tahunan enam digit. Konteks 2026 adalah harga inferensi LLM turun ~10x per tahun — token kelas-GPT-4 turun dari $20/M menjadi ~$0,40/M dari akhir 2022 ke 2026. Sebagian besar penurunan berasal dari stack serving yang lebih baik (Fase 17 · 04-09), bukan hardware. Routing adalah cara mengubah penurunan harga itu menjadi margin tanpa regresi produk. Mode kegagalannya adalah drift model-murah: rute mendorong 40% ke model lebih lemah, kualitas turun 3-5% pada task penalaran, tak ada yang sadar selama satu kuartal. Gerbangkan rute dengan metrik kualitas online, bukan sekadar eval set offline.

**Type:** Learn
**Languages:** Python (stdlib, toy cascading router simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan cascading model: murah-dulu dengan cek confidence, naikkan saat confidence rendah.
- Enumerasikan empat sinyal routing (klasifikasi task, panjang prompt, kemiripan embedding ke himpunan known-hard, self-confidence dari pass-pertama).
- Hitung biaya campuran ekspektasi pada split routing target dan toleransi kehilangan kualitas.
- Sebutkan metrik pemantauan-drift (gerbang kualitas online) yang menangkap creep model-murah.

## Masalahnya

Layananmu berharga $80k/bulan pada GPT-5. Analitikmu menunjukkan 70% query sederhana: "jam berapa kini di Paris?" "parafrasekan kalimat ini." Model kelas-Haiku menangani itu sempurna pada 3% dari biaya. 30% butuh penalaran GPT-5 — coding, matematika, perencanaan multi-langkah.

Kalau kamu merutekan 70% ke murah dan 30% ke mahal, tagihanmu turun ~65% pada kualitas produk sama. Inilah routing. Triknya adalah membangun broker tanpa meregresi kualitas.

## Konsepnya

### Empat sinyal routing

1. **Klasifikasi task**: sederhana/kompleks/codegen/matematika/chat. Bisa berupa klasifier berbasis-aturan, LLM kecil (kelas-Haiku pada $0,25/M), atau kemiripan embedding ke bucket berlabel. Output: rute = murah / seimbang / frontier.

2. **Panjang prompt**: prompt >4K token sering butuh frontier demi koherensi. Prompt <500 token biasanya tidak.

3. **Kemiripan embedding ke himpunan known-hard**: kalau query dekat (kosinus > 0,88) dengan bucket known-hard, naikkan langsung ke frontier.

4. **Self-confidence dari pass-pertama**: kirim ke murah; kalau log-prob model menunjukkan confidence rendah ATAU ia menolak ATAU mengeluarkan bahasa hedging, ulangi pada frontier. Menambah latensi P95 pada ~10% trafik tetapi menghemat 50%+ pada 90% lainnya.

### Tiga pola

**Pra-rute** (klasifier di depan): latensi tambahan ~5-10 ms; tercepat keseluruhan.

**Cascade** (murah-dulu, naikkan saat confidence rendah): latensi median ~1,2x (run murah plus verifikasi), ~2x pada yang dinaikkan. Lantai kualitas terbaik.

**Rute ensemble** (jalankan murah dan frontier paralel untuk sampel, reward-model memilih): kualitas tertinggi, biaya tertinggi; pakai hanya untuk A/B kritis.

### Implementasi

Gateway AI (Fase 17 · 19) mengekspos routing. LiteLLM punya konfigurasi `router` dengan fallback dan cost-routing. Portkey punya guard + routing. Kong AI Gateway punya routing berbasis-plugin. Marketplace model OpenRouter mengekspos API rekomendasi.

Open-source: RouteLLM (LMSYS), Not Diamond (komersial), Prompt Mule.

### Kurva harga 2026

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | new tier |

Sebagian besar perbaikan adalah efisiensi serving — pelajaran inti di Fase 17 · 04-09 berubah menjadi penurunan biaya sisi-provider. Routing memungkinkanmu menangkap gain itu pada lapisan aplikasi alih-alih menunggu semua penggunamu bermigrasi ke tier murah.

### Drift adalah risiko sesungguhnya

Rutemu mengirim 40% ke model murah. Selama enam bulan, distribusi task bergeser (pengguna makin sofisticated, bertanya lebih panjang). Router tidak sadar karena klasifiernya dilatih pada data Q1. Kualitas turun diam-diam. Tak ada yang mengeluh cukup keras. Kamu mengetahuinya dalam benchmark kompetitor yang kamu kalahkan.

Gerbangkan rute dengan metrik kualitas online:

- Thumbs-up / thumbs-down pengguna per rute.
- LLM-judge otomatis pada sampel held-out (5%) per rute.
- Escalation rate: kalau cascade menaikkan rute >30%, model murah di-over-route.
- Refusal rate per rute.

### Angka yang perlu kamu ingat

- Hemat routing 2026 pada kualitas-iso: studi kasus 20-60%.
- Penurunan harga LLM 2022-2026: agregat ~10x per tahun.
- GPT-4-level 2022 vs 2026: ~$20/M → ~$0,40/M.
- Dampak latensi cascade: median ~1,2x, yang dinaikkan ~2x (~10% trafik).

```figure
model-cascade-router
```

## Gunakan

`code/main.py` mensimulasikan pra-rute, cascade, dan ensemble pada workload campuran. Melaporkan biaya campuran, kehilangan kualitas, dan escalation rate.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-router-plan.md`. Dengan workload dan budget kualitas, ia memilih pola routing dan sinyal.

## Latihan

1. Jalankan `code/main.py`. Pada lantai akurasi berapa cascade mengalahkan pra-rute?
2. Basis penggunamu 30% enterprise (query kompleks), 70% tier gratis (sederhana). Rancang split routing. Metrik online apa yang menggerbangkannya?
3. Rute menurunkan kualitas 2% tetapi menghemat 40%. Apakah itu layak rilis? Tergantung produk — argumenkan keduanya.
4. Implementasikan cek confidence memakai logprob dari API OpenAI / Anthropic. Ambang berapa yang kamu mulai?
5. Selama enam bulan, escalation rate naik dari 8% menjadi 22%. Diagnosis tiga penyebab dan perbaikan untuk masing-masing.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Model routing | "cost broker" | Dynamic choice of model per request |
| Model cascade | "cheap-first escalate" | Run cheap, fall through to frontier on low confidence |
| Pre-route | "classify first" | Classifier up front; no re-run |
| Ensemble route | "parallel pick" | Run multiple, reward-model picks best |
| Escalation rate | "uprouted %" | Fraction of cascade requests that escalated |
| RouteLLM | "LMSYS router" | OSS router library |
| Not Diamond | "commercial router" | SaaS model-routing product |
| Drift | "cheap creep" | Distribution shift without router noticing |
| Online quality gate | "live check" | Automated LLM-judge sampling live traffic |

## Further Reading

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://web.archive.org/web/20260413143335/https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — multi-model gateway with routing primitives.
