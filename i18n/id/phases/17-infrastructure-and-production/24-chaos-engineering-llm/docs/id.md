# Chaos Engineering untuk Produksi LLM

> Chaos engineering untuk LLM adalah disiplin tersendiri pada 2026. Prasyarat sebelum menjalankan eksperimen di produksi: SLI/SLO terdefinisi, observabilitas trace+metrik+log, rollback otomatis, runbook, on-call. Arsitektur punya empat plane: kontrol (penjadwal eksperimen), target (layanan, infra, data store), keamanan (guard + abort + filter trafik), observabilitas (metrik + trace + log), umpan-balik (ke penyesuaian SLO). Guardrail wajib: alert burn-rate menjeda eksperimen kalau burn budget-error harian > 2x ekspektasi; window supresi + korelasi trace-ID mendedupe noise alert. Kadensi: canary kecil mingguan + review SLO; game day bulanan + postmortem; audit resiliensi lintas-tim kuartalan + pemetaan dependensi. Eksperimen spesifik-LLM: overload memori, kegagalan jaringan, outage provider, prompt malformed, badai eviksi KV cache. Tooling: Harness Chaos Engineering (rekomendasi turunan-LLM, downscaling blast-radius, integrasi tool MCP); LitmusChaos (CNCF); Chaos Mesh (CNCF Kubernetes-native).

**Type:** Learn
**Languages:** Python (stdlib, toy chaos experiment runner)
**Prerequisites:** Phase 17 · 23 (SRE for AI), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Sebutkan lima prasyarat chaos engineering (SLI/SLO, observabilitas, rollback, runbook, on-call) dan jelaskan mengapa melewatkan satu pun merusak praktiknya.
- Gambarkan empat plane (kontrol, target, keamanan, observabilitas) dan loop umpan-balik ke SLO.
- Enumerasikan lima eksperimen spesifik-LLM (overload memori, gagal jaringan, outage provider, prompt malformed, badai eviksi KV).
- Pilih tool — Harness, LitmusChaos, Chaos Mesh — dengan stack tertentu.

## Masalahnya

Chaos testing pada stack tradisional sudah mapan. Stack LLM menambah mode kegagalan baru. Prompt 4K-token dengan karakter racun membuat tokenizer macet 12 detik. Provider upstream 429; gateway-mu retry; layananmu OOM akibat konkurensi yang teramplifikasi-retry. Badai eviksi KV cache pada load burst menyebabkan kaskade prefill-ulang yang menjenuhkan komputasi.

Tak satu pun muncul dalam unit test. Chaos engineering adalah cara menemukan mereka sebelum pengguna menemukan.

## Konsepnya

### Prasyarat

Jangan jalankan chaos di produksi tanpa:

1. **SLI/SLO** — indikator dan objektif level-layanan terdefinisi.
2. **Observabilitas** — trace, metrik, log, dirangkai ke dasbor.
3. **Rollback otomatis** — rollback flag-kebijakan Fase 17 · 20.
4. **Runbook** — terstruktur, Fase 17 · 23.
5. **On-call** — seseorang untuk merespons.

Kehilangan satu pun berarti chaos menjadi insiden nyata.

### Empat plane + umpan-balik

**Control plane** — penjadwal eksperimen (workflow Litmus, jadwal Chaos Mesh, UI Harness).

**Target plane** — layanan, pod, node, load balancer, data store.

**Safety plane** — kill switch, window supresi, batas blast-radius, gerbang budget-error.

**Observability plane** — metrik normal + korelasi trace-ID untuk membedakan kegagalan akibat-chaos dari yang alami.

**Loop umpan-balik** — temuan mengalir balik ke penyesuaian SLO, update runbook, perbaikan kode.

### Guardrail wajib

- **Alert burn-rate**: jeda eksperimen kalau burn budget-error harian melebihi 2x ekspektasi.
- **Window supresi**: bisukan alert non-eksperimen dalam blast radius selama eksperimen.
- **Korelasi trace-ID**: semua error akibat-eksperimen membawa tag agar on-call bisa dedupe.

### Lima eksperimen spesifik-LLM

1. **Overload memori** — paksa badai preemption KV cache dengan mengirim request long-context pada konkurensi tinggi. Amati: apakah layanan shed dengan anggun atau crash?

2. **Kegagalan jaringan** — putus konektivitas antara gateway inferensi dan provider. Amati: apakah fallback bekerja dalam SLA? (Fase 17 · 19)

3. **Simulasi outage provider** — 429 100% dari OpenAI. Amati: apakah routing failover ke Anthropic? (Fase 17 · 16, 19)

4. **Prompt malformed** — injeksikan payload pemacet-tokenizer (mis. unicode nested dalam, codepoint UTF-8 raksasa). Amati: apakah satu request mengunci worker?

5. **Badai eviksi KV** — paksa eviksi dengan menjenuhkan budget blok vLLM. Amati: apakah LMCache pulih atau layanan terdegradasi?

### Kadensi

- **Mingguan** — eksperimen canary kecil di staging, mungkin 5% prod.
- **Bulanan** — game day terjadwal pada skenario spesifik; kehadiran lintas-tim; postmortem.
- **Kuartalan** — audit resiliensi lintas-tim; update peta dependensi.

### Tooling

- **Harness Chaos Engineering** — komersial; rekomendasi eksperimen turunan-AI; downscaling blast-radius; integrasi tool MCP.
- **LitmusChaos** — CNCF graduated; berbasis workflow Kubernetes.
- **Chaos Mesh** — CNCF sandbox; gaya CRD Kubernetes-native.
- **Gremlin** — komersial; dukungan luas.
- **AWS FIS** / **Azure Chaos Studio** — penawaran managed cloud.

### Mulai kecil

Eksperimen pertama: pod-kill satu replika decode pada trafik steady. Amati rerouting dan pemulihan. Kalau ini bekerja dan tampak aman, naik ke chaos jaringan.

Eksperimen spesifik-LLM pertama: injeksikan satu 429 provider selama 5 menit. Amati fallback. Kebanyakan tim menemukan fallback mereka belum sepenuhnya teruji.

### Angka yang perlu kamu ingat

- Empat plane: kontrol, target, keamanan, observabilitas.
- Jeda burn-rate: 2x burn budget harian ekspektasi.
- Kadensi: canary mingguan, game day bulanan, audit kuartalan.
- Lima eksperimen LLM: memori, jaringan, provider, prompt malformed, badai KV.

```figure
i4-chaos-guard
```

## Gunakan

`code/main.py` mensimulasikan tiga eksperimen chaos dengan gerbang safety plane. Melaporkan eksperimen mana yang akan memicu abort burn-rate.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-chaos-plan.md`. Dengan stack dan kematangan, ia memilih tiga eksperimen pertama dan tooling.

## Latihan

1. Jalankan `code/main.py`. Eksperimen mana yang memicu gerbang burn-rate dan mengapa?
2. Rancang lima eksperimen chaos pertama untuk layanan RAG berbasis-vLLM. Sertakan kriteria sukses.
3. Alert burn-rate-mu menjeda eksperimen. Bagaimana kamu menentukan root cause — chaos atau alami?
4. Argumenkan apakah chaos harus berjalan di produksi atau hanya staging. Kapan produksi adalah jawaban tepat?
5. Sebutkan tiga mode kegagalan spesifik-LLM yang tidak bisa direproduksi chaos jaringan generik.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| SLI / SLO | "service targets" | Indicator + objective; required prerequisite |
| Blast radius | "scope" | Set of services / users affected by experiment |
| Burn-rate alert | "budget gate" | Fires when error-budget burn rate > 2x expected |
| Game day | "monthly drill" | Scheduled cross-team chaos exercise |
| LitmusChaos | "CNCF workflow" | Graduated CNCF Kubernetes chaos tool |
| Chaos Mesh | "CNCF CRD" | CNCF sandbox Kubernetes-native chaos |
| Harness CE | "commercial AI-assisted" | Harness chaos with AI recommendations |
| Malformed prompt | "tokenizer bomb" | Input that stalls tokenization |
| KV eviction storm | "preemption cascade" | Mass eviction triggering re-prefills |

## Further Reading

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
