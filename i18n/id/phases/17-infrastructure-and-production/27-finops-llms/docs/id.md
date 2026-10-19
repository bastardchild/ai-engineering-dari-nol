# FinOps untuk LLM — Ekonomi Unit dan Atribusi Multi-Tenant

> FinOps tradisional rusak pada spend LLM. Biaya adalah transaksi-token, bukan uptime-resource. Tag tidak memetakan — panggilan API adalah transaksi, bukan aset. Keputusan engineering (desain prompt, window konteks, panjang output) adalah keputusan finansial. Playbook 2026 punya tiga dimensi atribusi yang diinstrumentasikan sejak hari pertama: per-pengguna (`user_id`) untuk harga kursi dan ekspansi, per-task (`task_id` + `route`) untuk biaya permukaan produk dan prioritisasi, per-tenant (`tenant_id`) untuk ekonomi unit dan renewal. Empat lapisan token — prompt, tool, memori, respons — satu bucket menyembunyikan spend. Tangga enforcement untuk produk multi-tenant: rate limit per tenant (2-3x puncak ekspektasi, 429 jelas + retry-after); spend cap harian (1,5-3x plafon terkontrak; memicu pengetatan rate + alert); kill switch pada z-score spend > 4 (auto-pause + page on-call). Pola atribusi: tag-dan-agregat, telemetry-joiner (trace-ID → billing; akurasi tertinggi), sampling-dan-ekstrapolasi, alokasi berbasis-model, event-sourced, streaming real-time. Metrik unit: biaya per query terselesaikan, biaya per artefak tergenerasi — bukan $/M token. Tagging retroaktif selalu meleset; instrumentasikan pada pembuatan request.

**Type:** Learn
**Languages:** Python (stdlib, toy cost-attribution simulator with kill switch)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 14 (Caching)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan mengapa FinOps tradisional (tag + tier) rusak pada spend LLM dan sebutkan tiga dimensi atribusi baru.
- Enumerasikan empat lapisan token (prompt, tool, memori, respons) dan mengapa billing satu-bucket menyembunyikan biaya.
- Rancang tangga enforcement (rate → spend cap → kill switch) untuk produk multi-tenant.
- Pilih metrik unit (biaya per query terselesaikan / artefak) alih-alih $/M token.

## Masalahnya

Tagihanmu berkata $40.000. Kamu tidak tahu:
- Tenant mana yang menghabiskannya.
- Fitur produk mana yang mendorongnya.
- Apakah ada pengguna individual yang abusif.
- Apakah prompt bloat, panggilan tool, atau amplifikasi memori pelakunya.

Tag-dan-agregat sisi-provider bekerja untuk resource cloud (EC2, S3) di mana tag merambat ke line item. Panggilan API LLM tidak auto-tag — kamu harus membubuhkan pengguna/task/tenant pada call site dan membawanya terus. Atribusi retroaktif selalu melewatkan edge case.

## Konsepnya

### Tiga dimensi atribusi

**Per-pengguna** (`user_id`): siapa menghabiskan apa. Mendorong harga kursi, percakapan ekspansi, mengidentifikasi power user.

**Per-task** (`task_id` + `route`): permukaan produk mana berharga berapa. Mendorong prioritisasi fitur, keputusan bunuh-fitur-mahal.

**Per-tenant** (`tenant_id`): pelanggan mana yang profitable. Mendorong ekonomi unit, harga renewal, ambang tier.

Instrumentasikan ketiganya pada call site sejak hari pertama. Retroaktif selalu lebih buruk.

### Empat lapisan token

| Layer | Example | Typical % of total |
|-------|---------|---------------------|
| Prompt | system + user input | 40-60% |
| Tool | tool-call results fed back | 20-40% (agent workloads) |
| Memory | prior conversation / retrieved docs | 10-30% |
| Response | model output | 10-30% |

Menggabungkan keempatnya membuat optimasi buta. Pecahkan dalam skema atribusimu.

### Tangga enforcement

1. **Rate limit** per tenant. 2-3x puncak ekspektasi. Kembalikan 429 dengan `Retry-After`. Tenant merasakan friksi; tanpa tagihan kejutan.

2. **Spend cap harian** per tenant. 1,5-3x plafon terkontrak. Pemicu: ketatkan rate limit + alert customer-success.

3. **Kill switch** pada z-score spend > 4 relatif terhadap baseline tenant. Auto-pause tenant; page on-call; eskalasikan ke ops + CS.

### Pola atribusi

- **Tag-dan-agregat**: bubuhkan header metadata; agregatkan belakangan. Sederhana; kasar.
- **Telemetry joiner**: gabungkan trace ke billing via trace ID. Akurasi tertinggi. Yang dilakukan tim matang.
- **Sampling + ekstrapolasi**: sampel 5-10%, kalikan. Hemat-biaya untuk spend kasar; melewatkan ekor.
- **Alokasi berbasis-model**: regresi untuk menginferensi pendorong biaya. Untuk data legacy tanpa tag.
- **Event-sourced**: biaya sebagai event dalam stream (Kafka / Kinesis). Real-time.
- **Streaming real-time**: dasbor update sub-detik.

### Biaya per X adalah metrik unit

$/M token adalah bahasa vendor. Metrik produk:

- Biaya per tiket support terselesaikan.
- Biaya per artikel tergenerasi.
- Biaya per task agen sukses.
- Biaya per menit-sesi-pengguna.

Ikatkan biaya ke outcome produk. Kalau tidak, optimasi tanpa jangkar.

### Bentuk trace atribusi biaya

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

Emisikan pada setiap panggilan. Simpan di data lake. Agregatkan per dimensi. Stack observabilitas Fase 17 · 13 adalah tempatnya.

### Stack hemat bertumpuk

Tumpukan: cache + batch + route + gateway. Dengan keempatnya:
- Cache L2 (Fase 17 · 14): input ~10x lebih murah.
- Batch (Fase 17 · 15): 50% off.
- Rute ke model murah (Fase 17 · 16): reduksi biaya 60%.
- Efisiensi gateway (Fase 17 · 19): redundansi + retry.

Best-case ditumpuk: ~5-10% dari baseline naif. Kebanyakan tim mengaktifkan 2-3 tuas; sedikit yang menumpuk keempatnya.

### Angka yang perlu kamu ingat

- Dimensi atribusi: per-pengguna, per-task, per-tenant.
- Empat lapisan token: prompt, tool, memori, respons.
- Kill switch: z-score spend > 4.
- Metrik unit: biaya per query terselesaikan, bukan $/M token.
- Optimasi ditumpuk: ~5-10% dari baseline dimungkinkan.

```figure
i4-spend-ladder
```

## Gunakan

`code/main.py` mensimulasikan layanan LLM multi-tenant dengan tangga enforcement tiga-tier. Menginjeksikan tenant abusif dan mendemonstrasikan kill switch menyala.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-finops-plan.md`. Dengan produk dan skala, ia merancang skema atribusi dan tangga enforcement.

## Latihan

1. Jalankan `code/main.py`. Pada z-score berapa kill switch menyala? Bagaimana kamu memilih ambangnya?
2. Rancang dasbor biaya per-tenant, per-task. Apa 5 view yang kamu bangun duluan?
3. Tenant terbesarmu negatif secara ekonomi-unit. Usulkan tiga intervensi diurutkan menurut dampak pelanggan.
4. Hitung biaya per tiket terselesaikan untuk produk support: 3M token/tiket, ~800 tiket/hari, tarif cached GPT-5.
5. Argumenkan apakah tagging retroaktif bisa berhasil. Kapan ia dapat diterima?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Per-user attribution | "user-level cost" | `user_id` stamped on every call |
| Per-task attribution | "feature cost" | `task_id` + `route` identify product surface |
| Per-tenant attribution | "customer cost" | `tenant_id`; drives unit economics |
| Four token layers | "cost layers" | prompt + tool + memory + response |
| Rate limit | "429 guard" | Per-tenant ceiling enforced at gateway |
| Daily spend cap | "daily ceiling" | Tenant-scoped budget with alert |
| Kill switch | "auto-pause" | Spend z-score > 4 triggers auto-suspension |
| Cost per resolved | "product unit metric" | Cost tied to product outcome, not tokens |
| Telemetry joiner | "trace-to-billing" | Highest-accuracy attribution pattern |
| Stacked optimization | "cache+batch+route+gateway" | Compounding savings to ~5-10% baseline |

## Further Reading

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
