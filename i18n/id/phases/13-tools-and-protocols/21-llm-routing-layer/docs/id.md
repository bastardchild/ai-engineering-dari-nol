# LLM Routing Layer — LiteLLM, OpenRouter, Portkey

> Provider lock-in itu mahal. Workload pemanggilan tool yang berbeda cocok dengan model yang berbeda. Routing gateway memberi satu permukaan API, retry, failover, pelacakan biaya, dan guardrails. Tiga arketip mendominasi 2026: LiteLLM (open-source self-hosted), OpenRouter (SaaS terkelola), Portkey (production-grade, open-sourced pada Maret 2026). Pelajaran ini menamai kriteria keputusannya dan menelusuri sebuah routing gateway berbasis stdlib.

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02 (function calling), Phase 13 · 17 (gateways)
**Time:** ~45 minutes

## Tujuan Pembelajaran

- Membedakan opsi routing self-hosted, terkelola, dan production-grade.
- Mengimplementasikan rantai fallback yang mencoba ulang pada kegagalan provider menurut urutan prioritas yang ditetapkan.
- Melacak biaya per-permintaan dan penggunaan token lintas provider.
- Memilih di antara LiteLLM, OpenRouter, dan Portkey untuk batasan production tertentu.

## Masalahnya

Skenario di mana routing provider menjadi penting:

1. **Biaya.** Claude Sonnet berbiaya 3x Haiku. Untuk tugas triage, Haiku sudah cukup; untuk tugas sintesis, Sonnet sepadan. Route per-permintaan.

2. **Failover.** OpenAI punya jam yang buruk. Setiap permintaan gagal. Kamu ingin fallback otomatis ke Anthropic tanpa deploy ulang.

3. **Latensi.** Sebuah UI chat langsung membutuhkan time-to-first-token yang cepat. Sebuah perangkum batch tidak. Route berdasarkan SLA latensi.

4. **Kepatuhan.** Pengguna UE harus tetap di wilayah UE. Route berdasarkan wilayah.

5. **Eksperimen.** A/B dua model pada workload yang sama. Route berdasarkan bucket pengujian.

Menulis manual semua ini per integrasi itu repetitif. Sebuah routing gateway memberi satu API yang kompatibel OpenAI dan menangani sisanya.

## Konsepnya

### Bentuk proxy kompatibel OpenAI

Semua orang berbicara bentuk OpenAI. Routing gateway mengekspos `/v1/chat/completions`, menerima skema OpenAI, dan di dalamnya mem-proxy ke Anthropic / Gemini / Cohere / Ollama / apa pun. Klien tidak peduli.

### Alias model

Alih-alih id snapshot yang ter-pin, kodenamu menulis `our_smart_model`. Gateway memetakan alias ke model nyata. Ketika sebuah provider merilis generasi baru, kamu mengubah alias di sisi server; kodenamu tidak menyentuh apa pun.

### Rantai fallback

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Gateway mendefinisikan ini dalam sebuah config. Retry dihitung terhadap sebuah anggaran sehingga kaskade fallback tidak meledakkan biaya.

### Caching semantik

Prompt yang identik atau nyaris identik menghantam cache alih-alih provider. Penghematan pada agent loop yang berulang bisa 30 sampai 60 persen. Kuncinya berbasis embedding; prompt nyaris-identik berbagi satu slot cache.

### Guardrails

Tingkat gateway:

- **Redaksi PII.** Pass berbasis regex atau ML sebelum mengirim prompt.
- **Pelanggaran kebijakan.** Tolak prompt dengan konten terlarang.
- **Filter output.** Bersihkan completion dari kebocoran.

Portkey dan Kong sama-sama mengirimkan guardrails yang beropini. LiteLLM membiarkannya opsional.

### Batas laju per-key

Satu API key = satu tim. Anggaran per-key mencegah satu tim mengonsumsi kuota bersama. Sebagian besar gateway mendukung ini.

### Trade-off self-hosted vs terkelola

| Factor | LiteLLM (self-hosted) | OpenRouter (managed) | Portkey (production) |
|--------|----------------------|----------------------|----------------------|
| Code | Open source, Python | Managed SaaS | Open source (Mar 2026) + managed |
| Setup | Deploy a proxy | Sign up | Either |
| Providers | 100+ | 300+ | 100+ |
| Billing | Your own keys | OpenRouter credits | Your own keys |
| Observability | OpenTelemetry | Dashboard | Full OTel + PII redaction |
| Best for | Teams that want full control | Rapid prototyping | Production with compliance |

LiteLLM menang ketika kamu punya tim SRE dan menginginkan kedaulatan data. OpenRouter menang ketika kamu menginginkan satu langganan tanpa infra. Portkey menang ketika kamu membutuhkan guardrails dan kepatuhan siap pakai.

### Pelacakan biaya

Setiap permintaan membawa `provider`, `model`, `input_tokens`, `output_tokens`. Kalikan dengan harga per-model per-token (diambil dari lembar harga yang dipegang gateway). Agregasi per-pengguna / per-tim / per-proyek.

### MCP plus routing

Sebuah gateway dapat merutekan panggilan LLM **dan** permintaan sampling MCP. Ketika modelPreferences sebuah permintaan sampling memilih model tertentu, gateway menerjemahkan ke backend yang tepat. Di sinilah Pelajaran 13 · 17 (MCP gateway) dan routing gateway pelajaran ini kadang menyatu menjadi satu layanan.

### Strategi routing

- **Prioritas statis.** Pertama dalam daftar; fallback pada error.
- **Load balancing.** Round-robin atau berbobot.
- **Sadaran biaya.** Pilih model termurah yang memenuhi latensi / kualitas.
- **Sadaran latensi.** Pilih model tercepat dalam N menit terakhir.
- **Sadaran tugas.** Klasifier prompt merutekan "coding" ke satu model, perangkum ke model lain.

```figure
tp-router-failover
```

## Gunakan

`code/main.py` mengimplementasikan sebuah routing gateway dalam ~150 baris: menerima permintaan berbentuk OpenAI, menerjemahkan ke stub per-provider, menjalankan rantai fallback berprioritas, melacak biaya per-permintaan, dan menerapkan pass redaksi PII pada input. Jalankan dengan tiga skenario: permintaan normal, gangguan provider utama yang memicu fallback, kebocoran PII yang tertangkap redaksi.

Yang perlu diperhatikan:

- Dict `ROUTES`: alias -> daftar provider konkret berurutan prioritas.
- Loop fallback mencoba ulang pada 5xx.
- Cost tracker mengalikan penggunaan token dengan tarif per-model.
- PII redactor membersihkan pola berbentuk SSN sebelum meneruskan.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-routing-config-designer.md`. Diberi profil workload (latensi, biaya, kepatuhan), skill ini memilih LiteLLM / OpenRouter / Portkey dan menghasilkan sebuah config routing.

## Latihan

1. Jalankan `code/main.py`. Picu skenario gangguan; konfirmasikan fallback mendarat di provider kedua dan biaya dikreditkan dengan benar.

2. Tambahkan caching semantik: SHA256 dari prompt adalah kunci lookup; cache hit mengembalikan secara instan. Ukur penghematan biaya pada panggilan berulang.

3. Tambahkan klasifier prompt yang merutekan prompt "code ..." ke sebuah alias yang mengutamakan kecerdasan dan prompt "summarize ..." ke sebuah alias yang mengutamakan kecepatan.

4. Rancang anggaran per-tim: setiap tim punya batas belanja bulanan; gateway menolak permintaan begitu cap tercapai. Pilih granularitas penegakan (per-permintaan atau berjendela).

5. Baca dokumentasi LiteLLM, OpenRouter, dan Portkey berdampingan. Sebutkan satu fitur yang dimiliki masing-masing tetapi tidak dimiliki dua lainnya.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Routing gateway | "LLM proxy" | One-API-surface layer in front of many providers |
| OpenAI-compatible | "Speaks the OpenAI schema" | Accepts `/v1/chat/completions` shape, translates to any backend |
| Model alias | "our_smart_model" | Name in your code that the gateway maps to a concrete model |
| Fallback chain | "Retry list" | Ordered list of providers attempted on failure |
| Semantic caching | "Prompt-embedding cache" | Key is embedding of the prompt; near-duplicates share a cache hit |
| Guardrails | "Input/output filters" | Redact PII, reject policy violations |
| Per-key rate limit | "Team budget" | Quota scoped to an API key |
| Cost tracking | "Per-request spend" | Aggregate token usage x price per model |
| LiteLLM | "The open proxy" | Self-hostable OSS routing gateway |
| OpenRouter | "The managed SaaS" | Hosted gateway with credit-based billing |
| Portkey | "The production option" | Open-source + managed with guardrails built in |

## Bacaan Lanjutan

- [LiteLLM — docs](https://docs.litellm.ai/) — routing gateway self-hosted
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — routing SaaS terkelola
- [Portkey — docs](https://portkey.ai/docs) — routing production dengan guardrails
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — panduan keputusan
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — survei vendor
