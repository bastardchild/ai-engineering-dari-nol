# Gateway AI — LiteLLM, Portkey, Kong AI Gateway, Bifrost

> Gateway berada di antara aplikasimu dan provider model. Fitur inti adalah routing provider, fallback, retry, rate limiting, referensi secret, observabilitas, guardrail. Pembagian pasar pada 2026: **LiteLLM** adalah OSS MIT dengan 100+ provider, kompatibel-OpenAI, tetapi mogok sekitar ~2000 RPS (memori 8 GB, kegagalan berjenjang dalam benchmark yang dipublikasikan); terbaik untuk Python, <500 RPS, dev/prototyping. **Portkey** berposisi control-plane (guardrail, redaksi PII, deteksi jailbreak, jejak audit), menjadi open-source Apache 2.0 Maret 2026, overhead latensi 20-40 ms, tier produksi $49/bulan. **Kong AI Gateway** dibangun di atas Kong Gateway — benchmark Kong sendiri pada 12 CPU yang sama: 228% lebih cepat daripada Portkey, 859% lebih cepat daripada LiteLLM; harga $100/model/bulan (maks 5 pada tier Plus); cocok-enterprise kalau kamu sudah memakai Kong. **Bifrost** (Maxim AI) — retry otomatis dengan backoff terkonfigurasi, fallback ke Anthropic saat OpenAI 429. **Cloudflare / Vercel AI Gateway** — terkelola, zero-ops, retry dasar. Residensi data mendorong keputusan self-host; Portkey dan Kong berada di tengah dengan OSS + managed opsional.

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Enumerasikan enam fitur inti gateway (routing, fallback, retry, rate limit, secret, observabilitas, guardrail).
- Petakan empat gateway 2026 (LiteLLM, Portkey, Kong AI, Bifrost) ke plafon skala dan kasus penggunaan.
- Kutip benchmark Kong (228% vs Portkey, 859% vs LiteLLM) dan jelaskan mengapa ia penting untuk >500 RPS.
- Pilih self-hosted vs managed dengan residensi data dan budget ops tertentu.

## Masalahnya

Produkmu memanggil OpenAI, Anthropic, dan Llama self-hosted. Tiap provider punya SDK, model error, rate limit, dan skema auth berbeda. Kamu menginginkan failover (kalau OpenAI 429, coba Anthropic), satu penyimpanan kredensial, observabilitas terpadu, dan rate limit per tenant.

Menciptakan ulang ini pada lapisan aplikasi menggandeng setiap layanan ke setiap provider. Lapisan gateway mengonsolidasikannya menjadi satu proses dengan satu API (biasanya kompatibel-OpenAI) yang bercabang ke provider.

## Konsepnya

### Enam fitur inti

1. **Routing provider** — OpenAI, Anthropic, Gemini, self-hosted, dsb. di balik satu API.
2. **Fallback** — saat 429, 5xx, atau kegagalan kualitas, ulangi di tempat lain.
3. **Retry** — backoff eksponensial, percobaan terbatas.
4. **Rate limit** — per-tenant, per-key, per-model.
5. **Referensi secret** — tarik kredensial dari vault saat runtime (tidak pernah di aplikasi).
6. **Observabilitas** — OTel + atribut GenAI (Fase 17 · 13) + atribusi biaya.
7. **Guardrail** — redaksi PII, deteksi jailbreak, filter topik-diizinkan.

### LiteLLM — OSS MIT, Python

- 100+ provider, kompatibel-OpenAI, konfigurasi router, fallback, observabilitas dasar.
- Mogok sekitar 2000 RPS dalam benchmark Kong; footprint memori 8 GB, kegagalan berjenjang pada load berkelanjutan.
- Paling cocok: aplikasi Python, <500 RPS, gateway dev/staging, routing eksperimental.
- Biaya: $0 untuk OSS; tier gratis cloud ada.

### Portkey — posisi control plane

- OSS Apache 2.0 per Maret 2026. Guardrail, redaksi PII, deteksi jailbreak, jejak audit.
- Overhead latensi 20-40 ms per-request.
- $49/bulan untuk tier produksi dengan retensi + SLA.
- Paling cocok: industri teregulasi yang butuh guardrail + observabilitas terbundel.

### Kong AI Gateway — permainan skala

- Dibangun di atas Kong Gateway (produk API gateway matang, lua+OpenResty).
- Benchmark Kong sendiri pada setara 12-CPU: 228% lebih cepat daripada Portkey, 859% lebih cepat daripada LiteLLM.
- Harga: $100/model/bulan, maks 5 pada tier Plus.
- Paling cocok: sudah memakai Kong; >1000 RPS; bersedia melisensi.

### Bifrost (Maxim AI)

- Retry otomatis dengan backoff terkonfigurasi.
- Fallback ke Anthropic saat OpenAI 429 adalah resep kanonis.
- Pendatang baru; komersial.

### Cloudflare AI Gateway / Vercel AI Gateway

- Terkelola, zero-ops. Retry dan observabilitas dasar.
- Paling cocok: aplikasi JavaScript yang di-serve di Edge pada Cloudflare/Vercel.
- Terbatas dibanding Kong/Portkey pada guardrail dan rate limit.

### Self-hosted vs managed

Residensi data adalah fungsi pemaksa. Healthcare dan finansial default self-host (LiteLLM atau Portkey OSS atau Kong). Produk konsumen default managed (Cloudflare AI Gateway) atau tier-tengah (Portkey managed). Hibrida: self-hosted untuk tenant teregulasi, managed untuk lainnya.

### Budget latensi

- LiteLLM: overhead tipikal 5-15 ms.
- Portkey: overhead 20-40 ms.
- Kong: overhead 3-8 ms.
- Cloudflare/Vercel: overhead 1-3 ms (keunggulan edge).

Latensi gateway langsung menambah TTFT. Untuk SLA TTFT P99 < 100 ms, Kong atau Cloudflare. Untuk P99 < 500 ms, apa pun boleh.

### Semantik rate-limit penting

Token-bucket sederhana bekerja hingga skala sedang. Multi-tenant butuh sliding-window + kelonggaran burst + tiering per-tenant. LiteLLM membawa token-bucket; Kong membawa sliding-window; Portkey membawa tiered.

### Gateway + observabilitas + routing tersusun

Fase 17 · 13 (observabilitas) + 16 (routing model) + 19 (gateway) adalah lapisan yang sama dalam produksi. Pilih satu tool yang mencakup ketiganya atau rangkai dengan hati-hati: kebanyakan deployment 2026 menggabung Helicone (observabilitas) atau Portkey (guardrail) dengan Kong (skala) untuk peran terpisah.

### Angka yang perlu kamu ingat

- LiteLLM: mogok pada ~2000 RPS, memori 8 GB.
- Portkey: overhead 20-40 ms; Apache 2.0 sejak Maret 2026.
- Kong: 228% lebih cepat daripada Portkey, 859% lebih cepat daripada LiteLLM.
- Harga Kong: $100/model/bulan, maks 5 pada tier Plus.
- Cloudflare/Vercel: overhead 1-3 ms di edge.

```figure
mx-gateway-fallback
```

## Gunakan

`code/main.py` mensimulasikan routing gateway dengan fallback lintas 3 provider pada injeksi 429/5xx. Melaporkan latensi, retry rate, dan hit rate fallback.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-gateway-picker.md`. Dengan skala, posture ops, compliance, budget latensi, ia memilih gateway.

## Latihan

1. Jalankan `code/main.py`. Konfigurasikan fallback dari OpenAI→Anthropic→self-hosted. Berapa hit rate ekspektasi pada error rate provider 5%?
2. SLA-mu TTFT P99 < 200 ms pada baseline 300 ms. Gateway mana yang tetap dalam budget?
3. Pelanggan healthcare mewajibkan self-hosted + redaksi PII + audit. Pilih Portkey OSS atau Kong.
4. Bandingkan LiteLLM vs Kong: pada plafon RPS berapa tim harus bermigrasi?
5. Rancang kebijakan rate-limit untuk SaaS multi-tenant: tier gratis, tier trial, tier berbayar. Token-bucket atau sliding-window?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Gateway | "API broker" | Process sitting between apps and providers |
| LiteLLM | "the MIT one" | Python OSS, 100+ providers, breaks at 2K RPS |
| Portkey | "guardrails gateway" | Control plane + observability, Apache 2.0 |
| Kong AI Gateway | "the scale one" | Built on Kong Gateway, benchmark leader |
| Bifrost | "Maxim's gateway" | Retries + Anthropic fallback recipe |
| Cloudflare AI Gateway | "edge managed" | Edge-deployed managed gateway, zero-ops |
| PII redaction | "data scrub" | Regex + NER mask before sending to model |
| Jailbreak detection | "prompt injection guard" | Classifier on user input |
| Audit trail | "regulated log" | Immutable record of every LLM call |
| Token-bucket | "simple rate limit" | Refill-based rate limiter |
| Sliding-window | "precise rate limit" | Time-windowed rate limiter; better fairness |

## Further Reading

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
