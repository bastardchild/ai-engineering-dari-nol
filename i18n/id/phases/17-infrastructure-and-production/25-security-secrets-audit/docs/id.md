# Keamanan — Secret, Rotasi API Key, Log Audit, Guardrail

> Hilangkan secret sprawl via vault terpusat (HashiCorp Vault, AWS Secrets Manager, Azure Key Vault). Jangan pernah menyimpan kredensial di file config, file env dalam VCS, spreadsheet. Pakai peran IAM alih-alih key statis; OIDC untuk CI/CD. Pola AI-gateway adalah solusi 2026: aplikasi → gateway → provider model, dengan gateway menarik kredensial dari vault saat runtime. Rotasi di vault dan semua aplikasi mengambil dalam hitungan menit — tanpa redeploy, tanpa pesan Slack "siapa punya key baru". Kebijakan rotasi ≤90 hari; pindai dengan TruffleHog / GitGuardian / Gitleaks pada setiap commit. Zero-trust: MFA, SSO, RBAC/ABAC, token berumur-pendek, posture perangkat. Scrubbing PII memakai pengenalan entitas untuk menyamarkan PHI/PII sebelum meneruskan; tokenisasi konsisten (pendekatan Mesh) memetakan nilai sensitif ke placeholder stabil agar LLM menjaga semantik kode/relasi. Egress jaringan: layanan LLM di subnet VPC/VNet khusus yang hanya memutihkan `api.openai.com`, `api.anthropic.com` dsb; blokir semua outbound lain. Pendorong insiden 2026: serangan supply-chain Vercel lewat kredensial CI/CD yang dikompromikan mengeksfiltrasi env var lintas ribuan deployment pelanggan.

**Type:** Learn
**Languages:** Python (stdlib, toy PII-scrubber + audit-log writer)
**Prerequisites:** Phase 17 · 19 (AI Gateways), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Enumerasikan empat antipola manajemen-secret (file config dalam VCS, env hardcoded, spreadsheet, key statis) dan sebutkan penggantinya.
- Jelaskan pola AI-gateway-menarik-dari-vault sebagai standar produksi 2026.
- Implementasikan scrubber PII dengan tokenisasi konsisten (nilai sama → placeholder sama) agar semantik bertahan.
- Sebutkan insiden supply-chain Vercel 2026 dan apa yang diajarkannya tentang higienitas kredensial CI/CD.

## Masalahnya

Seorang intern meng-commit `.env` berisi API key. Mereka menghapusnya cepat. Key sudah ada dalam histori git — pemindaian GitGuardian menangkapnya, proses rotasimu adalah "Slack tim, update 40 file config, redeploy semua layanan." 8 jam kemudian, separuh layananmu live dan separuhnya menunggu window deploy.

Terpisah, prompt pengguna mencakup "My SSN is 123-45-6789." Prompt terkirim ke OpenAI. Kamu punya BAA tetapi kebijakan internalmu adalah menyamarkan PII sebelum meneruskan. Kamu tidak melakukannya.

Terpisah, pod LLM klaster EKS-mu bisa mencapai host internet mana pun. Seseorang mengekstfiltrasi data via lookup DNS ke domain yang dikontrol penyerang. Tak ada yang memblokirnya.

Keamanan layanan LLM harus menangani ketiga vektor. Kredensial berbasis-vault. Scrubbing PII. Filtering egress jaringan. Log audit.

## Konsepnya

### Vault terpusat + tarikan peran-IAM

**Vault**: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, GCP Secret Manager. Satu sumber kebenaran.

**Peran IAM**: aplikasi/gateway mengautentikasi via identitas IAM-nya, bukan key statis. Vault mengembalikan secret seumur token.

**Pola AI-gateway**: gateway menarik `OPENAI_API_KEY` dari vault pada waktu-request. Rotasi di vault; request berikutnya mendapat key baru. Tanpa redeploy.

### Kebijakan rotasi ≤ 90 hari

Semua API key, token root vault, kredensial CI/CD. Rotasi otomatis bila mungkin. Rotasi manual dicatat dan dilacak.

### Pemindaian secret

- **TruffleHog** — regex + entropi pada commit.
- **GitGuardian** — komersial, akurasi tinggi.
- **Gitleaks** — OSS, berjalan di CI.

Jalankan pada setiap commit. Blokir PR kalau secret baru terdeteksi.

### Posture zero-trust

- MFA wajib pada semua akun.
- SSO via SAML/OIDC.
- RBAC (berbasis-peran) atau ABAC (berbasis-atribut) untuk akses berbutir-halus.
- Token berumur-pendek (jam, bukan hari).
- Posture perangkat — hanya perangkat korporat dengan enkripsi disk.

### Scrubbing PII / PHI

Sebelum prompt meninggalkan inframu:

1. Pengenalan entitas (spaCy NER, Presidio, komersial).
2. Samarkan entitas yang cocok: `"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`.
3. Tokenisasi konsisten (pendekatan Mesh): nilai sama dipetakan ke placeholder sama agar LLM menjaga relasi.
4. Pemetaan balik opsional untuk respons LLM.

Filter regex statis menangkap pola dasar; NER menangkap lebih banyak. Pakai keduanya.

### Guardrail input + output

Input: blokir jailbreak yang dikenal, topik terlarang; rate-limit per-pengguna.

Output: scrub regex untuk secret bocor (pola API key, pola email dalam konteks refusal), klasifier untuk pelanggaran kebijakan.

### Whitelist egress jaringan

Layanan LLM di subnet khusus:
- Whitelist: `api.openai.com`, `api.anthropic.com`, endpoint vector DB, endpoint vault.
- Sisanya: drop.
- DNS via resolver khusus-allowlist (hindari eksfiltrasi DNS-tunneling).

### Log audit

Log immutable setiap panggilan LLM berisi:
- Timestamp.
- Pengguna / tenant.
- Hash prompt (bukan prompt mentah demi privasi).
- Model + versi.
- Jumlah token.
- Biaya.
- Hash respons.
- Trip guardrail apa pun.

Retensi sesuai syarat regulasi (SOC 2 1 tahun, HIPAA 6 tahun).

### Insiden Vercel 2026

Serangan supply-chain: kredensial CI/CD yang dikompromikan mengeksfiltrasi env var lintas ribuan deployment pelanggan. Pelajaran: kredensial CI/CD setara-prod. Simpan di vault. Scope sempit. Rotasi agresif.

### Angka yang perlu kamu ingat

- Kebijakan rotasi: ≤ 90 hari.
- Pindai pada setiap commit: TruffleHog / GitGuardian / Gitleaks.
- Vercel 2026: kredensial CI/CD dikompromikan → ribuan env var pelanggan bocor.
- Retensi log audit: SOC 2 = 1 tahun, HIPAA = 6 tahun.

```figure
i4-vault-rotation
```

## Gunakan

`code/main.py` mengimplementasikan scrubber PII mainan dengan tokenisasi konsisten dan log audit append-only.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-llm-security-plan.md`. Dengan scope regulasi dan state kini, ia merencanakan migrasi vault, scrubber, egress, log audit.

## Latihan

1. Jalankan `code/main.py`. Kirim dua prompt yang merujuk SSN sama. Konfirmasi keduanya mendapat placeholder sama.
2. Rancang kebijakan egress jaringan untuk deployment vLLM-pada-EKS yang memanggil OpenAI + Anthropic + Weaviate.
3. Kamu menemukan key dalam histori git (berumur 2 tahun). Apa respons tepat — rotasi key, scrub histori, atau keduanya? Justifikasi.
4. Log audit-mu tumbuh 10 GB/hari. Rancang tier retensi (panas 30 hari, hangat 12 bulan, dingin 6 tahun).
5. Argumenkan apakah tokenisasi-balik (mensubstitusi nilai nyata kembali ke respons LLM) layak dengan kompleksitasnya dibanding membiarkan placeholder terlihat.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Vault | "secrets store" | Centralized credential management service |
| IAM role | "identity-based auth" | Role assumed by app; returns short-lived creds |
| OIDC for CI/CD | "cloud-issued tokens" | No static keys in CI — identity via OIDC |
| TruffleHog / GitGuardian / Gitleaks | "secret scanners" | Commit-time secret detection |
| RBAC / ABAC | "access control" | Role-based vs attribute-based |
| PII scrubbing | "data masking" | Remove or tokenize sensitive entities |
| Consistent tokenization | "stable placeholders" | Same value → same token each time |
| Mesh approach | "Mesh tokenization" | Semantic-preserving tokenization pattern |
| Egress whitelist | "outbound allowlist" | Only permitted domains reachable |
| Audit log | "immutable history" | Append-only record for compliance |

## Further Reading

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII detection and anonymization.
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
