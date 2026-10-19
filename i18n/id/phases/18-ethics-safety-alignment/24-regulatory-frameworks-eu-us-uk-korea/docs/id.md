# Kerangka Regulasi — EU, US, UK, Korea

> Empat rezim regulasi utama mendefinisikan lanskap tata kelola AI tahun 2026. EU AI Act (berlaku 1 Agustus 2024) — praktik terlarang dan AI literacy sejak 2 Februari 2025; kewajiban GPAI sejak 2 Agustus 2025; penerapan penuh dan transparansi Article 50 2 Agustus 2026; GPAI legacy dan sistem high-risk tertanam 2 Agustus 2027; denda hingga 15M EUR atau 3% dari turnover global. GPAI Code of Practice (10 Juli 2025): tiga bab — Transparency, Copyright, Safety and Security — 12 komitmen; penegakan dimulai Agustus 2026. UK AISI -> AI Security Institute (Februari 2025): pergantian nama menandai cakupan yang lebih sempit. US AISI -> CAISI (Juni 2025): Center for AI Standards and Innovation di bawah NIST; pergeseran ke postur pro-pertumbuhan. Korean AI Framework Act (disahkan Desember 2024, berlaku Januari 2026): Article 12 mendirikan AISI di bawah MSIT; mewajibkan perwakilan lokal bagi perusahaan AI asing, risk assessment, dan langkah keselamatan untuk AI berdampak-tinggi dan generatif.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 18 (frontier frameworks), Phase 18 · 27 (data governance)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan tier risiko EU AI Act (terlarang, high-risk, general-purpose, limited-risk) dan timeline Agustus 2025 / Agustus 2026 / Agustus 2027.
- Jelaskan tiga bab GPAI Code of Practice dan penyedia mana yang terikat oleh masing-masing.
- Jelaskan rebrand 2025: UK AISI -> AI Security Institute; US AISI -> CAISI; apa yang diimplikasikan setiap rebrand tentang arah kebijakan.
- Nyatakan ketentuan inti dari Korean AI Framework Act.

## Masalahnya

Kerangka lab (Pelajaran 18) bersifat sukarela. Kerangka regulasi bersifat wajib. Periode 2024-2026 menyaksikan gelombang pertama regulasi AI komprehensif mulai berlaku. Para deployer harus memetakan kontrol teknis ke kewajiban regulasi; pemetaannya berbeda menurut yurisdiksi.

## Konsepnya

### EU AI Act

**Berlaku 1 Agustus 2024.** Struktur tier risiko:

- **Prohibited practices** (Article 5). Social scoring, identifikasi biometrik jarak-jauh real-time di ruang publik (dengan pengecualian penegakan hukum), manipulasi eksploitatif kelompok rentan. Diterapkan 2 Februari 2025.
- **High-risk systems** (Annex III). Pekerjaan, pendidikan, kredit, penegakan hukum, keadilan, migrasi. Memerlukan conformity assessment, manajemen risiko, logging, transparansi.
- **Model General-Purpose AI (GPAI)**. Diterapkan 2 Agustus 2025. Semua penyedia GPAI memiliki kewajiban; GPAI ber-risiko-sistemik (>1e25 FLOP komputasi training) memiliki kewajiban tambahan.
- **Limited-risk systems.** Kewajiban transparansi di bawah Article 50 (pelabelan konten buatan-AI). Diterapkan 2 Agustus 2026.

Timeline:
- 2 Feb 2025: prohibited practices + AI literacy.
- 2 Agu 2025: GPAI + tata kelola.
- 2 Agu 2026: penerapan penuh + transparansi Article 50 + denda hingga 15M EUR / 3% turnover global.
- 2 Agu 2027: GPAI legacy + high-risk tertanam.

Komisi mengusulkan menyesuaikan timeline high-risk menjadi 16 bulan pada akhir 2025.

### GPAI Code of Practice

Dipublikasikan 10 Juli 2025. Tiga bab:

- **Transparency.** Semua penyedia GPAI.
- **Copyright.** Semua penyedia GPAI.
- **Safety and Security.** Penyedia GPAI ber-risiko-sistemik (diperkirakan 5-15 perusahaan).

12 komitmen total. Sebuah Signatory Taskforce yang dipimpin oleh AI Office mengelola implementasi. Penegakan dimulai 2 Agustus 2026; sampai saat itu, kepatuhan itikad baik diterima.

### Transparency Code untuk Article 50

Draf pertama 17 Desember 2025. Draf kedua Maret 2026. Versi final Juni 2026. Mencakup pelabelan konten buatan-AI termasuk deepfake — lapisan regulasi yang memerlukan teknologi watermarking dari Pelajaran 23.

### UK AI Security Institute (Februari 2025)

Dulu bernama AI Safety Institute. Rebrand menyempitkan cakupan: menjatuhkan bingkai bias algoritmik dan kebebasan berbicara; fokus pada keamanan kemampuan frontier. Me-release alat evaluasi Inspect dengan open-source (Mei 2024). Berkolaborasi dengan Redwood (Pelajaran 10) pada control safety cases.

### US CAISI (Juni 2025)

Administrasi Trump mengubah AI Safety Institute di NIST menjadi Center for AI Standards and Innovation. Pergeseran ke arah "kebijakan AI pro-pertumbuhan" menurut pernyataan VP Vance di Paris AI Action Summit. Penekanan berkurang pada evaluasi pra-deployment; penekanan pada standar dan dukungan inovasi. Penyeimbang domestik terhadap postur regulasi EU AI Act.

### Korean AI Framework Act

Disahkan Desember 2024. Diundangkan Januari 2025. Berlaku Januari 2026. Mengonsolidasikan 19 rancangan undang-undang AI yang terpisah.

Article 12 mendirikan sebuah AISI di bawah Ministry of Science and ICT (MSIT). Mewajibkan:
- Perwakilan lokal untuk perusahaan AI asing yang beroperasi di Korea.
- Risk assessment untuk sistem AI "high-impact".
- Langkah keselamatan untuk AI generatif dan AI berdampak-tinggi.

Yurisdiksi Asia pertama dengan regulasi AI horizontal yang komprehensif.

### Dinamika lintas-yurisdiksi

- EU: ketat, bertier-risiko, denda berat. Benchmark untuk regulasi yang berdekatan dengan privasi.
- US: pro-inovasi, terdesentralisasi, negara bagian (mis., California AB 2013 — Pelajaran 27) mengisi celah federal.
- UK: fokus keamanan yang sempit, infrastruktur evaluasi yang kuat.
- Korea: dipimpin oleh MSIT, berfokus pada penyedia asing.

Filosofi regulasi yang bersaing. Deployer di banyak yurisdiksi harus mematuhi yang paling ketat, yang pada tahun 2026 umumnya adalah EU AI Act.

### Di mana ini cocok di Fase 18

Pelajaran 18 adalah tata kelola sukarela-lab; Pelajaran 24 adalah regulasi; Pelajaran 25 adalah kelas CVE yang sedang muncul untuk sistem AI; Pelajaran 26-27 membahas dokumentasi (kartu) dan tata kelola data training.

```figure
an-eu-act-timeline
```

## Gunakan

Tanpa kode. Baca sumber primer EU AI Act: teks regulasi, GPAI Code of Practice, kerangka Inspect UK AISI. Petakan deployment-mu ke kewajiban yang berlaku untuk setiap yurisdiksi.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-regulatory-map.md`. Diberi deskripsi deployment, ia memetakan yurisdiksi yang berlaku, klasifikasi tier di masing-masing, kewajiban per-yurisdiksi, dan struktur tenggat waktu.

## Latihan

1. Baca EU AI Act (regulasi 2024/1689) dan GPAI Code of Practice (10 Juli 2025). Identifikasi tiga kewajiban yang berlaku untuk setiap penyedia GPAI dan tiga yang hanya berlaku untuk GPAI ber-risiko-sistemik.

2. Sebuah deployment dibuat oleh perusahaan AS, berjalan di infrastruktur EU, dan melayani pengguna Korea. Aturan dari yurisdiksi mana yang berlaku, dan aturan mana yang mengikat pada setiap pertanyaan substantif?

3. Pergantian nama UK AI Security Institute menyempitkan cakupan. Argumen untuk dan terhadap bingkai yang lebih sempit. Identifikasi asumsi kebijakan yang menjadi sandaran setiap posisi.

4. Bingkai "pro-pertumbuhan" CAISI adalah penyimpangan dari model lembaga keselamatan-AI 2022-2024. Identifikasi dua pergeseran kebijakan yang terukur yang akan mengikuti dari bingkai ini.

5. Korean AI Framework Act mewajibkan perwakilan lokal bagi penyedia asing. Jelaskan implikasi operasional bagi sebuah perusahaan Bay Area yang melayani pengguna Korea.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| EU AI Act | "the regulation" | Risk-tier-based horizontal AI regulation; in force Aug 2024 |
| GPAI | "general-purpose AI" | Large foundation models; systemic-risk subset has additional obligations |
| Article 50 | "transparency obligations" | AI-generated content labelling; applies Aug 2026 |
| UK AISI | "AI Security Institute" | Renamed Feb 2025; narrower frontier-security focus |
| CAISI | "US center for AI standards" | Renamed Jun 2025 from AI Safety Institute; pro-growth posture |
| Korean AI Framework Act | "MSIT horizontal regulation" | First Asian comprehensive AI law; effective Jan 2026 |
| Systemic-risk GPAI | "the 1e25 FLOP threshold" | Additional obligations tier; estimated 5-15 companies bound |

## Bacaan Lanjutan

- [EU AI Act text (Regulation 2024/1689)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — the regulation and timeline
- [GPAI Code of Practice (10 July 2025)](https://digital-strategy.ec.europa.eu/en/library/final-version-general-purpose-ai-code-practice) — three-chapter code
- [UK AI Security Institute (renamed Feb 2025)](https://www.gov.uk/government/organisations/ai-security-institute) — official page
- [CSET — South Korea AI Framework Act Analysis (2025)](https://cset.georgetown.edu/publication/south-korea-ai-law-2025/) — Korean framework analysis
