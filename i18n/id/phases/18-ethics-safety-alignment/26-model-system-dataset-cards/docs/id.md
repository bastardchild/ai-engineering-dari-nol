# Kartu Model, Sistem, dan Dataset

> Tiga format dokumentasi membingkai transparansi AI. Model Cards (Mitchell et al. 2019) — label gizi untuk model: data training, analisis terdisagregasi kuantitatif, pertimbangan etis, caveat; hanya 0.3% kartu model di Hugging Face yang mendokumentasikan pertimbangan etis (Oreamuno et al. 2023). Datasheets for Datasets (Gebru et al. 2018, CACM) — motivasi, komposisi, proses pengumpulan, pelabelan, distribusi, pemeliharaan; analogi datasheet elektronik. Data Cards (Pushkarna et al., Google 2022) — detail berlapis modular (telescopic, periscopic, microscopic) sebagai boundary object bagi pembaca yang beragam. Perkembangan 2024-2025: generasi otomatis via LLM (CardGen, Liu et al. 2024); detail kartu model berkorelasi dengan kenaikan unduhan hingga 29% di HF (Liang et al. 2024); attestations yang dapat diverifikasi (Laminator, Duddu et al. 2024); penambahan pelaporan keberlanjutan untuk karbon/air (Jouneaux et al. Juli 2025); kartu regulasi EU/ISO yang sedang muncul. System Cards (Sidhpurwala 2024; transparansi tingkat sistem Meta; "Blueprints of Trust" arXiv:2509.20394) — dokumentasi sistem AI end-to-end yang mencakup kemampuan keamanan, proteksi prompt-injection, deteksi eksfiltrasi data, dan alignment dengan nilai manusia.

**Type:** Build
**Languages:** Python (stdlib, model-card + datasheet + system-card generator)
**Prerequisites:** Phase 18 · 18 (safety frameworks), Phase 18 · 24 (regulatory)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan model card asli Mitchell et al. 2019 dan datasheet Gebru et al. 2018.
- Jelaskan layering telescopic/periscopic/microscopic dari Data Cards.
- Jelaskan System Cards dan cakupan end-to-end mereka.
- Nyatakan tiga perkembangan 2024-2025 (generasi otomatis, attestations yang dapat diverifikasi, pelaporan keberlanjutan).

## Masalahnya

Kerangka regulasi (Pelajaran 24) dan kebijakan keselamatan lab (Pelajaran 18) sama-sama memerlukan dokumentasi. Format dokumentasi berevolusi dari khusus-model (model cards) ke khusus-dataset (datasheets) ke khusus-sistem (system cards). Masing-masing mengatasi cakupan transparansi yang berbeda. Pekerjaan otomasi dan verifiable-attestation 2024-2025 mengatasi masalah adopsi yang sudah lama ada.

## Konsepnya

### Model Cards (Mitchell et al. 2019)

Bagian-bagian:
- Detail model.
- Penggunaan yang dimaksudkan.
- Faktor (faktor demografis atau lingkungan yang relevan untuk evaluasi).
- Metrik.
- Data evaluasi.
- Data training.
- Analisis kuantitatif (terdisagregasi berdasarkan faktor).
- Pertimbangan etis.
- Caveat dan rekomendasi.

Masalah adopsi: audit Oreamuno et al. 2023 atas kartu model Hugging Face menemukan hanya 0.3% yang mendokumentasikan pertimbangan etis.

### Datasheets for Datasets (Gebru et al. 2018)

Analogi datasheet elektronik. Bagian-bagian:
- Motivasi (mengapa dataset dibuat).
- Komposisi (apa isinya).
- Proses pengumpulan (bagaimana ia dirakit).
- Pelabelan (jika berlaku).
- Penggunaan (yang dimaksudkan, yang dilarang, risiko).
- Distribusi.
- Pemeliharaan.

Dipublikasikan di CACM 2021. Datasheet adalah dokumentasi hulu; kartu model bergantung pada akurasi datasheet.

### Data Cards (Pushkarna et al., Google 2022)

Detail berlapis modular. Tiga level zoom:
- **Telescopic.** Ringkasan tingkat tinggi untuk non-ahli.
- **Periscopic.** Gambaran tingkat menengah untuk praktisi ML.
- **Microscopic.** Dokumentasi terperinci level fitur untuk auditor.

Framing boundary-object: pembaca yang berbeda mengekstrak informasi yang berbeda dari dokumen yang sama.

### System Cards

Cakupan: sistem AI end-to-end termasuk model + stack keselamatan + konteks deployment. Bagian-bagian biasanya mencakup:
- Kemampuan keamanan.
- Proteksi prompt-injection.
- Deteksi eksfiltrasi data.
- Alignment dengan nilai manusia yang dinyatakan.
- Tanggapan insiden.

Pekerjaan Sidhpurwala 2024 dan pekerjaan transparansi tingkat sistem Meta. "Blueprints of Trust" (arXiv:2509.20394) memformalkan System Card sebagai pelengkap lapisan deployment dari Model Cards.

### Perkembangan 2024-2025

- **CardGen (Liu et al. 2024).** Generasi model card otomatis via LLM; melaporkan objektivitas lebih tinggi daripada banyak kartu yang ditulis manusia pada field standar Mitchell 2019.
- **Korelasi unduhan (Liang et al. 2024).** Kartu model yang terperinci berkorelasi dengan tingkat unduhan hingga 29% lebih tinggi di HF — tekanan adopsi kini digerakkan oleh pasar, bukan hanya oleh kepatuhan.
- **Laminator (Duddu et al. 2024).** Attestations yang dapat diverifikasi via hardware TEE / signature kriptografis — memungkinkan kartu model membawa sebuah proof-of-claim, bukan hanya sebuah klaim.
- **Keberlanjutan (Jouneaux et al. Juli 2025).** Penambahan untuk jejak karbon, air, dan energi komputasi; standar ISO yang sedang muncul.
- **Kartu regulasi.** EU AI Act (Pelajaran 24) GPAI Code of Practice bab Transparency mewajibkan model cards sebagai sebuah artefak kepatuhan.

### Di mana ini cocok di Fase 18

Pelajaran 24-25 adalah lapisan regulasi dan CVE. Pelajaran 26 adalah lapisan dokumentasi. Pelajaran 27 adalah tata kelola data training, yang merupakan hulu dari datasheet. Pelajaran 28 adalah ekosistem riset yang menghasilkan evaluasi yang dirujuk dalam kartu.

```figure
an-card-scopes
```

## Gunakan

`code/main.py` menghasilkan sebuah model card, datasheet, dan system card minimal untuk sebuah deployment mainan. Masing-masing mengikuti struktur bagian yang kanonis. Kamu dapat memeriksa formatnya dan membandingkan ketiga cakupan.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-card-audit.md`. Diberi sebuah model card, datasheet, atau system card, ia mengaudit cakupan bagian, disagregasi numerik, dan apakah attestations yang dapat diverifikasi hadir.

## Latihan

1. Jalankan `code/main.py`. Periksa kartu yang dihasilkan. Identifikasi bagian yang lemah (hanya placeholder) dan tentukan bukti apa yang akan memperkuatnya.

2. Perluas model card dengan sebuah analisis terdisagregasi kuantitatif di dua kelompok demografis (Pelajaran 20).

3. Baca Oreamuno et al. 2023 tentang tingkat adopsi 0.3%. Usulkan satu perubahan struktural pada spesifikasi model card yang akan meningkatkan adopsi pertimbangan etis.

4. Laminator (Duddu et al. 2024) menggunakan TEE untuk attestations yang dapat diverifikasi. Rancang sebuah field model card yang membawa sebuah attestation kriptografis atas hasil evaluasi dan jelaskan peran verifiernya.

5. Tulis sebuah System Card (System Card, bukan Model Card) untuk salah satu proyek lampau atau deployment hipotetismu. Identifikasi bagian dengan nilai tertinggi bagi auditor pihak ketiga.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Model Card | "the Mitchell card" | Mitchell et al. 2019 standard documentation for ML models |
| Datasheet | "the Gebru datasheet" | Gebru et al. 2018 standard documentation for datasets |
| Data Card | "the Pushkarna card" | Google 2022 modular layered data documentation |
| System Card | "the deployment card" | End-to-end AI system documentation including safety stack |
| Boundary object | "different readers, one doc" | Data Cards framing: same document serves diverse audiences |
| Verifiable attestation | "the Laminator attestation" | Cryptographic or TEE proof attached to a documentation claim |
| Sustainability field | "carbon / water footprint" | Emerging 2025 addition for environmental accounting |

## Bacaan Lanjutan

- [Mitchell et al. — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — the canonical model card
- [Gebru et al. — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — datasheet paper
- [Pushkarna et al. — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — layered data documentation
- [Sidhpurwala et al. — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — System Card formalization
