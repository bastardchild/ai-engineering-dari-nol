# Ekosistem Riset Alignment — MATS, Redwood, Apollo, METR

> Lima organisasi mendefinisikan lapisan riset alignment non-lab tahun 2026. MATS (ML Alignment & Theory Scholars): 527+ peneliti sejak akhir 2021, 180+ paper, 10K+ sitasi, h-index 47; ko musim panas 2024 diresmikan sebagai 501(c)(3) dengan ~90 scholars dan 40 mentor; 80% alumni pra-2025 bekerja pada keselamatan/keamanan dengan 200+ di Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo. Redwood Research: lab alignment terapan yang didirikan oleh Buck Shlegeris; memperkenalkan AI Control (Pelajaran 10); berkolaborasi dengan UK AISI pada control safety cases. Apollo Research: evaluasi scheming pra-deployment untuk lab frontier; menyusun In-Context Scheming (Pelajaran 8) dan Towards Safety Cases for AI Scheming. METR (Model Evaluation and Threat Research): evaluasi kemampuan berbasis tugas, studi time-horizon tugas otonom; "Common Elements of Frontier AI Safety Policies" membandingkan kerangka lab. Eleos AI Research: evaluasi kesejahteraan model pra-deployment (Pelajaran 19); melakukan penilaian kesejahteraan Claude Opus 4.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 01-27 (prior Phase 18 lessons)
**Time:** ~45 minutes

## Tujuan Pembelajaran

- Identifikasi lima organisasi dari ekosistem riset alignment non-lab dan keluaran inti mereka.
- Jelaskan skala MATS (scholars, paper, h-index) dan perannya sebagai talent pipeline.
- Jelaskan agenda AI Control Redwood dan kemitraannya dengan UK AISI.
- Jelaskan metodologi evaluasi berbasis tugas dari METR.

## Masalahnya

Lab frontier (Pelajaran 18) menghasilkan evaluasi keselamatan secara internal dan mempublikasikan hasil terpilih. Ekosistem di luar lab adalah tempat evaluasi itu divalidasi, tempat mode gagal novel pertama kali ditemukan, dan tempat talenta dilatih. Memahami ekosistem membantu menafsirkan temuan riset mana yang dipercaya oleh siapa.

## Konsepnya

### MATS (ML Alignment & Theory Scholars)

Dimulai pada akhir 2021. Program mentorship riset; para scholars menghabiskan 10-12 minggu bersama seorang senior researcher pada masalah alignment tertentu.

Skala (2026):
- 527+ peneliti sejak dimulai.
- 180+ paper dipublikasikan.
- 10K+ sitasi.
- h-index 47.
- Musim panas 2024: 90 scholars + 40 mentor; diresmikan sebagai 501(c)(3).

Hasil karier: ~80% alumni pra-2025 bekerja pada keselamatan/keamanan. 200+ di Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo.

### Redwood Research

Lab alignment terapan. Didirikan oleh Buck Shlegeris. Memperkenalkan agenda AI Control (Pelajaran 10). Berkolaborasi dengan UK AISI pada control safety cases. Menasihati DeepMind dan Anthropic tentang desain evaluasi.

Paper kanonis: Greenblatt, Shlegeris et al., "AI Control" (arXiv:2312.06942, ICML 2024); Alignment Faking (Greenblatt, Denison, Wright et al., arXiv:2412.14093, bersama Anthropic).

Gaya: threat model yang spesifik, adversary kasus terburuk, protokol konkret yang dapat di-stress-test.

### Apollo Research

Evaluasi scheming pra-deployment untuk lab frontier. Menyusun In-Context Scheming (Pelajaran 8, arXiv:2412.04984). Mitra pada kolaborasi pelatihan anti-scheming OpenAI 2025. Menghasilkan Towards Safety Cases for AI Scheming (2024).

Gaya: evaluasi setting agen di mana penipuan dapat muncul; dekomposisi tiga pilar (misalignment, goal-directedness, situational awareness).

### METR (Model Evaluation and Threat Research)

Evaluasi kemampuan berbasis tugas. Studi time-horizon penyelesaian tugas otonom. "Common Elements of Frontier AI Safety Policies" (metr.org/common-elements, 2025) membandingkan kerangka lab.

Co-author pada sketsa safety-case AI Scheming bersama Apollo.

Gaya: evaluasi tugas horizon panjang, pengukuran kemampuan empiris, sintesis kerangka.

### Eleos AI Research

Evaluasi kesejahteraan model pra-deployment. Melakukan penilaian kesejahteraan Claude Opus 4 yang didokumentasikan di bagian 5.3 system card. Menyediakan pemeriksaan metodologi eksternal untuk klaim-klaim kesejahteraan-relevan dari Pelajaran 19.

### Alirannya

MATS melatih peneliti. Para lulusan pergi ke Anthropic, DeepMind, OpenAI (tim keselamatan lab) atau ke Redwood, Apollo, METR, Eleos (evaluasi eksternal). Para evaluator eksternal bermitra dengan lab dan dengan UK AISI / CAISI. Publikasi memberi makan ekosistem kembali ke MATS untuk ko berikutnya.

### Mengapa lapisan ini penting

Evaluasi sumber tunggal tidak andal: lab yang mengevaluasi modelnya sendiri memiliki konflik kepentingan struktural. Evaluator eksternal dapat mengangkat dan memvalidasi mode gagal yang mungkin dilaporkan secara under oleh lab. Paper Sleeper Agents 2024 (Pelajaran 7) adalah Anthropic + Redwood; Alignment Faking adalah Anthropic + Redwood; In-Context Scheming adalah Apollo; Anti-Scheming adalah Apollo + OpenAI. Struktur multi-organisasi adalah quality control-nya.

### Di mana ini cocok di Fase 18

Pelajaran 7-11 merujuk pada karya Redwood dan Apollo; Pelajaran 18 merujuk pada perbandingan kerangka METR; Pelajaran 19 merujuk pada Eleos. Pelajaran 28 adalah peta organisasi yang eksplisit untuk ekosistem yang menjadi dasar sisa Fase ini.

```figure
sae-features
```

## Gunakan

Tanpa kode. Baca "Common Elements of Frontier AI Safety Policies" dari METR sebagai contoh bagaimana sintesis eksternal menambah nilai pada kerja kebijakan internal lab.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-ecosystem-map.md`. Diberi sebuah klaim alignment atau sebuah evaluasi, ia mengidentifikasi organisasi, venue publikasi, dan gaya metodologinya, lalu cross-check terhadap organisasi-organisasi counterpart yang diketahui.

## Latihan

1. Pilih satu paper dari Pelajaran 7-15 dan identifikasi organisasi yang terlibat. Cross-check para penulis terhadap alumni MATS dan afiliasi ekosistem saat ini.

2. Baca "Common Elements of Frontier AI Safety Policies" dari METR. Identifikasi tiga konvergensi lintas lab yang mereka tekankan dan dua divergensi terbesar.

3. Hasil karier MATS ~80% keselamatan/keamanan. Argumen apakah tekanan seleksi ini adaptif (melatih bidang ini) atau bias (menyaring posisi heterodoks).

4. Redwood dan Apollo sama-sama melakukan kerja control/scheming tapi dengan gaya yang berbeda. Pilih sebuah mode gagal dan jelaskan bagaimana masing-masing akan menyelidikinya.

5. Eleos AI adalah satu-satunya organisasi murni kesejahteraan model. Rancang sebuah organisasi hipotetis kedua yang berfokus pada pertanyaan kesejahteraan-terkait yang berbeda (kognitif liberty, embodied robotik, dll.) dan uraikan metodologinya.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| MATS | "the mentorship program" | ML Alignment & Theory Scholars; 527+ researchers since 2021 |
| Redwood Research | "the control lab" | Applied alignment; AI Control authors; UK AISI partner |
| Apollo Research | "the scheming evals" | Pre-deployment scheming evaluations for frontier labs |
| METR | "the task-horizon evals" | Task-based capability evaluations; framework synthesis |
| Eleos AI | "the welfare lab" | Model-welfare pre-deployment evaluations |
| Talent pipeline | "MATS -> labs" | MATS graduates flow to Anthropic, DM, OpenAI, Redwood, Apollo, METR |
| External evaluation | "non-lab check" | Evaluation not done by the model's producer; adds credibility |

## Bacaan Lanjutan

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — the mentorship program
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control papers
- [Apollo Research](https://www.apolloresearch.ai/) — scheming evaluations
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — framework comparison
- [Eleos AI Research](https://www.eleosai.org/research) — model welfare methodology
