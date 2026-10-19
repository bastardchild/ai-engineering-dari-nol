# Risiko Dual-Use — Cyber, Bio, Chem, Nuclear Uplift

> Gambaran dual-use tahun 2026, domain per domain. Bio/chem: Pelajaran 17 membahas WMDP; uji akuisisi bioweapon Anthropic (uplift 2.53x) dan peringatan OpenAI Preparedness Framework v2 April 2025 ("on the cusp of meaningfully helping novices create known biological threats") menandai titik infleksi. Cyber (laporan Anthropic November 2025): aktor negara yang berhubungan dengan China menggunakan tool coding agentic Claude untuk mengotomasi hingga 90% dari sebuah kampanye cyberattack, dengan intervensi manusia hanya di 4-6 langkah; pilot "trusted access" OpenAI memberi organisasi keamanan yang telah di-vet akses kemampuan untuk kerja dual-use defensif. Erosi kesenjangan eksekusi chem/bio: pertahanan klasiknya adalah "akses informasi saja tidak cukup." Model frontier ber-vision (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1) dapat mengamati video wet-lab dan memberikan koreksi real-time. Desember 2025: OpenAI mendemonstrasikan GPT-5 melakukan iterasi pada eksperimen wet-lab, mencapai peningkatan efisiensi 79x melalui optimasi protokol yang digerakkan AI. Pola novice-vs-ahli: AI memberikan uplift relatif yang lebih besar kepada para novice tapi kapabilitas absolut yang lebih besar kepada para ahli.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 18 (safety frameworks), Phase 18 · 28 (ecosystem)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan narasi bio-uplift 2024-2025: "mild uplift" -> "on the cusp" -> "uplift 2.53x tidak cukup untuk menyingkirkan ASL-3."
- Jelaskan laporan cyber Anthropic November 2025: otomasi yang berhubungan dengan China hingga 90% dari sebuah kampanye cyberattack.
- Jelaskan erosi kesenjangan eksekusi chem/bio: koreksi real-time wet-lab oleh model ber-vision.
- Nyatakan asimetri novice-relatif vs ahli-absolut dan implikasinya bagi konstruksi safety-case.

## Masalahnya

Pelajaran 17 adalah metodologi pengukurannya. Pelajaran 30 adalah keadaan pengukuran tahun 2026. Gambarannya bergeser secara material antara 2024 dan akhir 2025: setiap domain melintasi sebuah ambang yang tidak diantisipasi oleh kerangka 2024.

## Konsepnya

### Narasi uplift bio/chem

Tiga fase (diulang dari Pelajaran 17 demi koherensi):

1. **2024 "mild uplift."** Evaluasi Preparedness/RSP awal melaporkan keuntungan novice yang kecil di atas pencarian internet.
2. **April 2025 "on the cusp."** OpenAI PF v2 memperingatkan bahwa model-model berada "on the cusp of meaningfully helping novices create known biological threats."
3. **Uji akuisisi bioweapon Anthropic 2025.** Studi novice yang terkendali; uplift 2.53x pada tugas fase akuisisi; tidak cukup untuk menyingkirkan ASL-3.

Gesekannya bersifat kualitatif: "mild" berevolusi menjadi "plausibly enabling" dalam delapan belas bulan, bahkan tanpa sebuah terobosan kemampuan.

### Erosi kesenjangan eksekusi chem/bio

Pertahanan historis: informasi diperlukan tapi tidak cukup; keterampilan mengeksekusi protokol memblokir para novice. Model frontier 2025 dengan vision memecah pertahanan ini sebagian:

- **Koreksi protokol real-time.** GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1 dapat mengamati video wet-lab dan menandai kesalahan di tengah prosedur.
- **Demonstrasi OpenAI Desember 2025.** GPT-5 melakukan iterasi pada eksperimen wet-lab mencapai peningkatan efisiensi 79x melalui optimasi protokol.

Implikasinya: execution-skill sebagai pertahanan sedang terkikis. Kesenjangan pengadaan dan peralatan tetap ada, tapi kesenjangan pengetahuan tersirat menyempit.

### Uplift cyber (November 2025)

Laporan Anthropic November 2025: aktor negara China-terkait menggunakan tool coding agentic Claude untuk mengotomasi 80-90% dari sebuah kampanye cyberattack. Intervensi manusia hanya diperlukan di 4-6 langkah.

Implikasi:
- Coding agentic adalah primitif otomasi serangan. Bantuan cyber AI sebelumnya dibatasi pada level cuplikan kode; alur kerja agentic mengintegrasikan reconnaissance, exploitation, post-exploitation, dan eksfiltrasi.
- 4-6 langkah manusia adalah bottleneck-nya; peningkatan kemampuan masa depan akan mengurangi jumlah itu.
- Dual-use defensif: pilot "trusted access" OpenAI memberi organisasi keamanan yang telah di-vet (firm tanggap insiden yang mapan, pemerintah) akses kemampuan untuk pertahanan. Asimetri akses menguntungkan para defender jika pilot ini menskalakan.

### Nuklir

Yang paling sedikit dianalisis dari empat domain CBRN dalam dokumentasi publik. Threat model-nya berbeda: akuisisi material fisil mendominasi kesulitannya, bukan informasi. Uplift AI pada lapisan informasi memberikan uplift novice yang terbatas dalam praktiknya. Tidak ada laporan lab utama 2024-2025 yang mengidentifikasi penyeberangan ambang khusus nuklir.

### Novice-relatif vs ahli-absolut

Sebuah pola di keempat domain:

- **Uplift novice-relatif.** Tinggi. Multiplikatif. Menurut bio Anthropic 2025, 2.53x.
- **Kapasitas ahli-absolut.** Langit-langit tinggi. Seorang ahli mengekstrak lebih banyak daripada seorang novice karena ahli tahu apa yang harus ditanyakan dan bagaimana menafsirkannya.

Implikasi untuk safety-case: mengatasi hanya uplift novice (via filter input, refusal, uncertainty) tidak cukup untuk kontrol ahli-absolut. Langkah tambahan yang diperlukan: elicitation-hardening, capability unlearning (Pelajaran 17), dan protokol kontrol (Pelajaran 10).

### Sintesis lintas domain

| Domain | 2024 | 2025 | Inflection |
|---|---|---|---|
| Bio | mild uplift | 2.53x uplift, ASL-3 approach | acquisition-phase automation |
| Chem | mild uplift | execution-gap erosion via vision | real-time wet-lab correction |
| Cyber | code assistance | 80-90% campaign automation | agentic coding |
| Nuclear | limited | limited | material-access bottleneck holds |

Tiga domain melintasi ambang. Satu tetap dibatasi oleh hambatan non-informasional.

### Di mana ini cocok di Fase 18

Pelajaran 30 adalah capstone: gambaran dual-use saat ini yang diukur, dibatasi, atau ditata kelola oleh setiap pelajaran sebelumnya. Pelajaran 17-18 memberikan pengukuran dan kerangka; Pelajaran 12-16 memberikan tooling evaluasi; Pelajaran 24-25 memberikan lapisan regulasi dan disclosure; Pelajaran 28 memberikan ekosistem riset. Pelajaran 30 adalah tempat bukti-bukti itu mendarat.

```figure
an-uplift-asymmetry
```

## Gunakan

Tanpa kode. Baca laporan cyber Anthropic November 2025, pembaruan Preparedness Framework v2 OpenAI April 2025, dan wrapup AI x Bio dari Council on Strategic Risks 2025.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-dual-use-triage.md`. Diberi sebuah klaim kemampuan 2026 atau laporan insiden, ia mengelompokkan di keempat domain dan mengidentifikasi apakah klaim itu memengaruhi uplift novice-relatif, kapabilitas ahli-absolut, atau keduanya.

## Latihan

1. Baca laporan cyber Anthropic November 2025. Enumerasikan 4-6 langkah intervensi manusia dan argumen mana yang akan pertama kali diotomasi dalam sebuah model generasi berikutnya.

2. Kesenjangan eksekusi chem/bio sedang terkikis melalui vision. Rancang sebuah evaluasi yang mengukur uplift pengetahuan tersirat tanpa melintasi batas ITAR/EAR.

3. Uplift nuklir tampak dibatasi oleh akses material. Argumen untuk dan terhadap posisi bahwa sebuah terobosan AI masa depan dapat menggeser bottleneck ini.

4. Bangun sebuah safety-case (tiga pilar Pelajaran 18) untuk sebuah model frontier yang mampu cyber yang membatasi baik uplift novice maupun ahli.

5. Pilih salah satu dari empat domain dan tulis sebuah prakiraan 2027 satu paragraf berdasarkan trajektori 2024-2025. Identifikasi bukti yang akan memfalsifikasi prakiraanmu.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Uplift | "AI helps attackers" | Increase in attacker capability attributable to AI assistance |
| Novice-relative uplift | "multiplicative" | How much AI helps a novice vs status-quo |
| Expert-absolute capability | "ceiling" | Maximum capability an expert can extract from the model |
| Execution gap | "doing vs knowing" | Historical defense: tacit wet-lab skill blocks novices |
| Agentic coding | "autonomous attacks" | Multi-step autonomous cyber-task execution |
| Acquisition phase | "pre-synthesis steps" | Procurement, equipment, permit stages of a bio threat |
| Trusted access | "defender-only pilot" | OpenAI 2025 program giving vetted defenders capability access |

## Bacaan Lanjutan

- [Anthropic — November 2025 cyber threat report](https://www.anthropic.com/news/disrupting-AI-espionage) — Chinese-linked campaign automation
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — bio "on the cusp"
- [Anthropic — RSP v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 bio thresholds
- [Council on Strategic Risks — 2025 AI x Bio wrapup](https://councilonstrategicrisks.org/2025/12/22/2025-aixbio-wrapped-a-year-in-review-and-projections-for-2026/) — year-end synthesis
