# Framework Keselamatan Frontier — RSP, PF, FSF

> Tiga framework dari lab besar mendefinisikan tata kelola industri 2026 atas kapasitas frontier. Anthropic Responsible Scaling Policy v3.0 (Februari 2026) memperkenalkan AI Safety Levels bertingkat (ASL-1 sampai ASL-5+), dimodelkan pada biosafety levels, dengan ASL-3 diaktifkan Mei 2025 untuk model yang relevan-CBRN. OpenAI Preparedness Framework v2 (April 2025) mendefinisikan lima kriteria untuk kapasitas yang dilacak dan memisahkan Capabilities Reports dari Safeguards Reports. DeepMind Frontier Safety Framework v3.0 (September 2025) memperkenalkan Critical Capability Levels termasuk Harmful Manipulation CCL yang baru. Ketiganya kini menyertakan klausul penyesuaian-pesaing yang memungkinkan penundaan jika lab pesaing merilis tanpa safeguard yang sebanding. Penyelarasan antar-lab tetap bersifat struktural, bukan terminologis: "Capability Thresholds," "High Capability thresholds," dan "Critical Capability Levels" menandai konstruk yang analog.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 07-09 (deception failures)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan struktur tingkat ASL milik Anthropic dan apa yang mengaktifkan ASL-3.
- Sebutkan lima kriteria OpenAI Preparedness Framework v2 untuk kapasitas yang dilacak.
- Jelaskan struktur Critical Capability Level milik DeepMind dan Harmful Manipulation CCL.
- Jelaskan klausul penyesuaian-pesaing dan mengapa itu penting bagi dinamika perlombaan.
- Definisikan safety case dan jelaskan struktur tiga pilar (monitoring, illegibility, incapability).

## Masalahnya

Pelajaran 7-17 menetapkan bahwa penipuan itu mungkin, kapasitas dual-use itu ada, dan evaluasi punya batasnya. Sebuah lab dengan model berkapasitas-frontier butuh struktur tata kelola internal yang:
- Mendefinisikan ambang kapan safeguard baru diperlukan.
- Mendefinisikan evaluasi yang wajib sebelum scaling.
- Mendeskripsikan seperti apa safety case itu.
- Menangani masalah dinamika perlombaan (jika pesaing merilis tanpa safeguard, apa yang kamu lakukan?).

Ketiga framework 2025-2026 adalah state of the art — tidak sempurna, terus berkembang, dan cukup selaras antar-lab sehingga pertanyaan tata kelolanya kini adalah apakah framework-nya memadai, bukan apakah framework itu ada.

## Konsepnya

### Anthropic Responsible Scaling Policy v3.0 (Februari 2026)

Struktur ASL:
- ASL-1: bukan model frontier (diksubsumasi oleh baseline lebih-lemah-dari-frontier).
- ASL-2: baseline frontier saat ini; dideploy dengan safeguard biasa.
- ASL-3: risiko penyalahgunaan bencana jauh lebih tinggi; kapasitas relevan-CBRN. Diaktifkan Mei 2025.
- ASL-4: ambang AI R&D-2; model yang bisa mengotomatiskan riset AI level pemula.
- ASL-5+: AI R&D lanjutan; yang secara drastis mempercepat scaling yang efektif.

Baru di v3.0:
- Frontier Safety Roadmaps (publik dalam bentuk yang disunting).
- Risk Reports (kuartalan, sebagian ditinjau eksternal).
- AI R&D dipecah menjadi AI R&D-2 dan AI R&D-4.
- Setelah AI R&D-4 terlampaui, sebuah safety case afirmatif wajib ada, yang mengidentifikasi risiko misalignment dari model yang mengejar tujuan yang misaligned.

### OpenAI Preparedness Framework v2 (15 April 2025)

Lima kriteria untuk kapasitas yang dilacak:
- **Plausible.** Model ancaman yang masuk akal ada.
- **Measurable.** Evaluasi empiris mungkin.
- **Severe.** Kerugiannya besar.
- **Net-new.** Bukan risiko lama yang diskalakan.
- **Instantaneous-or-irremediable.** Kerugian terjadi cepat atau tidak bisa dibatalkan.

Kapasitas yang memenuhi kelima kriteria itu dilacak. Yang lain tidak.

Struktur PF v2 lainnya:
- Memisahkan Capabilities Reports (apa yang bisa dilakukan model) dari Safeguards Reports (kontrol apa yang ada).
- Tinjauan Safety Advisory Group.
- Pemimpin menyetujui; Safety & Security Committee mengawasi.
- "Adjustment clause": OpenAI boleh mengurangi persyaratan jika lab lain merilis tanpa safeguard yang sebanding.

### DeepMind Frontier Safety Framework v3.0 (September 2025)

Critical Capability Levels (CCL) per domain:
- Bioweapon Uplift
- Cyber Uplift
- ML R&D Acceleration
- Harmful Manipulation (baru di v3.0): model yang bisa mengubah keyakinan/perilaku secara substansial dalam konteks berisiko-tinggi.

v2.0 (Februari 2025) menambahkan bagian Deceptive Alignment dan level keamanan lebih tinggi untuk CCL ML R&D.

### Penyelarasan antar-lab

- Anthropic "Capability Thresholds."
- DeepMind "Critical Capability Levels."
- OpenAI "High Capability thresholds."

Tidak ada terminologi standar industri. Secara struktural selaras: tiga tingkat kapasitas frontier dengan kriteria evaluasi yang dipublikasikan. Ketiganya menyertakan klausul penyesuaian-pesaing (2025 ke atas).

### Safety case

Safety case adalah argumen tertulis bahwa sebuah deployment aman secara dapat-diterima di bawah asumsi kasus-terburuk. Struktur standarnya menyasar tiga pilar:

- **Monitoring.** Bisakah kita mendeteksi perilaku buruk jika itu terjadi?
- **Illegibility.** Apakah model kekurangan kemampuan untuk mengeksekusi rencana yang koheren untuk menyebabkan kerugian?
- **Incapability.** Apakah model kekurangan kapasitas untuk menyebabkan kerugian yang dimaksud?

Safety case berbeda menyasar pilar berbeda. Untuk kasus CBRN ASL-3, incapability (lewat unlearning) adalah sasaran utama. Untuk deceptive alignment, monitoring dan illegibility adalah sasaran. Untuk cyber uplift, ketiganya relevan.

### Masalah dinamika perlombaan

Klausul penyesuaian-pesaing kontroversial. Kritikus berargumen klausul itu menciptakan perlombaan ke bawah: jika ketiga lab akan mengurangi persyaratan saat pesaing mendefeksi, ekuilibrium bergeser ke arah defeksi. Para pembela berargumen alternatifnya (safeguard sepihak) menghasilkan hasil yang lebih buruk jika lab yang mendefeksi kurang peduli pada keselamatan.

UK AISI, US CAISI, dan EU AI Office (Pelajaran 24) adalah mitra tata kelola eksternalnya. Framework lab bersifat sukarela; framework regulasinya sedang muncul.

### Di mana ini cocok di Phase 18

Pelajaran 17-18 adalah lapisan pengukuran-dan-tata-kelola di atas analisis penipuan dan red-team. Pelajaran 19-24 membahas welfare, bias, privasi, watermarking, dan struktur regulasi. Pelajaran 28 memetakan ekosistem riset (MATS, Redwood, Apollo, METR) yang mengoperasionalkan evaluasi-evaluasi itu.

```figure
al-asl-ladder
```

## Gunakan

Tanpa kode untuk pelajaran ini. Baca tiga sumber primer: RSP v3.0, PF v2, FSF v3.0. Petakan struktur tingkat tiap lab ke lab lainnya dan identifikasi satu ambang yang didefinisikan tiap lab tapi tidak dimiliki yang lain.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-framework-diff.md`. Diberi sebuah framework keselamatan atau catatan rilis, file itu membandingkan definisi ambang framework, evaluasi yang disyaratkan, dan struktur safety case terhadap RSP v3.0, PF v2, FSF v3.0 serta menandai kesenjangan antar-lab.

## Latihan

1. Baca RSP v3.0, PF v2, dan FSF v3.0. Susun tabel ambang CBRN tiap lab, ambang AI R&D tiap lab, dan evaluasi pra-deployment yang disyaratkan tiap lab.

2. Klausul penyesuaikan-pesaing ada di ketiga framework (2025+). Tulis satu paragraf yang mendukungnya; tulis satu paragraf yang menentangnya. Identifikasi asumsi yang ditopang tiap posisi.

3. Rancang sebuah safety case untuk model yang melampaui ambang AI R&D-4 milik Anthropic. Sebutkan bukti yang dibutuhkan masing-masing dari tiga pilar (monitoring, illegibility, incapability).

4. FSF v3.0 milik DeepMind memperkenalkan Harmful Manipulation CCL. Usulkan tiga pengukuran empiris yang akan menunjukkan sebuah model telah melampaui ambang ini.

5. Baca "Common Elements of Frontier AI Safety Policies" milik METR (2025). Sebutkan tiga konvergensi antar-lab terkuat dan dua divergensi terbesar.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| RSP | "Anthropic's framework" | Responsible Scaling Policy; ASL tiers; v3.0 February 2026 |
| PF | "OpenAI's framework" | Preparedness Framework; five criteria; v2 April 2025 |
| FSF | "DeepMind's framework" | Frontier Safety Framework; CCLs; v3.0 September 2025 |
| ASL-3 | "biosafety level 3-analog" | Anthropic tier for CBRN-relevant capabilities; activated May 2025 |
| CCL | "critical capability level" | DeepMind's threshold construct; per-domain |
| Safety case | "the formal argument" | Written argument that deployment is acceptably safe under worst-case U |
| Adjustment clause | "competitor defection allowance" | Framework provision for reducing requirements if competitors ship without comparable safeguards |

## Bacaan Lanjutan

- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL tiers, roadmaps, AI R&D disaggregation
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — five criteria, adjustment clause
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0, Harmful Manipulation
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — cross-lab comparison
