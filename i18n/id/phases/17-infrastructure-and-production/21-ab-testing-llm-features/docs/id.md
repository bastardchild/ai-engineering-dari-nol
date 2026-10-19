# Pengujian A/B Fitur LLM — GrowthBook, Statsig, dan Masalah Firasat

> Pengujian A/B tradisional tidak dibangun untuk LLM non-deterministik. Distingsi kritis: eval menjawab "bisakah model mengerjakan tugas?" Uji A/B menjawab "apakah pengguna peduli?" Keduanya diwajibkan; merilis berdasarkan vibe check sudah selesai. Apa yang diuji pada 2026: prompt engineering (wording), seleksi model (GPT-4 vs GPT-3.5 vs OSS; akurasi vs biaya vs latensi), parameter generasi (temperature, top-p). Kasus nyata: varian reward-model chatbot menghasilkan +70% panjang percakapan dan +30% retensi; eksperimen subject-line AI Nextdoor menghasilkan +1% CTR setelah refinement fungsi-reward; Khan Academy Khanmigo beriterasi pada sumbu latensi-vs-akurasi-matematika. Pembagian platform: **Statsig** (diakuisisi OpenAI seharga $1,1B pada September 2025) — sequential testing, CUPED, all-in-one. **GrowthBook** — open-source, warehouse-native, engine Bayesian + Frequentist + Sequential, CUPED, cek SRM, koreksi Benjamini-Hochberg + Bonferroni. Kamu memilih berdasarkan preferensi warehouse-SQL dan apakah "diakuisisi OpenAI" penting bagi organisasimu.

**Type:** Learn
**Languages:** Python (stdlib, toy sequential test simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 20 (Progressive Deployment)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Bedakan eval ("bisakah model mengerjakan tugas") dari uji A/B ("apakah pengguna peduli").
- Enumerasikan tiga sumbu yang bisa diuji (prompt, model, parameter) dan pilih metrik untuk masing-masing.
- Jelaskan koreksi CUPED, sequential testing, dan Benjamini-Hochberg untuk perbandingan-ganda.
- Pilih Statsig atau GrowthBook berdasarkan posture warehouse-SQL dan sikap akuisisi korporat.

## Masalahnya

Kamu men-tune system prompt dengan tangan. Rasanya lebih baik. Kamu merilisnya. Konversi berubah sebesar noise. Kamu menyalahkan metrik. Atau kamu merilis model baru dan konversi tidak bergerak — apakah model terdegradasi atau perubahannya terlalu kecil untuk terdeteksi? Kamu tidak tahu, karena kamu merilis tanpa A/B.

Eval menjawab apakah model bisa mengerjakan task pada himpunan berlabel. Mereka tidak menjawab apakah pengguna menyukai output. Hanya eksperimen online terkontrol yang menjawab itu, dan hanya jika eksperimen punya power cukup, mengontrol non-determinisme, dan mengoreksi perbandingan ganda.

## Konsepnya

### Eval vs uji A/B

**Eval** — offline, himpunan berlabel, judge (rubrik atau LLM-as-judge atau manusia). Jawaban: "Apakah output benar / membantu / aman pada distribusi tetap ini?"

**Uji A/B** — online, pengguna live, teracak. Jawaban: "Apakah varian baru menggerakkan metrik level-pengguna yang penting?"

Keduanya diwajibkan. Eval menangkap regresi sebelum eksposur; A/B mengonfirmasi dampak produk sesudahnya.

### Apa yang diuji

1. **Prompt engineering** — wording, struktur system-prompt, contoh. Metrik: keberhasilan task, retensi pengguna, biaya/request.
2. **Seleksi model** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS. Metrik: akurasi (task) + biaya/request + latensi P99. Multi-objektif.
3. **Parameter generasi** — temperature, top-p, max_tokens. Metrik: spesifik-task (diversitas output vs determinisme).

### CUPED — reduksi varians

Controlled-experiments Using Pre-Experiment Data. Regresikan varians pra-periode sebelum membandingkan pasca-periode. Reduksi varians tipikal: 30-70%. Ukuran sampel efektif naik gratis.

Implementasi: Statsig maupun GrowthBook mengimplementasikan.

### Sequential testing

A/B klasik mengasumsikan ukuran sampel tetap. Uji sekuensial ("peek-and-decide") mengontrol false-positive rate pada pengamatan berulang. Prosedur sekuensial always-valid (mSPRT, confidence sequence Howard) memungkinkanmu berhenti dini pada pemenang yang jelas.

### Koreksi perbandingan-ganda

Menjalankan 20 uji A/B pada confidence 95% menghasilkan satu false positive secara kebetulan. Koreksi Bonferroni mengetatkan α per-test; Benjamini-Hochberg mengontrol false-discovery rate. GrowthBook mengimplementasikan keduanya.

### SRM — ketidakcocokan rasio sampel

Hash assignment mengacak pengguna ke varian. Kalau split 50/50 menghasilkan 47/53, sesuatu rusak — cek SRM menandainya. Kedua platform mengimplementasikan.

### Statsig vs GrowthBook

**Statsig**:
- Diakuisisi OpenAI seharga $1,1B (September 2025). Hosted, SaaS.
- Sequential testing, CUPED, populasi held-out.
- All-in-one: feature flag + eksperimen + observabilitas.
- Paling cocok: tim yang sudah menginginkan produk terbundel, tidak peduli kepemilikan OpenAI.

**GrowthBook**:
- Open-source (MIT); warehouse-native (membaca langsung dari Snowflake/BigQuery/Redshift).
- Banyak engine: Bayesian, Frequentist, Sequential.
- CUPED, SRM, Bonferroni, koreksi BH.
- Self-host atau cloud managed.
- Paling cocok: shop warehouse-SQL, tim data mengontrol lapisan metrik, menginginkan OSS.

### Non-determinisme mempersulit power

Prompt sama menghasilkan output bervariasi. Perhitungan power tradisional mengasumsikan observasi IID. Dengan non-determinisme LLM, ukuran sampel efektif lebih rendah daripada nominal. Kalikan ukuran sampel yang dibutuhkan dengan ~1,3-1,5x sebagai margin aman.

### Hasil kasus nyata

- Varian reward model chatbot: +70% panjang percakapan, +30% retensi.
- Subject line Nextdoor: +1% CTR setelah refinement fungsi-reward.
- Khan Academy Khanmigo: tradeoff latensi-vs-akurasi-matematika iteratif.

### Antipola: merilis berdasarkan firasat

Setiap engineer senior bisa menyebut fitur yang dirilis karena "terasa lebih baik" tanpa A/B. Kebanyakan meregresi metrik produk yang tidak disadari tim selama berbulan-bulan. A/B adalah fungsi pemaksanya.

### Angka yang perlu kamu ingat

- Statsig diakuisisi OpenAI: $1,1B, September 2025.
- GrowthBook: open-source MIT; Bayesian + Frequentist + Sequential.
- Reduksi varians CUPED: 30-70%.
- Non-determinisme LLM → buffer ukuran-sampel +30-50%.

```figure
mx-sequential-test
```

## Gunakan

`code/main.py` mensimulasikan uji A/B sekuensial dengan batas tetap dan sekuensial. Menunjukkan bagaimana sekuensial memungkinkanmu berhenti dini.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-ab-plan.md`. Dengan perubahan fitur, workload, baseline, ia memilih platform, gerbang, ukuran sampel.

## Latihan

1. Jalankan `code/main.py`. Untuk lift ekspektasi 5% dengan baseline konversi 3%, berapa ukuran sampel hingga power 80%?
2. Pilih Statsig atau GrowthBook untuk pelanggan on-prem teregulasi-healthcare.
3. Rancang A/B yang menguji GPT-4 vs GPT-3.5 pada biaya-per-tiket-terselesaikan. Apa metrik primer, metrik guardrail, sekunder?
4. Canary-mu lolos tetapi A/B menunjukkan konversi -1,2%. Apakah kamu merilis? Tulis kriteria eskalasi.
5. Terapkan CUPED pada pra-periode dengan 60% varians pasca. Hitung boost ukuran-sampel-efektif.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Eval | "offline test" | Labeled-set evaluation of model capability |
| A/B test | "experiment" | Live randomized comparison on users |
| CUPED | "variance reduction" | Pre-period regression to reduce variance |
| Sequential test | "peek-ok test" | Always-valid procedure allowing early stop |
| Multiple comparison | "the family error" | Running many tests inflates false positives |
| Bonferroni | "tight correction" | Divide α by number of tests |
| Benjamini-Hochberg | "BH FDR" | False-discovery-rate control, less conservative |
| SRM | "bad split" | Sample ratio mismatch; assignment bug |
| Statsig | "OpenAI owned" | Commercial all-in-one, acquired 2025 |
| GrowthBook | "the OSS one" | MIT warehouse-native platform |
| mSPRT | "sequential probability ratio test" | Classical sequential procedure |

## Further Reading

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
