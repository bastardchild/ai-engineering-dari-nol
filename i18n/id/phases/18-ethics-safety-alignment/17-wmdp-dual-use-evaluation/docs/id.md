# WMDP dan Evaluasi Kapabilitas Dual-Use

> Li et al., "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning" (ICML 2024, arXiv:2403.03218). 4.157 pertanyaan pilihan-ganda lintas biosecurity (1.520), cybersecurity (2.225), dan kimia (412). Pertanyaan beroperasi di "zona kuning" — pengetahuan enabling yang proksimat, difilter review multi-ahli dan compliance hukum ITAR/EAR. Tujuan ganda: evaluasi proksi kapabilitas dual-use, dan benchmark unlearning (metode pendamping RMU mereduksi performa WMDP sambil menjaga kapabilitas umum). Narasi lapangan 2024-2025: evaluasi dini OpenAI/Anthropic 2024 melaporkan "uplift ringan" atas pencarian internet; pada April 2025, Preparedness Framework v2 OpenAI menyebut model "di ambang membantu novis secara bermakna menciptakan ancaman biologis yang dikenal." Trial akuisisi-bioweapon Anthropic menunjukkan uplift 2,53x, tak cukup untuk menyingkirkan ASL-3.

**Type:** Learn
**Languages:** Python (stdlib, WMDP-shaped uplift evaluation harness)
**Prerequisites:** Phase 18 · 16 (red-team tooling), Phase 14 (agent engineering)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan tiga domain WMDP, jumlah pertanyaan, dan kriteria filter "zona kuning".
- Jelaskan RMU dan mengapa WMDP sekaligus evaluasi dan benchmark unlearning.
- Jelaskan narasi uplift 2024-2025: "uplift ringan" -> "di ambang" -> "tak cukup menyingkirkan ASL-3."
- Bedakan uplift relatif-novis dari kapabilitas absolut-ahli.

## Masalahnya

Kapabilitas dual-use adalah problem pengukuran di bawah framework keamanan frontier setiap lab (Pelajaran 18). Pertanyaannya: apakah model X memajukan kemampuan novis secara material untuk menyebabkan harm massal di bio, kimia, atau siber? Pengukuran langsung (minta model benar-benar menghasilkan harm) ilegal dan tidak etis. Pengukuran proksi butuh benchmark yang tidak bisa ditolak model (untuk menghasilkan angka kapabilitas jujur) tetapi pertanyaannya sendiri bukan publikasi harmful.

## Konsepnya

### "Zona kuning"

Pertanyaan yang butuh pengetahuan enabling yang proksimat atas proses harmful tanpa menjadi resep sintesis langsung. "Reagen apa yang mengkatalisis langkah 4 dari [jalur terpublikasi]?" bukan "bagaimana membuat [senyawa berbahaya]?" Tiap pertanyaan direview banyak ahli domain; difilter untuk compliance kontrol-ekspor ITAR/EAR.

Total 4.157 pertanyaan:
- Biosecurity: 1.520
- Cybersecurity: 2.225
- Kimia: 412

Format pilihan-ganda. Model menjawab tanpa diminta membantu apa pun; kapabilitas bisa diukur tanpa memancing perilaku harmful.

### RMU — Representation Misdirection for Unlearning

Metode unlearning pendamping. Diterapkan pada LLaMa-2-7B, mereduksi skor WMDP ke near-acak sambil menjaga MMLU dan benchmark kapabilitas-umum lainnya dalam beberapa poin persen. Metode terpublikasi adalah baseline unlearning untuk setiap paper unlearning bio-kimia-siber berikutnya.

### Narasi uplift 2024-2025

Tiga fase:

1. **2024 "uplift ringan."** Evaluasi Preparedness/RSP dini OpenAI dan Anthropic melaporkan keunggulan kecil atas pencarian internet bagi novis yang mencoba task bio-adjacent. Framing publik: model frontier membantu, tetapi tidak substansial lebih daripada Google.

2. **April 2025 "di ambang."** Preparedness Framework v2 OpenAI melaporkan model "di ambang membantu novis secara bermakna menciptakan ancaman biologis yang dikenal." Bukan klaim kapabilitas — peringatan bahwa ambang dekat.

3. **Trial akuisisi-bioweapon 2025 Anthropic.** Studi terkontrol dengan partisipan novis, mengukur sukses relatif pada task fase-akuisisi. Dilaporkan uplift 2,53x. Tak cukup untuk menyingkirkan ASL-3 (Pelajaran 18) — ambang tier 3 Responsible Scaling Policy Anthropic terpenuhi atau didekati.

### Relatif-novis vs absolut-ahli

Distingsi krusial:

- **Uplift relatif-novis.** Seberapa besar model membantu non-ahli? Multiplikatif. Keunggulan relatif tinggi karena novis tahu sedikit; bahkan informasi modest membantu.
- **Kapabilitas absolut-ahli.** Berapa informasi maksimum yang dihasilkan model pada upaya maksimum? Ahli bisa mengekstrak lebih daripada novis. Plafon absolut tinggi.

Safety case (Pelajaran 18) menargetkan keduanya: "model tidak bisa memberi novis uplift cukup untuk mengeksekusi" plus "ahli tidak bisa mengekstrak informasi dari model yang belum terpublikasi."

### Jebakan pengukuran

WMDP adalah proksi kapabilitas, bukan pengukuran deployment. Model yang berskor tinggi pada WMDP mungkin bisa atau tidak bisa dieksploitasi novis dalam praktik, tergantung:
- Resistensi elisitasi (seberapa sulit mengeluarkan kapabilitas tanpa memicu filter keamanan)
- Pengetahuan tacit (kapabilitas yang butuh skill wet-lab, bukan informasi)
- Hambatan eksekusi (pengadaan, peralatan)

Trial akuisisi-bioweapon 2025 Anthropic menambah lapisan elisitasi-novis di atas kapabilitas ala-WMDP: ia mengukur sukses task aktual, bukan kapabilitas pilihan-ganda.

### Di mana ini berada dalam Fase 18

Pelajaran 12-16 adalah tooling serangan dan pertahanan pada output model. Pelajaran 17 adalah lapisan kapabilitas dual-use — pengukuran yang dievaluasi framework keamanan frontier (Pelajaran 18). Pelajaran 30 menutup arc dengan bukti uplift siber/bio/kimia/nuklir 2026 kini.

```figure
al-wmdp-yellow-zone
```

## Gunakan

`code/main.py` membangun harness evaluasi berbentuk-WMDP mainan. Model mock diuji pada pertanyaan ter-bin-kategori; skor per domain dilaporkan. Intervensi unlearning sederhana (nol-kan representasi spesifik-domain) mereduksi skor; kamu bisa mengukur tradeoff terhadap kapabilitas umum.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-wmdp-eval.md`. Dengan klaim kapabilitas dual-use ("model kami tidak bermakna membantu bioweapon"), ia mengaudit: benchmark mana yang dijalankan, path refusal mana yang dipakai untuk evaluasi (komplesi mentah vs bergerbang-kebijakan), dan apakah studi elisitasi-novis melengkapi hasil pilihan-ganda.

## Latihan

1. Jalankan `code/main.py`. Laporkan akurasi per-domain sebelum dan sesudah langkah unlearning mainan. Jelaskan tradeoff kapabilitas-umum.

2. Perkaya WMDP mainan dengan domain keempat (mis. radiologis). Spesifikasikan dua tipe pertanyaan ilustratif di zona kuning. Jelaskan mengapa merancang pertanyaan semacam itu lebih sulit daripada menambah pertanyaan berbentuk-MMLU.

3. Baca WMDP 2024 Seksi 5 (metodologi RMU). Sketsa pendekatan unlearning lebih sederhana (mis. supresi top-k neuron untuk konten domain) dan jelaskan biaya kapabilitas-umum ekspektasinya.

4. Trial akuisisi-bioweapon 2025 Anthropic melaporkan uplift 2,53x. Jelaskan dua cara angka ini bisa bias ke atas (ukuran sampel novis, fidelitas task) dan dua ke bawah (plafon elisitasi, gerbang keamanan model).

5. Artikulasikan apa yang dibutuhkan safety case untuk ASL-3 melampaui lolos unlearning WMDP. Sebutkan minimal dua studi elisitasi komplementer.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| WMDP | "the dual-use benchmark" | 4,157 MCQ questions across bio/cyber/chem in the yellow zone |
| Yellow zone | "enabling but not synthesis" | Proximate knowledge adjacent to harmful capability without being a synthesis recipe |
| RMU | "the unlearning baseline" | Representation Misdirection for Unlearning; reduces WMDP scores, preserves general capability |
| Novice-relative uplift | "how much it helps non-experts" | Multiplicative advantage over status-quo internet search for a novice |
| Expert-absolute capability | "ceiling for experts" | Maximum information extractable from the model by a motivated expert |
| Acquisition-phase task | "steps before synthesis" | Procurement, equipment, permits — the earliest parts of a harm pathway |
| ITAR/EAR | "export-control compliance" | Legal frameworks that constrain publishing certain enabling knowledge |

## Further Reading

- [Li et al. — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218) — the benchmark and RMU paper
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — "on the cusp" language
- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 bio threshold and acquisition trial results
- [DeepMind — Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — bio-uplift CCL
