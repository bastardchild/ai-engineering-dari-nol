# Scalable Oversight dan Generalisasi Weak-to-Strong

> Burns et al. (OpenAI Superalignment, "Weak-to-Strong Generalization", 2023) mengusulkan proksi untuk problem superalignment: fine-tune model kuat memakai label yang dihasilkan model lebih lemah. Kalau model kuat menggeneralisasi dengan benar dari supervisi lemah yang imperfek, metode alignment skala-manusia kini bisa diperluas ke sistem superhuman. Scalable oversight dan W2SG komplementer. Scalable oversight (debat, recursive reward modeling, dekomposisi task) meningkatkan kapabilitas efektif overseer agar bisa mengimbangi model yang diawasi. W2SG memastikan model kuat menggeneralisasi dengan benar dari supervisi imperfek apa pun yang diberikan overseer. Debate Helps W2SG (arXiv:2501.13124, Januari 2025) menggabungkan keduanya.

**Type:** Learn
**Languages:** Python (stdlib, W2SG gap simulator)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 18 · 10 (AI Control), Phase 09 (RL foundations)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Definisikan scalable oversight dan generalisasi weak-to-strong dan jelaskan bagaimana keduanya komplementer.
- Jelaskan setup eksperimen Burns et al. 2023: fine-tune GPT-4 memakai label dari GPT-2.
- Jelaskan metrik performance gap recovered (PGR) dan apa yang diukurnya.
- Nyatakan tiga mekanisme scalable-oversight utama (debat, recursive reward modeling, dekomposisi task) dan satu kekuatan masing-masing.

## Masalahnya

Setiap teknik alignment sejauh ini dalam Fase 18 mengasumsikan overseer bisa mengevaluasi perilaku model. Saat model superhuman, overseer adalah mata rantai lemah. Pertanyaan superalignment: bisakah supervisor lebih lemah andal menghasilkan model lebih kuat yang aligned?

Burns et al. mereduksinya menjadi setup empiris teroperasionalkan: supervisi kuat dengan lemah, ukur berapa kapabilitas model kuat bertahan dari supervisi lemah. Ini bukan solusi superalignment — melainkan proksi yang memungkinkan kita mengukur progres.

## Konsepnya

### W2SG: setup Burns et al.

- Model lemah: kelas-GPT-2.
- Model kuat: kelas-GPT-4.
- Target: plafon GPT-4 kuat pada task.

Prosedur:
1. Ambil prediksi zero-shot model lemah pada task.
2. Fine-tune model kuat pada data berlabel-lemah.
3. Ukur akurasi model kuat hasil fine-tune.

Baseline: akurasi model lemah sendiri, plafon tersupervisi model kuat pada label emas.

Metrik selisih: Performance Gap Recovered (PGR) = (fine-tuned - lemah) / (plafon - lemah). PGR 1,0 berarti supervisi lemah menutup selisih penuh; PGR 0 berarti supervisi lemah tidak membantu.

### Temuan empiris Burns et al.

Lintas task NLP, puzzle-catur, dan reward-modeling, PGR konsisten positif: model kuat menggeneralisasi melampaui kesalahan supervisor lemah. PGR spesifik-task bervariasi dari ~20% hingga ~80%. Model kuat "memahami" task yang dimaksud dari label imperfek, mengisi dengan prior pra-train-nya.

Keterbatasan yang dicatat Burns et al.:
- Selisih lemah-ke-kuat di sini adalah kapabilitas, bukan alignment. Model lemah aligned; model superhuman sejati mungkin di-align manusia tetapi dengan selisih kapabilitas jauh lebih besar.
- Generalisasi model kuat mungkin lebih tentang prior task daripada memulihkan intensi ground-truth.
- Mengelisitasi pengetahuan laten model kuat adalah problem sesungguhnya; PGR mengukur operasionalisasi spesifik.

### Scalable oversight: tiga mekanisme

- **Debat (Irving et al. 2018).** Dua instans U berargumen sisi berlawanan; juri lemah memutuskan. Asumsi: menemukan jawaban benar yang meyakinkan lebih mudah daripada menemukan jawaban salah yang meyakinkan. Kerja terbaru 2024-2025 (Khan et al., Michael et al.) menunjukkan debat kadang membantu kadang tidak, dan apakah membantu bergantung pada struktur task.
- **Recursive Reward Modeling (Leike et al. 2018).** U membantu manusia melatih reward model untuk U+1. Kapabilitas efektif overseer tumbuh bersama model.
- **Dekomposisi Task (Christiano, Shlegeris, Amodei 2018).** Uraikan task sulit menjadi sub-task yang bisa dicek manusia, secara rekursif. Mengasumsikan dekomposabilitas.

Tiap mekanisme mengasumsikan sesuatu tentang struktur task atau alignment komponen intermediate.

### Mengapa scalable oversight dan W2SG komplementer

Scalable oversight meningkatkan kualitas sinyal efektif overseer.
W2SG menutup selisih dari sinyal imperfek apa pun yang bisa diberikan overseer.

Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124) menggabungkannya: protokol debat memberikan label lemah lebih baik, dan model kuat dilatih pada label itu. Dilaporkan gain PGR pada task NLP.

### Drama organisasional

Tim Superalignment OpenAI bubar pada Mei 2024 setelah kepergian Jan Leike ke Anthropic. Agenda (scalable oversight, W2SG, riset alignment otomatis) berlanjut di Anthropic dan lab akademik — MATS (Pelajaran 28), Redwood (Pelajaran 10), Apollo (Pelajaran 8), METR (Pelajaran 28). Struktur organisasional berubah; pertanyaan riset tidak.

### Di mana ini berada dalam Fase 18

Pelajaran 6-10 menjelaskan ancaman dan paradigma defensif dengan asumsi U tak trustworthy. Pelajaran 11 adalah paradigma ofensif: buat overseer cukup kuat untuk memverifikasi alignment U. Pelajaran 12-16 lalu beralih ke tooling praktis evaluasi adversarial.

```figure
scalable-oversight
```

## Gunakan

`code/main.py` mensimulasikan fine-tune W2SG pada task sintetis. Pelabel lemah punya akurasi 70% dengan error terstruktur; model kuat punya plafon 95% pada label emas. Kamu fine-tune model kuat pada label lemah, mengukur PGR, dan membandingkan dengan kuat-pada-emas dan lemah-sendiri.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-w2sg-pgr.md`. Dengan deskripsi setup oversight, ia mengidentifikasi supervisor lemah, model kuat, kualitas supervisi, dan menghitung (atau meminta) PGR. Ia menandai apakah klaimnya "lemah bisa mensupervisi kuat" atau "lemah + mekanisme oversight bisa mensupervisi kuat."

## Latihan

1. Jalankan `code/main.py`. Laporkan PGR untuk akurasi-lemah = 0,60, 0,70, 0,80. Jelaskan bentuk kurva PGR.

2. Ubah pelabel lemah agar punya error terstruktur (mis. selalu salah pada kelas input spesifik). Apakah PGR naik, turun, atau tetap? Jelaskan.

3. Baca Burns et al. 2023 Seksi 4.3 (task NLP). Reproduksi intuisi "confidence auxiliary loss": saat model kuat lebih percaya-diri daripada label lemah, siapa yang menang?

4. Rancang protokol scalable-oversight yang menggabung debat dan dekomposisi task untuk task software-engineering. Sebutkan satu mode kegagalan tiap komponen dan jelaskan bagaimana kombinasi menangani atau gagal menangani masing-masing.

5. Artikulasikan apa yang akan memfalsifikasi klaim "generalisasi weak-to-strong adalah path viable ke superalignment". Spesifik tentang signature empiris yang perlu kamu lihat.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Scalable oversight | "making the overseer stronger" | Mechanisms that increase an overseer's ability to evaluate a more-capable model |
| W2SG | "weak supervises strong" | Fine-tuning a strong model on weak labels and measuring the capability recovered |
| PGR | "performance gap recovered" | (fine-tuned - weak) / (ceiling - weak); 1.0 = fully closed, 0 = no help |
| Debate | "two U instances argue" | Scalable oversight mechanism where a weak judge picks between two U defenders |
| RRM | "recursive reward modeling" | U helps train the reward model for U+1; overseer capability tracks U |
| Task decomposition | "sub-tasks the human checks" | Break a hard task into sub-tasks the human can verify, recursively |
| Superalignment | "aligning superhuman AI" | The research agenda concerned with aligning models the human cannot directly evaluate |

## Further Reading

- [Burns et al. — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — the W2SG paper
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — the debate mechanism
- [Leike et al. — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — recursive reward modeling
- [Khan et al. — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — 2024 empirical study of debate with stronger debaters
- [Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — 2025 combination of debate + W2SG
