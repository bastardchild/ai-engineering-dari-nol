# STaR, V-STaR, Quiet-STaR — Penalaran yang Mengajari Dirinya Sendiri

> Loop self-improvement terkecil berada di dalam rationale. Sebuah model menghasilkan rantai berpikir, menyimpan yang mendarat pada jawaban benar, dan fine-tuning pada hasil itu. Itulah STaR. V-STaR menambahkan verifier sehingga seleksi pada waktu inference lebih baik. Quiet-STaR mendorong rationale ke setiap token. Ketiganya berhasil. Tidak ada yang ajaib — loop ini melestarikan setiap jalan pintas yang kebetulan mencapai jawaban benar.

**Type:** Learn
**Languages:** Python (stdlib, bootstrap-loop simulator)
**Prerequisites:** Phase 13 · 01-03 (Reasoning and CoT), Phase 15 · 01 (long-horizon framing)
**Time:** ~60 minutes

## Masalahnya

Cara paling langsung mengajari model bernalar adalah mengumpulkan jejak penalaran yang ditulis manusia. Itu mahal, lambat, dan dibatasi oleh seberapa banyak chain-of-thought berkualitas tinggi yang bersedia ditulis manusia.

STaR (Self-Taught Reasoner, Zelikman et al., 2022) bertanya: bagaimana jika model menulis rationale-nya sendiri dan menilainya terhadap jawaban yang diketahui? Loop-nya adalah:

1. Sampel jejak penalaran plus jawaban.
2. Jika jawaban akhirnya benar, simpan jejaknya.
3. Fine-tuning pada jejak yang disimpan.
4. Ulangi.

Ia berhasil. GSM8K dan CommonsenseQA sama-sama membaik tanpa anotasi manusia baru. Tetapi loop ini punya bias bawaan: rationale apa pun yang menghasilkan jawaban benar dipertahankan, terlepas dari apakah penalarannya sendiri sehat. V-STaR (Hosseini et al., 2024) menambalnya dengan verifier yang dipelajari; Quiet-STaR (Zelikman et al., 2024) menggeneralisasi idenya menjadi rationale internal per token.

## Konsepnya

### STaR: bootstrap pada apa yang berhasil

Mulai dari model dasar dengan kemampuan penalaran yang masih lemah. Pada setiap soal training, sampel rationale plus jawaban. Jika jawabannya cocok dengan label, simpan triple (soal, rationale, jawaban). Fine-tuning model pada himpunan yang disimpan. Ulangi.

Satu trik penting. Jika model tidak pernah bisa menjawab suatu soal dengan benar, loop tidak bisa belajar darinya. STaR menambahkan **rationalization**: untuk soal yang gagal dijawab model, suntikkan jawaban benar sebagai petunjuk dan prompt ulang model untuk menghasilkan rationale yang mengarah ke sana. Rationale hasil rationalization ditambahkan ke set training.

Hasil dalam makalah aslinya (Zelikman et al., 2022): model dasar GPT-J membaik pada GSM8K dari 5,8% menjadi 10,7% melalui putaran STaR berulang dengan rationalization — sekitar 5 poin persentase absolut. Pada CommonsenseQA, GPT-J 6B yang dilatih STaR mencapai 72,5%, sebanding dengan GPT-3 175B yang di-fine-tuning (~73%) — model yang kira-kira 30x lebih besar yang dilatih pada rationale beranotasi manual.

### V-STaR: latih verifier dengan DPO

STaR membuang rationale yang salah. Hosseini et al. (2024) mengamati bahwa itu juga data: setiap pasangan (rationale, "apakah ini benar") bisa melatih verifier. Mereka memakai Direct Preference Optimization atas solusi benar maupun salah untuk membangun ranker. Pada waktu inference, sampel N rationale dan pilih pilihan teratas menurut verifier.

Delta yang dilaporkan: +4 hingga +17 poin persentase atas baseline self-improvement sebelumnya pada GSM8K dan MATH, dengan sebagian besar gain berasal dari pemakaian verifier untuk seleksi waktu-inference ketimbang untuk fine-tuning generator tambahan.

### Quiet-STaR: rationale internal per token

Zelikman et al. (2024) bertanya: bagaimana jika model belajar menghasilkan rationale internal pendek pada setiap posisi token, bukan hanya di antara soal dan jawaban? Quiet-STaR melatih model untuk memancarkan "thought" tersembunyi sebelum setiap token yang diprediksi, lalu mencampur prediksi yang sadar-thought dengan prediksi baseline lewat bobot yang dipelajari.

Hasilnya: Mistral 7B memperoleh peningkatan absolut zero-shot pada GSM8K dari 5,9% menjadi 10,9% dan CommonsenseQA dari 36,3% menjadi 47,2% tanpa fine-tuning khusus tugas. Model ini belajar "kapan harus berpikir" — token sulit mendapat rationale internal lebih panjang; yang mudah hampir tidak mendapat apa-apa.

### Mengapa ketiganya berbagi satu masalah keamanan

Ketiga metode memakai jawaban akhir sebagai sinyal gradien. Rationale yang mencapai jawaban benar lewat penalaran cacat — mengeksploitasi jalan pintas, menebak, atau memakai pola yang tidak menggeneralisasi — diperkuat secara positif. Pada soal in-distribution jalan pintas itu berhasil. Pada soal out-of-distribution ia rusak secara senyap.

Verifier V-STaR memitigasi dengan belajar merangking rationale, tetapi verifier dilatih pada set label yang sama. Ia bisa belajar lebih menyukai penalaran salah yang terformat rapi dibanding ketidakpastian yang jujur. Desain yang lebih aman adalah menggabungkan data gaya STaR dengan (a) process-supervised reward model (memberi reward pada langkah antara, bukan hanya jawaban) dan (b) evaluasi OOD held-out yang mematahkan jalan pintas sederhana.

### Perbandingan

| Method | Training signal | Inference cost | Data waste | Known failure mode |
|---|---|---|---|---|
| STaR | keep (rationale, answer) if correct | 1x | discards all incorrect rationales | shortcut rationales |
| STaR + rationalization | above + correct-answer hinted retries | 1x | less | rationalized rationales may be implausible |
| V-STaR | STaR + DPO verifier from both classes | Nx (best-of-N) | minimal | verifier can reinforce confident wrongness |
| Quiet-STaR | per-token rationale + mixing weight | 1.5-3x | minimal | still answer-conditioned gradient |

### Di mana posisinya dalam stack 2026

STaR itu tua. Tetapi polanya muncul kembali di mana-mana pada 2025-2026. RL pada soal matematika yang verifiable (DeepSeek-R1, Kimi-k1.5, o1) adalah sinyal gradien STaR yang dikondisikan jawaban, diskalakan. Process reward model (Lightman et al., 2023; "Let's verify step by step" dari OpenAI) adalah alternatif yang disupervisi-proses. AlphaEvolve (Pelajaran 3) adalah STaR untuk kode, dengan evaluator program sebagai pengganti label. Darwin Godel Machine (Pelajaran 4) adalah STaR untuk scaffolding agen itu sendiri.

Memahami STaR membuat semuanya klik. Ia adalah loop self-improvement minimum yang viable.

```figure
reflection-loop
```

## Gunakan

`code/main.py` menjalankan loop STaR tersimulasi pada tugas aritmetika mainan. Kamu bisa mengamati:

- Bagaimana akurasi menanjak selama putaran bootstrap.
- Bagaimana jalan pintas menyelinap masuk: simulator menyertakan kelas rationale "malas" yang mendapat jawaban benar 40% dari waktu tetapi menggeneralisasi dengan buruk. Perhatikan apakah STaR menyimpannya.
- Bagaimana verifier (gaya V-STaR) membantu pada waktu inference tetapi tidak bisa sepenuhnya memangkas jalan pintas yang diperkenalkan selama training.

## Rilis

`outputs/skill-star-loop-reviewer.md` membantumu mengaudit pipeline self-taught-reasoning yang diusulkan sebelum kamu melatihnya.

## Latihan

1. Jalankan simulator. Atur frekuensi jalan pintas ke nol, lalu ke 0,4. Seberapa besar divergensi akurasi akhirnya antara kedua proses, padahal keduanya mencapai >90% pada distribusi training?

2. Tambahkan pengujian OOD held-out ke simulator. Ambil soal dari distribusi berbeda dan evaluasi model hasil bootstrap pada set in-distribution maupun OOD. Kuantifikasi kesenjangannya.

3. Baca makalah Quiet-STaR (arXiv:2403.09629) Bagian 3. Jelaskan token "end-of-thought" dan head mixing-weight masing-masing dalam tiga kalimat.

4. Bandingkan filter keep-if-correct STaR dengan alternatif yang disupervisi-proses yang memberi reward pada setiap langkah rationale secara independen. Identifikasi perbedaan biaya pelabelan dan perbedaan kualitas yang masuk akal.

5. Rancang satu evaluasi yang akan menangkap rationale jalan pintas pada model yang di-deploy. Ia tidak harus sempurna — ia harus mematahkan jalan pintas paling sederhana yang akan diperkuat loop STaR.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| STaR | "Self-Taught Reasoner" | Fine-tune on model-generated rationales that land correct answers; repeat |
| Rationalization | "Hinted retry" | Inject the correct answer and re-prompt for a rationale on problems the base model fails |
| V-STaR | "Verifier STaR" | DPO-train a verifier on both correct and incorrect rationales, use it for inference-time selection |
| Quiet-STaR | "Per-token rationales" | Generate hidden thoughts at every token position; mix with baseline prediction |
| Answer-conditioned gradient | "Outcome-based signal" | The training loop rewards final answers, not reasoning steps |
| Process reward model | "Step-level verifier" | Reward model trained on per-step correctness, not outcome — contrasts with STaR |
| Shortcut rationale | "Right answer, wrong reasoning" | A rationale that reaches the label via a non-generalizing pattern; STaR keeps these |

## Further Reading

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — the original paper.
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) — adds a DPO verifier for inference-time selection.
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — per-token internal rationales.
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — process reward models, the alternative gradient signal.
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — RL on verifiable tasks, STaR scaled to frontier training.
