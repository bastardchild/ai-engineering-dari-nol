# Society of Mind dan Debat Multi-Agent

> Premis Minsky 1986 — kecerdasan adalah masyarakat spesialis — ditemukan kembali setiap dekade. Pada 2023 Du et al. mengubahnya menjadi algoritma konkret: beberapa instance LLM mengusulkan jawaban, membaca jawaban satu sama lain, mengkritik, dan memperbarui. Selama N putaran mereka menyimpulkan pada konsensus yang mengalahkan zero-shot CoT dan reflection pada enam tugas penalaran dan faktualitas. Dua temuan penting: baik **banyak agent** maupun **banyak putaran** berkontribusi secara independen. Society mengalahkan monolog single-agent; pertukaran multi-putaran mengalahkan voting satu-shot.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Masalahnya

Self-consistency — sampel satu model banyak kali dan ambil jawaban mayoritas — adalah peningkatan penalaran termurah yang bisa kamu tempelkan. Ia bekerja, tetapi cepat jenuh. Kamu bisa menggandakan sampelmu dan tidak melihat lompatan bermakna lainnya.

Debat memecah saturasi itu. Alih-alih N sampel independen dari satu model, N agent membaca penalaran satu sama lain dan merevisi. Korelasi antar sampel turun (mereka tidak lagi i.i.d.), dan titik konvergensi sering benar di tempat voting i.i.d. dengan yakin salah.

## Konsepnya

### Algoritma Du et al. 2023

Dari arXiv:2305.14325 (ICML 2024):

1. Masing-masing dari N agent menghasilkan jawaban awal untuk pertanyaan itu.
2. Untuk putaran r = 2..R: setiap agent ditunjukkan jawaban putaran r-1 dari agent lain dan ditanya "mempertimbangkan ini, berikan jawabanmu yang diperbarui."
3. Setelah R putaran, lakukan voting mayoritas pada jawaban akhir.

Paper ini menguji pada MMLU, GSM8K, biografi, MATH, dan benchmark faktualitas. Debat secara konsisten mengalahkan CoT dan Self-Reflection.

### Dua knob independen

Ablasi dari paper yang sama:

- **Jumlah agent saja** (1 putaran, voting mayoritas N) mengalahkan single-agent pada sebagian besar tugas, tetapi mendatar.
- **Jumlah putaran saja** (1 agent melihat penalaran sebelumnya miliknya sendiri) nyaris tidak membantu — kelemahan reflection yang sudah diketahui.
- **Keduanya bersama** menghasilkan lompatan besar. Pertukaran multi-putaran antara banyak agent mendorong peningkatannya.

### Mengapa ini bekerja

Dua mekanisme:

1. **Paparan terhadap ketidaksepakatan.** Ketika sebuah agent melihat rantai penalaran agent lain dengan kesimpulan berbeda, ia harus membenarkan atau memperbarui. Apa pun jalannya, context untuk putaran r+1 lebih kaya daripada putaran r.
2. **Pengurangan error terkorelasi.** Dalam self-consistency, semua sampel berasal dari model yang sama, sehingga error-nya berkorelasi — kamu merata-ratakan menjadi jawaban yang dengan yakin salah. Model berbeda atau seed berbeda mengdekorrelasi. *Tampilan yang diperdebatkan* berbeda mengdekorrelasi lebih jauh.

### Debat heterogen

A-HMAD dan tindak lanjut terkait menggunakan *model dasar berbeda* untuk agent berbeda. Debat Llama + Claude + GPT mengurangi runtuhnya monokultur (Lesson 26) karena error terkorelasi dari satu keluarga model tidak dibagikan oleh yang lain.

Kekurangan: model lemah yang berpartisipasi dalam debat bisa menarik konsensus ke arah jawaban salahnya (lihat "Should we be going MAD?", arXiv:2311.17371).

### NLSOM — ekstensi 129-agent

Zhuge et al. ("Mindstorms in Natural Language-Based Societies of Mind," arXiv:2305.17066) menskalakan ide ini ke society beranggota 129. Hasilnya: spesialisasi dan self-organization muncul seiring skala, dan sistemnya mengungguli single-agent pada tugas seperti visual question answering.

### Mode kegagalan

- **Kaskade sycophancy.** Semua agent tunduk pada agent mana pun yang terdengar paling percaya diri. Debat runtuh ke suara paling keras. Prompting untuk peran adversarial ("satu agent harus berargumen posisi berlawanan") membantu.
- **Drift topik.** Debat selama banyak putaran menyimpang dari pertanyaan asli. Mitigasi: menyuntikkan kembali pertanyaannya setiap putaran.
- **Ledakan komputasi** N agent × R putaran = N·R panggilan LLM, masing-masing dengan context yang tumbuh. Debat 5-agent, 5-putaran adalah 25 panggilan dengan context yang tumbuh. Biaya per pertanyaan bisa melebihi 10× satu panggilan CoT.

```figure
multi-agent-debate
```

## Bangun

`code/main.py` menjalankan debat 3-agent × 3-putaran pada pertanyaan matematika di mana setiap agent memulai dengan jawaban berbeda (yang mungkin salah). Agent di-skrip — masing-masing "memperbarui" dengan merata-ratakan jawaban tetangga yang ditimbang oleh confidence ber-skrip. Konvergensi terlihat di log per-putaran.

Demo menunjukkan dua efek kunci:

- Satu putaran pertukaran menggeser agent lebih dekat ke jawaban yang benar.
- Putaran ekstra setelah putaran 2 menunjukkan hasil yang berkurang (cocok dengan dataran Du et al.).

Jalankan:

```
python3 code/main.py
```

## Gunakan

`outputs/skill-debate-configurator.md` mengonfigurasi sebuah debat untuk tugas baru: jumlah agent, jumlah putaran, heterogenitas (model sama vs campuran), penetapan peran (simetris vs satu-adversarial). Ia juga memperkirakan biaya token sebelum kamu menjalankannya.

## Rilis

Jika kamu merilis debat:

- **Cap putaran di 3.** Du et al. menunjukkan 3 putaran menangkap sebagian besar peningkatannya. Lebih banyak adalah biaya, bukan kualitas.
- **Cap agent di 5.** Di atas 5, pembengkakan context dan biaya mendominasi.
- **Heterogen secara default.** Setidaknya dua model dasar berbeda di pool.
- **Slot adversarial.** Satu agent di-prompt untuk tidak setuju terlepas dari apa pun. Mematahkan sycophancy.
- **Log setiap putaran.** Sistem debat yang menyembunyikan putaran antara tidak bisa di-debug atau diaudit.

## Latihan

1. Jalankan `code/main.py`, lalu set jumlah putaran ke 5 dan amati hasil yang berkurang. Pada putaran berapa konvergensi tambahan berhenti?
2. Tambahkan agent keempat dengan peran adversarial: selalu tidak setuju dengan mayoritas saat ini. Apakah ini mematahkan atau memperbaiki konvergensi?
3. Plot (cetak) skor kesepakatan per putaran (fraksi agent pada jawaban mayoritas). Kapan skor itu mencapai 1,0 dan apakah itu ekuivalen dengan "benar"?
4. Baca ablasi Bagian 4 Du et al. Replikasi hasil "agents-only" vs "rounds-only" vs "both" menggunakan kode ini.
5. Baca "Should we be going MAD?" (arXiv:2311.17371) dan daftarkan dua varian debat di luar round-robin — misalnya, judge-led, chain-of-debate, adversarial.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Society of Mind | "Minsky's idea" | Intelligence as interacting specialists; 1986 framing now operationalized via LLM debate. |
| Multi-agent debate | "Agents argue" | N agents propose, critique each other, revise over R rounds, majority-vote. |
| Consensus | "They agree" | Not epistemic truth — just fraction-on-majority-answer. Can be confidently wrong. |
| Rounds | "Exchange steps" | One round = each agent reads the others and updates once. |
| Heterogeneous debate | "Mix model families" | Using different base models to decorrelate errors. |
| Sycophancy cascade | "Everyone agrees with the loud one" | Debate failure where agents defer to the most confident agent regardless of correctness. |
| NLSOM | "129-agent society" | Natural-language society of mind; Zhuge et al.'s scaled version. |
| Correlated error | "Same model, same bug" | Why self-consistency saturates; debate across different views decorrelates. |

## Bacaan Lanjutan

- [Du et al. — Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) — the reference paper, ICML 2024
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066) — 129-agent NLSOM
- [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) — benchmarks debate variants
- [Debate project page](https://composable-models.github.io/llm_debate/) — Du et al.'s code, demos, and ablation details
