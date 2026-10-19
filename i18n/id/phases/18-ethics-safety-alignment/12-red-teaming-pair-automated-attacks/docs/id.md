# Red-Teaming: PAIR dan Serangan Otomatis

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419). PAIR — Prompt Automatic Iterative Refinement — adalah jailbreak black-box otomatis kanonis. LLM penyerang dengan system prompt red-team iteratif mengusulkan jailbreak untuk LLM target, mengakumulasi upaya dan respons dalam histori chat-nya sendiri sebagai umpan-balik in-context. PAIR biasanya sukses dalam 20 query, berorde besaran lebih efisien daripada GCG (pencarian gradien level-token Zou et al.) dan tanpa butuh akses white-box. PAIR kini baseline standar di JailbreakBench (arXiv:2404.01318) dan HarmBench, berdampingan GCG, AutoDAN, TAP, dan Persuasive Adversarial Prompt.

**Type:** Build
**Languages:** Python (stdlib, mock PAIR loop against a toy target)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 14 (agent engineering)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan algoritma PAIR: system prompt penyerang, refinement iteratif, umpan-balik in-context.
- Jelaskan mengapa PAIR strictly lebih efisien daripada GCG saat target black-box.
- Sebutkan empat baseline serangan-otomatis lain (GCG, AutoDAN, TAP, PAP) dan nyatakan satu fitur pembeda masing-masing.
- Jelaskan protokol evaluasi JailbreakBench dan HarmBench dan arti "attack success rate" pada masing-masing.

## Masalahnya

Red-teaming dulu aktivitas manual. Segelintir tester ahli mengkonstruksi prompt adversarial dan melacak mana yang bekerja. Ini tidak berskala: attack success rate butuh sampel statistik, dan target adalah target bergerak pada setiap rilis model. PAIR mengoperasionalkan red-teaming sebagai problem optimasi dengan target black-box.

## Konsepnya

### Algoritma PAIR

Input:
- LLM Target T (model yang kita serang).
- LLM Juri J (menskor apakah respons adalah jailbreak).
- LLM Penyerang A (optimizer red-team).
- String tujuan G: "respond with [harmful instruction]."
- Budget K (biasanya 20 query).

Loop, untuk k dalam 1..K:
1. A di-prompt dengan tujuan G dan histori pasangan (prompt, respons) sejauh ini.
2. A mengemisikan prompt baru p_k.
3. Submit p_k ke T; terima respons r_k.
4. J menskor (p_k, r_k) pada tujuan.
5. Kalau skor >= ambang, berhenti — jailbreak ditemukan.
6. Kalau tidak, tambahkan (p_k, r_k) ke histori A; lanjutkan.

Hasil empiris (NeurIPS 2023): attack success rate >50% terhadap GPT-3.5-turbo, Llama-2-7B-chat; rata-rata query-ke-sukses pada rentang 10-20.

### Mengapa PAIR efisien

GCG (Zou et al. 2023) mencari atas sufiks adversarial token demi gradien; ia butuh akses model white-box dan menghasilkan sufiks tak terbaca. PAIR black-box dan menghasilkan serangan bahasa-alami yang mentransfer lintas model. Umpan-balik in-context PAIR memungkinkan penyerang belajar dari tiap penolakan; GCG tidak punya padanan (setiap update token baru harus menemukan ulang progres sebelumnya).

### Serangan otomatis terkait

- **GCG (Zou et al. 2023, arXiv:2307.15043).** Pencarian gradien level-token untuk sufiks adversarial. White-box, transferable, menghasilkan string tak terbaca.
- **AutoDAN (Liu et al. 2023).** Pencarian evolusioner atas prompt, dipandu objektif hierarkis.
- **TAP (Mehrotra et al. 2024).** Tree-of-attack dengan pruning — mencabangkan banyak rollout ala-PAIR.
- **PAP (Zeng et al. 2024).** Persuasive Adversarial Prompt — mengenkode teknik persuasi manusia sebagai template prompt.

### JailbreakBench dan HarmBench

Keduanya (2024) menstandarkan evaluasi:

- JailbreakBench (arXiv:2404.01318). 100 perilaku harmful lintas 10 kategori kebijakan-OpenAI. Attack success rate (ASR) sebagai metrik primer. Butuh juri (GPT-4-turbo, Llama Guard, atau StrongREJECT).
- HarmBench (Mazeika et al. 2024). 510 perilaku lintas 7 kategori, dengan tes harm semantik dan fungsional. Membandingkan 18 serangan terhadap 33 model.

ASR biasanya dilaporkan pada budget query tetap. Membandingkan serangan butuh menyamakan budget; ASR 90% pada 200 query tidak sebanding dengan ASR 85% pada 20.

### Alasan pentingnya bagi deployment 2026

Setiap lab frontier kini menjalankan PAIR dan TAP terhadap model produksi sebelum rilis. Trajektori ASR muncul dalam model card (Pelajaran 26) dan apendiks safety-case (Pelajaran 18). Serangannya tidak eksotis — melainkan infrastruktur standar.

### Di mana ini berada dalam Fase 18

Pelajaran 12 adalah fondasi serangan-otomatis. Pelajaran 13 (Many-Shot Jailbreaking) adalah eksploit panjang komplementer. Pelajaran 14 (ASCII Art / Visual) adalah serangan encoding. Pelajaran 15 (Indirect Prompt Injection) adalah permukaan serangan produksi 2026. Pelajaran 16 membahas padanan tooling-defensif (Llama Guard, Garak, PyRIT).

```figure
al-pair-loop
```

## Gunakan

`code/main.py` membangun loop PAIR mainan. Target adalah klasifier mock yang menolak prompt harmful yang "obvious" (filter-kata-kunci). Penyerang adalah refiner berbasis-aturan yang mencoba parafrasa, framing roleplay, dan encoding. Juri menskor respons. Kamu menyaksikan penyerang sukses dalam ~5-15 iterasi melawan filter kata-kunci dan gagal melawan filter semantik.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-attack-audit.md`. Dengan laporan evaluasi red-team, ia mengaudit: serangan mana yang dijalankan (PAIR, GCG, TAP, AutoDAN, PAP), pada budget berapa masing-masing, dengan juri mana, pada himpunan perilaku-harmful mana (JailbreakBench, HarmBench, internal).

## Latihan

1. Jalankan `code/main.py`. Ukur rata-rata-query-ke-sukses untuk tiga strategi penyerang bawaan. Jelaskan asumsi pertahanan-target mana yang dieksploitasi masing-masing.

2. Implementasikan strategi penyerang keempat (mis. terjemahan ke bahasa lain, encoding base64). Laporkan rata-rata-query-ke-sukses baru terhadap target filter-kata-kunci dan target filter-semantik.

3. Baca Chao et al. 2023 Gambar 5 (perbandingan PAIR vs GCG). Jelaskan dua skenario di mana GCG lebih disukai meski keunggulan efisiensi PAIR.

4. JailbreakBench melaporkan ASR terhadap himpunan tujuan tetap. Rancang metrik tambahan yang mengukur diversitas serangan (varians pada prompt sukses). Jelaskan mengapa diversitas penting untuk evaluasi pertahanan.

5. TAP (Mehrotra 2024) memperluas PAIR dengan branching + pruning. Sketsa ekstensi ala-TAP ke `code/main.py` dan jelaskan tradeoff biaya komputasi vs success-rate.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| PAIR | "automated jailbreak" | Prompt Automatic Iterative Refinement; attacker-LLM + judge-LLM loop |
| GCG | "gradient jailbreak" | White-box token-level gradient search for adversarial suffixes |
| Attack success rate (ASR) | "% jailbreaks at k queries" | Primary metric; must be reported with query budget and judge identity |
| Judge LLM | "the scorer" | LLM that grades whether a response satisfies the harmful goal |
| JailbreakBench | "the evaluation" | Standardized harmful-behaviour set with tagged categories |
| HarmBench | "the broader bench" | 510 behaviours, functional + semantic harm tests |
| TAP | "tree of attacks" | PAIR with branching + pruning; better ASR at higher compute |

## Further Reading

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — PAIR paper, NeurIPS 2023
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — GCG paper
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — standardized evaluation
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — broader evaluation
