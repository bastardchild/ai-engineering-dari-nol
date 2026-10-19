# Jailbreak ASCII Art dan Visual

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753). Samarkan token relevan-keamanan dalam request harmful, ganti dengan rendering ASCII-art huruf yang sama, dan kirim prompt terselubung. GPT-3.5, GPT-4, Gemini, Claude, Llama-2 semuanya gagal mengenali token ASCII-art secara robust. Serangan mem-bypass filter PPL (perplexity), pertahanan Paraphrase, dan Retokenization. Terkait: benchmark ViTC mengukur pengenalan prompt visual non-semantik; StructuralSleight menggeneralisasi ke Uncommon Text-Encoded Structures (tree, graf, JSON nested) sebagai keluarga serangan encoding.

**Type:** Build
**Languages:** Python (stdlib, ArtPrompt token-masking harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan serangan ArtPrompt: langkah identifikasi-kata, substitusi ASCII-art, prompt terselubung final.
- Jelaskan mengapa pertahanan standar (PPL, Paraphrase, Retokenization) gagal pada ArtPrompt.
- Definisikan ViTC dan jelaskan apa yang diukurnya.
- Jelaskan StructuralSleight sebagai generalisasi ke Uncommon Text-Encoded Structures arbitrer.

## Masalahnya

Serangan via parafrasa dan roleplay (Pelajaran 12) dan via konteks panjang (Pelajaran 13) beroperasi pada pola level-teks. ArtPrompt beroperasi pada level pengenalan: model tidak mem-parse token terlarang. Ia mem-parse gambar yang dirender dalam karakter. Filter keamanan melihat pungtuasi harmless. Model melihat kata.

## Konsepnya

### ArtPrompt, dua langkah

Langkah 1. Identifikasi Kata. Dengan request harmful, penyerang memakai LLM untuk mengidentifikasi kata relevan-keamanan (mis. "bomb" dalam "how to make a bomb").

Langkah 2. Generasi Prompt Terselubung. Ganti tiap kata teridentifikasi dengan rendering ASCII-art-nya (blok karakter 7x5 atau 7x7 membentuk bentuk huruf). Model menerima grid pungtuasi dan spasi yang bisa dikenali model yang cukup kapabel sebagai kata; filter keamanan hanya melihat grid.

Hasil: GPT-4, Gemini, Claude, Llama-2, GPT-3.5 semuanya gagal. Attack success rate di atas 75% pada subset benchmark mereka.

### Mengapa pertahanan standar gagal

- **PPL (filter perplexity).** ASCII art punya perplexity tinggi — tetapi begitu pula semua input novel. Pilihan ambang yang memblokir ArtPrompt juga memblokir input terstruktur legit.
- **Paraphrase.** Memparafrasa prompt menghancurkan ASCII art. Dalam praktik, LLM parafrasa sering menjaga atau merekonstruksi art.
- **Retokenization.** Memisahkan token secara berbeda tidak mengubah bahwa visi model mengenali bentuk huruf.

Isu dasarnya adalah filter keamanan bersifat level-token atau level-semantik; ArtPrompt beroperasi pada level pengenalan visual.

### Benchmark ViTC

Pengenalan prompt visual non-semantik. Mengukur kemampuan model membaca ASCII-art, wingdings, dan konten visual non-teks-semantik lainnya. Efektivitas ArtPrompt berkorelasi dengan akurasi ViTC: makin baik model membaca teks visual, makin baik ArtPrompt bekerja padanya. Ini tradeoff kapabilitas-keamanan.

### StructuralSleight

Menggeneralisasi ArtPrompt: Uncommon Text-Encoded Structures (UTES). Tree, graf, JSON nested, CSV-dalam-JSON, blok kode ala-diff. Kalau struktur langka dalam data keamanan training tetapi bisa di-parse model, ia bisa menyembunyikan konten harmful.

Implikasi pertahanan: keamanan harus menggeneralisasi lintas representasi terstruktur yang bisa di-parse model. Himpunannya besar dan tumbuh.

### Analog modalitas-gambar

LLM visual (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1) memperluas permukaan serangan. Serangan ala-ArtPrompt dengan gambar aktual lebih kuat daripada analog ASCII-art karena encoder gambar menghasilkan sinyal lebih kaya.

### Di mana ini berada dalam Fase 18

Pelajaran 12-14 menjelaskan tiga vektor serangan ortogonal: refinement iteratif (PAIR), panjang konteks (MSJ), dan encoding (ArtPrompt/StructuralSleight). Pelajaran 15 bergeser dari serangan model-sentris ke serangan batas-sistem (indirect prompt injection). Pelajaran 16 menjelaskan respons tooling-defensif.

```figure
al-ascii-cloak
```

## Gunakan

`code/main.py` membangun ArtPrompt mainan. Kamu bisa menyamarkan kata spesifik dalam query harmful dengan glyph ASCII-art, verifikasi string terselubung lolos filter kata-kunci, dan (opsional) decode string terselubung kembali memakai recognizer sederhana.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-encoding-audit.md`. Dengan laporan pertahanan-jailbreak, ia mengenumerasi keluarga serangan encoding yang dicakup (ASCII art, base64, leet-speak, homoglyph UTF-8, UTES) dan lapisan pertahanan yang menangkap masing-masing.

## Latihan

1. Jalankan `code/main.py`. Verifikasi string terselubung lolos filter kata-kunci sederhana. Laporkan perubahan level-karakter yang dibutuhkan.

2. Implementasikan encoding kedua: base64 untuk kata target sama. Bandingkan rate bypass-filter terhadap ArtPrompt dan kesulitan pemulihan.

3. Baca Jiang et al. 2024 Seksi 4.3 (hasil lima-model). Usulkan alasan mengapa resistensi-ArtPrompt Claude lebih tinggi daripada Gemini pada benchmark sama.

4. Rancang pertahanan pra-generasi yang mendeteksi region berbentuk-ASCII-art dalam prompt. Ukur false-positive rate pada kode, tabel, dan notasi matematika legit.

5. StructuralSleight mendaftar 10 struktur encoding. Sketsa pertahanan tergeneralisasi yang menangani kesepuluhnya dan estimasi biaya komputasi per prompt yang dipertahankan.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| ArtPrompt | "the ASCII-art attack" | Two-step jailbreak that masks safety words with ASCII-art renderings |
| Cloaking | "hide the word" | Replace a forbidden token with a visual representation the model reads but the filter does not |
| UTES | "uncommon structure" | Uncommon Text-Encoded Structure — tree, graph, nested JSON, etc. used to smuggle content |
| ViTC | "visual-text capability" | Benchmark for model's ability to read non-semantic visual encoding |
| Perplexity filter | "PPL defense" | Reject prompts with high perplexity; fails because legitimate structured input also scores high |
| Retokenization | "tokenizer shift defense" | Pre-process the prompt with a different tokenizer; fails because recognition is visual |
| Homoglyph | "lookalike characters" | Unicode characters that look identical to Latin letters; bypass substring checks |

## Further Reading

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — the ASCII-art jailbreak paper
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES generalization
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — complementary iterative attack
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — complementary length attack
