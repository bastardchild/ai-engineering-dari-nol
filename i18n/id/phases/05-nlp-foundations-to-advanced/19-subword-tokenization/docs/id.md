# Tokenisasi Subword — BPE, WordPiece, Unigram, SentencePiece

> Tokenizer kata tersedak pada kata tak terlihat. Tokenizer karakter meledakkan panjang sekuens. Tokenizer subword membagi selisihnya. Setiap LLM modern dikirim di atasnya.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## Masalahnya

Vocabulary-mu punya 50.000 kata. Pengguna mengetik "untokenizable". Tokenizermu mengembalikan `[UNK]`. Model kini tak punya sinyal tentang kata itu. Lebih buruk: dokumen persentil-90 di korpusmu punya 40 kata jarang, yang berarti 40 bit informasi terbuang per dokumen.

Tokenisasi subword memecahkannya. Kata umum tetap token tunggal. Kata jarang terurai menjadi potongan bermakna: `untokenizable` → `un`, `token`, `izable`. Data training mencakup segalanya karena string apa pun pada akhirnya adalah sekuens byte.

Setiap LLM frontier di 2026 dikirim pada satu dari tiga algoritme (BPE, Unigram, WordPiece), dibungkus satu dari tiga pustaka (tiktoken, SentencePiece, HF Tokenizers). Kamu tidak bisa mengirim model bahasa tanpa memilih satu.

## Konsepnya

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).** Mulai dengan vocabulary level-karakter. Hitung setiap pasangan bersebelahan. Gabungkan pasangan tersering menjadi token baru. Ulangi sampai mencapai ukuran vocabulary target. Algoritme dominan: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**BPE level-byte.** Algoritme sama tetapi atas byte mentah (256 token dasar) alih-alih karakter Unicode. Menjamin nol token `[UNK]` — sekuens byte apa pun ter-encode. GPT-2 memakai 50.257 token (256 byte + 50.000 merge + 1 spesial).

**Unigram.** Mulai dengan vocabulary besar. Tetapkan tiap token probabilitas unigram. Pangkas iteratif token yang penghapusannya paling sedikit menaikkan log-likelihood korpus. Probabilistik saat inference: bisa menyampel tokenisasi (berguna untuk augmentasi data via regularisasi subword). Dipakai T5, mBART, ALBERT, XLNet, Gemma.

**WordPiece.** Gabungkan pasangan yang memaksimalkan likelihood korpus training alih-alih frekuensi mentah. Dipakai BERT, DistilBERT, ELECTRA.

**SentencePiece vs tiktoken.** SentencePiece adalah pustaka yang *melatih* vocabulary (BPE atau Unigram) langsung pada teks Unicode mentah, meng-encode spasi sebagai `▁`. tiktoken adalah *encoder* cepat OpenAI terhadap vocabulary pra-bangun; ia tidak melatih.

Aturan praktis:

- **Melatih vocabulary baru:** SentencePiece (multilingual, tanpa pra-tokenisasi) atau HF Tokenizers.
- **Inference cepat terhadap vocab GPT:** tiktoken (cl100k_base, o200k_base).
- **Keduanya:** HF Tokenizers — satu pustaka, training + serving.

```figure
bpe-merge
```

## Bangun Sendiri

### Langkah 1: BPE dari nol

Lihat `code/main.py`. Loop-nya:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Tiga fakta yang dikodekan algoritme. `</w>` menandai akhir kata supaya "low" (sufiks) dan "lower" (prefiks) tetap berbeda. Pembobotan frekuensi membuat pasangan frekuensi-tinggi menang duluan. Daftar merge berurutan — inference menerapkan merge sesuai urutan training.

### Langkah 2: encode dengan merge terpelajari

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

Naif O(n·|merges|). Implementasi produksi (tiktoken, HF Tokenizers) memakai lookup merge-rank dengan priority queue dan berjalan nyaris-linear.

### Langkah 3: SentencePiece dalam praktik

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

Perhatikan: tanpa pra-tokenisasi, spasi di-encode sebagai `▁`, `character_coverage` mengontrol seberapa agresif karakter jarang dipertahankan vs dipetakan ke `<unk>`.

### Langkah 4: tiktoken untuk vocab kompatibel-OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Hanya-encoding. Cepat (backend Rust). Cocok persis dengan tokenisasi GPT-4/5 untuk penghitungan byte, estimasi biaya, budgeting context-window.

## Jebakan yang masih lolos di 2026

- **Drift tokenizer.** Training pada vocab A, deploy melawan vocab B. ID token berbeda; output model sampah. Cek hash `tokenizer.json` di CI.
- **Ambiguitas spasi.** BPE "hello" vs " hello" menghasilkan token berbeda. Selalu spesifikasikan `add_special_tokens` dan `add_prefix_space` secara eksplisit.
- **Undertraining multilingual.** Korpus berat-Inggris menghasilkan vocabulary yang memecah skrip non-Latin menjadi 5-10x lebih banyak token. Prompt sama biayanya 5-10x lebih mahal dalam bahasa Jepang/Arab pada GPT-3.5. o200k_base sebagian memperbaikinya.
- **Pecahan emoji.** Satu emoji bisa memakan 5 token. Checkpoint penanganan emoji saat budgeting konteks.

## Gunakan

Stack 2026:

| Situation | Pick |
|-----------|------|
| Training a monolingual model from scratch | HF Tokenizers (BPE) |
| Training a multilingual model | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Serving an OpenAI-compatible API | tiktoken (`o200k_base` for GPT-4+) |
| Domain-specific vocab (code, math, protein) | Train custom BPE on domain corpus, merge with base vocab |
| Edge inference, small model | Unigram (smaller vocabularies work better) |

Ukuran vocabulary adalah keputusan scaling, bukan konstanta. Heuristik kasar: 32k untuk <1B param, 50-100k untuk 1-10B, 200k+ untuk multilingual/frontier.

## Rilis

Simpan sebagai `outputs/skill-bpe-vs-wordpiece.md`:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## Latihan

1. **Mudah.** Latih BPE 500-merge pada korpus mungil `code/main.py`. Encode tiga kata held-out. Berapa yang menghasilkan tepat 1 token vs >1 token?
2. **Sedang.** Bandingkan jumlah token pada 100 kalimat Wikipedia Inggris antara `cl100k_base`, `o200k_base`, dan BPE SentencePiece yang kamu latih dengan vocab=32k. Laporkan rasio kompresi masing-masing.
3. **Sulit.** Latih korpus yang sama dengan BPE, Unigram, dan WordPiece. Ukur akurasi downstream ketika memakai masing-masing pada pengklasifikasi sentimen kecil. Apakah pilihannya menggeser jarum lebih dari 1 poin F1?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Greedy merge of most-frequent character pairs until target vocab size hit. |
| Byte-level BPE | No unknown tokens ever | BPE over raw 256 bytes; GPT-2 / Llama use this. |
| Unigram | Probabilistic tokenizer | Prunes from a large candidate set using log-likelihood; used by T5, Gemma. |
| SentencePiece | The whitespace one | Library that trains BPE/Unigram on raw text; space encoded as `▁`. |
| tiktoken | The fast one | OpenAI's Rust-backed BPE encoder for pre-built vocabs. No training. |
| Merge list | The magic numbers | Ordered list of `(a, b) → ab` merges; inference applies in order. |
| Character coverage | How rare is too rare? | Fraction of characters in training corpus the tokenizer must cover; ~0.9995 typical. |

## Bacaan Lanjutan

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — the BPE paper.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — the Unigram paper.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — the library.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — concise reference.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — cookbook + encoding list.
