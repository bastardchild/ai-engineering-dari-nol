# BERT — Masked Language Modeling

> GPT memprediksi kata berikutnya. BERT memprediksi kata yang hilang. Satu kalimat perbedaan — dan setengah dekade dari segala yang berbentuk embedding.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**Time:** ~45 minutes

## Masalahnya

Pada 2018 setiap task NLP — sentimen, NER, QA, entailment — melatih modelnya sendiri dari nol pada data berlabelnya sendiri. Tidak ada checkpoint "memahami bahasa Inggris" terlatih yang bisa kamu fine-tune. ELMo (2018) menunjukkan kamu bisa melatih embedding kontekstual dengan LSTM dua arah; itu membantu tapi tidak menggeneralisasi.

BERT (Devlin et al. 2018) bertanya: bagaimana kalau kita ambil encoder transformer, melatihnya pada setiap kalimat di internet, dan memaksanya memprediksi kata yang hilang dari konteks di kedua sisi? Lalu kamu fine-tune satu head pada downstream task-mu. Efisiensi parameternya adalah sebuah wahyu.

Hasilnya: dalam 18 bulan BERT dan variannya (RoBERTa, ALBERT, ELECTRA) mendominasi setiap leaderboard NLP yang ada. Pada 2020 setiap mesin pencari, pipeline moderasi konten, dan sistem pencarian semantik di bumi punya BERT di dalamnya.

Pada 2026 model encoder-only masih tool yang tepat untuk klasifikasi, retrieval, dan ekstraksi terstruktur — mereka berjalan 5–10× lebih cepat per token dibanding decoder dan embedding mereka adalah tulang punggung setiap stack retrieval modern. ModernBERT (Des 2024) mendorong arsitekturnya ke konteks 8K dengan Flash Attention + RoPE + GeGLU.

## Konsepnya

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### Sinyal training

Ambil sebuah kalimat: `the quick brown fox jumps over the lazy dog`.

Mask 15% token secara acak:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

Latih model untuk memprediksi token orisinal pada posisi ter-mask. Karena encoder itu dua arah, memprediksi `[MASK]` pada posisi 1 bisa memakai `brown fox jumps` pada posisi 2+. Itulah hal yang tidak bisa dilakukan GPT.

### Aturan mask BERT

Dari 15% token yang dipilih untuk prediksi:

- 80% diganti dengan `[MASK]`.
- 10% diganti dengan token acak.
- 10% dibiarkan tidak berubah.

Mengapa tidak selalu `[MASK]`? Karena `[MASK]` tidak pernah muncul saat inference. Melatih model untuk mengharapkan `[MASK]` pada 100% posisi ter-mask akan menciptakan pergeseran distribusi antara pretraining dan fine-tuning. Kombinasi 10% acak + 10% tidak berubah membuat model tetap jujur.

### Next Sentence Prediction (NSP) — dan mengapa ia dibuang

BERT orisinal juga dilatih pada NSP: diberikan dua kalimat A dan B, prediksi apakah B mengikuti A. RoBERTa (2019) mengablasinya dan menunjukkan NSP merugikan, bukan membantu. Encoder modern melewatkannya.

### Apa yang berubah pada 2026: ModernBERT

Makalah ModernBERT 2024 membangun ulang block dengan primitif 2026:

| Component | Original BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Positional | Learned absolute | RoPE |
| Activation | GELU | GeGLU |
| Normalization | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | Alternating local (128) + global |
| Context length | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

Dan tidak seperti stack 2018, ia Flash-Attention-native. Inference 2–3× lebih cepat pada panjang sequence 8K dibanding DeBERTa-v3 dengan skor GLUE yang lebih baik.

### Kasus pakai yang masih memilih encoder pada 2026

| Task | Why encoder beats decoder |
|------|---------------------------|
| Retrieval / semantic search embeddings | Bidirectional context = better embedding quality per token |
| Classification (sentiment, intent, toxicity) | One forward pass; no generation overhead |
| NER / token labeling | Per-position output, natively bidirectional |
| Zero-shot entailment (NLI) | Classifier head on top of encoder |
| Reranker for RAG | Cross-encoder scoring, 10x faster than LLM rerankers |

```figure
transformer-residual
```

## Bangun Sendiri

### Langkah 1: logika masking

Lihat `code/main.py`. Fungsi `create_mlm_batch` menerima daftar ID token, ukuran vocab, dan probabilitas mask. Mengembalikan ID input (dengan mask diterapkan) dan label (hanya pada posisi ter-mask, -100 di tempat lain — konvensi ignore index PyTorch).

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Langkah 2: jalankan prediksi MLM pada korpus mungil

Latih encoder 2 lapis + head MLM pada vocabulary 20 kata, 200 kalimat. Tanpa gradien — kita melakukan sanity check forward-pass. Training penuh butuh PyTorch.

### Langkah 3: bandingkan tipe mask

Tunjukkan bagaimana aturan tiga arah membuat model tetap bisa dipakai tanpa `[MASK]`. Prediksi pada kalimat tanpa mask dan pada kalimat ber-mask. Keduanya seharusnya menghasilkan distribusi token yang masuk akal karena model melihat kedua pola dalam training.

### Langkah 4: head fine-tune

Ganti head MLM dengan head klasifikasi pada dataset sentimen mainan. Hanya head yang berlatih; encoder dibekukan. Ini adalah pola yang diikuti setiap aplikasi BERT.

## Gunakan

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Model embedding adalah BERT yang di-fine-tune.** Model `sentence-transformers` seperti `all-MiniLM-L6-v2` adalah BERT yang dilatih dengan contrastive loss. Encoder-nya sama. Loss-nya yang berubah.

**Reranker cross-encoder juga BERT yang di-fine-tune.** Klasifikasi pasangan pada `[CLS] query [SEP] doc [SEP]`. Attention dua arah antara query dan doc persis inilah yang memberi cross-encoder keunggulan kualitas atas biencoder.

**Kapan tidak memilih BERT pada 2026.** Apa pun yang generatif. Encoder tidak punya cara yang masuk akal untuk menghasilkan token secara autoregresif. Juga: apa pun di bawah 1B param di mana decoder kecil bisa menyamai kualitas dengan fleksibilitas lebih (Phi-3-Mini, Qwen2-1.5B).

## Rilis

Lihat `outputs/skill-bert-finetuner.md`. Skill ini merancang fine-tune BERT (pilihan backbone, spek head, data, eval, stopping) untuk task klasifikasi atau ekstraksi baru.

## Latihan

1. **Mudah.** Jalankan `code/main.py` dan cetak distribusi mask pada 10.000 token. Konfirmasi ~15% terpilih, dan dari jumlah itu ~80% menjadi `[MASK]`.
2. **Sedang.** Implementasikan whole-word masking: kalau sebuah kata ditokenisasi menjadi subword, mask semua subword bersamaan atau tidak sama sekali. Ukur apakah ini meningkatkan akurasi MLM pada korpus 500 kalimat.
3. **Sulit.** Latih BERT mungil (2 lapis, d=64) pada 10.000 kalimat dari dataset publik. Fine-tune token `[CLS]` untuk sentimen SST-2. Bandingkan dengan baseline decoder-only pada param yang sebanding — mana yang menang?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MLM | "Masked language modeling" | Training signal: randomly replace 15% of tokens with `[MASK]`, predict the originals. |
| Bidirectional | "Looks both ways" | Encoder attention has no causal mask — every position sees every other position. |
| `[CLS]` | "The pooler token" | A special token prepended to every sequence; its final embedding is used as the sentence-level representation. |
| `[SEP]` | "Segment separator" | Separates paired sequences (e.g. query/doc, sentence A/B). |
| NSP | "Next sentence prediction" | BERT's second pretraining task; shown to be useless in RoBERTa, dropped after 2019. |
| Fine-tuning | "Adapt to a task" | Keep the encoder mostly frozen; train a small head on top for the downstream task. |
| Cross-encoder | "A reranker" | A BERT that takes both query and doc as input, outputs a relevance score. |
| ModernBERT | "2024 refresh" | Encoder rebuilt with RoPE, RMSNorm, GeGLU, alternating local/global attention, 8K context. |

## Bacaan Lanjutan

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — makalah orisinal.
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — cara melatih BERT dengan benar; membunuh NSP.
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — replaced-token detection mengalahkan MLM pada komputasi sebanding.
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — makalah ModernBERT.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — referensi encoder kanonis.
