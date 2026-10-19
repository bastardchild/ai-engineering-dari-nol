# T5, BART — Model Encoder-Decoder

> Encoder memahami. Decoder membangkitkan. Satukan kembali dan kamu dapat model yang dibangun untuk task input → output: terjemahkan, ringkas, tulis ulang, transkripsikan.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## Masalahnya

GPT decoder-only dan BERT encoder-only masing-masing memangkas arsitektur 2017 untuk tujuan berbeda. Tapi banyak task secara alami input-output:

- Penerjemahan: Inggris → Prancis.
- Ringkasan: artikel 5.000 token → ringkasan 200 token.
- Pengenalan speech: token audio → token teks.
- Ekstraksi terstruktur: prosa → JSON.

Untuk ini, encoder-decoder adalah kecocokan paling bersih. Encoder menghasilkan representasi padat dari source. Decoder membangkitkan output, cross-attend ke representasi itu di setiap langkah. Training adalah shift-by-one di sisi output. Loss yang sama dengan GPT, hanya dikondisikan pada output encoder.

Dua makalah mendefinisikan playbook modern:

1. **T5** (Raffel et al. 2019). "Text-to-Text Transfer Transformer." Setiap task NLP dibingkai ulang sebagai teks-masuk, teks-keluar. Satu arsitektur, satu vocabulary, satu loss. Pretraining pada prediksi span ter-mask (span rusak di input, decode di output).
2. **BART** (Lewis et al. 2019). "Bidirectional and Auto-Regressive Transformer." Autoencoder denoising: rusak input dengan banyak cara (acak, mask, hapus, putar), minta decoder merekonstruksi yang orisinal.

Pada 2026 format encoder-decoder bertahan di tempat struktur input penting:

- Whisper (speech → teks).
- Stack terjemahan Google.
- Sebagian model penyelesaian/perbaikan kode yang punya struktur konteks-dan-edit berbeda.
- Flan-T5 dan varian untuk task penalaran terstruktur.

Decoder-only memenangkan sorotan, tapi encoder-decoder tidak pernah pergi.

## Konsepnya

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### Loop forward

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

Yang penting, encoder berjalan sekali per input. Decoder berjalan secara autoregresif tapi cross-attend ke output encoder yang *sama* di setiap langkah. Menyimpan cache output encoder adalah speedup gratis untuk input panjang.

### Pretraining T5 — span corruption

Pilih span acak dari input (rata-rata panjang 3 token, total 15%). Ganti setiap span dengan sentinel unik: `<extra_id_0>`, `<extra_id_1>`, dst. Decoder mengeluarkan hanya span yang rusak dengan prefix sentinelnya:

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

Sinyal lebih murah daripada memprediksi seluruh sequence. Kompetitif dengan MLM (BERT) dan prefix-LM (UniLM) dalam ablasi makalah T5.

### Pretraining BART — denoising multi-noise

BART mencoba lima fungsi noise:

1. Token masking.
2. Token deletion.
3. Text infilling (mask a span, decoder inserts the right length).
4. Sentence permutation.
5. Document rotation.

Kombinasi text infilling + sentence permutation menghasilkan angka downstream terbaik. Decoder selalu merekonstruksi yang orisinal. Output BART adalah sequence penuh, bukan hanya span yang rusak — jadi komputasi pretraining lebih tinggi daripada T5.

### Inference

Pembangkitan autoregresif yang sama dengan GPT. Sampling greedy / beam / top-p berlaku. Beam search (lebar 4–5) standar untuk terjemahan dan ringkasan karena distribusi output lebih sempit daripada chat.

### Kapan memilih tiap varian pada 2026

| Task | Encoder-decoder? | Why |
|------|------------------|-----|
| Translation | Yes, usually | Clear source sequence; fixed output distribution; beam search works |
| Speech-to-text | Yes (Whisper) | Input modality differs from output; encoder shapes audio features |
| Chat / reasoning | No, decoder-only | No persistent "input" — the conversation is the sequence |
| Code completion | Usually no | Decoder-only with long context wins; code models like Qwen 2.5 Coder are decoder-only |
| Summarization | Either works | BART, PEGASUS beat earlier decoder-only baselines; modern decoder-only LLMs match them |
| Structured extraction | Either | T5 is clean because "text → text" absorbs any output format |

Tren sejak ~2022: decoder-only mengambil alih task yang dulu dimiliki encoder-decoder karena (a) LLM decoder-only yang di-instruction-tune menggeneralisasi ke apa pun via prompting, (b) satu arsitektur berskala lebih mudah daripada dua, (c) RLHF mengasumsikan decoder. Encoder-decoder bertahan di tempat modalitas input berbeda (speech, gambar) atau di tempat kualitas beam search penting.

```figure
encoder-decoder
```

## Bangun Sendiri

Lihat `code/main.py`. Kita mengimplementasikan span corruption ala T5 untuk korpus mainan — bagian tunggal paling berguna dari pelajaran ini karena ia muncul di setiap resep pretraining encoder-decoder sejak itu.

### Langkah 1: span corruption

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

Format target adalah konvensi T5: `<sent0> span0 <sent1> span1 ...`. Input rusak menyisipkan token tak berubah dengan token sentinel di lokasi span.

### Langkah 2: verifikasi round-trip

Diberikan input rusak dan target, rekonstruksi kalimat orisinal. Kalau corruption-mu reversibel, forward pass terdefinisi dengan baik. Ini sanity check — training nyata tidak pernah melakukan ini, tapi tesnya murah dan menangkap bug off-by-one dalam pembukuan span-mu.

### Langkah 3: noise BART

Lima fungsi: `token_mask`, `token_delete`, `text_infill`, `sentence_permute`, `document_rotate`. Gabungkan dua di antaranya dan tunjukkan hasilnya.

## Gunakan

Referensi HuggingFace:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

Trik T5: nama task masuk ke teks input. Model yang sama menangani puluhan task karena setiap task adalah teks-masuk, teks-keluar. Pada 2026 pola ini telah digeneralisasi oleh model decoder-only yang di-instruction-tune, tapi T5 mengkodifikasikannya pertama.

## Rilis

Lihat `outputs/skill-seq2seq-picker.md`. Skill ini memilih antara encoder-decoder dan decoder-only untuk task baru berdasarkan struktur input-output, latensi, dan target kualitas.

## Latihan

1. **Mudah.** Jalankan `code/main.py`, terapkan span corruption ke kalimat 30 token, verifikasi bahwa menggabungkan token source non-sentinel dengan span target yang di-decode mereproduksi yang orisinal.
2. **Sedang.** Implementasikan noise `text_infill` milik BART: ganti span acak dengan satu token `<mask>`, dan decoder harus menyimpulkan panjang span yang benar plus isinya. Tunjukkan satu contoh.
3. **Sulit.** Fine-tune `flan-t5-small` pada korpus mungil Inggris → pig-Latin (200 pasang). Ukur BLEU pada 50 pasang held-out. Bandingkan dengan fine-tuning `Llama-3.2-1B` pada data yang sama dengan komputasi yang sama.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Encoder-decoder | "Seq2seq transformer" | Two stacks: bidirectional encoder for input, causal decoder with cross-attention for output. |
| Cross-attention | "Where source talks to target" | Decoder's Q × encoder's K/V. The only place encoder information enters the decoder. |
| Span corruption | "T5's pretraining trick" | Replace random spans with sentinel tokens; decoder outputs the spans. |
| Denoising objective | "BART's game" | Apply a noise function to the input, train the decoder to reconstruct the clean sequence. |
| Sentinel token | "The `<extra_id_N>` placeholder" | Special tokens that tag corrupted spans in the source and re-tag them in the target. |
| Flan | "Instruction-tuned T5" | T5 fine-tuned on >1,800 tasks; made encoder-decoder competitive at instruction-following. |
| Beam search | "Decoding strategy" | Keep top-k partial sequences at each step; standard for translation/summarization. |
| Teacher forcing | "Training-time input" | During training, feed the true previous output token to the decoder, not the sampled one. |

## Bacaan Lanjutan

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART.
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5.
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper, encoder-decoder kanonis 2026.
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — implementasi referensi.
