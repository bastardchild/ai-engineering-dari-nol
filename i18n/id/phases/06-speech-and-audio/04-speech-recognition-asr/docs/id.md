# Pengenalan Pertuturan (ASR) — CTC, RNN-T, Attention

> Pengenalan pertuturan adalah klasifikasi audio di setiap timestep, disatukan oleh model sekuens yang tahu bahasa Inggris dan keheningan. CTC, RNN-T, dan attention adalah tiga cara melakukannya. Pilih satu dan pahami mengapa.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**Time:** ~45 minutes

## Masalahnya

Kamu punya klip 10 detik 16 kHz. Kamu ingin sebuah string: "turn on the kitchen lights". Tantangannya bersifat struktural: frame audio tidak selaras satu-ke-satu dengan karakter. Kata "okay" bisa memakan 200 ms atau 1200 ms. Keheningan memecah tuturan. Beberapa fonem lebih panjang dari yang lain. Jumlah token output tidak diketahui sebelumnya.

Tiga formulasi menyelesaikan ini:

1. **CTC (Connectionist Temporal Classification).** Keluarkan probabilitas token per frame termasuk *blank* khusus. Repeats dan blank diruntuhkan saat decode. Non-autoregressive, cepat. Dipakai wav2vec 2.0, MMS.
2. **RNN-T (Recurrent Neural Network Transducer).** Joint network memprediksi token berikutnya mengingat frame encoder dan token sebelumnya. Bisa di-stream. Dipakai ASR on-device Google, NVIDIA Parakeet.
3. **Attention encoder-decoder.** Encoder mengompres audio ke hidden states, decoder cross-attend untuk menghasilkan token secara autoregressive. Dipakai Whisper, SeamlessM4T.

Pada 2026, SOTA WER pada LibriSpeech test-clean adalah 1,4% (Parakeet-TDT-1.1B, NVIDIA) dan 1,58% (Whisper-Large-v3-turbo). Perbedaannya kecil; perbedaan deployment-nya besar.

## Konsepnya

![Tiga formulasi ASR: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**Intuisi CTC.** Biarkan encoder mengeluarkan `T` distribusi frame-level di atas `V+1` token (V karakter + blank). Untuk target string `y` dengan panjang `U < T`, alignment frame apa pun yang diruntuhkan menjadi `y` dihitung. CTC loss menjumlahkan semua alignment semacam itu. Inference: argmax per frame, runtuhkan repeats, hapus blank.

Kelebihan: non-autoregressive, bisa di-stream, nol lookahead. Kekurangan: *asumsi kemandirian bersyarat* — prediksi setiap frame mandiri dari yang lain, jadi tidak ada bahasa internal. Perbaiki dengan LM eksternal via beam search atau shallow fusion.

**Intuisi RNN-T.** Menambahkan jaringan *predictor* yang meng-embed riwayat token dan *joiner* yang menggabungkan state predictor dengan frame encoder menjadi distribusi gabungan di atas `V+1` (yang `+1` adalah null / no-emit). Memodelkan secara eksplisit ketergantungan bersyarat yang diabaikan CTC. Bisa di-stream karena setiap langkah hanya bersyarat pada frame lampau dan token lampau.

Kelebihan: bisa di-stream + LM internal. Kekurangan: training lebih kompleks dan boros memori (lattice loss 3D); kernel loss RNN-T adalah kategori pustaka tersendiri.

**Attention encoder-decoder.** Encoder (6-32 layer transformer) di atas frame log-mel. Decoder (6-32 layer transformer) cross-attend ke output encoder untuk menghasilkan token secara autoregressive. Tanpa batasan alignment — attention bisa melihat ke mana pun dalam audio. Tidak bisa di-stream kecuali kamu membatasi attention (Whisper-Streaming berbasis chunk, 2024).

Kelebihan: kualitas tertinggi untuk ASR offline, mudah dilatih dengan tooling seq2seq standar. Kekurangan: latensi autoregressive sebanding dengan panjang output; tidak bisa streaming tanpa rekayasa.

### WER: satu angka itu

**Word Error Rate** = `(S + D + I) / N`, di mana S=substitusi, D=deletions, I=insertions, N=jumlah kata referensi. Cocok dengan Levenshtein edit distance di tingkat kata. Makin rendah makin baik. WER di atas 20% umumnya tidak bisa dipakai; di bawah 5% adalah paritas-manusia untuk speech yang dibacakan. Angka 2026 pada benchmark standar:

| Model | LibriSpeech test-clean | LibriSpeech test-other | Size |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

Semuanya berbasis encoder-decoder atau RNN-T. Sistem CTC murni (wav2vec 2.0) berada di kisaran 1,8–2,1% pada test-clean.

```figure
ctc-collapse
```

## Bangun Sendiri

### Langkah 1: decode CTC greedy

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

Dua aturan: runtuhkan repeats berurutan, buang blank. Contoh: `a a _ _ a b b _ c` → `a a b c`.

### Langkah 2: beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

Produksi memakai beam search prefix tree dengan LM fusion; ini adalah kerangka konseptualnya.

### Langkah 3: WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Langkah 4: inference terhadap Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

One-liner untuk ASR umum terkuat di 2026. Berjalan di GPU 24 GB dengan kecepatan ~20× realtime.

### Langkah 5: streaming dengan Parakeet atau wav2vec 2.0

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

ASR streaming membutuhkan attention encoder yang di-chunk dan state carryover; pakai pustaka yang mendukungnya (NeMo untuk Parakeet, pipeline `transformers` dengan `chunk_length_s`).

## Gunakan

Stack 2026:

| Situation | Pick |
|-----------|------|
| English, offline, max quality | Whisper-large-v3-turbo |
| Multilingual, robust | SeamlessM4T v2 |
| Streaming, low latency | Parakeet-TDT-1.1B or Riva |
| Edge, mobile, <500 ms latency | Whisper-Tiny quantized or Moonshine (2024) |
| Long-form | Whisper with VAD-based chunking (WhisperX) |
| Domain-specific (medical, legal) | Fine-tune wav2vec 2.0 + domain LM fusion |

## Jebakan yang masih rilis di 2026

- **Tanpa VAD.** Menjalankan Whisper pada keheningan menghasilkan halusinasi ("Thanks for watching!"). Selalu gerbangkan dengan VAD.
- **WER karakter vs kata vs subword.** Laporkan WER tingkat kata *setelah* normalisasi (huruf kecil, tanda baca dibuang).
- **Drift identifikasi bahasa.** Auto LID Whisper salah mengarahkan klip berisik ke bahasa Jepang atau Wales; paksa `language="en"` saat kamu tahu.
- **Klip panjang tanpa chunking.** Whisper punya jendela 30 detik. Pakai `chunk_length_s=30, stride=5` untuk apa pun yang lebih panjang.

## Rilis

Simpan sebagai `outputs/skill-asr-picker.md`. Pilih model, strategi decoding, chunking, dan LM fusion untuk target deployment tertentu.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ia men-decode secara greedy keluaran CTC yang disusun tangan dan menghitung WER terhadap referensi.
2. **Sedang.** Implementasikan beam search prefix tree di Langkah 2 dengan benar (perhatikan aturan merge blank). Bandingkan dengan greedy pada dataset sintetis 10 contoh.
3. **Sulit.** Pakai `whisper-large-v3-turbo` pada [LibriSpeech test-clean](https://www.openslr.org/12). Hitung WER pada 100 tuturan pertama. Bandingkan dengan angka yang dipublikasikan.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| CTC | The blank-token loss | Marginal over all frame-to-token alignments; non-AR. |
| RNN-T | The streaming loss | CTC + next-token predictor; handles word-order. |
| Attention enc-dec | Whisper-style | Encoder + cross-attending decoder; best offline quality. |
| WER | The number you report | `(S+D+I)/N` at word level. |
| Blank | The emptiness | Special token in CTC signalling "no emission this frame". |
| LM fusion | External language model | Add weighted LM log-probs during beam search. |
| VAD | The silence gate | Voice activity detector; trims non-speech. |

## Bacaan Lanjutan

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — makalah CTC-nya.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — makalah RNN-T-nya.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — makalah kanonis 2022; ekstensi v3-turbo pada 2024.
- [NVIDIA NeMo — kartu Parakeet-TDT](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — pemimpin Open ASR Leaderboard 2026.
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — benchmark langsung lintas 25+ model.
