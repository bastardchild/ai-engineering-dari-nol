# Whisper — Arsitektur & Fine-Tuning

> Whisper adalah transformer encoder-decoder berjendela 30 detik, dilatih pada 680 ribu jam pasangan audio-teks multilingual lemah-terawasi. Satu arsitektur, banyak tugas, tangguh di 99 bahasa. Baseline ASR referensi 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 5 · 10 (Attention), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## Masalahnya

Whisper, dirilis OpenAI pada September 2022, adalah model ASR pertama yang ship sebagai komoditas: tempel audio, dapat teks, 99 bahasa, tahan bising, berjalan di laptop. Pada 2024 OpenAI telah merilis varian Large-v3 dan Turbo; pada 2026, Whisper adalah baseline bawaan untuk semuanya mulai transkripsi podcast, asisten suara, sampai subtitle YouTube.

Tetapi Whisper bukan pipeline yang boleh kamu perlakukan sebagai black box selamanya. Domain shift membunuhnya — jargon teknis, aksen pembicara, nama diri, klip pendek, hening. Kamu perlu tahu:

1. Apa isi dalamnya sebenarnya.
2. Cara memberinya audio chunked, streaming, atau bentuk-panjang dengan benar.
3. Kapan fine-tune dan bagaimana caranya.

## Konsepnya

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**Arsitektur.** Transformer encoder-decoder standar.

- Input: spectrogram log-mel 30 detik, 80 mels, hop 10 ms → 3000 frame. Klip lebih pendek di-zero-pad, klip lebih panjang di-chunk.
- Encoder: conv-downsample (stride 2) + `N` blok transformer. Untuk Large-v3: 32 layer, dimensi 1280, 20 head.
- Decoder: `N` blok transformer dengan self-attn kausal + cross-attn ke output encoder. Ukuran sama dengan encoder.
- Output: token BPE di atas vocabulary berjumlah 51.865 token.

Large-v3 punya 1,55 miliar parameter. Turbo memakai decoder 4 layer (dari 32), memotong latensi 8× dengan regresi WER <1%.

**Format prompt.** Whisper adalah model multitask yang diarahkan oleh token khusus pada prompt decoder:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — tag bahasa; memaksa perilaku terjemahan-vs-transkripsi.
- `<|transcribe|>` atau `<|translate|>` — transliterasi output bahasa Inggris dari input bahasa apa pun, atau verbatim.
- `<|notimestamps|>` — lewati timestamp tingkat kata (lebih cepat).

Prompt itulah yang memungkinkan satu model mengerjakan banyak tugas. Ganti `<|en|>` ke `<|fr|>` dan ia men-transkrip bahasa Prancis.

**Jendela 30 detik.** Semua dipaku pada 30 detik. Klip lebih panjang perlu di-chunk; klip lebih pendek di-pad. Jendela tidak di-stream secara native — inilah alasan WhisperX, Whisper-Streaming, dan faster-whisper ada.

**Normalisasi log-mel.** `(log_mel - mean) / std` di mana statistiknya berasal dari korpus training Whisper sendiri. Kamu *harus* memakai preprocessing Whisper (`whisper.audio.log_mel_spectrogram`), bukan `librosa.feature.melspectrogram`.

### Varian pada 2026

| Variant | Params | Latency (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | streaming | 2.0% |

### Fine-tuning

Alur kerja kanonis pada 2026:

1. Kumpulkan 10–100 jam audio domain-target dengan transkrip yang selaras.
2. Jalankan `transformers.Seq2SeqTrainer` dengan callback `generate_with_loss`.
3. Parameter-efficient: LoRA pada `q_proj`, `k_proj`, `v_proj` layer attention mengurangi memori GPU 4× dengan biaya WER <0,3.
4. Bekukan encoder jika kamu punya <10 jam. Hanya tune decoder-nya.
5. Pakai tokenizer dan format prompt milik Whisper sendiri; jangan pernah menukar tokenizer.

Hasil komunitas: fine-tuning Medium pada 20 jam dikte medis menurunkan WER dari 12% ke 4,5% pada kosakata medis. Fine-tuning Turbo pada 4 jam bahasa Islandia menurunkan WER dari 18% ke 6%.

```figure
sp-asr-attention
```

## Bangun Sendiri

### Langkah 1: jalankan Whisper out of the box

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

Default penting yang selalu harus kamu override: `temperature=0.0` (rantai fallback sampling default 0,0 → 0,2 → 0,4 …), `condition_on_previous_text=False` (mencegah masalah halusinasi berantai), dan `no_speech_threshold=0.6` (deteksi hening).

### Langkah 2: long-form ter-chunk

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX menambahkan (1) gating Silero VAD, (2) alignment tingkat kata via wav2vec 2.0, (3) diarization via `pyannote.audio`. Kuda kerja produksi transkripsi tahun 2026.

### Langkah 3: fine-tune dengan LoRA

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

Lalu loop Trainer standar. Checkpoint setiap 1000 langkah. Evaluasi dengan WER pada held-out.

### Langkah 4: inspeksi apa yang dipelajari tiap layer

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

Visualisasikan dengan heatmap — kamu akan melihat alignment diagonal saat langkah decoder menelusuri frame encoder. Diagonal itulah konsep timestamp kata menurut Whisper.

## Pakai

Stack tahun 2026:

| Situation | Pick |
|-----------|------|
| General English, offline | Large-v3-turbo via `whisperx` |
| Mobile / edge | Whisper-Tiny quantized (int8) or Moonshine |
| Multilingual long-form | Large-v3 via `whisperx` + diarization |
| Low-resource language | Fine-tune Medium or Turbo with LoRA |
| Streaming (2 s latency) | Whisper-Streaming or Parakeet-TDT |
| Word-level timestamps | WhisperX (forced alignment via wav2vec 2.0) |

`faster-whisper` (backend CTranslate2) adalah runtime inferensi CPU+GPU tercepat pada 2026 — 4× lebih cepat dari vanilla dengan output identik.

## Jebakan yang masih ship di 2026

- **Teks terhalusinasi saat hening.** Whisper yang dilatih pada caption menyertakan "Thanks for watching!", "Subscribe!", lirik lagu. Selalu VAD-gate sebelum memanggil.
- **Kaskade `condition_on_previous_text`.** Satu halusinasi mencemari jendela berikutnya. Set `False` kecuali kamu butuh kelancaran lintas chunk.
- **Padding klip pendek.** Klip 2 detik yang di-pad ke 30 detik bisa menghalusinasi di hening ekornya. Pakai `pad=False` atau VAD-gate.
- **Statistik mel yang salah.** Memakai mel librosa alih-alih milik Whisper menghasilkan output hampir-acak. Pakai `whisper.audio.log_mel_spectrogram`.

## Ship It

Simpan sebagai `outputs/skill-whisper-tuner.md`. Rancang pipeline fine-tune atau inferensi Whisper untuk domain tertentu.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ia men-tokenisasi prompt gaya Whisper, menghitung anggaran bentuk decoded, dan mencetak jadwal chunk untuk klip 10 menit.
2. **Sedang.** Install `faster-whisper`, transkrip podcast 10 menit, bandingkan WER terhadap transkrip manusia. Coba `language="auto"` vs `language="en"` yang dipaksa.
3. **Sulit.** Pakai HF `datasets`, pilih bahasa yang sulit bagi Whisper (misal Urdu), fine-tune Medium dengan LoRA 2 epoch pada 2 jam, dan laporkan delta WER.

## Istilah Kunci

| Istilah | Yang orang bilang | Arti sebenarnya |
|---------|-------------------|-----------------|
| 30-sec window | Batas Whisper | Cap input keras; chunk audio lebih panjang. |
| SOT | Start-of-transcript | `<\|startoftranscript\|>` memulai prompt decoder. |
| Timestamps token | Alignment temporal | Setiap offset 0,02 s adalah token khusus di vocabulary 51k. |
| Turbo | Varian cepat | Decoder 4 layer, 8× lebih cepat, regresi WER <1%. |
| WhisperX | Wrapper long-form | VAD + Whisper + alignment wav2vec + diarization. |
| LoRA fine-tune | Tuning efisien | Tambahkan adapter low-rank ke attention; latih ~0,3% parameter. |
| Halusinasi | Kegagalan diam | Whisper menghasilkan bahasa Inggris lancar dari noise/hening. |

## Bacaan Lanjutan

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — arsitektur dan resep training asli.
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — decoder 4 layer, speedup 8×.
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — long-form, word-aligned, terdiarization.
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — berbasis CTranslate2, 4× lebih cepat.
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — panduan kanonis LoRA / full-FT.
