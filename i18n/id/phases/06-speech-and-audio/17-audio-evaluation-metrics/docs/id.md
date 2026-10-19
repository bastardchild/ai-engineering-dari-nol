# Evaluasi Audio — WER, MOS, UTMOS, MMAU, FAD, dan Leaderboard Terbuka

> Kamu tidak bisa merilis apa yang tidak bisa kamu ukur. Pelajaran ini menyebut metrik 2026 untuk setiap tugas audio: ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, WER-on-ASR-round-trip), audio-language (MMAU, LongAudioBench), musik (FAD, CLAP), dan speaker (EER). Ditambah leaderboard tempat kamu membandingkan.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 minutes

## Masalahnya

Setiap tugas audio punya banyak metrik, masing-masing mengukur sumbu yang berbeda. Memakai metrik yang salah adalah cara kamu merilis model yang terlihat hebat di dashboard-mu dan mengerikan di produksi. Daftar kanonis 2026:

| Task | Primary | Secondary |
|------|---------|-----------|
| ASR | WER | CER · RTFx · first-token latency |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| Voice cloning | SECS (ECAPA cosine) | MOS · CER |
| Speaker verification | EER | minDCF · FAR / FRR at operating point |
| Diarization | DER | JER · speaker confusion |
| Audio classification | top-1 · mAP | macro F1 · per-class recall |
| Music generation | FAD | CLAP · listening panel MOS |
| Audio language model | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S | latency P50/P95 | WER · MOS |

## Konsepnya

![Matriks evaluasi audio — metrik vs tugas vs leaderboard 2026](../assets/eval-landscape.svg)

### Metrik ASR

**WER (Word Error Rate).** `(S + D + I) / N`. Lowercase, buang tanda baca, normalkan angka sebelum skoring. Pakai `jiwer` atau `whisper_normalizer` dari OpenAI. &lt; 5% = bicara-baca tingkat kesetaraan-manusia.

**CER (Character Error Rate).** Rumus sama, di level karakter. Dipakai untuk bahasa bernada (Mandarin, Kanton) di mana segmentasi kata ambigu.

**RTFx (inverse real-time factor).** Detik audio yang diproses per detik wall-clock. Makin tinggi makin baik. Parakeet-TDT mencapai 3380×. Whisper-large-v3 ~30×.

**First-token latency.** Wall-clock dari input audio ke token transkrip pertama. Kritis untuk streaming. Deepgram Nova-3: ~150 ms.

### Metrik TTS

**MOS (Mean Opinion Score).** Rating manusia 1-5. Standar emas tetapi lambat. Kumpulkan 20+ pendengar per sampel, 100+ sampel per model.

**UTMOS (2022-2026).** Prediktor MOS yang dipelajari. Berkorelasi ~0.9 dengan MOS manusia pada benchmark standar. F5-TTS: UTMOS 3.95; ground truth: 4.08.

**SECS (Speaker Encoder Cosine Similarity).** Untuk voice cloning. Cosine embedding ECAPA antara referensi dan output hasil-kloning. &gt; 0.75 = clone yang bisa dikenali.

**WER-on-ASR-round-trip.** Jalankan Whisper di atas output TTS, hitung WER terhadap teks input. Menangkap regresi intelligibility. SOTA 2026: CER &lt; 2%.

**TTFA (time-to-first-audio).** Latensi wall-clock. Kokoro-82M: ~100 ms; F5-TTS: ~1 s.

### Khusus voice-cloning

**SECS + MOS + CER** sebagai triplet. Cloning yang skor SECS-nya tinggi tetapi MOS rendah berarti timbre-tetapi-tidak-alami; kebalikannya berarti suara alami tetapi pembicara salah.

### Verifikasi speaker

**EER (Equal Error Rate).** Ambang di mana False Accept Rate sama dengan False Reject Rate. ECAPA pada VoxCeleb1-O: 0.87%.

**minDCF (min Detection Cost).** Biaya berbobot pada titik operasi yang dipilih (sering FAR=0.01). Lebih relevan untuk produksi daripada EER.

### Diarization

**DER (Diarization Error Rate).** `(FA + Miss + Confusion) / total_speaker_time`. Bicara terlewat + bicara false-alarm + kebingungan pembicara, masing-masing sebagai pecahan. Pertemuan AMI: DER ~10-20% itu realistis. pyannote 3.1 + Precision-2 komersial: DER &lt;10% pada audio yang direkam baik.

**JER (Jaccard Error Rate).** Alternatif untuk DER, robust terhadap bias segmen pendek.

### Klasifikasi audio

Multi-label: **mAP (mean Average Precision)** di seluruh kelas. AudioSet: mAP 0.548 untuk BEATs-iter3.

Multi-class eksklusif: **akurasi top-1, top-5**. Speech Commands v2: top-1 99.0% (Audio-MAE).

Tidak seimbang: **macro F1** + **recall per-kelas**. Laporkan per-kelas — akurasi agregat menyembunyikan kelas mana yang gagal.

### Generasi musik

**FAD (Fréchet Audio Distance).** Jarak antara distribusi embedding VGGish audio asli vs hasil generasi. MusicGen-small pada MusicCaps: 4.5. MusicLM: 4.0. Makin rendah makin baik.

**CLAP Score.** Skor keselarasan teks-audio memakai embedding CLAP. &gt; 0.3 = keselarasan wajar.

**MOS listening panel.** Masih kata terakhir untuk musik kelas-konsumen. Suno v5 ELO 1293 di TTS Arena (dari preferensi manusia berpasangan).

### Benchmark audio-language

**MMAU (Massive Multi-Audio Understanding).** 10k pasang audio-QA.

**MMAU-Pro.** 1800 butir sulit, empat kategori: speech / sound / music / multi-audio. Peluang acak 25% pada 4 arah. Gemini 2.5 Pro keseluruhan ~60%; multi-audio ~22% di semua model.

**LongAudioBench.** Klip multi-menit dengan query semantik. Audio Flamingo Next mengalahkan Gemini 2.5 Pro.

**AudioCaps / Clotho.** Benchmark captioning. Metrik SPICE, CIDEr, FENSE.

### Speech-to-speech streaming

**Latensi P50 / P95 / P99.** Wall-clock dari akhir bicara-pengguna ke respons terdengar pertama. Moshi: 200 ms; GPT-4o Realtime: 300 ms.

**WER / MOS** pada output-nya.

**Responsivitas barge-in.** Waktu dari interupsi pengguna hingga asisten senyap. Target &lt; 150 ms.

### Leaderboard 2026

| Leaderboard | Tracks | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | English + multilingual + long-form | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO from paired votes | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM reasoning | `sonalkum.github.io/mmau-pro` |
| SpeakerBench / VoxSRC | Speaker recognition | `voxsrc.github.io` |
| MMAU music subset | Music LALM | (within MMAU) |
| HEAR benchmark | Self-supervised audio | `hearbenchmark.com` |

```figure
sp-wer-align
```

## Bangun Sendiri

### Langkah 1: WER dengan normalisasi

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Langkah 2: WER round-trip TTS

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Langkah 3: SECS untuk voice cloning

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Langkah 4: FAD untuk generasi musik

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Langkah 5: EER untuk verifikasi speaker (kode sama dengan Pelajaran 6)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## Gunakan

Pasangkan setiap deploy dengan harness evaluasi tetap yang berjalan di setiap pembaruan model. Tiga aturan kardinal:

1. **Normalkan sebelum skoring.** Lowercase, buang tanda baca, ekspansi angka. Laporkan aturan normalisasinya.
2. **Laporkan distribusi, bukan rata-rata.** P50/P95/P99 untuk latensi. Recall per-kelas untuk klasifikasi. Per-kategori untuk MMAU.
3. **Jalankan satu benchmark publik kanonis.** Bahkan jika data produksimu berbeda, pelaporan di Open ASR / TTS Arena / MMAU memungkinkan reviewer membandingkan apple-to-apple.

## Jebakan

- **Ekstrapolasi UTMOS.** Dilatih pada bicara bersih ala VCTK; menilai audio berisik / hasil-kloning / emosional dengan buruk.
- **Bias panel MOS.** 20 pekerja Amazon Mechanical Turk ≠ 20 pengguna target. Bayar untuk panel domain jika taruhannya tinggi.
- **FAD bergantung pada set referensi.** Bandingkan terhadap distribusi referensi yang sama di lintas model.
- **WER agregat.** WER 5% secara keseluruhan bisa menyembunyikan WER 30% pada bicara beraksen. Laporkan per potongan demografi.
- **Saturasi benchmark publik.** Kebanyakan model frontier berada di dekat plafon pada benchmark standar. Bangun set held-out in-house yang mencerminkan trafikmu.

## Rilis

Simpan sebagai `outputs/skill-audio-evaluator.md`. Pilih metrik, benchmark, dan format pelaporan untuk rilis model audio apa pun.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Hitung WER / CER / EER / SECS / FAD-ish / MMAU-ish pada input mainan.
2. **Sedang.** Bangun harness WER round-trip TTS. Jalankan output Kokoro atau F5-TTS-mu lewat Whisper. Hitung WER atas 50 prompt. Tandai prompt dengan WER &gt; 10%.
3. **Sulit.** Skor pilihan LALM Pelajaran 10-mu pada subset speech + multi-audio MMAU-Pro (50 butir masing-masing). Laporkan akurasi per-kategori dan bandingkan dengan angka yang dipublikasikan.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| WER | ASR score | `(S+D+I)/N` at word level after normalization. |
| CER | Character WER | For tone languages or char-level systems. |
| MOS | Human opinion | 1-5 rating; 20+ listeners × 100 samples. |
| UTMOS | ML MOS predictor | Learned model; correlates ~0.9 with human MOS. |
| SECS | Voice-clone similarity | ECAPA cosine between reference and clone. |
| EER | Speaker verif score | Threshold where FAR = FRR. |
| DER | Diarization score | (FA + Miss + Confusion) / total. |
| FAD | Music-gen quality | Fréchet distance on VGGish embeddings. |
| RTFx | Throughput | Audio seconds per wall-clock second. |

## Bacaan Lanjutan

- [jiwer](https://github.com/jitsi/jiwer) — WER/CER library with normalization utilities.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — learned MOS predictor.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — the music-gen standard.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 live rankings.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — human-vote TTS leaderboard.
- [MMAU-Pro benchmark](https://sonalkum.github.io/mmau-pro/) — LALM reasoning leaderboard.
- [HEAR benchmark](https://hearbenchmark.com/) — audio SSL benchmarks.
