# Voice Cloning & Konversi Suara

> Voice cloning membacakan teksmu dengan suara orang lain. Voice conversion menuliskan ulang suaramu menjadi suara orang lain sambil mempertahankan apa yang kamu ucapkan. Keduanya bertumpu pada dekomposisi yang sama: memisahkan identitas pembicara dari konten.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## Masalahnya

Pada 2026, cuplikan audio 5 detik sudah cukup untuk menghasilkan clone suara siapa pun berkualitas tinggi dengan GPU konsumen. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox semuanya mengirimkan cloning zero-shot atau few-shot. Teknologi ini adalah berkah (TTS untuk aksesibilitas, dubbing, suara asistif) dan senjata (telepon penipuan, deepfake politik, pencurian IP).

Dua tugas yang sangat erat:

- **Voice cloning (sisi TTS):** teks + suara referensi 5 detik → audio dengan suara itu.
- **Voice conversion (sisi bicara):** audio sumber (orang A mengucapkan X) + suara referensi orang B → audio B mengucapkan X.

Keduanya memfaktorkan bentuk gelombang menjadi (konten, pembicara, prosodi) lalu mengombinasikan ulang konten dari satu sumber dengan pembicara dari sumber lain.

Kendala utama yang harus kamu patuhi pada 2026: **watermarking dan gerbang persetujuan diwajibkan secara hukum di UE (AI Act, berlaku Agustus 2026) dan di California (AB 2905, efektif 2025)**. Pipa produksimu harus memancarkan watermark yang tak terdengar dan menolak clone tanpa persetujuan.

## Konsepnya

![Voice cloning vs conversion: faktorisasi, tukar pembicara, kombinasikan ulang](../assets/voice-cloning.svg)

**Cloning zero-shot.** Berikan cuplikan 5 detik ke model yang telah dilatih pada ribuan pembicara. Speaker encoder memetakan cuplikan itu ke speaker embedding; decoder TTS memberi kondisi pada embedding tersebut beserta teks.

Digunakan oleh: F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024).

**Fine-tuning few-shot.** Rekam 5-30 menit suara target. LoRA-fine-tune sebuah model dasar selama satu jam. Kualitas melompat dari "lumayan" menjadi "tak terbedakan". Coqui dan ElevenLabs keduanya mendukung pola ini; komunitas memakainya bersama F5-TTS.

**Voice conversion (VC).** Dua keluarga:

- **Recognition-synthesis.** Jalankan model ala ASR untuk mengekstrak representasi konten (misalnya, posterior fonem lunak, PPGs), lalu resintesis dengan speaker embedding target. Tahan terhadap bahasa dan aksen. Digunakan oleh KNN-VC (2023), Diff-HierVC (2023).
- **Disentanglement.** Latih autoencoder yang memisahkan konten, pembicara, dan prosodi di ruang laten pada bottleneck. Tukar speaker embedding saat inference. Kualitas lebih rendah tetapi lebih cepat. Digunakan oleh AutoVC (2019), varian VITS-VC.

**Cloning berbasis neural codec (2024+).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox — memperlakukan audio sebagai token diskrit dari SoundStream / EnCodec, melatih model autoregresif besar atau flow-matching di atas token codec. Kualitas sebanding dengan ElevenLabs pada prompt pendek.

### Bagian etikanya, bukan tambahan belakangan

**Watermarking.** PerTh (Perth) dan SilentCipher (2024) menanamkan ID ~16-32 bit secara tak terdeteksi di dalam audio. Bertahan melalui re-encoding, streaming, dan penyuntingan umum. Open source siap produksi.

**Gerbang persetujuan.** Setiap output clone harus dipasangkan dengan catatan persetujuan yang dapat diverifikasi. "Saya, Rohit, pada 2026-04-22, memberi wewenang suara ini untuk tujuan X." Simpan di log yang tahan-tambal (tamper-evident).

**Deteksi.** AASIST, RawNet2, dan Wav2Vec2-AASIST tersedia sebagai detektor. Tantangan ASVspoof 2025 mempublikasikan EER 0,8–2,3% untuk detektor state-of-the-art terhadap output ElevenLabs, VALL-E 2, dan Bark.

### Angka (2026)

| Model | Zero-shot? | SECS (target sim) | WER (intel.) | Params |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Yes | 0.72 | 2.1% | 335M |
| XTTS v2 | Yes | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Yes | 0.70 | 2.8% | 220M |
| VALL-E 2 | Yes | 0.77 | 2.4% | 370M |
| VoiceBox | Yes | 0.78 | 2.1% | 330M |

SECS > 0,70 umumnya tak terbedakan dari target bagi kebanyakan pendengar.

```figure
sp-voice-factorize
```

## Bangun Sendiri

### Langkah 1: dekomposisi dengan recognition-synthesis (demo berbasis kode saja di main.py)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

Sederhana secara konsep; massa implementasinya ada di `tts_model` dan speaker encoder.

### Langkah 2: clone zero-shot dengan F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Transkrip referensi harus persis cocok dengan audionya; ketidakcocokan merusak alignment.

### Langkah 3: voice conversion dengan KNN-VC

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC menjalankan WavLM untuk mengekstrak embedding per-frame pada sumber dan kumpulan target, lalu mengganti setiap frame sumber dengan tetangga terdekatnya di kumpulan itu. Non-parametrik, bekerja dengan satu menit bicara target.

### Langkah 4: tanamkan watermark

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

~32 bit payload, dapat dideteksi setelah re-encode MP3 dan noise ringan.

### Langkah 5: gerbang persetujuan

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## Gunakan

Stack 2026:

| Situation | Pick |
|-----------|------|
| 5-sec zero-shot clone, open-source | F5-TTS or OpenVoice v2 |
| Commercial production cloning | ElevenLabs Instant Voice Clone v2.5 |
| Voice conversion (rewriting) | KNN-VC or Diff-HierVC |
| Many-speaker fine-tune | StyleTTS 2 + speaker adapter |
| Cross-lingual cloning | XTTS v2 or VALL-E X |
| Deepfake detection | Wav2Vec2-AASIST |

## Jebakan

- **Transkrip referensi tak sejajar.** F5-TTS dan sejenisnya menuntut teks referensi cocok persis dengan audio referensi, termasuk tanda baca.
- **Referensi ber-reverberasi.** Gema membunuh clone-nya. Rekam kering, mikrofon dekat.
- **Ketidakcocokan emosi.** Referensi training "ceria" menghasilkan clone ceria atas segalanya. Cocokkan emosi referensi dengan penggunaan target.
- **Kebocoran bahasa.** Mengclone penutur Inggris lalu meminta model berbicara Prancis sering tetap membawa aksennya; pakai model lintas-bahasa (XTTS, VALL-E X).
- **Tanpa watermark.** Secara hukum tak bisa dirilis di UE mulai Agustus 2026.

## Rilis

Simpan sebagai `outputs/skill-voice-cloner.md`. Rancang pipa cloning atau conversion dengan gerbang persetujuan + watermark + target kualitas.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Mendemonstrasikan pertukaran speaker-embedding dengan menghitung kosinus antara dua "pembicara" sebelum dan sesudah pertukaran.
2. **Sedang.** Pakai OpenVoice v2 untuk mengclone suaramu sendiri. Ukur SECS antara referensi dan clone. Ukur CER lewat Whisper.
3. **Sulit.** Terapkan watermark SilentCipher ke 20 clone, jalankan melalui encode+decode MP3 128 kbps, deteksi payload-nya. Laporkan akurasi bit.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Zero-shot clone | 5 seconds is enough | Pretrained model + speaker embedding; no training. |
| PPG | Phonetic posteriorgram | Per-frame ASR posteriors used as language-agnostic content rep. |
| KNN-VC | Nearest-neighbor conversion | Replace each source frame with nearest target-pool frame. |
| Neural codec TTS | VALL-E style | AR model over EnCodec/SoundStream tokens. |
| Watermark | Inaudible signature | Bits embedded in audio, survive re-encode. |
| SECS | Cloning fidelity | Cosine between target and clone speaker embeddings. |
| AASIST | Deepfake detector | Anti-spoof model; detects synthesized speech. |

## Bacaan Lanjutan

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — open-source SOTA zero-shot cloning.
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) and [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — neural-codec TTS.
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — disentanglement-based voice conversion.
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — retrieval-based VC.
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — production-ready 32-bit audio watermark.
- [ASVspoof 2025 results](https://www.asvspoof.org/) — detector vs synthesizer arms race, updated 2026.
