# Capstone 12 — Pipeline Pemahaman Video (Scene, QA, Search)

> Twelve Labs memproduksikan Marengo + Pegasus. VideoDB mengirimkan API CRUD-for-video. Molmo 2 dari AI2 menerbitkan checkpoint VLM terbuka. Gemini long-context menangani berjam-jam video secara native. TimeLens-100K mendefinisikan temporal grounding dalam skala besar. Pipeline 2026 sudah final: segmentasi scene, caption + embedding per scene, alignment transkrip, index multi-vektor, dan query yang menjawab dengan timestamp (start, end) plus pratinjau frame. Capstone-nya adalah meng-ingest 100 jam, mencapai benchmark publik, dan mengukur halusinasi pada pertanyaan penghitungan dan aksi.

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (UI)
**Prerequisites:** Phase 4 (CV), Phase 6 (speech), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Phases exercised:** P4 · P6 · P7 · P11 · P12 · P17
**Time:** 30 hours

## Masalahnya

QA video berdurasi panjang adalah masalah multimodal paling rakus bandwidth pada skala 2026. Gemini 2.5 Pro bisa membaca video 2 jam secara native, tetapi meng-ingest 100 jam video ke dalam korpus yang bisa ditanya tetap membutuhkan index tingkat scene. Bentuk produksinya menggabungkan segmentasi scene (TransNetV2 atau PySceneDetect), captioning per scene dengan VLM (Gemini 2.5, Qwen3-VL-Max, atau Molmo 2), alignment transkrip (Whisper-v3-turbo dengan timestamp kata), dan index multi-vektor yang menyimpan caption, embedding frame, dan transkrip berdampingan. Pipeline query menjawab dengan timestamp (start, end) plus pratinjau frame.

Benchmark-nya publik (ActivityNet-QA, NeXT-GQA) ditambah set kustom 100-query milikmu sendiri. Halusinasi pada pertanyaan tipe-penghitungan dan tipe-aksi adalah kelas kegagalan yang terkenal sulit; capstone ini secara eksplisit mengukurnya.

## Konsepnya

Tiga pipeline berjalan paralel saat ingest. **Scene segmentation** memotong video menjadi scene. **VLM captioning** menghasilkan caption per scene dan embedding frame dari sebuah keyframe. **ASR alignment** menghasilkan timestamp tingkat kata. Tiga aliran itu di-join berdasarkan (scene_id, rentang waktu). Setiap scene mendapat tiga tipe vektor di index multi-vektor (Qdrant): embedding caption, embedding keyframe, embedding transkrip.

Saat query, pertanyaan bahasa-alamiah ditembakkan ke ketiga vektor itu; hasil digabung dengan RRF; adapter temporal grounding (gaya TimeLens) memperhalus jendela (start, end) di dalam scene teratas. Synthesizer VLM (Gemini 2.5 Pro atau Qwen3-VL-Max) mengambil query + scene teratas + frame yang dipotong dan menjawab dengan timestamp yang dikutip serta pratinjau frame.

Pengukuran halusinasi itu penting. Pertanyaan penghitungan ("berapa orang masuk ruangan?") dan tipe-aksi ("apakah koki menuang sebelum mengaduk?") terkenal tidak andal. Laporkan akurasi terpisah dari pertanyaan deskriptif.

## Arsitektur

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## Stack

- Segmentasi scene: TransNetV2 (state-of-the-art 2024-26) atau PySceneDetect
- ASR: Whisper-v3-turbo via faster-whisper dengan timestamp kata
- Captioner + penjawab VLM: Gemini 2.5 Pro atau Qwen3-VL-Max atau Molmo 2
- Temporal grounding: adapter yang dilatih pada TimeLens-100K atau VideoITG
- Index: Qdrant dengan dukungan multi-vektor (caption / frame / transkrip)
- UI: Next.js 15 dengan video player HTML5 dan thumbnail scene
- Eval: ActivityNet-QA, NeXT-GQA, set 100-pertanyaan kustom berlabel manual
- Benchmark halusinasi: subset penghitungan dan tipe-aksi dengan label manual

```figure
cf-scene-index
```

## Bangun Sendiri

1. **Walker ingest.** Menerima URL YouTube atau MP4 lokal. Turunkan resolusi ke 720p jika perlu. Simpan `{video_id, file_path}`.

2. **Segmentasi scene.** Jalankan TransNetV2 atau PySceneDetect untuk menghasilkan `[{scene_id, start_ms, end_ms, keyframe_path}]`. Target 100 jam: ~6k-8k scene.

3. **Pass ASR.** Jalankan Whisper-v3-turbo pada audio; ekspor timestamp tingkat kata; pecah menjadi irisan transkrip per scene.

4. **Captioning VLM.** Per scene, panggil Gemini 2.5 Pro (atau Qwen3-VL-Max) dengan keyframe dan template caption pendek. Hasilkan caption + embedding frame.

5. **Index multi-vektor.** Collection Qdrant dengan tiga vektor bernama. Payload: `{video_id, scene_id, start_ms, end_ms, keyframe_url}`.

6. **Query.** Pertanyaan bahasa-alamiah menembakkan tiga query dense; gabungkan dengan reciprocal rank fusion; top-k=5 scene.

7. **Temporal grounding.** Jalankan adapter bergaya TimeLens pada scene teratas untuk memperhalus jendela (start, end) di dalam scene.

8. **Synth VLM.** Panggil Gemini 2.5 Pro dengan query + 3 klip scene teratas (sebagai gambar atau klip pendek) + transkrip. Wajibkan sitasi `(video_id, start_ms, end_ms)`.

9. **Eval.** Jalankan ActivityNet-QA dan NeXT-GQA. Bangun set kustom 100-query. Laporkan akurasi keseluruhan + rincian per kelas (penghitungan, aksi, deskriptif).

## Gunakan

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## Rilis

`outputs/skill-video-qa.md` adalah deliverable-nya. Diberi URL YouTube atau video yang diunggah, pipeline mengindeks scene dan menjawab pertanyaan dengan sitasi bertimestamp.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Temporal grounding IoU | Intersection-over-union on held-out grounding set |
| 20 | QA accuracy | NeXT-GQA and custom 100-query |
| 20 | Ingest throughput | Hours of video per dollar spent |
| 20 | UI and citation UX | Timestamp links, thumbnail strip, jump-to-frame |
| 15 | Hallucination rate | Counting and action-type accuracy separately |
| **100** | | |

## Latihan

1. Tukar Gemini 2.5 Pro dengan Qwen3-VL-Max pada pass captioning. Laporkan delta kualitas caption pada sampel 50 scene yang dinilai manusia.

2. Turunkan embedding frame per scene menjadi satu vektor gabungan alih-alih multi-vektor. Ukur regresi retrieval-nya.

3. Buat mode "counting strict": synthesizer mengekstrak setiap instance yang dihitung beserta timestamp dan pengguna mengklik untuk memverifikasi. Ukur apakah verifikasi-pengguna mengurangi halusinasi.

4. Benchmark biaya ingest: jam-video-per-dolar di tiga pilihan VLM. Pilih titik optimalnya.

5. Tambahkan transkrip berdiarization pembicara: jalankan pyannote speaker diarization pada audio dan embed transkrip per pembicara. Demonstrasikan query "apa yang dikatakan Alice tentang X?".

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Scene segmentation | "Shot detection" | Cutting video into scenes at shot boundaries |
| Multi-vector index | "Caption + frame + transcript" | Qdrant collection with named vectors per representation |
| Temporal grounding | "When exactly did it happen" | Refining the (start, end) window for a query answer |
| Frame embedding | "Visual representation" | A vector embedding of a keyframe; used for scene-visual similarity |
| RRF fusion | "Reciprocal rank fusion" | Merge strategy across multiple ranked lists; a classic hybrid-retrieval trick |
| Counting hallucination | "Miscount" | Known failure mode of VLMs on "how many X" questions |
| ActivityNet-QA | "Video-QA benchmark" | Long-form video QA accuracy benchmark |

## Bacaan Lanjutan

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — open VLM checkpoints
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — temporal grounding at scale
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — the hosted reference
- [VideoDB](https://videodb.io) — CRUD-for-video API reference
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — commercial reference
- [TransNetV2](https://github.com/soCzech/TransNetV2) — scene segmentation model
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — classic open alternative
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — reference eval benchmark
