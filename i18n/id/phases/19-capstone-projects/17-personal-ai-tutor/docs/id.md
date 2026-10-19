# Capstone 17 — Tutor AI Pribadi (Adaptif, Multimodal, dengan Memori)

> Khanmigo (Khan Academy), Duolingo Max, Google LearnLM / Gemini for Education, Quizlet Q-Chat, dan Synthesis Tutor semuanya mengirimkan tutoring multimodal adaptif berskala besar pada 2026. Bentuk yang sama adalah kebijakan Socratic (jangan pernah sekadar menumpahkan jawaban), model pembelajar yang diperbarui setiap interaksi (gaya Bayesian knowledge tracing), input suara + teks + foto-matematika, retrieval kurikulum graph, penjadwalan spaced-repetition, dan filter keselamatan ketat untuk konten yang sesuai usia. Capstone-nya adalah mengirimkan sebuah tutor spesifik-subjek (aljabar K-12 atau Python dasar), menjalankan studi efikasi dua minggu dengan 10 pembelajar, dan lulus audit keselamatan konten.

**Type:** Capstone
**Languages:** Python (backend, learner model), TypeScript (web app), SQL (curriculum graph via Postgres + Neo4j)
**Prerequisites:** Phase 5 (NLP), Phase 6 (speech), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 14 (agents), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**Time:** 30 hours

## Masalahnya

Tutoring adaptif dulu adalah ceruk riset ed-tech. Menjelang 2026 ia menjadi produk konsumen. Khanmigo dideploy di sebagian besar distrik sekolah AS. Duolingo Max mencapai puluhan juta MAU. LearnLM / Gemini for Education dari Google menggerakkan tutoring di Google Classroom. Quizlet Q-Chat berdampingan dengan flashcard. Synthesis Tutor menjadi viral sebagai tutor-anak-penaksir. Elemen-elemen yang sama: input multimodal (ketik, bicara, fotografi persamaan), pedagogi Socratic (tanya dulu, jelaskan kemudian), model pembelajar yang diperbarui setiap interaksi, dan keselamatan sesuai-usia yang ketat.

Kamu akan membangun salah satu dari ini untuk sebuah kohort tertentu. Ukurannya adalah sebuah studi efikasi yang nyata: skor pre-test dan post-test selama dua minggu dengan 10 pembelajar. Loop suaranya harus terasa natural (sub-stack capstone 03). Memorinya harus menghormati privasi. Filter keselamatan-nya harus lulus red-team yang sadar-COPPA untuk K-12.

## Konsepnya

Empat komponen. **Tutor policy** adalah sebuah loop Socratic: saat pembelajar meminta jawaban, kebijakan itu mengajukan pertanyaan pengarah; saat mereka benar, ia berpindah ke konsep berikutnya; saat mereka buntu, ia menawarkan petunjuk ber-scaffold. **Learner model** adalah Bayesian knowledge tracing (atau varian sederhana) yang memperbarui probabilitas mastery per node kurikulum setiap interaksi. **Curriculum graph** adalah sebuah Neo4j berisi konsep dengan edge prasyarat; kebijakan itu berjalan di graph untuk memilih konsep berikutnya. **Memory** adalah sebuah store episodik + semantik (gaya agentmemory) yang menyimpan interaksi, kesalahan, dan preferensi lampau.

UX-nya multimodal. Input teks untuk jawaban ketikan. Input suara via LiveKit + Whisper (gunakan ulang capstone 03). Input foto untuk masalah matematika via dots.ocr atau PaliGemma 2. Output suara via Cartesia Sonic-2. Keselamatan memakai Llama Guard 4 plus sebuah filter sesuai-usia (memblokir konten dewasa, kekerasan, self-harm) dan kebijakan retensi memori yang sadar-COPPA.

Studi efikasi adalah deliverable-nya. 10 pembelajar, pre-test dan post-test, dua minggu. Laporkan delta learning gain dan confidence interval. Bandingkan dengan sebuah baseline non-adaptif (konten yang sama disampaikan secara linier tanpa kebijakan tutor).

## Arsitektur

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## Stack

- Pilihan subjek: aljabar K-12 atau Python dasar (pilih satu untuk kedalaman)
- Kebijakan tutor: LangGraph di atas Claude Sonnet 4.7 (dengan prompt caching)
- Model pembelajar: Bayesian knowledge tracing (klasik) atau FSRS untuk spacing
- Curriculum graph: Neo4j berisi konsep + edge prasyarat + konten OER
- Memori: store vektor persisten bergaya agentmemory + episodik + semantik
- Suara: LiveKit Agents 1.0 + Cartesia Sonic-2 (gunakan ulang sub-stack capstone 03)
- Foto matematika: dots.ocr atau PaliGemma 2 untuk pengenalan persamaan
- Keselamatan: Llama Guard 4 + filter sesuai-usia kustom
- Eval: pembuatan pertanyaan level Bloom, harness tes pre/post, tooling studi efikasi

```figure
cf-tutor-loop
```

## Bangun Sendiri

1. **Curriculum graph.** Bangun sebuah Neo4j dengan 50-150 node konsep (mis., aljabar K-12 dari "garis bilangan" sampai "rumus kuadrat") beserta edge prasyaratnya. Lampirkan konten OER per node (Open Textbook, OpenStax).

2. **Model pembelajar.** Inisialisasi Bayesian knowledge tracing dengan prior: guess, slip, learn-rate. Perbarui mastery per konsep setiap interaksi. Persistensikan per pembelajar.

3. **Kebijakan tutor.** LangGraph dengan node: `read_signal` (apakah jawaban pembelajar benar / sebagian / buntu?), `select_concept` (jelajahi curriculum graph memilih konsep berprioritas tertinggi), `scaffold` (prompt Socratic), `update_mastery`.

4. **Memori.** Setiap interaksi ditulis ke store episodik. Kesalahan dan preferensi dipromosikan ke memori semantik. Kebijakan retensi sadar-COPPA: hapus otomatis setelah 1 tahun, dapat diakses orang tua.

5. **Jalur suara.** Worker LiveKit Agents yang tertaut ke kebijakan tutor. ASR via Whisper-v3-turbo. TTS via Cartesia Sonic-2. Mendukung barge-in (gunakan ulang mekanika capstone 03).

6. **Jalur foto-matematika.** Unggah atau jepret gambar; jalankan dots.ocr atau PaliGemma 2 untuk mengenali persamaan itu; berikan ke tutor sebagai input terstruktur.

7. **Keselamatan.** Setiap output model melewati Llama Guard 4 + sebuah filter sesuai-usia (memblokir self-harm, konten dewasa, kekerasan). Akses memori di-scope berdasarkan ID pembelajar; permukaan akses orang tua untuk penghapusan.

8. **Studi efikasi.** 10 pembelajar, pre-test (baseline 30-pertanda standar), dua minggu interaksi tutor (3 sesi/minggu), post-test. Bandingkan dengan kohort baseline non-adaptif berisi 10 pembelajar pada konten yang sama.

9. **Laporan progres mingguan.** Per pembelajar, hasilkan otomatis ringkasan PDF berisi topik yang dieksplorasi, trajektori mastery, dan langkah berikutnya yang direkomendasikan.

## Gunakan

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## Rilis

`outputs/skill-ai-tutor.md` adalah deliverable-nya. Sebuah tutor adaptif spesifik-subjek dengan input multimodal, model pembelajar, memori, keselamatan, dan efikasi yang terukur.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Learning gain delta | Pre/post-test delta in a 10-learner two-week study |
| 20 | Socratic fidelity | Rubric score on transcript samples |
| 20 | Multimodal UX | Voice + photo + text coherence end to end |
| 20 | Safety + privacy posture | Llama Guard 4 pass rate + COPPA-aware retention |
| 15 | Curriculum breadth and graph quality | Concept coverage + prerequisite graph consistency |
| **100** | | |

## Latihan

1. Jalankan studi efikasi dengan dan tanpa model pembelajar adaptif (urutan konsep acak). Laporkan delta-nya. Ekspektasinya adaptif menang, tetapi besarnya adalah angka yang menarik.

2. Tambahkan sebuah probe multimodal: pertanyaan konsep yang sama disampaikan sebagai teks, suara, dan foto. Ukur apakah pembelajar konvergen lebih cepat dengan modality yang mereka sukai.

3. Bangun dashboard orang tua: topik yang dipraktikkan, trajektori mastery, konsep yang akan datang, event keselamatan (setiap hit guardrail). Selaras dengan COPPA.

4. Tambahkan mode pergantian bahasa: tutor menerima input bahasa Spanyol dan mengajar dalam bahasa Spanyol. Ukur cakupan X-Guard.

5. Uji tekanan privasi memori: verifikasi bahwa pembelajar A tidak dapat melihat data pembelajar B bahkan melalui serangan re-ingest klip suara. Catat upaya akses itu dan beri alert.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Socratic policy | "Ask, do not dump" | Tutor asks a leading question rather than giving the answer |
| Bayesian knowledge tracing | "BKT" | Classic learner-model equations for mastery probability per concept |
| FSRS | "Free Spaced Repetition Scheduler" | 2024 spaced-repetition scheduler, better than SM-2 |
| Curriculum graph | "Concept DAG" | Neo4j of concepts with prerequisite edges |
| Episodic memory | "Per-interaction log" | Every interaction stored for later retrieval |
| Semantic memory | "Learned pattern store" | Compacted mistakes and preferences promoted from episodic |
| COPPA | "Kids privacy law" | US law restricting data collection from children under 13 |

## Bacaan Lanjutan

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — reference consumer K-12 tutor
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — reference language-learning tutor
- [Google LearnLM / Gemini for Education](https://blog.google/products-and-platforms/products/education/google-learnlm-gemini-generative-ai/) — hosted reference model
- [Quizlet Q-Chat](https://quizlet.com) — alternate reference
- [Synthesis Tutor](https://www.synthesis.com) — startup reference
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — spaced-repetition scheduler
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — learner-model classic
- [LiveKit Agents](https://github.com/livekit/agents) — voice stack
