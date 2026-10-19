# Capstone 04 — QA Dokumen Multimodal (PDF Vision-First, Tabel, Grafik)

> Garis depan document-QA pada 2026 berpindah dari OCR-lalu-teks menuju late interaction vision-first. ColPali, ColQwen2.5, dan ColQwen3-omni memperlakukan setiap halaman PDF sebagai gambar, meng-embed-nya dengan late interaction multi-vektor, dan membiarkan query attend ke patch secara langsung. Pada laporan keuangan 10-K, makalah ilmiah, dan catatan tulisan tangan, pola ini mengalahkan OCR-first dengan margin besar. Bangun pipeline dari ujung ke ujung di atas 10k halaman dan terbitkan perbandingan berdampingan melawan OCR-lalu-teks.

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (viewer UI)
**Prerequisites:** Phase 4 (computer vision), Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Phases exercised:** P4 · P5 · P7 · P11 · P12 · P17
**Time:** 30 hours

## Masalahnya

Perusahaan duduk di atas PDF yang dirusak pipeline OCR: 10-K hasil scan dengan tabel yang miring, makalah ilmiah padat persamaan, grafik yang hanya masuk akal sebagai gambar, dan anotasi tulisan tangan. Memperlakukan ini sebagai teks-dulu berarti kehilangan separuh sinyal. Jawaban 2026 adalah retrieval multi-vektor late-interaction di atas gambar halaman mentah. ColPali (Illuin Tech) memperkenalkannya; ColQwen2.5-v0.2 dan ColQwen3-omni mendorong akurasinya. Di ViDoRe v3, retrieval vision-first memperoleh skor di atas OCR-lalu-teks dengan margin yang berarti — dan jaraknya melebar pada grafik, tabel, dan tulisan tangan.

Trade-off-nya adalah penyimpanan dan latensi. Sebuah embedding ColQwen adalah ~2048 patch vektor per halaman, bukan satu vektor 1024-dim. Penyimpanan mentah membengkak. DocPruner (2026) menghadirkan pruning 50% tanpa penurunan akurasi yang terukur. Kamu akan mengindeks 10k halaman, mengukur nDCG@5 ViDoRe v3, menyajikan jawaban di bawah 2s, dan membandingkannya langsung dengan baseline OCR-lalu-teks.

## Konsepnya

Late interaction berarti setiap token query menilai terhadap setiap token patch, dan skor maksimum per token query dijumlahkan. Kamu mendapat pencocokan fine-grained tanpa perlu satu vektor hasil pooling. Indeks multi-vektor (Vespa, Qdrant multi-vector, atau AstraDB) menyimpan embedding per-patch dan menjalankan MaxSim saat retrieval.

Penjawabnya adalah sebuah vision-language model yang menerima query plus halaman top-k hasil retrieval sebagai gambar lalu menulis jawaban beserta region bukti (bounding box atau referensi halaman). Qwen3-VL-30B, Gemini 2.5 Pro, dan InternVL3 adalah pilihan frontier 2026. Untuk persamaan dan notasi ilmiah, sebuah fallback OCR (Nougat, dots.ocr) disisipkan sebagai kanal teks opsional.

Evaluasinya adalah matriks dua dimensi. Satu sumbu: jenis konten (paragraf teks biasa, tabel padat, grafik batang/garis, catatan tulisan tangan, persamaan). Sumbu lain: pendekatan retrieval (late-interaction vision-first vs OCR-lalu-teks vs hibrida). Setiap sel mendapat nDCG@5 dan akurasi jawaban. Laporan tersebut adalah deliverable-nya.

## Arsitektur

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## Stack

- Rendering halaman: PyMuPDF (fitz) pada 180 DPI, dinormalkan potret
- Model late-interaction: ColQwen2.5-v0.2 atau ColQwen3-omni (tim vidore di Hugging Face)
- Indeks: Vespa dengan field multi-vektor, atau Qdrant multi-vector, atau AstraDB dengan MaxSim
- Pruning: kebijakan DocPruner 2026 (simpan patch varians tinggi, kompresi 50% dengan kerugian akurasi < 0.5%)
- Fallback OCR (persamaan / tabel padat): dots.ocr atau Nougat
- Penjawab VLM: Qwen3-VL-30B self-hosted atau Gemini 2.5 Pro hosted; InternVL3 sebagai fallback
- Evaluasi: benchmark ViDoRe v3, M3DocVQA untuk penalaran multi-halaman
- UI viewer: Next.js 15 dengan overlay canvas untuk region bukti

```figure
ce-late-interaction
```

## Bangun Sendiri

1. **Ingest.** Telusuri korpus 10k halaman PDF dari 10-K, makalah ilmiah, dan dokumen hasil scan. Render tiap halaman menjadi PNG 1536x2048. Persist `{doc_id, page_num, image_path}`.

2. **Embed.** Jalankan ColQwen2.5-v0.2 pada setiap gambar halaman. Bentuk keluaran ~2048 embedding patch dengan dim 128. Terapkan DocPruner untuk menyimpan separuh dengan sinyal tertinggi. Tulis ke field multi-vektor Vespa atau multi-vector Qdrant.

3. **Query.** Untuk setiap query masuk, embed dengan menara query (embedding level token). Jalankan MaxSim terhadap indeks: untuk setiap token query, ambil dot-product maksimum atas embedding patch halaman, lalu jumlahkan. Kembalikan halaman top-k.

4. **Sintesis.** Panggil Qwen3-VL-30B dengan query dan gambar 5 halaman teratas. Prompt: "Answer using only the supplied pages. Cite each claim by (doc_id, page) and name the region (figure, table, paragraph)."

5. **Region bukti.** Pasca-proses jawaban untuk mengekstrak region yang dikutip. Bila VLM mengeluarkan bounding box (Qwen3-VL melakukannya), render sebagai overlay di viewer.

6. **Fallback OCR.** Untuk halaman yang diidentifikasi padat-persamaan (heuristik pada varians gambar), jalankan Nougat atau dots.ocr dan berikan teks OCR sebagai kanal tambahan di samping gambar.

7. **Eval.** Jalankan ViDoRe v3 (nDCG@5 retrieval) dan M3DocVQA (akurasi QA multi-halaman). Jalankan juga pipeline OCR-lalu-teks pada korpus yang sama dengan synthesizer yang sama. Hasilkan matriks jenis-konten × pendekatan.

8. **UI.** Prototipe Streamlit dulu; viewer produksi Next.js 15 dengan overlay region bukti per-halaman.

## Gunakan

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## Rilis

`outputs/skill-doc-qa.md` menjelaskan deliverablenya: sebuah sistem QA dokumen multimodal vision-first yang disetel ke korpus tertentu dan dievaluasi terhadap baseline OCR-lalu-teks di ViDoRe v3.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA accuracy | Benchmark numbers vs OCR-text baseline and published leaderboard |
| 20 | Evidence-region grounding | Fraction of cited regions that actually contain the answer span |
| 20 | Storage and latency engineering | DocPruner compression ratio, index p95, answer p95 |
| 20 | Multi-page reasoning | Accuracy on a hand-labeled 100-question multi-page set |
| 15 | Source-inspection UX | Viewer clarity, overlay fidelity, side-by-side comparison tools |
| **100** | | |

## Latihan

1. Ukur ColQwen2.5-v0.2 versus ColQwen3-omni pada korpus yang sama. Halaman mana yang satu benar dan yang lain meleset? Tambahkan tag "content class" ke indeks untuk routing berdasarkan jenis.

2. Pruning embedding secara agresif (75%, 90%). Temukan tebing kompresinya: titik di mana nDCG@5 ViDoRe jatuh di bawah baseline OCR.

3. Bangun yang hibrida: jalankan OCR-lalu-teks dan ColQwen secara paralel, gabungkan dengan RRF, lalu rerank dengan cross-encoder. Apakah yang hibrida mengalahkan masing-masing secara tunggal? Di mana ia paling membantu?

4. Tukar Qwen3-VL-30B dengan VLM yang lebih kecil (Qwen2.5-VL-7B). Ukur kurva akurasi-per-dolar.

5. Tambahkan dukungan catatan tulisan tangan. Render korpus tulisan tangan, embed dengan ColQwen, ukur retrieval-nya. Bandingkan dengan pipeline OCR tulisan tangan.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Late interaction | "ColPali-style retrieval" | Query tokens score against page patches independently; MaxSim aggregates |
| Multi-vector | "Per-patch embedding" | Each document has many vectors, not one pooled vector |
| MaxSim | "Late-interaction scoring" | For every query token, take max similarity over document vectors; sum |
| DocPruner | "Patch compression" | 2026 pruning that keeps 50% of patches with negligible accuracy loss |
| ViDoRe v3 | "Document-retrieval benchmark" | The 2026 standard for measuring visual-document retrieval |
| Evidence region | "Cited bounding box" | A bbox on the source page that localizes the answer span |
| OCR fallback | "Equation channel" | Text pipeline used alongside vision for equation- or table-heavy pages |

## Bacaan Lanjutan

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — referensi doc retrieval late-interaction
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — makalah metode dasarnya
- [ColQwen family on Hugging Face](https://huggingface.co/vidore) — checkpoint siap produksi
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — baseline RAG multimodal multi-halaman
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html) — stack penyajian referensi
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — indeks alternatif
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — indeks managed alternatif
- [Nougat OCR](https://github.com/facebookresearch/nougat) — fallback OCR yang mampu menangani persamaan
