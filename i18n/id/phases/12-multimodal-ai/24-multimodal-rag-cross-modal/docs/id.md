# Multimodal RAG dan Cross-Modal Retrieval

> RAG dokumen vision-native hanyalah satu irisan. RAG multimodal produksi melangkah lebih jauh — me-retrieve di lintas teks, gambar, audio, dan video untuk alur kerja seperti perencanaan perjalanan ("cari brunch vegan yang tenang dengan cahaya alami"), triase medis ("cedera apa yang cocok dengan foto ini + catatan-catatan ini"), e-commerce ("outfit mirip selfie ini, dengan ukuranku"), dan layanan lapangan ("diagnosis suara mesin ini plus foto komponennya"). Tiga survei 2025 — Abootorabi et al., Mei et al., Zhao et al. — mengodifikasi sub-masalahnya: retrieval cross-modal, fusion retrieval, generation grounding, evaluasi multimodal. Pelajaran ini membaca survei-survei tersebut dan merancang sebuah pipeline produksi.

**Type:** Build
**Languages:** Python (stdlib, cross-modal retriever with fusion + grounded generator)
**Prerequisites:** Phase 12 · 23 (ColPali), Phase 11 (RAG basics)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Merancang retrieval cross-modal: teks → gambar, gambar → teks, audio → video, dll.
- Membandingkan tiga strategi fusion: score fusion, attention-based fusion, MoE fusion.
- Menjelaskan generation grounding: seperti apa "cantumkan sumbermu" ketika sumbernya merupakan campuran modalitas.
- Menyebutkan tiga survei RAG multimodal kanonis tahun 2025 dan taksonomi sub-masalahnya.

## Masalahnya

RAG modalitas-tunggal adalah pola yang sudah selesai: embed query, embed chunk, retrieve, masukkan ke LLM. RAG multimodal menuntut:

1. Beberapa retrieval head (setiap modalitas membutuhkan embedding di ruang yang kompatibel).
2. Fusion hasil retrieval di lintas modalitas.
3. Generation grounding yang mengutip sumber di lintas modalitas.
4. Metrik evaluasi yang mencakup sinyal cross-modal.

Ketiga survei 2025 tiba pada taksonomi yang sama.

## Konsepnya

### Cross-modal retrieval

Me-retrieve dokumen modalitas B diberi query modalitas A. Tiga pola:

1. Ruang embedding bersama. CLIP dan CLAP menghasilkan embedding teks + gambar / teks + audio di ruang bersama. Cosine similarity di lintas modalitas bekerja langsung. Dibatasi pada pasangan yang dilatih CLIP.

2. Encoder per-modalitas + terjemahan. Encoder teks + encoder gambar + modul penerjemah kecil yang memetakan di antara ruang. Sen2Sen oleh Gupta et al. dan desain-desain 2024 lainnya. Fleksibel tetapi menambah kompleksitas.

3. VLM sebagai encoder. Gunakan hidden state VLM sebagai representasi retrieval. Modalitas apa pun yang didukung VLM berfungsi. Kualitas lebih tinggi, lebih mahal.

Pilihan: CLIP / SigLIP 2 untuk teks+gambar; CLAP untuk teks+audio; hidden-state-VLM untuk cross-modal pada kualitas frontier.

### Strategi fusion

Kamu telah me-retrieve 10 hasil: 5 gambar, 3 potongan teks, 2 klip audio. Bagaimana kamu menggabungkannya?

Score fusion (termurah). Setiap modalitas punya retriever-nya sendiri, masing-masing mengembalikan skor. Normalisasi skor dalam-modalitas lalu jumlahkan. Sederhana, sering berhasil.

Attention-based fusion. Gabungkan semua item yang di-retrieve, biarkan jaringan attention kecil memberi bobot padanya. Butuh pelatihan.

MoE fusion. Jaringan gating mengarahkan ke expert khusus-modalitas. Tipe query berbeda diarahkan berbeda — sebuah pertanyaan visual memberi bobot lebih tinggi pada gambar.

Default produksi: score fusion dengan sedikit bias ke modalitas dominan query-nya. Naik ke MoE jika A/B menunjukkan kemenangan jelas di domainmu.

### Generation grounding

LLM harus mengutip item retrieval mana yang mendorong setiap klaim. Untuk multimodal:

- Sumber teks: sitasi standar `[1]`.
- Sumber gambar: `[img 3]` dengan caption singkat.
- Audio: `[audio 2 at 0:34]`.

Latih generator dengan data yang sadar-grounding: setiap klaim dalam target training ditandai dengan indeks sumber. Saat inference, model secara alami mengeluarkan sitasi.

### Survei-survei 2025

Abootorabi et al. (arXiv:2502.08826, "Ask in Any Modality"): taksonomi untuk RAG multimodal. Mencakup retrieval, fusion, generasi. Cakupan paling luas.

Mei et al. (arXiv:2504.08748, "A Survey of Multimodal RAG"): fokus pada benchmark sub-tugas dan mode kegagalan. Berguna untuk perancangan evaluasi.

Zhao et al. (arXiv:2503.18016): survei berfokus-visi. Kuat pada karya keluarga ColPali.

Membaca ketiganya memberimu state of the art per musim semi 2025. Sebagian besar sub-masalah masih terbuka.

### MuRAG — paper fondasionalnya

MuRAG (Chen et al., 2022) adalah RAG multimodal pertama. Me-retrieve gambar + teks dari sebuah KB multimodal, menghasilkan jawaban. Menunjukkan kelayakan sebelum gelombang VLM. Sistem-sistem modern (REACT, VisRAG, M3DocRAG) dibangun di atasnya.

### Contoh produksi perencana-perjalanan

Query: "cari brunch vegan yang tenang dengan cahaya alami."

Pipeline:

1. Dekomposisi query. "tenang" → kata kunci audio/ulasan; "brunch vegan" → item menu; "cahaya alami" → fitur gambar.
2. Retrieve per modalitas:
   - Retrieval teks pada ulasan: "brunch vegan, suasana tenang."
   - Retrieval gambar pada foto restoran: "cahaya alami, terang."
   - Retrieval audio pada klip suara-lingkungan: "desibel rendah, tanpa musik."
3. Fuse skor. Setiap restoran punya skor komposit.
4. Restoran top-k → generator VLM dengan semua bukti → jawaban dengan sitasi.

Ini jauh melampaui text-RAG. Setiap modalitas menambahkan sinyal yang terlewat oleh teks semata.

### RAG multimodal agentic

Multi-hop: jika retrieval pertama tidak mengembalikan jawaban berkeyakinan-tinggi, LLM memformulasi-ulang dan me-retrieve lagi. Pola RAG agentic dari Phase 14 berlaku di sini. Contoh:

- Retrieve top-10 awal → LLM bertanya "terlalu bising, filter <40 dB" → retrieve ulang.
- Retrieve gambar → LLM melihat satu punya menu → retrieve teks menu → jawab.

Menambah kompleksitas tetapi menangani query yang tidak bisa ditangani oleh retrieval single-shot.

### Evaluasi

Evaluasi cross-modal masih belum matang. Proxy yang umum:

- Recall@k per modalitas.
- Akurasi top-k hasil-fusion.
- Kepuasan end-to-end yang dinilai manusia.
- Spesifik-tugas (reservasi selesai, pembelian dilakukan).

Tidak ada benchmark standar yang mencakup semua modalitas. Sebagian besar paper mengevaluasi pada tugas spesifik-domain.

```figure
contrastive-matrix
```

## Gunakan

`code/main.py`:

- Tiga retriever mock (teks, gambar, audio) yang beroperasi pada korpus bersama restoran.
- Score fusion yang menggabungkan skor modalitas dengan bobot yang dapat dikonfigurasi.
- Stub generator yang mengeluarkan jawaban akhir dengan sitasi.
- Loop agentic sederhana yang memformulasi-ulang query jika keyakinan rendah.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-multimodal-rag-designer.md`. Diberi spesifikasi produk dengan alur query multimodal, merancang retriever, fusion, generator, dan evaluasi.

## Latihan

1. Usulkan RAG multimodal triase-medis: query = foto cedera + gejala teks. Modalitas apa me-retrieve dari KB apa?

2. Score fusion adalah jumlah berbobot sederhana. Mode kegagalan apa yang dimilikinya yang dihindari oleh MoE fusion?

3. Baca taksonomi Abootorabi et al. (Bagian 3). Apa tiga sub-masalah kanonis dan bagaimana ia memetakan ke produk pilihanmu?

4. Rancang spesifikasi evaluasi untuk RAG multimodal perencana-perjalanan. Metrik apa yang mencakup recall gambar, recall audio, dan kebenaran komposit?

5. RAG agentic multi-hop punya pajak latensi per round-trip. Pada kesulitan query berapa peningkatan akurasi membenarkan latensinya?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Cross-modal retrieval | "Query one modality, retrieve another" | Text query retrieves images; image query retrieves text; requires a shared space or translator |
| Score fusion | "Combine scores" | Weighted sum of per-modality retrieval scores; simplest fusion |
| MoE fusion | "Modality-routed experts" | Gating network picks which modality's scores to trust per query |
| Grounded generation | "Cite your sources" | Each claim in the answer tagged with the source index |
| MuRAG | "First multimodal RAG" | 2022 paper that established the multimodal RAG pattern |
| Agentic multi-hop | "Reformulate and retry" | LLM re-queries retrievers when first-pass confidence is low |

## Bacaan Lanjutan

- [Abootorabi et al. — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei et al. — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao et al. — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen et al. — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu et al. — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)
