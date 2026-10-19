# LLaVA dan Penyetelan Instruksi Visual

> LLaVA(April 2023) adalah arsitektur multimoda yang paling banyak ditiru di planet ini. Ini menggantikan Q-Former BLIP-2 dengan MLP 2 lapis, menggantikan Flamingo gated cross-attention dengan penggabungan token yang naif, dan dilatih pada 158 ribu putaran instruksi visual yang dihasilkan oleh GPT-4 dari teks hanya teks. Setiap praktisi yang membuat VLM antara tahun 2023 dan 2026 membuat beberapa varian LLaVA.LLaVA-1.5 menambahkan AnyRes.LLaVA-Resolusi NeXT terbentur.LLaVA-OneVision menyatukan gambar, multi-gambar, dan video dalam satu resep. Pelajaran ini membaca resep, mengimplementasikan proyektor, dan menjelaskan mengapa "yang lebih sederhana menang".

**Type:** Build
**Languages:** Python (stdlib, projector + instruction-template builder)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — instruction tuning)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Buat proyektor MLP 2 lapis yang memetakan ViT patch embedding(dim 1024) ke LLM embedding dim (dim 4096).
- Berjalanlah LLaVA resep dua tahap: (1) penyelarasan proyektor pada 558 ribu pasang teks, (2) visual instruction tuning pada 158k putaran yang dihasilkan GPT-4.
- Membangun sebuah LLaVA-format prompt dengan placeholder token gambar, prompt sistem, dan giliran pengguna/asisten.
- Jelaskan mengapa komunitas berpindah dari Q-Former ke MLP meskipun Q-Former memenangkan token-budget.

## Masalahnya

Q-Former BLIP-2 (Pelajaran 12.03) mengompresi gambar menjadi 32 token. Bersih, efisien, bagus untuk benchmark. Tapi ada dua masalah.

Pertama, Q-Former bisa dilatih tetapi bisa loss bukanlah tugas akhir. Tahap 1 melatih ITC+ITM+ITG. Tahap 2 melatih LM loss. Kueri mempelajari beberapa representasi perantara yang kemudian harus didekodekan oleh LLM. Informasi hilang dalam kemacetan.

Kedua, Q-Former membutuhkan 188 juta parameter, dan pada LLaVA skala 2023 kamu harus merancangnya bersama dengan target LLM kamu. Ubah LLM, latih kembali Q-Former. Ubah visi encoder, berlatih kembali. Setiap kombinasi adalah proyek R&D yang terpisah.

Itu LLaVA jawabannya memalukan dalam kesederhanaannya: ambillah ViT 576 patch token, lewati masing-masing melalui MLP 2 lapis (`1024 → 4096 → 4096`), dan membuang semua 576 ke dalam urutan input LLM. Tidak ada hambatan. Tidak ada pelatihan awal tahap 1 dengan tujuan aneh. Latih saja MLP di LM langsung loss.

Dari mana datanya berasal?LLaVA Wawasan kedua: gunakan GPT-4 (hanya teks) untuk menghasilkan data instruksi. Masukkan GPT-4 keterangan COCO dan data kotak pembatas untuk sebuah gambar, minta gambar tersebut menghasilkan percakapan, deskripsi, dan pertanyaan penalaran yang kompleks. 158k instruksi-respons berubah secara gratis. Tidak ada anotasi manusia.

Hasilnya: VLM yang berjalan pada 8 A100 selama satu hari, mengalahkan Flamingo di MMMU, dan mengirimkan pos pemeriksaan terbuka yang dapat diperluas oleh komunitas. Pada akhir tahun 2023, ia telah menghasilkan 50+ cabang.

## Konsepnya

### Arsitektur

LLaVA-1,5 pada 13B:
- Penglihatan encoder: CLIP ViT-L/14 @ 336 (dibekukan selama tahap 1, opsional dicairkan tahap 2).
- Proyektor: MLP 2 lapis dengan aktivasi GELU, `1024 → 4096 → 4096`.
- LLM: Vicuna-13B (kemudian Llama-3.1-8B).

Meneruskan prompt gambar + teks:

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

Gambar tersebut menempati 576 token konteks LLM. Pada konteks 2048, tersisa 1.472 token untuk teks. Pada konteks 32k, ini adalah kesalahan pembulatan.

### Tahap 1: penyelarasan proyektor

Membekukan ViT. Bekukan LLM. Latih hanya MLP 2 lapis. Dataset: 558 ribu pasangan keterangan gambar (LAION-CC-SBU).Loss: pemodelan bahasa pada keterangan, dikondisikan pada token gambar yang diproyeksikan.

Dalam satu epoch di batch 128 hal ini dilakukan dalam beberapa jam. Proyektor belajar memetakan ViT-spasi ke ruang LLM. Tidak ada pengawasan khusus tugas.

### Tahap 2: visual instruction tuning

Cairkan proyektor (masih bisa dilatih). Cairkan LLM (biasanya seluruhnya, terkadang LoRA). Berlatihlah pada 158k putaran instruksi visual.

Data instruksi adalah triknya. Liu dkk. dihasilkan oleh:
1. Ambil gambar COCO.
2. Ekstrak deskripsi teks (5 keterangan manusia + daftar kotak pembatas).
3. Kirim ke GPT-4 dengan tiga templat cepat:
   - Percakapan: "Buat dialog bolak-balik antara pengguna dan asisten tentang gambar ini."
   - Deskripsi mendetail: "Berikan deskripsi gambar yang kaya dan mendetail."
   - Penalaran kompleks: "Ajukan pertanyaan yang memerlukan penalaran tentang gambar tersebut, lalu jawablah."
4. Parsing output GPT-4 menjadi pasangan (instruksi, respons).

Semua ini tidak menyentuh gambar secara langsung — hanya deskripsi teks. GPT-4 berhalusinasi konten gambar yang masuk akal. Sedikit berisik, tapi berhasil: 158 ribu putaran sudah cukup untuk membuka dialog.

### Mengapa komunitas meniru ini

- Tidak ada tahap-1 khusus loss untuk menyetel. LM loss selama.
- Proyektor berlatih dalam hitungan jam, bukan hari.
- LLM dapat ditukar (LLaVA-Llama2, LLaVA-mistral, LLaVA-Llama3) dengan melatih ulang proyektor saja.
- Pipeline data instruksi visual menggunakan GPT-4 dan murah untuk dibuat ulang untuk domain baru.

### LLaVA-1,5 dan LLaVA-Berikutnya

LLaVA-1.5 (Oktober 2023) ditambahkan:
- Data tugas akademik (VQA, OKVQA, RefCOCO) dicampur ke dalamnya instruction tuning.
- Prompt sistem yang lebih baik.
- 2048 → konteks 32k.

LLaVA-NeXT (Januari 2024) menambahkan:
- AnyRes: membagi gambar beresolusi tinggi menjadi kisi 2x2 atau 1x3 dengan potongan 336x336, ditambah satu gambar mini global beresolusi rendah. Setiap hasil panen menjadi 576 token; total sekitar 2880 token visual per gambar. Tugas OCR dan grafik melonjak.
- Campuran data instruksi yang lebih baik dengan ShareGPT4V (teks GPT-4V berkualitas tinggi).
- LLM basis yang lebih kuat (Mistral-7B, Yi-34B).

### LLaVA-Satu Visi

Pelajaran 12.08 membahas OneVision secara mendalam. Versi pendek: proyektor yang sama, tetapi dilatih dengan kurikulum yang mencakup gambar tunggal, multi-gambar, dan video dalam satu model dengan anggaran token visual bersama.

### Perbandingannya dengan Q-Former

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| Visual tokens per image | 32 | 576 (base) or 2880 (AnyRes) |
| Trainable params | 188M + LM | 40M + LM |
| Stage 1 loss | ITC+ITM+ITG | LM only |
| LLM drop-in | Requires retrain | Swap with minimal retrain |
| Multi-image | Awkward | Natural (concat) |
| Video | Awkward | Natural (per-frame concat) |
| Token budget | Small | Large |

MLP unggul dalam kesederhanaan dan fleksibilitas token. Q-Mantan menang dengan anggaran token. Pada akhir tahun 2023, anggaran token tidak lagi menjadi kendala yang mengikat (konteks LLM tumbuh menjadi 32k-128k+) dan kesederhanaan mendominasi.

### Format cepat

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` adalah token pengganti. Sebelum tokenisasi, diganti dengan 576 token visual (atau 2880 dengan AnyRes). Tokenizer melihat urutan yang sedikit lebih panjang daripada saat dilatih, namun LLM menangani masukan baru karena tahap 1 mengajarkannya.

### Parameter ekonomi

LLaVA-1.5-7B rincian:
- CLIP ViT-L/14 @ 336: 303M (dibekukan tahap 1, sering dicairkan tahap 2).
- Proyektor (2x linier): ~22 juta dapat dilatih.
- Llama-7B: 7B.
- Total: 7,3 miliar parameter. Dapat dilatih selama tahap 2: proyektor penuh 7B + 22M.

Biaya pelatihan untuk tahap 2: ~20 jam pada 8xA100. Ini adalah nomor kuncinya — satu hari, satu node, dapat direproduksi. Itulah sebabnya LLaVA menyebar.

```figure
mm-llava-projector
```

## Gunakan

`code/main.py` mengimplementasikan:

1. Proyektor MLP 2 lapis (redupkan 16 → 32 → 32 untuk skala mainan) dengan Python murni.
2. Alur pembuatan prompt: prompt sistem +`<image>` diganti dengan N token yang diproyeksikan + giliran pengguna + placeholder pembuatan asisten.
3. Sebuah visualisator untuk tampilan blok visual 576 token dalam konteks LLM (persentase konteks 2k / 32k / 128k yang dikonsumsi).

## Rilis

Pelajaran ini menghasilkan `outputs/skill-llava-vibes-eval.md`. Diberikan sebuah LLaVA-pos pemeriksaan keluarga, ia menjalankan rangkaian evaluasi getaran 10 cepat (3 teks, 3 VQA, 2 alasan, 2 penolakan) dan melaporkan kartu skor yang dapat dibaca manusia. Bukan patokan; tes asap untuk memastikan proyektor dan LLM terhubung dengan baik.

## Latihan

1. Hitung jumlah parameter yang dapat dilatih untuk proyektor MLP 2 lapis di `1024 → 4096 → 4096`. Dengan GELU dan bias, berapa persennya LLaVA-13B mewakilinya?

2. Membangun sebuah LLaVA permintaan untuk kasus "penolakan" — gambar tersebut berisi individu pribadi. Tulis tanggapan asisten yang diharapkan. kenapa harus LLaVA menolak ini zero-shot dan data pelatihan apa yang diperlukan untuk memperkuat penolakan tersebut?

3. Baca bagian AnyRes di LLaVA-Blog Berikutnya. Hitung jumlah token visual untuk gambar 1344x672 di AnyRes. Bandingkan dengan token dasar 576 pada 336x336.

4. Itu LLaVA proyektor tahap-1 dilatih dengan LM loss pada keterangan. Apa yang terjadi jika kamu melewatkan tahap 1 dan langsung ke tahap 2 (visual instruction tuning)? Kutip ablasi VLM Prismatik (arXiv:2402.07865) untuk jawabannya.

5. LLaVA-Instruct-150k menggunakan GPT-4 dengan teks COCO untuk menghasilkan instruksi. Untuk domain baru (sinar X medis, citra satelit), jelaskan alur data empat langkah untuk menghasilkan instruksi domain. Apa yang salah pada setiap langkah?

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Projector | "MLP bridge" | 2-layer MLP with GELU mapping ViT dim to LLM dim |
| Image token | "<image> placeholder" | Prompt marker replaced by N projected visual tokens before inference |
| Visual instruction tuning | "LLaVA stage 2" | Training on GPT-4-generated (image, instruction, response) triplets |
| Stage 1 alignment | "Projector pretraining" | Freeze ViT and LLM, train projector with LM loss on captions |
| AnyRes | "Multi-crop tiling" | Split high-res image into a tile grid and concatenate each tile's visual tokens |
| LLaVA-Instruct | "GPT-4-generated" | 158k instruction-response pairs synthesized from COCO captions + GPT-4 |
| Vision encoder freeze | "Backbone locked" | CLIP weights do not update in stage 1, sometimes not in stage 2 either |
| ShareGPT4V | "Better captions" | 1M dense captions generated by GPT-4V, used for higher-quality alignment |
| VQA | "Visual question answering" | Task of answering a free-form question about an image |
| Prismatic VLMs | "Design-space paper" | Karamcheti 2024 ablation systematically testing projector and data choices |

## Bacaan Lebih Lanjut

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485)— itu LLaVA kertas.
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744)—LLaVA-1.5.
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793)— dataset teks padat.
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)— ablasi ruang desain.
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)— gambar tunggal, multi-gambar, video terpadu.
