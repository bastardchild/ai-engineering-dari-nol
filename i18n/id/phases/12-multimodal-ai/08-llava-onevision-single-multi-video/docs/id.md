# LLaVA-OneVision: Gambar-Tunggal, Multi-Gambar, Video dalam Satu Model

> Sebelum LLaVA-OneVision (Li et al., Agustus 2024) dunia VLM-terbuka memiliki garis keturunan terpisah: LLaVA-1.5 untuk gambar tunggal, model multi-gambar seperti Mantis dan VILA, model video seperti Video-LLaVA dan Video-LLaMA. Masing-masing memenangkan benchmarknya sendiri dan gagal di yang lain. LLaVA-OneVision berargumen satu curriculum bisa melatih satu model untuk mendominasi ketiga skenario itu, dan bahwa efek transfer-tugas emergen (keterampilan gambar-tunggal diekspor ke video, penalaran multi-gambar diekspor ke gambar-tunggal) mengalahkan jumlah para spesialis. Resepnya sederhana menipu: token budget visual yang konstan lintas skenario, plus curriculum eksplisit yang bergerak dari gambar-tunggal ke OneVision (multi-gambar) ke video. Pelajaran ini membaca budget, curriculum, dan perilaku emergen itu.

**Type:** Build
**Languages:** Python (stdlib, token budget solver + curriculum planner)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 06 (any-resolution)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Merancang token budget visual yang tetap konstan lintas input gambar-tunggal, multi-gambar, dan video.
- Mengurutkan curriculum training yang mentransfer keterampilan dari gambar-tunggal ke video tanpa catastrophic forgetting.
- Menjelaskan mengapa satu model mengalahkan spesialis pada jumlah parameter yang sama saat curriculum dilakukan dengan benar.
- Menyebutkan tiga kemampuan emergen yang dilaporkan LLaVA-OneVision: penalaran multi-kamera, set-of-mark prompting, agen tangkapan-layar iPhone.

## Masalahnya

Gambar, multi-gambar, dan video masing-masing menekan model secara berbeda.

Gambar-tunggal menginginkan token resolusi tinggi (AnyRes, ~2880 token visual) untuk menangkap OCR dan detail halus. Budget per sampel: satu gambar, 2880 token.

Multi-gambar menginginkan beberapa gambar pada resolusi sedang (~576 token masing-masing) agar penalaran lintas-gambar muat dalam konteks. Budget per sampel: 4-8 gambar, 576 masing-masing, 2300-4600 token.

Video menginginkan banyak frame pada resolusi rendah (~196 token per frame setelah pooling) untuk menangkap dinamika temporal. Budget per sampel: 8-32 frame, 196 masing-masing, 1600-6200 token.

Jika kamu melatih model terpisah, kamu memilih satu budget. Jika kamu melatih satu model, kamu membutuhkan budget yang menskala dengan wajar lintas skenario tanpa meledakkan konteks.

Pra-OneVision, jawaban defaultnya adalah "latih satu skenario, abaikan yang lain." Video-LLaVA memasang video pada model gambar dengan tahap training tambahan. LLaVA-NeXT menambahkan dukungan multi-gambar dengan tiling. Tidak ada yang menangani ketiganya dengan bersih.

## Konsepnya

### Token budget OneVision

LLaVA-OneVision memilih token budget visual terpadu sekitar 3000-4000 token per sampel, dialokasikan berbeda per skenario:

- Gambar tunggal: AnyRes-9 (3x3 tile + thumbnail), setiap tile pada 384 dengan 729 patch, bilinear pooling agresif 2x2 → 182 per tile. Total: 9 * 182 + 182 = 1820 token. Atau AnyRes-4 pada 729-per-tile = 2916 + 729.
- Multi-gambar: setiap gambar pada resolusi sedang (384, tanpa tiling), 729 token tanpa pooling. Budget 6 gambar → 4374 token.
- Video: 32 frame pada resolusi 384 dengan bilinear pool agresif 3x3 → 81 token per frame. Total: 32 * 81 = 2592 token.

Alokasi mempertahankan total token yang kira-kira konstan. LLM tidak pernah melihat batch yang meledakkan konteksnya. Encoder menghasilkan geometri berbeda per skenario, tetapi LLM mengonsumsi budget yang sama.

### Curriculum tiga-tahap

LLaVA-OneVision melatih dalam tiga tahap:

1. SFT gambar-tunggal (tahap SI). Semua data adalah gambar-tunggal-plus-teks. Latih pada input AnyRes resolusi tinggi. Ini mengajarkan persepsi, OCR, dan pemahaman granular. Menggunakan data LLaVA-NeXT plus data gambar-tunggal spesifik OneVision.
2. SFT OneVision (tahap OV). Campur gambar-tunggal + multi-gambar + video (frame yang di-sample merata). Latih pada token budget terpadu. Ini mengajarkan model menangani bentuk batch heterogen. Tanpa reset bobot — lanjut dari tahap SI.
3. Transfer tugas (tahap TT). Lanjutkan dengan campuran tugas target, biasanya lebih berat pada multi-gambar atau video tergantung produk. Fine-tune opsional untuk deployment.

Kritis: urutan curriculum itu penting. Melatih video-dulu atau multi-gambar-dulu menghasilkan kinerja gambar yang lebih buruk daripada gambar-tunggal-dulu, bahkan dengan data yang sama. Paper meng-ablate ini secara eksplisit.

### Mengapa curriculum bekerja

Training gambar-tunggal membangun dasar persepsi. Patch token membawa fitur visual granular; LLM belajar mengintegrasikannya dengan teks. Multi-gambar dan video memperkenalkan tantangan struktural (gambar mana yang mana, apa yang terjadi lebih dulu) yang sulit dipelajari tanpa dasar persepsi yang kuat.

Jika kamu melatih semua skenario dari nol bersama, model underfit pada persepsi (data gambar-tunggal terbatas per batch) dan overfit pada struktur (banyak data multi-gambar / video). Hasilnya: model yang mengikuti pola penalaran lintas-gambar tetapi secara visual dangkal.

Urutan curriculum memberimu kekuatan persepsi dari tahap SI, lalu penalaran kompositional/temporal dari tahap OV, tanpa kehilangan keduanya.

### Keterampilan emergen lintas-skenario

Paper LLaVA-OneVision melaporkan tiga kemampuan emergen:

1. Penalaran multi-kamera. Dilatih pada multi-gambar + video secara terpisah; saat inference, diminta bernalar tentang scene mengemudi multi-kamera. Model mengintegrasikan sudut pandang dengan benar meskipun tidak pernah melihat format persis itu dalam training.
2. Set-of-mark prompting. Pengguna menganotasi objek dalam gambar dengan tanda bernomor; model bernalar tentang "apa yang dilakukan tanda 3 relatif terhadap tanda 7." Tidak dilatih pada tanda maupun anotasi; dipelajari dari kombinasi spatial grounding + referensi multi-gambar.
3. Agen tangkapan-layar iPhone. Pengguna memberikan tangkapan layar layar iPhone dan meminta merencanakan klik berikutnya. Dilatih pada tangkapan layar UI, video alur kerja pengguna, dan pasangan sebelum/sesudah multi-gambar. Mengeneralisasi ke kasus penggunaan agen.

Ini bukan tugas yang dilatih; ia muncul dari struktur kompositional curriculum.

### Pooling visual-token

Token budget membutuhkan pooling. OneVision menggunakan interpolasi bilinear pada grid patch 2D: 24x24 = 576 patch menjadi 12x12 = 144 (faktor 2x) atau 8x8 = 64 (faktor 3x). Pooling dilakukan di ruang patch-grid, bukan ruang token, untuk mempertahankan lokalitas.

Pilihan faktor pooling per skenario itu sendiri adalah hyperparameter. Lebih sedikit pooling = lebih banyak token = representasi lebih kaya. Lebih banyak pooling = lebih sedikit token = lebih banyak frame / gambar yang muat.

### LLaVA-OneVision-1.5

Lanjutan 2025 (LLaVA-OneVision-1.5, arXiv 2509.23661) "fully open" dalam data training, bobot model, dan kode. Menyamai kesenjangan proprietri pada beberapa benchmark dan mendemokratisasi resepnya. Curriculum yang sama, lebih banyak data, LLM basis lebih baik. Tanpa perubahan arsitektur.

### Kontras dengan Qwen2.5-VL

Qwen2.5-VL (Pelajaran 12.09) membuat pilihan berbeda. Ia menggunakan M-RoPE dan FPS dinamis alih-alih pooling tetap. Budget-nya menskala dengan input — video 1 menit menggunakan lebih banyak token daripada video 5 detik. LLaVA-OneVision memfixkan budget dan menskala pooling. Keduanya bekerja; mereka menukar konfigurabilitas dengan prediktabilitas.

```figure
l5-onevision-budget
```

## Gunakan

`code/main.py` adalah perencana curriculum dan budget untuk VLM bergaya OneVision. Diberi token budget per sampel dan campuran skenario target (katakan 40% gambar-tunggal, 30% multi-gambar, 30% video), ia:

- Mengalokasikan resolusi, faktor pooling, dan frame per skenario.
- Memeriksa bahwa setiap skenario muat dalam budget bersama.
- Melaporkan jumlah token yang diharapkan, FLOPs LLM, dan skenario mana yang under-tokenized.
- Mencetak jadwal training tahap demi tahap.

Gunakan untuk merencanakan fine-tune OneVision atau untuk memeriksa kewarasan biaya per-request deployment VLM.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-onevision-budget-planner.md`. Diberi distribusi tugas target dan budget per-sampel, ia menghasilkan faktor AnyRes, pooling per-frame, jumlah frame video, dan bobot tahap curriculum. Gunakan ini setiap kali kamu melatih atau fine-tune VLM skenario-terpadu.

## Latihan

1. Produkmu mendukung 80% gambar-tunggal, 10% multi-gambar (2-4 gambar), 10% video (8-16 frame). Rancang token budget. Ke mana kamu menaruh budget ekstra yang kamu hemat dari tidak melakukan multi-gambar berat?

2. Baca LLaVA-OneVision Bagian 4.3 (kemampuan emergen). Usulkan keterampilan emergen keempat yang kemungkinan akan dibuka curriculum tetapi tidak dilaporkan paper.

3. Tukar urutan curriculum — latih multi-gambar lebih dulu, lalu gambar-tunggal, lalu video. Prediksi benchmark mana yang menurun dan mengapa.

4. Paper melaporkan benchmark video yang dilatih hanya pada 8 frame per sampel. Apakah itu mengeneralisasi ke video 30 detik saat inference? Apa yang rusak lebih dulu — token budget atau penalaran temporal?

5. Bilinear pooling patch 24x24 ke 12x12 adalah reduksi 4x per dimensi. Implementasikan pooling dalam Python stdlib dan verifikasi bahwa rata-rata setiap blok 2x2 cocok dengan output bilinear.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| OneVision scenario | "Single-image, multi-image, or video" | One of three input shapes the unified VLM handles; the budget stays constant across |
| Token budget | "How many tokens per sample" | Total visual tokens the LLM sees per training / inference sample, typically 3000-4000 |
| Curriculum | "Training order" | Stage ordering (single-image → multi-image → video) chosen for emergent transfer |
| Bilinear pooling | "Token shrink" | Applying bilinear interpolation to the patch grid (2D) to reduce token count while preserving locality |
| Emergent skill | "Not trained, still works" | Capability that appears at inference without matching training data, due to curriculum composition |
| AnyRes-k | "k-tile setup" | k sub-tiles of fixed resolution plus one thumbnail, typical k ∈ {4, 9} |
| Task transfer | "Cross-scenario generalization" | Skills learned on single-image that apply to video (and vice versa) via shared backbone |

## Bacaan Lanjutan

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
