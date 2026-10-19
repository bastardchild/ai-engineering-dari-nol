# Visi Any-Resolution: Patch-n'-Pack dan NaFlex

> Gambar nyata bukan kotak 224x224. Struk 9:16, grafik 16:9, pemindaian medis mungkin 4096x4096, tangkapan layar ponsel 9:19.5. Jawaban VLM pra-2024 — ubah ukuran semuanya ke kotak tetap — membuang sinyal yang membuat OCR, pemahaman dokumen, dan parsing scene beresolusi tinggi bekerja. NaViT (Google, 2023) menunjukkan kamu bisa mem-patch resolusi-variasi menjadi satu batch transformer dengan masking block-diagonal. M-RoPE Qwen2-VL (2024) membuang tabel posisi absolut sepenuhnya. AnyRes LLaVA-NeXT men-tile gambar resolusi tinggi menjadi basis + sub-gambar. Varian NaFlex SigLIP 2 (2025) kini menjadi encoder default untuk VLM terbuka yang menginginkan satu checkpoint melayani setiap aspect ratio. Pelajaran ini mengimplementasikan patch-n'-pack end to end.

**Type:** Build
**Languages:** Python (stdlib, patch packer + block-diagonal mask)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 12 · 05 (LLaVA)
**Time:** ~120 minutes

## Tujuan Pembelajaran

- Mem-patch dari batch gambar resolusi-variasi menjadi satu sekuens dan membangun attention mask block-diagonal.
- Memilih di antara tiling AnyRes (LLaVA-NeXT), NaFlex (SigLIP 2), dan M-RoPE (Qwen2-VL) untuk tugas tertentu.
- Menghitung token budget untuk OCR, grafik, dan fotografi tanpa resizing.
- Menyebutkan tiga mode kegagalan square-resize: teks tergencet, konten terpotong, token terbuang untuk padding.

## Masalahnya

Transformer mengharapkan sebuah sekuens. Batch adalah tumpukan sekuens dengan panjang sama. Jika gambarmu 224x224, kamu mendapat 196 patch token setiap kali, padding tidak diperlukan, selesai. Latih di 224, infer di 224, tidak pernah memikirkan resolusi lagi.

Dunia tidak bekerja seperti itu. Dokumen bersifat potret (8.5x11 inci, kira-kira 2:3). Tangkapan layar grafik lanskap (16:9). Struk tinggi dan tipis (1:3). Imaging medis dikirim pada 2048x2048 atau lebih besar. Tangkapan layar perangkat ponsel 1170x2532 (0.46:1).

Tiga opsi pra-2024 dan mengapa masing-masing gagal:

1. Ubah ukuran ke kotak tetap (224x224 atau 336x336). Gencetan mendistorsi teks dan wajah. Penurunan skala menghancurkan label grafik dan konten OCR. Praktik standar sampai LLaVA-1.5.
2. Crop ke aspect ratio tetap. Kamu membuang sebagian besar gambar, dan memilih lokasi crop adalah masalah visi tersendiri.
3. Padding ke sisi terpanjang. Memperbaiki distorsi tetapi membuang 50%+ token untuk padding pada gambar potret. Biaya attention kuadratik pada semua token padding itu.

Jawaban 2024-2025: biarkan transformer mengonsumsi patch pada resolusi asli gambar, dan cari cara mem-patch batch heterogen menjadi satu sekuens tanpa komputasi terbuang.

## Konsepnya

### NaViT dan patch-n'-pack

NaViT (Dehghani et al., 2023) adalah paper yang menunjukkan ini bekerja pada skala besar. Idanya bersifat mekanis:

1. Untuk setiap gambar dalam batch, hitung grid patch aslinya pada patch size tertentu (katakan 14).
2. Ratakan patch setiap gambar menjadi sekuens variabel-panjangnya sendiri.
3. Gabungkan patch semua gambar menjadi satu sekuens panjang untuk batch.
4. Bangun attention mask block-diagonal agar patch gambar A hanya attend di dalam gambar A.
5. Bawa informasi posisi per-patch (2D RoPE atau fractional position embeddings).

Sebatch tiga gambar pada 336x336 (576 token), 224x224 (256 token), dan 448x336 (768 token) menjadi satu sekuens 1600-token dengan mask block-diagonal 1600x1600. Tanpa padding. Tanpa komputasi terbuang. Transformer menangani aspect ratio apa pun.

NaViT juga memperkenalkan fractional patch dropping saat training — jatuhkan 50% patch secara acak di seluruh batch — yang sekaligus meregularisasi dan mempercepat training. SigLIP 2 mewarisi ini.

### AnyRes (LLaVA-NeXT)

AnyRes LLaVA-NeXT adalah alternatif pragmatis. Diberi gambar resolusi tinggi dan encoder tetap (CLIP atau SigLIP pada 336), tile gambar tersebut:

1. Pilih layout grid dari set yang sudah ditetapkan — (1x1), (1x2), (2x1), (1x3), (3x1), (2x2), dll. — yang paling cocok dengan aspect ratio gambar.
2. Tile gambar penuh ke dalam grid; setiap tile menjadi crop 336x336.
3. Juga hasilkan thumbnail: seluruh gambar diubah ukurannya ke 336x336 sebagai token konteks-global.
4. Encode setiap tile melalui encoder 336 yang dibekukan. Gabungkan token tile + token thumbnail.

Untuk gambar 672x672 pada grid 2x2 plus thumbnail: 4 * 576 + 576 = 2880 token visual. Mahal tetapi efektif — LLM melihat detail lokal dan konteks global sekaligus.

AnyRes adalah jalur pilihan saat encoder-mu dibekukan dan hanya mendukung satu resolusi. Ia meledakkan jumlah token untuk gambar besar (gambar 1344x1344 pada grid 4x4 adalah 9216 + 576 ≈ 9800 token, yang mengisi sebagian besar context 8k LLM).

### M-RoPE (Qwen2-VL)

Qwen2-VL memperkenalkan Multimodal Rotary Position Embedding. Alih-alih posisi fraksional NaViT atau tile-and-thumbnail AnyRes, setiap patch membawa posisi 3D (temporal, tinggi, lebar). Rotasi query/key menangani H, W, dan panjang temporal apa pun.

M-RoPE mengirim resolusi dinamis native tanpa retraining. Saat inference kamu memberi gambar HxW apa pun, patch embedder menghasilkan token H/14 x W/14, setiap token mendapat posisi (t=0, r=baris, c=kolom), RoPE memutar attention dengan frekuensi yang tepat, selesai. Qwen2.5-VL dan Qwen3-VL melanjutkan ini. V2PE InternVL3 adalah ide yang sama dengan encoding variabel per modality.

Berbeda dengan AnyRes, M-RoPE adalah O(H x W / P^2) token pada resolusi asli — tanpa overhead tile multipikatif. Berbeda dengan NaViT, ia tetap mengharapkan satu gambar per forward. Batching lintas resolusi tetap membutuhkan patch-n'-pack di atasnya.

### NaFlex (SigLIP 2)

NaFlex adalah mode native-flex checkpoint SigLIP 2. Satu model melayani beberapa panjang sekuens (256, 729, 1024 token) saat inference. Di dalamnya menggunakan patch-n'-pack gaya NaViT saat training dan posisi fraksional absolut per patch. Keunggulannya: satu checkpoint, pilih token budget-mu saat inference berdasarkan tugas.

Untuk tugas semantik (klasifikasi, retrieval), 256 token. Untuk OCR atau pemahaman grafik, 1024 token. Tanpa retraining.

### Mask packing

Mask block-diagonal adalah tempat sebagian besar implementasi tersandung. Untuk sekuens ter-pack dengan panjang `N_total` yang mencakup gambar `i=0..B-1` dengan panjang `n_i`, mask `M` berbentuk `(N_total, N_total)` bernilai 1 jika kedua indeks jatuh di blok gambar yang sama, selain itu 0. Kamu bisa membangunnya dari daftar panjang kumulatif:

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

Ini satu baris di PyTorch dengan `torch.block_diag` atau gather eksplisit. Jalur variable-length FlashAttention (`cu_seqlens`) melewatkan mask sepenuhnya dan attend di dalam sekuens menggunakan tensor panjang-kumulatif langsung — ~10x lebih cepat daripada mask padat untuk batch biasa.

### Token budget

Pilih strategimu berdasarkan tugas:

- OCR / dokumen: 1024-4096 token. SigLIP 2 NaFlex pada 1024, atau AnyRes 3x3 + thumbnail.
- Grafik dan UI: 729-1024 token pada 384-448 native. Resolusi dinamis Qwen2.5-VL dengan cap max pixels.
- Foto alami: 256-576 token sudah cukup. LLM hilir melihat cukup. Bayar token di mana kepadatan konten tinggi.
- Video: 64-128 token per frame setelah spatial pooling, 2-8 FPS. Pelajaran 12.17 membahas ini.

Aturan produksi 2026: pilih cap max-pixels per-tugas, encode pada aspect ratio asli hingga cap itu, pack batch-nya, dan lewati padding. Qwen2.5-VL mengekspos `min_pixels` dan `max_pixels` untuk knob persis ini.

```figure
mm-patch-n-pack
```

## Gunakan

`code/main.py` mengimplementasikan patch-n'-pack untuk batch gambar heterogen dengan koordinat piksel integer. Ia:

- Mengambil daftar ukuran gambar (H, W).
- Menghitung panjang sekuens patch setiap gambar pada patch size 14.
- Mem-patchnya menjadi satu sekuens dengan total panjang `sum(n_i)`.
- Membangun attention mask block-diagonal (padat, untuk kejelasan).
- Membandingkan biaya ter-pack vs square-resize dan tiling AnyRes.
- Mencetak tabel token budget untuk batch campuran (struk, grafik, tangkapan layar, foto).

Jalankan. Angka yang keluar adalah alasan setiap VLM terbuka 2026 menggunakan patch-n'-pack.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-resolution-budget-planner.md`. Diberi beban kerja aspect-ratio-campuran (OCR, grafik, foto, frame video) dan total-token budget, ia memilih strategi yang tepat (NaFlex, AnyRes, M-RoPE, atau fixed-square) dan menghasilkan konfigurasi per-request. Gunakan skill ini saat kamu menentukan ukuran VLM untuk produk — ia mencegah ledakan token 10x diam-diam yang membunuh budget latensi.

## Latihan

1. Sebuah struk 600x1500 (1:2.5). Pada patch size 14, berapa token resolusi asli? Berapa setelah square-resize ke 336? Mana yang kehilangan akurasi OCR lebih banyak dalam praktiknya?

2. Bangun mask block-diagonal untuk batch empat gambar dengan panjang 256, 576, 729, 1024. Verifikasi bahwa matriks attention 2585x2585 dan memiliki tepat `256^2 + 576^2 + 729^2 + 1024^2` entri non-nol.

3. Untuk gambar 1792x896 pada patch 14, bandingkan: (a) square-resize ke 336 lalu encode, (b) AnyRes 2x1 + thumbnail, (c) M-RoPE pada native. Mana yang menggunakan token paling sedikit? Mana yang mempertahankan detail paling banyak?

4. Implementasikan fractional patch dropping: diberi sekuens ter-pack, jatuhkan 50% token secara merata acak, dan perbarui mask block-diagonal sesuai ukuran. Ukur perubahan sparsity mask-nya.

5. Baca Bagian 3.2 paper Qwen2-VL (arXiv:2409.12191). Jelaskan dalam dua kalimat apa yang dikontrol `min_pixels` dan `max_pixels` dan mengapa kedua batas itu penting.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Patch-n'-pack | "NaViT-style packing" | Concatenate variable-length patch sequences from different images into one batch dimension |
| Block-diagonal mask | "Packing mask" | Attention mask that confines each image's patches to attend only to themselves, not neighbors in the pack |
| AnyRes | "LLaVA-NeXT tiling" | Split a high-res image into a grid of fixed-size tiles plus a global thumbnail; encode every tile with a fixed encoder |
| NaFlex | "SigLIP 2 native-flex" | Single SigLIP 2 checkpoint that serves 256/729/1024-token budgets at inference without retraining |
| M-RoPE | "Multimodal RoPE" | 3D rotary position encoding (time, row, column) that handles arbitrary H, W, T without position tables |
| cu_seqlens | "FlashAttention packing" | Cumulative-length tensor the FlashAttention varlen path uses instead of a dense block-diagonal mask |
| min_pixels / max_pixels | "Resolution bounds" | Qwen2.5-VL per-request knobs capping token count on very small or very large inputs |
| Visual token budget | "How many tokens per image" | Rough count of patch tokens emitted per image; sets the LLM's prompt budget and attention cost |

## Bacaan Lanjutan

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
