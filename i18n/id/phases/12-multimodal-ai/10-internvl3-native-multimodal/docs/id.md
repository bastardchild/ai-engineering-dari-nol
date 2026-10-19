# InternVL3: Pretraining Multimodal Native

> Setiap VLM terbuka sebelum InternVL3 mengikuti resep tiga-langkah yang sama: ambil LLM teks yang dilatih pada triliunan token teks, pasang sebuah vision encoder, lalu fine-tune bagian sambungannya. Ini bekerja tetapi memiliki utang alignment — LLM teks telah menghabiskan seluruh budget pretraining-nya pada teks murni dan tidak secara native memahami token visual. Saat kamu menambahkan visi secara post-hoc, LLM harus belajar ulang cara menghubungkan input visual dengan penalaran teksnya tanpa melupakan teks. InternVL3 (Zhu et al., April 2025) menolak pendekatan post-hoc: satu kali pretraining, teks dan multimodal berselang-seling sejak langkah pertama. Hasilnya menyamai Gemini 2.5 Pro pada MMMU-Pro pada 78B param terbuka. Pelajaran ini membaca kasus untuk pretraining native dan apa yang berubah saat kamu melakukannya.

**Type:** Learn
**Languages:** Python (stdlib, training-corpus mixer)
**Prerequisites:** Phase 12 · 05, Phase 12 · 07 (recipes)
**Time:** ~120 minutes

## Tujuan Pembelajaran

- Menjelaskan mengapa training VLM post-hoc mengakumulasikan utang alignment, dengan mengutip tiga gejala terukur (catastrophic forgetting, answer drift, inkonsistensi visual-teks).
- Mendeskripsikan campuran korpus pretraining native InternVL3 dan mengapa rasio teks : interleaved : caption itu penting.
- Membandingkan V2PE (variable visual position encoding) dengan M-RoPE Qwen2-VL.
- Menyebutkan optimasi deployment Visual Resolution Router (ViR) dan Decoupled Vision-Language (DvD).

## Masalahnya

Training VLM post-hoc adalah defaultnya. LLaVA, BLIP-2, Qwen-VL, Idefics — semuanya mengambil LLM yang sudah di-pretrain (Llama, Vicuna, Qwen, Mistral) dan menambahkan visi. Tahap training biasanya tampak seperti:

1. LLM dibekukan + vision encoder dibekukan + projector yang bisa dilatih, dilatih pada caption pair untuk menyelaraskan embedding.
2. Buka pembekuan LLM, latih pada data instruction (LLaVA-Instruct, ShareGPT4V).
3. Fine-tune spesifik-tugas opsional.

Tiga gejala utang alignment muncul:

- Catastrophic forgetting. VLM post-hoc melupakan keterampilan teks-murni. Skor GSM8K turun 5-10 poin. Skor Hellaswag turun. Agen teks-murni merosot.
- Answer drift. Rumusan berbeda dari pertanyaan visual yang sama mendapat jawaban berbeda. Vision encoder terhubung ke LLM dengan binding yang lebih lemah daripada token LLM itu sendiri.
- Inkonsistensi visual-teks. VLM bisa mendeskripsikan gambar dengan benar lalu menjawab pertanyaan yang membantah deskripsinya sendiri. Token visual tidak berpartisipasi dalam pemeriksaan konsistensi internal LLM dengan cara yang sama seperti teks.

Gejala-gejala ini terdokumentasi dengan baik. MM1.5 Bagian 4 mengkuantifikasinya. Ablation LLaVA-OneVision memberi petunjuk ke arahnya. Pretraining native adalah jawapannya.

## Konsepnya

### Pretraining multimodal native

InternVL3 melatih dari nol pada korpus yang native multimodal sejak langkah pertama. Campurannya:

- 40% data teks-murni (FineWeb, Proof-Pile-2, dll.)
- 35% data image-text interleaved (OBELICS, gaya MMC4)
- 20% data pasangan image-caption
- 5% data video-teks

Token visual, token teks, dan interaksi lintas-modal semuanya berpartisipasi dalam loss yang sama sejak langkah gradien pertama. Tanpa pretraining alignment, tanpa tahap pembekuan projector, tanpa catastrophic forgetting untuk dipulihkan.

Training adalah satu tahap untuk model basis. Instruction tuning mengikuti, tetapi model basis sudah memahami token visual sebagai warga kelas satu.

### V2PE (variable visual position encoding)

Qwen2-VL menggunakan M-RoPE dengan alokasi sumbu tetap. InternVL3 memperkenalkan V2PE: positional encoding bervariasi per tipe modality (teks, gambar, video) dengan penskalaan yang bisa dipelajari. Dalam praktiknya:

- Token teks mendapat posisi 1D (indeks teks).
- Patch gambar mendapat posisi 2D (baris, kolom).
- Frame video mendapat posisi 3D (waktu, baris, kolom).

Ketiganya berbagi basis frekuensi RoPE yang sama, tetapi alokasi hidden-dim per band adalah parameter yang dipelajari, bukan pembagian tetap. Kebebasan menukar resolusi frekuensi temporal vs spatial selama pretraining.

Klaim ablation V2PE: 1-2 poin pada benchmark video di atas M-RoPE pada komputasi yang sama. Bukan revolusi, tetapi lebih bersih.

### Visual Resolution Router (ViR)

Optimasi deployment. Tidak semua gambar membutuhkan encoding resolusi penuh. Foto dengan satu objek pada detail rendah membuang token saat di-encode pada 1280px native. ViR adalah classifier kecil yang memprediksi resolusi minimum yang dibutuhkan untuk menjawab pertanyaan, sebelum encoding.

Routing memiliki tiga tier: resolusi-rendah (256 token), sedang (576), tinggi (2048+). Untuk 60% kueri dalam traffic produksi, rendah atau sedang sudah cukup. Efek bersih: throughput 2-3x pada kualitas setara.

### Decoupled Vision-Language deployment (DvD)

Saat kamu menyajikan VLM besar, vision encoder berjalan sekali per gambar tetapi LLM berjalan autoregresif untuk setiap token output. Dua komponen memiliki bottleneck berbeda (visi = GPU memory bandwidth untuk conv + attention; LLM = KV cache). DvD memisahkannya ke GPU terpisah dengan streaming di antaranya.

Untuk model encoder 8B + 400M, DvD secara kasar menggandakan throughput per-node dibanding co-located.

### Kualitas single-stage vs multi-stage

Klaim benchmark utama InternVL3: pada 78B param, samakan MMMU-Pro Gemini 2.5 Pro. Pada 38B, samakan GPT-4o. Pada 8B, pimpin papan peringkat open-8B. Semuanya pada resep single-stage pretrain + instruction-tune.

Hipotesis utang-alignment terukur: InternVL3-8B kehilangan lebih sedikit poin benchmark teks (MMLU, GSM8K) daripada Qwen2.5-VL-7B per unit keuntungan benchmark visual. Model lebih bersifat generalis karena training adalah satu kesatuan, bukan dua.

### InternVL3.5 dan InternVL-U

InternVL3.5 (Agustus 2025) menskala resepnya. Pendekatan native-pretrain yang sama, lebih banyak data, lebih banyak param. Peningkatan MMMU bersifat inkremental.

InternVL-U (2026) menambahkan generation terpadu — output gambar melalui head MMDiT di atas backbone yang sama. "U" berarti "Understanding + generation," mengejar model terpadu bergaya Transfusion (Pelajaran 12.13). Backbone native-pretrain yang sama mendukung baik head pemahaman maupun generasi.

### Trade-off pretraining native

Pretraining native tidak gratis:

- Komputasi. Melatih VLM baru dari nol berbiaya sama dengan melatih LLM teks — jutaan GPU-hour. Adaptasi post-hoc menggunakan ulang bobot LLM yang ada, menghemat sebagian besar biayanya.
- Data. Korpus image-text interleaved berskala besar langka. OBELICS 141M dokumen; MMC4 571M. Teks saja dikirim pada 15T token. Kelangkaan data pretraining multimodal adalah batasan keras.
- Penggunaan ulang base-LLM. Pretraining native menyerahkan opsi untuk memasang LLM baru nanti. Post-hoc memungkinkan kamu menukar Llama-3.1 untuk Llama-4 dengan melatih ulang hanya adapternya.

Taruhan yang dibuat InternVL3: utang alignment lebih buruk daripada kerugian penggunaan-ulang. Benchmark mendukung klaimnya. Biaya produksi menghalangi lab masa depan mereplikasi dengan murah. VLM post-hoc akan tetap ada karena tetap lebih murah untuk sebagian besar proyek.

```figure
l5-native-pretrain
```

## Gunakan

`code/main.py` adalah mixer korpus-training dan simulator router ViR. Ia:

- Mengambil campuran korpus target (%teks, %interleaved, %caption, %video) dan menghitung langkah yang diharapkan per modality.
- Mensimulasikan routing ViR pada batch kueri (distribusi: 50% detail-rendah, 30% sedang, 20% detail-tinggi) dan melaporkan jumlah token rata-rata.
- Melaporkan perkiraan throughput DvD diberi FLOPs encoder vs LLM.
- Mencetak perbandingan berdampingan post-hoc vs pretraining native dalam param, komputasi, data, dan gejala utang-alignment yang diharapkan.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-native-vs-posthoc-auditor.md`. Diberi rencana training VLM yang diusulkan, ia mengaudit apakah pergi native atau post-hoc, menandai risiko utang-alignment, dan merekomendasikan campuran korpus. Gunakan saat kamu menentukan ukuran proyek VLM-terbuka baru dan perlu memilih strategi training.

## Latihan

1. Perkirakan delta komputasi antara InternVL3-8B (native pretrain) dan LLaVA-OneVision-7B (post-hoc). Perkiraan rasio GPU-hour? Apa yang menjelaskan kesenjangan itu?

2. InternVL3 melaporkan 40% teks / 35% interleaved / 20% caption / 5% video. Jika tugas targetmu berat-video, usulkan rasio baru dan berargumen mengapa model basis tetap membutuhkan data teks dan caption yang substansial.

3. Baca MM1.5 Bagian 4 tentang forgetting. Sebutkan benchmark persis di mana training post-hoc menunjukkan regresi terbesar. Seberapa besar regresi itu berbiaya?

4. ViR merutekan 60% traffic ke encoding resolusi-rendah. Kueri seperti apa yang salah-rutekan (dikirim ke resolusi-rendah saat resolusi-tinggi dibutuhkan)? Usulkan tiga mode kegagalan router.

5. DvD memisahkan visi dan LLM ke GPU terpisah. Pola traffic seperti apa yang justru merugikan throughput DvD alih-alih membantunya?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Native multimodal pretraining | "From scratch together" | Text + image + video tokens participate in the loss from step 1, not bolted on later |
| Alignment debt | "Post-hoc penalty" | Measurable regression in text skills and answer consistency that comes from bolting vision onto a frozen LLM |
| V2PE | "Variable visual pos encoding" | Per-modality learnable position encoding allocation; InternVL3's M-RoPE successor |
| ViR | "Resolution router" | Small classifier that picks minimum resolution needed per query before encoding, saving inference tokens |
| DvD | "Decoupled deployment" | Vision encoder on one GPU, LLM on another, with stream handoff; doubles throughput for large VLMs |
| InternVL-U | "Unified understanding + generation" | 2026 follow-up that adds image-generation heads to the native-pretrain backbone |
| Interleaved corpus | "OBELICS / MMC4" | Documents with text and images in natural reading order; the raw material for native pretraining |

## Bacaan Lanjutan

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)
