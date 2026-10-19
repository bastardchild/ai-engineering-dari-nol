# Embodied VLA: RT-2, OpenVLA, π0, GR00T

> Kali pertama sebuah model membaca resep dari situs web dan mengeksekusinya di robot dapur adalah RT-2 (Google DeepMind, Juli 2023). RT-2 mendiskretisasi aksi sebagai token teks, melakukan co-fine-tune VLM pada data web plus data aksi robot, dan membuktikan bahwa pengetahuan vision-language skala-web mentransfer ke kontrol robotik. OpenVLA (Juni 2024) mengirimkan referensi open 7B. Seri π0 dari Physical Intelligence (2024-2025) menambahkan expert aksi flow-matching. GR00T N1 dari NVIDIA (Maret 2025) menghadirkan kontrol dual-system (System 1 / System 2) untuk humanoid skala besar. Primitif VLA — vision-language-action, satu model yang melihat, membaca, dan bertindak — adalah jembatan antara model-model pemahaman di phase ini dan sistem otonom di Phase 15.

**Type:** Learn
**Languages:** Python (stdlib, action tokenizer + VLA inference skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 15 (Autonomous Systems, referenced)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Mendeskripsikan tokenisasi aksi: encoding bin diskrit (RT-2), FAST efficient action tokens, aksi flow-matching kontinu (π0).
- Menjelaskan mengapa co-fine-tuning pada data web + robot mempertahankan transfer pengetahuan-umum ke tugas-tugas baru.
- Membandingkan OpenVLA (open 7B Llama+VLM), π0 (flow-matching), dan GR00T N1 (dual-system) pada tugas robot yang sama.
- Menyebutkan dataset Open X-Embodiment dan perannya sebagai korpus training RT-X.

## Masalahnya

Robot yang melakukan pekerjaan rumah dari instruksi bahasa alami telah menjadi target riset sejak 1970-an. Jawaban dekade 2020-an: model vision-language-action (VLA). Arsitektur VLM yang sama dipakai untuk VQA, tetapi outputnya adalah aksi (torka sendi, pose end-effector, perintah diskrit) alih-alih teks.

Tantangan khusus VLA:

1. Ruang aksi bersifat kontinu (sudut sendi, gaya) dan berdimensi tinggi (lengan 7-DOF + gripper 3-DOF = 10 dimensi pada 30 Hz).
2. Data training khusus robot langka. Open X-Embodiment punya ~1M trajektori; teks-gambar web ada 5B+.
3. Frekuensi kontrol berarti. Loop kontrol 30 Hz berarti anggaran 33ms per aksi.
4. Keselamatan. Aksi yang salah merusak perangkat keras, manusia, atau properti.

## Konsepnya

### Tokenisasi aksi (RT-2)

Trik RT-2: merepresentasikan setiap target sendi sebagai token teks yang terkuantisasi. Mendiskretisasi rentang ternormalisasi [-1, 1] menjadi 256 bin, memetakan setiap bin ke sebuah ID vocabulary. Aksi 10-DOF menjadi 10 token pada setiap langkah kontrol.

Co-fine-tune VLM PaLM-X pada campuran:

- Pasangan gambar-teks web (captioning, VQA).
- Demonstrasi robot, aksi sebagai token.

Model melihat "pick up the red cube" (bahasa) → gambar (vision) → sekuens aksi 10-token (target sendi yang terdiskretisasi). Pretraining web mempertahankan transfer pengetahuan-umum: RT-2 bisa mengikuti "move towards the fast-moving object" meskipun "fast-moving" tidak ada dalam data training.

Inferensi pada 3-5 Hz di paper RT-2, dibatasi oleh decode autoregressive VLM.

### OpenVLA — referensi open 7B

OpenVLA (Kim et al., Juni 2024) adalah setara RT-2 dengan bobot terbuka. Backbone Llama 7B, dual vision encoder DINOv2 + SigLIP, tokenisasi aksi di atas 256 bin.

Dilatih pada Open X-Embodiment (970k trajektori dari 22 robot). Datang dengan dukungan fine-tuning LoRA untuk adaptasi ke robot baru.

Inferensi: 4-5 Hz pada A100 dengan kuantisasi. Cukup cepat untuk manipulasi lambat, tidak untuk kontrol frekuensi tinggi.

### Tokenizer FAST — decode aksi lebih cepat

Pertsch et al. (2024) menunjukkan bahwa tokenisasi bin diskrit tidak efisien — sebagian besar aksi mengelompok di wilayah kecil ruang-bin. FAST (Frequency-domain Action Sequence Tokenizer) mengompresi sekuens aksi via DCT dan mengkuantisasi koefisiennya.

Trajektori aksi 30 langkah menjadi ~10 token FAST alih-alih 300 token bin diskrit. Inferensi 3-5x lebih cepat tanpa penurunan kualitas.

### π0 dan aksi flow-matching

π0 dari Physical Intelligence (Black et al., Oktober 2024) mengganti token aksi diskrit dengan expert aksi flow-matching:

- Sebuah transformer aksi kecil membaca hidden state VLM dan mengeluarkan sekuens aksi kontinu 50 langkah via rectified flow.
- Kepala aksi dilatih dengan loss flow-matching; pretraining VLM tetap tidak berubah.
- Inferensi: sekuens aksi penuh dikeluarkan dalam ~5 langkah denoising, efektif kontrol 50 Hz.

Klaim π0: mengalahkan OpenVLA dan Octo pada berbagai besar tugas manipulasi. Formulasi aksi-kontinu mempertahankan kehalusan yang dihancurkan oleh diskretisasi.

π0.5 dan π0-FAST adalah peningkatan inkremental. π0-FAST menggabungkan tokenisasi FAST dengan flow matching.

### GR00T N1 — dual-system untuk humanoid

GR00T N1 dari NVIDIA (Maret 2025) dibangun untuk robot humanoid (>30 DOF, seluruh tubuh):

- System 2: VLM besar yang membaca adegan + instruksi, menghasilkan subgoal tingkat-tinggi pada ~1 Hz.
- System 1: transformer kepala-aksi kecil yang menghasilkan perintah sendi tingkat-rendah 50-100 Hz yang dibingkai oleh subgoal.

Pemisahan ini memetakan ke berpikir cepat-dan-lambat milik Kahneman: System 2 merencanakan, System 1 bertindak. Manfaat: perencanaan seukuran VLM yang lambat tidak memblokir kontrol cepat; System 1 tetap kecil demi latensi.

GR00T N1.7 (akhir 2025) meningkatkan penskalaan data. GR00T di-fine-tune dengan data sim-to-real dari Omniverse.

### Open X-Embodiment

Data trainingnya. RT-X (Oktober 2023) menghimpun 22 dataset yang mencakup 1M trajektori di 22 robot. Open X-Embodiment adalah korpus yang dipakai semua orang:

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table.
- Setiap sampel: (state robot, view kamera, instruksi, sekuens aksi).
- Higiene training: menyatukan ruang aksi, menormalisasi rentang sendi, mengubah ukuran kamera.

OpenVLA dan π0 dilatih pada Open X-Embodiment. Gap domain ke robot spesifik ditutup dengan fine-tuning LoRA pada 100-1000 demo spesifik-tugas.

### Co-fine-tuning vs robot-saja

Co-fine-tuning mencampur data VQA web dengan trajektori robot. Rasionya berarti: terlalu banyak VQA dan model melupakan aksi; terlalu banyak data robot dan model kehilangan pengetahuan umum.

Rasio RT-2: ~1:1. OpenVLA: ~0,5:1 web-ke-robot. π0: serupa. Rasio pasti adalah hiperparameter yang di-tune per ukuran dataset.

Training robot-saja menghasilkan model spesifik-tugas yang gagal pada instruksi out-of-distribution. Co-fine-tuning adalah perbedaan antara "pick up the red cube (in demo)" dan "pick up the third largest object from the left (novel phrasing)."

### Keselamatan dan batasan aksi

Setiap VLA produksi datang dengan:

- Batasan sendi keras (tidak bisa menorka melewati spesifikasi).
- Batasan kecepatan (soft clipping).
- Batas ruang kerja (end-effector tidak bisa meninggalkan meja).
- Persetujuan human-in-the-loop untuk tugas-tugas baru.

Ini berada di luar VLA sebagai pemeriksaan control-layer. Output VLA adalah saran, bukan perintah.

```figure
mm-action-tokens
```

## Gunakan

`code/main.py`:

- Mengimplementasikan tokenisasi aksi 256-bin dan de-tokenisasi.
- Mensketsakan tokenizer FAST berbasis DCT + kuantisasi.
- Membandingkan jumlah-token per langkah aksi di antara (bin diskrit, FAST, aliran kontinu).
- Mencetak ringkasan garis keturunan RT-2 → OpenVLA → π0 → GR00T.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-vla-action-format-picker.md`. Diberi tugas robot (manipulasi, navigasi, humanoid whole-body), memilih di antara bin diskrit + RT-2, FAST + OpenVLA, flow-matching + π0, atau dual-system + GR00T.

## Latihan

1. Lengan 10-DOF pada laju kontrol 30 Hz. Tokenisasi bin diskrit dengan 256 bin mengeluarkan berapa token per detik? Bisakah VLM 7B mengikuti?

2. Tokenisasi FAST mengompresi trajektori 30-langkah menjadi ~10 token. Apa yang hilang bagi pengguna jika trajektori punya gerakan frekuensi-tinggi (misalnya, bermain drum)?

3. Kepala flow-matching π0 melakukan denoising dalam ~5 langkah. Bandingkan throughput dengan decode autoregressive OpenVLA pada 4-5 Hz.

4. Pemisahan System 1 / System 2 milik GR00T memetakan ke Kahneman. Usulkan pemisahan berbeda (System 3?) yang mungkin membantu berjalan bipedal.

5. Baca Bagian 4 Open X-Embodiment tentang kurasi dataset. Sebutkan tiga aturan kurasi yang mencegah kebocoran domain.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| VLA | "Vision-language-action" | Model that takes image + instruction and outputs action commands |
| Action tokenization | "Discrete bins" | Quantize continuous joint targets into 256 bins per dim, each a vocab ID |
| FAST tokenizer | "Frequency action tokens" | DCT + quantize to compress 30-step trajectories to ~10 tokens |
| Co-fine-tune | "Mix web + robot" | Train on web VQA data alongside robot demos to preserve general knowledge |
| Flow-matching action head | "π0 continuous output" | Small transformer that outputs a 50-step action sequence via rectified flow |
| System 1 / System 2 | "Dual-system control" | Large VLM plans slowly, small action head acts quickly; GR00T pattern |
| Open X-Embodiment | "RT-X dataset" | 1M-trajectory cross-robot dataset; the training corpus |

## Bacaan Lanjutan

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
