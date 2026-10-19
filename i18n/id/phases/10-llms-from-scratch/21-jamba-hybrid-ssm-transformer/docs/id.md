# Jamba — HibridaSSM-Transformator

> Model ruang negara (SSM) dan transformator menginginkan hal yang berbeda. Transformer membeli kualitas melalui perhatian dengan biaya kuadrat. SSM membeli inferensi waktu linier dan memori konstan melalui pengulangan tetapi kualitas lag.AI21Jamba (Maret 2024) dan Jamba 1.5 (Agustus 2024) menempatkannya dalam model yang sama: 1 lapisan Transformer untuk setiap 7 lapisan Mamba, MoE di setiap blok lainnya, dan jendela konteks 256k yang muat pada satu 80GBGPU. Mamba-3 (ICLR2026) memperketatSSMbersisian dengan ruang negara bernilai kompleks danMIMOproyeksi. Pelajaran ini membaca kedua arsitektur secara menyeluruh dan menjelaskan mengapa resep hibrida dapat bertahan selama tiga tahun penskalaan ketika dimurnikanSSMdan upaya konteks panjang Transformer murni belum dilakukan.

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan tiga primitif dalam blok Jamba — lapisan Transformer, lapisan Mamba, MoE — dan resep interleaving genap 1:7:.
- Nyatakan apa yangSSMPerulangan sepertinya berada pada tingkat tinggi dan mengapa hal ini memungkinkan inferensi memori konstan.
- Hitung jejak cache KV model Jamba pada konteks 256k dan bandingkan dengan kebutuhan model Transformer murni.
- Sebutkan tiga inovasi Mamba-3 (diskritisasi trapesium eksponensial, pembaruan status bernilai kompleks,MIMO) dan masalah yang menjadi target masing-masing.

## Masalahnya

Perhatian berbentuk kuadrat dalam panjang barisan. Model ruang negara bersifat linier. Perbedaan tersebut bertambah: pada 256 ribu token, peta perhatian Transformer adalah 65 miliar entri per kepala; sebuahSSMKeadaan berulang adalah ukuran tetap terlepas dari panjang urutan.

Murni-SSMmodel (Mamba, Mamba-2) cocok dengan kebingungan Transformer pada skala kecil tetapi tertinggal dalam tugas pelacakan status dan gagal pada beberapa kategori pengambilan dalam konteks. Intuisinya: SSM memampatkan riwayat ke keadaan tetap, dan ketika sejarah panjang, informasi bocor. Perhatian mengingat semuanya dengan tepat tetapi membayar biaya kuadrat.

Perbaikan yang jelas: gunakan keduanya. Letakkan lapisan Transformer di tempat yang penting untuk mengingat kembali. GunakanSSMlapisan di tempat lain. Sesuaikan rasionya. Jamba adalah model tingkat produksi pertama yang mengirimkan resep hybrid ini dalam skala besar (total 52 miliar, aktif 12 miliar, konteks 256 ribu, 80 GB tunggalGPU). Jamba 1.5 menambah jumlah keluarga menjadi 398 miliar / 94 miliar aktif. Mamba-3 (ICLR2026) adalah yang terbaik saat iniSSMdasar bahwa hibrida dapat dibangun kembali.

Pelajaran ini membaca ketiga makalah dan menghasilkan model mental untuk "memilih rasio yang tepat".

## Konsepnya

### SebuahSSMdalam satu halaman

Model ruang negara memproses suatu urutan`x_1, ..., x_N`melalui keadaan ukuran tetap`h`:

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

Pada setiap langkah, negara berkembang melalui dinamika linier`A`, menerima masukan`B x_t`, dan mengeluarkan output`C h_t`.`A, B, C`bisa dipelajari. Perhatikan properti penting: komputasi`y_t`kebutuhan saja`h_{t-1}`dan`x_t`, tidak lebih awal`x`. Memori itu konstan. Inferensinya adalah O(1) per token.

Trik untuk kualitas pemodelan adalah strukturnya`A`. S4 (Gu 2021) menggunakan matriks yang sangat terstruktur yang dapat dievaluasi secara efisien sebagai konvolusi panjang selama pelatihan. Mamba (Gu, Dao 2023) menggantikan yang tetap`A, B, C`dengan yang bergantung pada data (bagian "selektif"). Mamba-2 (2024) semakin menyederhanakan strukturnya. Mamba-3 (2026) kembali menambahkan kompleksitas di tempat tertentu.

Properti kunci: untuk decoderLLM, sebuahSSMlapisan adalah pengganti drop-in untuk lapisan perhatian, dengan status per lapisan berukuran tetap, bukan cache KV yang bertambah.

### Blok Jamba

Blok Jamba menyisipkan lapisan menurut dua angka:

- `l`: rasio perhatian terhadap Mamba. Jamba menggunakan`l = 8`, artinya 1 lapisan Transformer untuk setiap 7 lapisan Mamba (7 Mamba + 1 Perhatian = 8 lapisan per grup).
- `e`: frekuensi MoE. Jamba menggunakan`e = 2`, artinya setiap lapisan lainnya menerapkan MoE.

Urutan lapisan dalam satu blok:

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

Setiap blok Jamba terdiri dari 8 lapisan. Pada kedalaman 4 blok (total 32 lapisan), kamu mendapatkan 28 lapisan Mamba dan 4 lapisan Perhatian. 16 di antaranya menggunakan MoE.

### Kenapa rasionya 1:7

AI21menjalankan ablasi: berapa rasio perhatian terhadap Mamba yang memberikan kebingungan per parameter terbaikANDmengingat kembali konteksnya pada evaluasi konteks panjangnya?

- Terlalu banyak perhatian (1:1): kualitas meningkat tetapi memori dan kecepatan menurun.
- Terlalu sedikit perhatian (1:15): memori bagus tetapi pengambilan dalam konteks gagal.
- Titik manis: 1:7 atau 1:8.

Intuisinya: lapisan Transformer menangani penarikan kembali dan pelacakan status. Lapisan Mamba menangani sebagian besar pemrosesan yang murah.

### Pengkodean posisi

Lapisan Mamba sendiri sadar akan posisi (melalui pengulangan). Lapisan perhatian pada hibrida asli berbasis Mamba tidak menggunakan RoPE — ituSSMlapisan memberikan info posisi. Jamba 1.5 menambahkan RoPE ke lapisan perhatian untuk generalisasi konteks yang lebih panjang, penyempurnaan post-hoc berdasarkan evaluasi empiris konteks panjang.

### Anggaran memori

Untuk bentuk Jamba-1 (32 lapisan: 28 Mamba + 4 Perhatian, tersembunyi 4096, 32 kepala perhatian):

- Cache KV (hanya lapisan perhatian):`2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`di 256kBF16. Hanya 4 lapisan perhatian yang berkontribusi.
- SSMnegara bagian:`28 * hidden * state_size`per awalan token, tetapi ini adalah ukuran tetap per lapisan, tidak diskalakan dengan panjang urutan. Status Mamba pada umumnya adalah 16 per fitur, tersembunyi 4096:`28 * 4096 * 16 * 2 = 3.7 MB`jumlah.

Bandingkan dengan Transformer murni dengan 32 lapisan, sama tersembunyinya, penuhMHAdi 32 kepala:`2 * 32 * 32 * 128 * 256k * 2 = 128 GB`di 256kBF16. Pengurangan 8x dalam cache KV. Bahkan melawanGQA(8) dasar yang digunakan sebagian besar model tahun 2024 (`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`), Hibrida 1:7 Jamba pada 16 GB masih 2x lebih kecil.

Itulah yang terjadiAI21artinya dengan "konteks 256k pada satu 80GBGPU." Cache KV dari full-MHATransformer murni tidak akan cocok; bahkan sebuahGQAbaseline tidak memberikan ruang untuk bobot dan aktivasi; Jamba melakukannya.

### Mamba-3: yang murni-SSMdasar pada tahun 2026

Mamba-3 (ICLR2026, arXiv:2603.15569) memperkenalkan tiga inovasi pada pure-SSMsisi:

1. **Diskritisasi trapesium eksponensial.** Menggantikan diskritisasi metode Euler di Mamba-2 dengan pengulangan yang lebih ekspresif. Operasi mirip konvolusi diterapkan pada masukan keadaan dalam pengulangan inti, bukan sebagai konvolusi luar pada`x_t`.

2. **Pembaruan status bernilai kompleks.** Mamba sebelumnya mereduksi matriks status dari kompleks (S4) menjadi diagonal nyata (Mamba) hingga identitas berskala (Mamba-2). Mamba-3 menambahkan kembali nilai-nilai kompleks — setara dengan penyematan putaran yang bergantung pada data pada negara bagian. Hal ini memulihkan kemampuan pelacakan keadaan yang sebelumnya memerlukan biaya penyederhanaan yang bernilai riil.

3. **Multi-masukan multi-output (MIMO) proyeksi.** Daripada menggunakan proyeksi skalar per fitur, gunakan proyeksi bernilai matriks. Meningkatkan kekuatan pemodelan dan pemanfaatan perangkat keras waktu inferensi tanpa meningkatkan latensi dekode.

Pada parameter 1,5 miliar, Mamba-3 meningkatkan rata-rata akurasi hilir sebesar 0,6 poin dibandingkan Gated DeltaNet; ituMIMOvarian menambahkan 1,2 lebih banyak untuk total perolehan 1,8 poin. Pada ukuran negara bagian yang sama, Mamba-3 cocok dengan Mamba-2 dengan separuh negara bagian.

Mamba-3 belum dikirimkan dalam produksi hibrida dalam skala besar – tetapi ini adalah kandidat yang jelas untuk ituSSMsisi model kelas Jamba berikutnya.

### Kapan harus memilih hibrida

Hibrida menang ketika:

- Konteksnya cukup panjang sehingga cache Transformer KV murni menjadi sulit (64k+).
- Tugas menggabungkan struktur jangka pendek (baik untukSSM) dengan penarikan jangka panjang (membutuhkan Transformer).
- kamu ingin menerapkan pada single-GPUanggaran memori yang mana cache Transformer KV saja tidak akan muat.

Hibrida kalah ketika:

- Konteksnya pendek (di bawah 16k). ItuSSMoverhead terbuang; Transformer murni baik-baik saja.
- Tugas memerlukan perhatian dari mana saja (penalaran mendalam, referensi silang multi-dokumen). Jarangnya lapisan perhatian pada hibrida itu menyakitkan.
- kamu menskalakan ke model perbatasan triliunan parameter. Transformator Murni +MLA+ MoE (gaya DeepSeek-V3) saat ini memenangkan perlombaan kemampuan.

### Lanskap kompetitif

| Model | Family | Scale | Unique claim |
|-------|--------|------|-------------|
| Mamba-2 | pure SSM | 3B | linear time, constant memory |
| Jamba | hybrid | 52B/12B | 256k on 80GB |
| Jamba 1.5 Large | hybrid | 398B/94B | enterprise-grade long-context |
| Mamba-3 | pure SSM | 1.5B (paper) | state-tracking restored |
| DeepSeek-V3 | pure Transformer + MoE | 671B/37B | frontier capability |

Lanskap tahun 2026: Transformator murni MoE mendominasi wilayah terdepan, namun hibrida memiliki ceruk konteks lebih dari 256 ribu. Kemenangan pelacakan negara bagian Mamba-3 dapat mendorong rasio hibrida lebih rendah (lebih besarSSM, kurang mendapat perhatian) pada generasi selanjutnya.

```figure
swiglu-ffn
```

## Gunakan

`code/main.py`adalah kalkulator memori untuk arsitektur hybrid. Diberikan sebuahSSM-Rasio transformator dan konfigurasi ukuran tersembunyi/jumlah lapisan, ini menghitung:

- Cache KV pada konteks target.
- SSMmemori negara.
- Total memori pada konteks N untuk berbagai bentuk model.

Kalkulator mendukung:

- Garis dasar Pure-Transformer (cache KV bertambah dengan N).
- Hibrida 1:7 ala Jamba.
- Murni-SSM(tidak ada cache KV sama sekali).

Angka-angka tersebut diambil langsung dari makalah Jamba-1 dan Jamba-1.5 untuk bentuk yang dipublikasikan dan diekstrapolasi untuk varian hipotetis.

Pertimbangan integrasi untuk penerapan nyata:

- Sebagian besar server inferensi produksi (vLLM, SGLang) mendukung Jamba dan Mamba. Periksa versi spesifiknya.
- Pada konteks 256k, keunggulan memori Jamba muncul dalam throughput permintaan bersamaan. Pada hal yang samaVRAMkamu memasukkan lebih banyak rangkaian Jamba daripada rangkaian Transformer.
- Mamba-3 sebagai model mandiri belum dikirimkan dalam tahap produksi — pratinjau penelitian pada 1,5 miliar.

## Rilis

Pelajaran ini menghasilkan`outputs/skill-hybrid-picker.md`. Mengingat spesifikasi beban kerja (profil panjang konteks, campuran tugas, anggaran memori), ini merekomendasikan antara Transformer murni, hybrid gaya Jamba, dan murniSSM, dengan alasan eksplisit tentang memori dan pengorbanan kualitas.

## Latihan

1. Jalankan`code/main.py`untuk menghitung cache KV pada konteks 256k untuk Transformer murni 32 lapis (tersembunyi 4096, 32 kepala) dan untuk hibrida Jamba-1 dengan bentuk yang sama. Verifikasi pengurangan memori ~8xAI21klaim kertas.

2. Ubah kalkulator untuk memodelkan hibrida 1:3 (4 Mamba : 1 Perhatian) dan hibrida 1:15 (14 Mamba : 1 Perhatian). Plot cache KV vs rasio. Pada rasio berapa cache KV sama denganSSMmemori negara?

3. Baca Bagian 3 makalah Jamba (arXiv:2403.19887). Jelaskan alasannyaAI21menggunakan Mamba-1 daripada Mamba-2 meskipun Mamba-2 lebih cepat. Petunjuk: bagian ablasi hibrid mendokumentasikan hal ini.

4. Hitung parameter overhead MoE-setiap-lapisan lainnya di Jamba 1.5 Besar (total 398B, aktif 94B). Bandingkan rasio aktif dengan DeepSeek-V3 (37B/671B) dan jelaskan mengapa arsitektur Jamba mendorong rasio aktif lebih tinggi.

5. Baca Bagian 3 makalah Mamba-3 (arXiv:2603.15569). Jelaskan dalam tiga kalimat mengapa pembaruan status bernilai kompleks setara dengan penyematan putar yang bergantung pada data. Ikat jawaban ke Fase 7 · Derivasi RoPE Pelajaran 04.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State space model (SSM) | "Recurrence with a fixed state" | A layer with a learned recurrence `h_t = A h_{t-1} + B x_t`; constant memory per token |
| Selective SSM | "Mamba's trick" | Data-dependent A, B, C parameters that give the model gating-like selectivity at linear time |
| Attention-to-Mamba ratio | "How many attention layers" | In Jamba, `l = 8` means 1 attention layer per 7 Mamba layers |
| Jamba block | "The 8-layer group" | One attention + seven Mamba + MoE on alternate positions |
| SSM state | "The hidden buffer" | Fixed-size per-layer state that replaces the KV cache for Mamba layers |
| 256k context | "Jamba's flagship number" | The sequence length Jamba-1 fits on a single 80GB GPU; pure Transformer cannot at that size |
| Mamba-3 | "2026 pure SSM" | Current-best pure-SSM architecture with complex state + MIMO; the baseline hybrids rebuild around |
| MIMO | "Multi-input multi-output" | Mamba-3 innovation using matrix-valued projections instead of scalar per-feature |
| Exponential-trapezoidal discretization | "Mamba-3's recurrence" | More expressive recurrence that subsumes Mamba-2's Euler-method discretization |
| Hybrid architecture | "Mix attention and SSM" | Any model that interleaves Transformer and SSM layers; Jamba is the production archetype |

## Bacaan Lebih Lanjut

- [Lieber dkk. — Jamba: Model Bahasa Transformator-Mamba Hibrida (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — makalah Jamba asli, ablasi rasio, klaim konteks 256k
- [AI21— Jamba 1.5: Hybrid Transformer-Mamba dalam Skala Besar (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — keluarga yang diperluas,398B/94Bdan12B/52Brilis publik
- [Gu, Dao — Mamba: Pemodelan Urutan Waktu Linier dengan Ruang Keadaan Selektif (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — yang selektifSSMkertas yang dibangun Jamba
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — penerus ruang negara terstruktur yang disederhanakan
- [Lahoti dkk. — Mamba-3 (arXiv:2603.15569,ICLR2026)](https://arxiv.org/abs/2603.15569) — keadaan bernilai kompleks,MIMO, murni tahun 2026-SSMperbatasan
- [Gu dkk. — Memodelkan Urutan Panjang secara Efisien dengan Ruang Keadaan Terstruktur (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — kertas S4, ituSSMtitik awal silsilah untuk LLM
