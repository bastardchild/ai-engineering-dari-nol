# Paralelisme Pipa Ganda

> DeepSeek-V3 dilatih pada 2.048H800GPU dengan pakar MoE tersebar di seluruh node. Biaya komunikasi all-to-all ahli lintas node 1GPU-jam komunikasi untuk setiap 1GPU-jam komputasi. GPU menganggur separuh waktu. DualPipe (DeepSeek, Des 2024) adalah pipeline dua arah yang tumpang tindih dengan komputasi maju dan mundur dengan komunikasi semua-ke-semua yang dipicunya. Penurunan gelembung, peningkatan throughput, dan penyimpanan dua salinan parameter model ("ganda" yang memberi nama) menjadi murah karena Paralelisme Pakar sudah menyebarkan pakar ke seluruh peringkat. Pelajaran ini adalah panduan tipe Pelajari tentang apa yang sebenarnya dilakukan DualPipe dan mengapa penyempurnaan DualPipeV di Sea AI Lab menurunkan biaya parameter 2x dengan mengorbankan gelembung yang sedikit lebih ketat.

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open-model architectures and MoE)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Sebutkan empat komponen potongan maju-mundur DualPipe dan mengapa masing-masing komponen memiliki jendela tumpang tindihnya sendiri.
- Jelaskan masalah gelembung saluran pipa dalam skala besar, dan apa arti "bebas gelembung" dalam praktik dibandingkan dalam pemasaran.
- Lacak jadwal DualPipe dengan tangan untuk 8 peringkat PP dan 16 batch mikro dan konfirmasikan aliran maju dan mundur saling mengisi slot menganggur.
- Nyatakan trade-off yang dilakukan DualPipeV (Sea AI Lab, 2025): menghilangkan replikasi parameter 2x dengan mengorbankan gelembung yang sedikit lebih besar saat Paralelisme Pakar tidak aktif.

## Masalahnya

Melatih model MoE 671B pada 2kH800GPU mengalami tiga hambatan yang semakin besar:

1. **Tekanan memori.** Masing-masingGPUmemegang sepotong model. Memori aktivasi pada urutan 8k di 61 lapisan pada 128 kepala sangatlah besar.
2. **Gelembung alur.** Paralelisme alur tradisional (GPipe, 1F1B) membuat GPU menganggur saat menunggu input atau gradien tahapannya. Pada 8 tahap, sekitar 12% dariGPUwaktu dapat menggelembung bahkan dengan penjadwalan 1F1B.
3. **Cross-node all-to-all.** KLH dengan paralelisme pakar menyebarkan pakar ke seluruh node. Setiap forward pass memicu all-to-all untuk mengirimkan token ke ahlinya, dan token lainnya untuk digabungkan. Pada GPU 2k, ini dengan mudah menjadi rasio komputasi-ke-komunikasi 1:1.

Masing-masing memiliki solusi terpisah: pos pemeriksaan gradien untuk memori, Zero Bubble (Sea AI Lab, 2023) untuk gelembung pipa, kernel komunikasi paralel ahli untuk semua-ke-semua. Apa yang dilakukan DualPipe adalah membuat mereka bermain bersama. Jadwal tersebut tumpang tindih dengan komputasi dan komunikasi dalam satu potongan maju-mundur, memasukkan batch mikro dari kedua ujung pipeline secara bersamaan, dan menggunakan jadwal yang dihasilkan untuk menyembunyikan semuanya di dalam jendela komputasi.

Hasil yang dilaporkan: hampir menghilangkan gelembung pipa, lebih dari 95%GPUpemanfaatan dalam pelatihan token 14,8T DeepSeek-V3.

## Konsepnya

### Penyegaran paralelisme saluran pipa

Pisahkan model lapisan-N di seluruh perangkat P. Perangkat`i`memegang lapisan`i * N/P .. (i+1) * N/P - 1`. Batch mikro mengalir maju melalui perangkat 0 ke P-1, lalu mundur dari P-1 ke 0. Setiap perangkat hanya dapat memulai tahap majunya ketika perangkat sebelumnya mengirimkan outputnya dan hanya dapat memulai mundur ketika perangkat hilir mengirimkan gradien hulu.

GPipe (Huang et al., 2019) menjadwalkan satu batch mikro dalam satu waktu, yang menghabiskan sebagian besar waktuGPUwaktu. 1F1B (Narayanan et al., 2021) menyisipkan lintasan maju dan mundur untuk beberapa mikro-batch. Zero Bubble (Qi dkk., 2023) membagi gerakan mundur menjadi dua bagian — gerakan mundur untuk masukan (B) dan gerakan mundur untuk beban (W) — dan menjadwalkannya untuk mengisi gelembung. Setelah Zero Bubble, jalur pipa hampir rapat.

DualPipe adalah langkah selanjutnya. Itu menambahkan dua ide di atas:

### Ide 1: dekomposisi bongkahan

Setiap potongan maju dibagi menjadi empat komponen:

- **Perhatian.**Q/K/Vproyeksi, perhatian, proyeksi keluaran.
- **Pengiriman menyeluruh.** Komunikasi lintas node yang mengirimkan token ke pakarnya.
- **MLP.** Perhitungan ahli KLH.
- **Penggabungan menyeluruh.** Komunikasi lintas node yang menghadirkan kembali keluaran ahli.

Potongan mundur menambahkan versi gradien dari masing-masingnya. DualPipe menjadwalkannya sehingga pengiriman semua ke semua terjadi secara paralel dengan komputasi perhatian pada potongan berikutnya, dan penggabungan semua ke semua terjadi secara paralel denganMLPmenghitung potongan berikut.

### Ide 2: penjadwalan dua arah

Sebagian besar jadwal saluran pipa memasukkan batch mikro dari tahap 0 dan mengalir menuju tahap P-1. DualPipe menyuntikkan batch mikro dariBOTHberakhir. Tahap 0 melihat kumpulan mikro maju yang berasal dari sana; tahap P-1 melihat kumpulan mikro ke depan yang berasal dari sana juga. Kedua aliran itu bertemu di tengah.

Agar ini berfungsi, perangkat`i`harus dipegangBOTHlapisan pipa awal`i` ANDlapisan pipa akhir`P - 1 - i`. Itu adalah bagian "ganda" dari DualPipe: setiap perangkat menyimpan dua salinan lapisan model yang perlu dilayani (satu untuk setiap arah). Pada skala DeepSeek-V3, biaya replikasi parameternya 2x lipat. Hal ini terjangkau karena Paralelisme Pakar sudah menyebarkan tenaga ahli di Kementerian Lingkungan Hidup begitu sedikit sehingga mereplikasi lapisan non-ahli sebanyak dua kali adalah hal yang sulit.

Yang terpenting, aliran maju dalam satu arah dan aliran mundur dalam arah lain saling tumpang tindih tepat di tempat gelembung berada dalam jadwal satu arah. Gelembungnya hilang.

### Jadwal yang ditelusuri dengan tangan

Misalkan P = 4 peringkat, 8 mikro-batch, dibagi 4 maju / 4 mundur. Waktu bergerak dari kiri ke kanan; baris adalah peringkat perangkat.

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

Membaca "F4/F5R" notasi: peringkat 1 berjalan di depan mikro-batch 4 (bergerak dari kiri ke kanan dalam pipa)ANDmaju dari micro-batch 5 (dari kanan ke kiri) di slot waktu yang sama. Itulah yang dimaksud dengan "dua arah" secara operasional.

Pada peringkat 2 aliran silang saling tumpang tindih lebih cepat, pada peringkat 0 dan P-1 aliran silang tersebut tumpang tindih paling lambat. Dalam jadwal fase tengah yang stabil, setiap peringkat berjalan maju dari arah X tumpang tindih dengan mundur dari arah Y. Komputasi sedang sibuk. Pengiriman semua-ke-semua untuk pass maju disembunyikan di dalam komputasi mundur. Kombinasi all-to-all sembunyikan di dalam komputasi maju. Gelembungnya diperas.

### Akuntansi gelembung

Gelembung pipa standar 1F1B (waktu terbuang per peringkat):

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Penyempurnaan Zero Bubble menurunkannya tetapi tidak menjadi nol. DualPipe, dalam fase stabil, memiliki gelembung nol jika jumlah mikro-batch habis dibagi 2 kali kedalaman pipa. Di luar fase stabil (pemanasan dan pendinginan), ada beberapa gelembung tetapi tidak bertambah seiring dengan jumlah batch mikro — sebuah properti utama yang disoroti makalah ini.

Dalam istilah pemasaran: "bebas gelembung". Dalam istilah teknis: gelembung tidak tumbuh dengan jumlah mikro-batch. Analisis lanjutan Sea AI Lab (DualPipeV / Cut-in-half) menunjukkan zero-bubble penuh hanya jika Paralelisme Pakar bukan penghambatnya; dengan all-to-all yang digerakkan oleh EP, beberapa kompromi penjadwalan selalu ada.

### DualPipeV — penyempurnaan

Sea AI Lab (2025) mengamati bahwa replikasi parameter 2x akan sia-sia jika komunikasi EP tidak tumpang tindih. Jadwal DualPipeV melipat injeksi dua arah menjadi jadwal "bentuk V" yang berjalan pada salinan parameter tunggal. Gelembungnya sedikit lebih besar dibandingkan DualPipe, namun penghematan memori cukup besar. DeepSeek mengadopsi DualPipeV dalam implementasi DualPipe sumber terbuka sebagai mode EP-off.

Pengorbanannya:

| Feature | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Param copies per device | 2 | 1 | 1 | 1 |
| Bubble vs micro-batches | constant | small growth | grows | grows |
| Compute-comm overlap | full | partial | minimal | partial |
| Use when | EP-heavy MoE | dense or EP-light | baseline | any pipeline |

### Apa artinya menjalankan token 14,8T

Pra-pelatihan DeepSeek-V3 menghabiskan 14,8T token pada 2.048H800GPU sekitar 2,8 jutaGPU-jam. Dengan 1F1B yang naif, mereka akan kehilangan 12-15% dari jumlah tersebut karena gelembung pipa — 340-420 ribuGPU-jam, cukup untuk melatih model 70B penuh. DualPipe memulihkan sebagian besar dari itu. Mengukur kontribusi secara langsung sulit dilakukan tanpa catatan internal, namun klaim dalam makalah ini lebih dari 95%GPUpemanfaatan rata-rata di seluruh pelatihan.

Untuk proses yang lebih kecil (di bawah 1.000 GPU), DualPipe berlebihan — gelembung pipeline lebih kecil dibandingkan total biaya, dan pelatihan model yang padat jarang mencapai hambatan yang menyeluruh. Untuk pelatihan MoE perbatasan dalam jumlah ribuanGPUskala besar, hal ini diperlukan secara efektif.

### Di mana ia berada di tumpukan

- Pelengkap **FSDP** (Tahap 10 · 05).FSDPmemecah parameter model di seluruh peringkat; DualPipe menjadwalkan komputasi di seluruh peringkat. Mereka bergabung.
- Kompatibel dengan pecahan gradien **ZeRO-3**. Pembukuan untuk replikasi dua salinan perlu bekerja sama dengan gradien pecahan ZeRO.
- Memerlukan **kernel khusus untuk semua** yang disesuaikan dengan topologi kluster tertentu. Kernel sumber terbuka DeepSeek adalah implementasi referensi.

```figure
expert-capacity
```

## Gunakan

`code/main.py`adalah simulator jadwal saluran pipa. Dibutuhkan`(P, n_micro_batches, schedule)`dan mencetak pemanfaatan fase stabil untuk masing-masing 1F1B, Zero Bubble, DualPipe, dan DualPipeV. Ini adalah alat pengajaran — angka-angka tersebut sesuai dengan klaim kualitatif di koran, namun bukan merupakan klaim tentang percepatan produksi yang diukur.

Nilai simulator: jalankan dengan jumlah P dan mikro-batch yang berbeda dan perhatikan bagaimana fraksi gelembung tumbuh untuk 1F1B tetapi tidak untuk DualPipe.

Pertimbangan integrasi untuk pelaksanaan pelatihan nyata:

- Pilih kedalaman paralel pipa yang terbagi rapi ke dalam jumlah mikro-batch kamu.
- Pastikan mesh paralel ahli kamu mendukung all-to-all dua arah. Kernel DeepSeek adalah referensinya.
- Berharap untuk menghabiskan waktu debug selama seminggu pada jadwal itu sendiri untuk pertama kalinya. Pembukuannya rumit.
- MemantauGPUpemanfaatan per peringkat, bukan hanya agregat. Manfaat DualPipe berasal dari pengetatan yang tersesat.

## Rilis

Pelajaran ini menghasilkan`outputs/skill-dualpipe-planner.md`. Diberikan spesifikasi cluster pelatihan (GPUhitungan, topologi, interkoneksi, bentuk model), ini merekomendasikan strategi paralelisme saluran pipa, algoritma penjadwalan yang akan digunakan, dan fraksi gelembung yang diharapkan pada skala target.

## Latihan

1. Jalankan`code/main.py`pada`(P=8, micro_batches=16, schedule=dualpipe)`dan`(P=8, micro_batches=16, schedule=1f1b)`. HitungGPUperbedaan pemanfaatan dan nyatakan sebagai pulihGPU-jam per juta token pelatihan.

2. Buat sketsa tabel jadwal untuk`(P=4, micro_batches=8, schedule=dualpipe)`dengan tangan. Tandai setiap slot waktu dengan ID mikro-batch dan arahnya. Identifikasi slot waktu pertama di mana tidak ada gelembung.

3. Baca Gambar 5 laporan teknis DeepSeek-V3 (arXiv:2412.19437). Identifikasi jendela yang tumpang tindih untuk pengiriman semua ke semua di dalam potongan maju DualPipe. Jelaskan bagaimana jadwal komputasi menyembunyikannya.

4. Hitung 2x parameter overhead DualPipe untuk model padat 70B dengan tahapan pipeline P=8 dan model MoE 671B dengan tahapan pipeline P=16. Tunjukkan mengapa overhead kasus Kementerian Lingkungan Hidup secara proporsional lebih kecil (sebagian besar parameternya adalah para ahli, yang dibagi ke dalam kelompok EP yang besar).

5. Bandingkan DualPipe dengan Chimera (penjadwal dua arah yang bersaing mulai tahun 2021). Identifikasi dua properti spesifik yang ditambahkan DualPipe yang tidak dimiliki Chimera, dengan menggunakan Bagian 3.4 makalah sebagai referensi.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Pipeline bubble | "Idle time per rank" | GPU cycles wasted because a pipeline stage is waiting for its input or gradient |
| 1F1B | "Default pipeline schedule" | One forward / one backward interleaved scheduling; the baseline DualPipe beats |
| Zero Bubble | "Sea AI Lab 2023" | Splits backward into B (input gradient) and W (weight gradient); almost fully tightens the pipeline |
| DualPipe | "DeepSeek-V3 schedule" | Bidirectional pipeline + compute-comm overlap; bubbles do not grow with micro-batch count |
| DualPipeV | "Cut-in-half" | V-shape refinement that drops the 2x parameter replication at the cost of slightly larger bubbles |
| Chunk | "Unit of pipeline work" | A forward or backward pass of one micro-batch through one pipeline stage |
| All-to-all dispatch | "Send tokens to experts" | Cross-node comm that routes tokens to their assigned MoE experts |
| All-to-all combine | "Bring expert outputs back" | Cross-node comm that gathers expert outputs after the MLP |
| Expert Parallelism (EP) | "Experts across GPUs" | Shards MoE experts across ranks so different GPUs hold different experts |
| Pipeline Parallelism (PP) | "Layers across GPUs" | Shards model layers across ranks; the dimension DualPipe schedules |
| Bubble fraction | "Wasted GPU time" | (bubble_time / total_time); the fraction DualPipe drives toward zero |

## Bacaan Lebih Lanjut

- [DeepSeek-AI — Laporan Teknis DeepSeek-V3 (arXiv:2412.19437), Bagian 3.3.2 dan Gambar 5](https://arxiv.org/abs/2412.19437) — referensi DualPipe utama
- [DeepSeek — Repositori DualPipe GitHub](https://github.com/deepseek-ai/DualPipe) — implementasi referensi sumber terbuka, termasuk mode DualPipeV (Cut-in-half).
- [Qi dkk. — Paralelisme Saluran Pipa Nol Gelembung (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — pendahulu Zero Bubble
- [Sea AI Lab — DualPipe bisa lebih baik tanpa Dual](https://sail.sea.com/blog/articles/63) — analisis DualPipeV yang menginformasikan mode EP-off DeepSeek
- [Narayanan dkk. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — jadwal 1F1B yang dibandingkan dengan DualPipe
- [Huang dkk. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) - masalah kertas dan gelembung paralelisme pipa asli
