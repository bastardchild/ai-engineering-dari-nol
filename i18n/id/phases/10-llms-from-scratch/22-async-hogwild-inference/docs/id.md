# Asinkron dan Hogwild! Kesimpulan

> Penguraian kode spekulatif (Fase 10 · 15) memparalelkan token dalam satu urutan. Kerangka kerja multi-agen memparalelkan seluruh rangkaian tetapi memaksakan koordinasi eksplisit (pemungutan suara, pemisahan sub-tugas). liar! Inferensi (Rodionov et al., arXiv:2504.06261) melakukan hal lain: jalankan N instance yang samaLLMsecara paralel melawan aSHAREDcache nilai kunci. Setiap pekerja melihat token yang dihasilkan setiap pekerja lainnya secara instan. Model penalaran modern — QwQ, DeepSeek-R1 — dapat berkoordinasi sendiri melalui cache bersama tersebut tanpa penyesuaian apa pun. Pendekatan ini bersifat eksperimental tetapi membuka sumbu paralelisme inferensi yang benar-benar baru dan ortogonal terhadap dekode spesifikasi. Pelajaran ini mengimplementasikan Hogwild dengan dua pekerja! simulator di stdlib Python dan menjelaskan mengapa kolaborasi cache bersama muncul dari kemampuan penalaran model yang ada.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12 (inference optimization), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan tiga persamaan paralel yang umumLLMtopologi (pemungutan suara, subtugas, Hogwild!) dan sebutkan masalah yang menjadi target masing-masing.
- Nyatakan inti Hogwild! penyiapan: banyak pekerja, satu cache KV bersama, koordinasi yang muncul melalui dorongan mandiri.
- Hitung percepatan waktu dinding Hogwild! sebagai fungsi dari jumlah pekerja`N`, paralelisme tingkat tugas`p`, dan overhead koordinasi`c`.
- Terapkan Hogwild dengan dua pekerja! simulator pada masalah mainan dan amati pembagian tugas yang muncul.

## Masalahnya

LLM modern memecahkan masalah sulit dengan menghasilkan rantai penalaran yang panjang — 5000 token logika langkah demi langkah adalah hal biasa, puluhan ribu token terjadi pada soal matematika yang mendalam. Pada usia 35tokens/secdecode pada model 70B, 50k token adalah 24 menit. Interaktif modelnya tidak.

Penguraian kode spekulatif (Fase 10 · 15) memberi kamu percepatan 3-5x dengan memparalelkan dalam satu urutan. Di luar itu, ketergantungan sekuensial dari decoding autoregresif adalah batas yang sulit. Setiap token baru bergantung pada setiap token sebelumnya.

Pertanyaan yang jelas: bisakah kita memparalelkan seluruh rangkaian? Jalankan banyak salinan dari model yang sama pada masalah yang sama, biarkan mereka bekerja sama, apakah mereka membagi pekerjaan?

Pekerjaan sebelumnya: ansambel pemungutan suara (jalankan model N, pilih jawaban mayoritas), pohon pemikiran (jalur penalaran cabang dan gabungkan kembali), dan kerangka kerja multi-agen (tetapkan sub-tugas untuk setiap agen, gunakan koordinator). Ini semua membantu dalam domain tugas tertentu. Semuanya juga memperkenalkan mesin koordinasi yang eksplisit – aturan pemungutan suara, logika cabang dan pemangkasan, protokol pengiriman pesan antar agen.

liar! Inferensi mengambil pendekatan yang berbeda. N pekerja berbagi satu cache KV. Setiap pekerja segera melihat token yang dihasilkan setiap pekerja lainnya, seolah-olah itu adalah konteksnya sendiri. Para pekerja – tanpa pelatihan atau penyesuaian apa pun – mencari cara untuk membagi pekerjaan. Model penalaran modern (QwQ, DeepSeek-R1, mode penalaran keluarga Claude) dapat membaca cache bersama dan mengatakan hal-hal seperti "Saya melihat pekerja 2 sudah menangani kasus dasar, jadi saya akan mengerjakan langkah induktif."

Percepatannya bergantung pada beban kerja dan eksperimental pada April 2026. Namun gagasan ini layak untuk diketahui karena membuka sumbu baru paralelisme inferensi.

## Konsepnya

### Pengaturan

Inisialisasi N proses pekerja, semuanya berjalan samaLLM. Alih-alih cache KV per pekerja, pertahankanONEcache bersama. Ketika pekerja`i`menghasilkan token`t_j`, token ditulis ke dalam cache bersama di posisi berikutnya. Ketika pekerja`k`mengambil langkah berikutnya, ia membaca status cache saat ini (yang mencakup semua yang telah dihasilkan oleh semua N pekerja sejauh ini).

Pada saat step time, para pekerja berlomba menulis token. Tidak ada indeks posisi per pekerja — cache adalah satu urutan yang terus bertambah. Pesanan ditentukan oleh waktu kedatangan tertulis.

### Mengapa koordinasi muncul

Para pekerja membagikan perintah. Biasanya sesuatu seperti "kamu adalah salah satu dari N instance yang bekerja bersama dalam masalah ini. Setiap instance membaca memori bersama dan dapat melihat apa yang telah ditulis oleh instance lain. Hindari pekerjaan yang berlebihan." Prompt ditambah cache bersama sudah cukup. Model penalaran membaca cache, memperhatikan bagian mana dari masalah yang telah dicoba, dan (seringkali namun tidak selalu) beralih ke bagian yang belum dijelajahi.

Alam liar! makalah (Rodionov et al., 2025) melaporkan pengamatan seperti:

- Pekerja merumuskan rencana dan mengkomunikasikannya kepada pekerja lain melalui cache.
- Pekerja memperhatikan kesalahan dalam penalaran pekerja lain dan memberitahukannya.
- Pekerja beradaptasi ketika sebuah rencana gagal dan mengusulkan alternatif.
- Saat diminta untuk memeriksa redundansi, pekerja mendeteksinya dan melakukan pivot.

Semua ini tidak memerlukan penyesuaian. Perilaku yang muncul berasal dari kemampuan penalaran yang sudah dimiliki model.

### Penamaan

Nama surat kabar tersebut mengacu pada Hogwild!SGD(Recht et al., 2011), pengoptimal pembaruan asinkron. Analoginya:SGDsemua pekerja asinkron menulis ke vektor parameter bersama; liar! Semua pekerja inferensi menulis ke cache KV bersama. Keduanya mengandalkan konvergensi empiris daripada jaminan sinkronisasi.

### RoPE membuat ini mudah diatur

Rotary Position Embeddings (RoPE, Su et al. 2021) mengkodekan informasi posisi melalui rotasi pada vektor Q dan K. Karena posisi adalah rotasi dan bukan offset yang dimasukkan, posisi token dapat bergeser tanpa menghitung ulang entri cache KV. Ketika pekerja`i`menulis ke dalam cache bersama pada posisi`p`, pekerja lain yang membaca posisi tersebut dapat menggunakan entri cache secara langsung — tidak perlu rotasi ulang.

Dalam model posisi terpelajar atau posisi absolut, Hogwild! akan memerlukan pembatalan cache pada setiap penulisan bersamaan. RoPE memungkinkan cache tetap stabil.

### Matematika waktu dinding

Biarkan`T_serial`menjadi waktu bagi satu pekerja untuk menyelesaikan masalahnya sendirian. Membiarkan`p`menjadi pecahan tingkat tugas yang dapat diparalelkan. Biarkan`c`menjadi overhead koordinasi per langkah (membaca cache yang diperluas, memutuskan apa yang akan ditulis).

Waktu pekerja tunggal:`T_serial`.
N-pekerja Hogwild! waktu, jika koordinasi bebas:`T_serial * ((1 - p) + p / N)`. Amdahl Klasik.
Dengan overhead koordinasi:`T_serial * ((1 - p) + p / N) + c * steps_per_worker`.

Agar seorang pekerja menjadi produktif,`c`harus relatif kecil terhadap waktu dekode per langkah. Pada model penalaran yang menghasilkan 5k+ token, para pekerja mampu membayar ratusan token untuk overhead koordinasi dan tetap menjadi yang terdepan. Pada tugas obrolan singkat, koordinasi mendominasi dan Hogwild! lebih buruk dari serial.

### Contoh konkrit

Masalah penalaran: 10 ribu token rantai pemikiran. Misalkan masalahnya ada`p = 0.7`konten yang dapat diparalelkan (strategi pembuktian berbeda, analisis kasus berbeda) dan`c = 200`token overhead koordinasi per pekerja. Dengan`N = 4`pekerja:

- Waktu serial: 10.000 langkah dekode.
- liar! waktu: 10.000 * (0,3 + 0,7 / 4) + 200 * 4 = 10.000 * 0,475 + 800 = 5550 langkah dekode.
- Kecepatan: 10.000/5550 = 1,8x.

Itu sederhana. Namun pada masalah penalaran yang lebih panjang (50 ribu token), biaya koordinasi diamortisasi dan percepatannya meningkat 2,5-3x. liar! adalah inferensi yang setara dengan paralelisme tingkat thread dalam bahasa yang memungkinkan kamu menulis kode multi-thread secara alami.

### Kapan harus meraih Hogwild!

- Masalah penalaran yang panjang (ribuan token) di mana tugas dapat diparalelkan ke seluruh sub-tujuan yang independen.
- Model penalaran yang telah dilatih berpikir langkah demi langkah. Model non-penalaran tidak dapat berkoordinasi dengan baik.
- Penerapan node tunggal sudah cukupVRAMuntuk menyimpan cache bersama ditambah N proses pekerja. Cache dibagikan, tetapi setiap pekerja memiliki memori aktivasi sendiri.

### Kapan tidak

- Obrolan interaktif singkat. Overhead koordinasi mendominasi.
- Tugas yang tidak diparalelkan (bukti linier tunggal, kompilasi tunggal). N=1 adalah maks.
- Model non-penalaran. Tidak ada koordinasi yang muncul.
- Penerapan multi-node. Cache bersama memerlukan sinkronisasi lintas pekerja yang sangat cepat. Intra-node baik-baik saja; cross-node adalah bencana latensi.

### Status eksperimental

Pada April 2026, Hogwild! adalah metode penelitian dengan implementasi PyTorch open-source. Adopsi produksi belum terjadi. Tiga pemblokir:

1. Manajemen cache KV bersama di seluruh proses bersamaan adalah rekayasa yang tidak sepele.
2. Koordinasi yang muncul bergantung pada tugas; tolok ukur masih dibangun.
3. Percepatannya tidak terlalu besar jika dibandingkan dengan apa yang telah dihasilkan oleh decoding spekulatif, dan keduanya dapat digabungkan tetapi teknik gabungannya adalah lapisan yang lain.

Perlu diketahui. Layak untuk dicoba. Belum layak untuk dipertaruhkan suatu produk.

```figure
continuous-batching
```

## Bangun Sendiri

`code/main.py`mengimplementasikan mainan Hogwild! simulator:

- Dua proses pekerja, masing-masing bersifat deterministik "LLM" yang menghasilkan salah satu dari beberapa kategori token (token kerja, token observasi, token koordinat) dengan probabilitas yang diketahui.
- Cache bersama (hanya daftar token) yang dibaca dan ditulis oleh kedua pekerja.
- Logika koordinasi sederhana: ketika seorang pekerja melihat bahwa pekerja lain telah menghasilkan cukup token kerja dalam suatu kategori, ia akan memilih kategori yang berbeda.

Simulator berjalan dengan anggaran langkah tetap dan melaporkan:

- Total token kerja yang dihasilkan.
- Total waktu dinding (jumlah langkah pekerja).
- Percepatan efektif pada satu pekerja.
- Jejak pekerja mana yang menulis token yang mana.

### Langkah 1: cache bersama

Daftar yang ditambahkan oleh kedua pekerja. Penguncian sederhana (Python`threading.Lock`) dalam pelaksanaan nyata; kami mensimulasikan dengan counter.

### Langkah 2: lingkaran pekerja

Setiap pekerja, di setiap langkah:

- Membaca cache bersama saat ini.
- Memutuskan kategori token apa yang akan ditulis berdasarkan apa yang sudah ada.
- Menulis satu token.

### Langkah 3: heuristik koordinasi

Jika kategori X sudah memiliki token K dalam cache dan kategori yang dimaksudkan pekerja adalah X, pekerja beralih ke kategori Y. Ini adalah mainan pengganti untuk perilaku model penalaran "perhatikan ini sudah tercakup, lakukan sesuatu yang lain sebagai gantinya."

### Langkah 4: percepatan terukur

Jalankan simulator dengan N=1 pekerja dan dengan N=2 pekerja, total anggaran langkah yang sama. Hitung token kerja yang dihasilkan. N=2 akan menghasilkan token kerja sekitar 1,5-1,8x lebih banyak karena pembagian tugas yang didorong oleh koordinasi.

### Langkah 5: menekankan koordinasi

Kurangi sensitivitas heuristik koordinasi. Jalankan lagi. Perhatikan bahwa tanpa koordinasi yang baik, N=2 secara berlebihan menghasilkan token yang sama dan percepatannya turun di bawah 1. Hal ini sesuai dengan pengamatan makalah: trik ini hanya berhasil jika pekerja memiliki kapasitas penalaran untuk berkoordinasi sendiri.

## Gunakan

liar! integrasi dalam produksi pada April 2026 adalah tingkat penelitian. Implementasi referensi dariYandex/HSE/ISTberbasis PyTorch dan menargetkan pengaturan multi-proses node tunggal pada model DeepSeek-R1 dan QwQ.

Jalur adopsi pragmatis:

1. Profil beban kerja tugas penalaran kamu. Ukur pecahan token yang bersifat eksploratif (berbagai strategi, analisis kasus, pencarian) vs linier.
2. Jika eksplorasi mendominasi, jalankan Hogwild yang beranggotakan dua orang! percobaan. Ukur peningkatan waktu dinding.
3. Jika peningkatannya di bawah 1,3x, kamu berada dalam rezim yang didominasi koordinasi. Kembali ke pekerja tunggal.
4. Jika peningkatannya lebih dari 1,5x, tekan ke N=4 dan ukur lagi. Pendapatan yang semakin berkurang biasanya mencapai sekitar N=4-8.

Kombinasikan dengan decoding spekulatif: masing-masing Hogwild! pekerja dapat secara mandiri menggunakan dekode spesifikasi. Kedua percepatan tersebut berlipat ganda (kira-kira), menghasilkan dekode spesifikasi 3x dan 1,8x Hogwild! hingga 5,4x lebih efektif dibandingkan decoding pekerja tunggal yang naif.

## Rilis

Pelajaran ini menghasilkan`outputs/skill-parallel-inference-router.md`. Mengingat profil beban kerja yang masuk akal (anggaran token, profil paralelisme tugas, kelompok model, target penerapan), ini mengarahkan antara strategi pemungutan suara, pohon pemikiran, multi-agen, Hogwild!, dan penguraian kode spekulatif.

## Latihan

1. Jalankan`code/main.py`dengan pengaturan default. Konfirmasikan N=2 Hogwild! konfigurasi menghasilkan lebih banyak token kerja daripada garis dasar N=1 dalam waktu dinding yang sama.

2. Kurangi kekuatan heuristik koordinasi (set`coordination_weight=0.1`). Memutarkan lagi. Tunjukkan bahwa percepatannya runtuh. Jelaskan alasannya: para pekerja menggandakan upaya ketika mereka tidak dapat berkoordinasi.

3. Hitung Hogwild yang diharapkan! mempercepat tugas penalaran 50 ribu token dengan`p=0.8, c=500`dan N=4 pekerja. Lakukan hal yang sama untuk tugas obrolan 1k-token dengan`p=0.3, c=200`dan N=4. Mengapa yang satu menang dan yang lainnya kalah?

4. Baca Hogwild! makalah Bagian 4 (evaluasi awal). Identifikasi dua mode kegagalan yang penulis laporkan. Jelaskan bagaimana koordinasi yang lebih baik dapat memitigasi masing-masing hal tersebut.

5. Gabungkan Hogwild! dengan decoding spekulatif di dalam mainan: setiap pekerja menggunakan decode spesifikasi 2 token secara internal. Laporkan percepatan perkalian. Masalah pembukuan apa yang muncul ketika dua pekerja ingin memperluas awalan cache bersama yang sama?

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Hogwild! | "Parallel workers, shared cache" | N instances of the same LLM running concurrently with one shared KV cache; emergent coordination via self-prompting |
| Shared KV cache | "The coordination medium" | A single growing KV buffer that all workers read and write; enables instant token visibility across workers |
| Emergent coordination | "No training needed" | Reasoning-capable LLMs can read the shared cache and divide work without any fine-tuning or explicit protocol |
| Coordination overhead (c) | "Tokens spent orienting" | The per-worker cost of reading the extended cache and deciding what to do; must stay small vs total decode time |
| Parallelizable fraction (p) | "What can run in parallel" | Task-level parallelism: the fraction of the total work that is not intrinsically sequential |
| RoPE enables Hogwild! | "Rotary positions are shift-invariant" | Because positions are rotations, writing into a shared cache does not require recomputing prior tokens |
| Voting ensemble | "Run N, pick the majority" | The simplest parallel inference topology; useful for classification, less for long-form reasoning |
| Tree of thought | "Branch and prune" | Reasoning strategy that explores multiple branches and prunes; explicit coordination logic |
| Multi-agent framework | "Assign sub-tasks" | Each agent gets a role; a coordinator orchestrates; heavy protocol overhead |

## Bacaan Lebih Lanjut

- [Rodionov dkk. — Alam liar! Inferensi: ParalelLLMPembuatan melalui Perhatian Bersamaan (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) — Alam liar! makalah, evaluasi awal pada QwQ dan DeepSeek-R1
- [Recht, Re, Wright, Niu — Hogwild!: Pendekatan Tanpa Kunci untuk Memparalelkan Penurunan Gradien Stokastik (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — Hogwild! yang asli, asal penamaannya
- [Su dkk. — RoFormer: Transformator yang Ditingkatkan dengan Penyematan Posisi Putar (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE, properti yang membuat inferensi cache bersama dapat dilakukan
- [Yao dkk. — Pohon Pemikiran: Pemecahan Masalah yang Disengaja dengan Model Bahasa Besar (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — strategi penalaran pohon pemikiran Hogwild! duduk ortogonal ke
- [Leviathan dkk. — Inferensi Cepat dari Transformers melalui Penguraian Kode Spekulatif (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — penguraian kode spekulatif, paralelisme dalam urutan Hogwild! menulis dengan
- [Hogwild! referensi implementasi PyTorch](https://github.com/eqimp/hogwild_llm) — satu-satunya sumber kebenaran untuk eksperimen makalah ini
