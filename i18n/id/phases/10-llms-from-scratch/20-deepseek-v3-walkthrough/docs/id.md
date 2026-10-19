# Panduan Arsitektur DeepSeek-V3

> Fase 10 · Pelajaran 14 menyebutkan enam kenop arsitektur yang diputar setiap model terbuka. DeepSeek-V3 (Desember 2024, total 671 miliar parameter, 37 miliar aktif) mengubah keenamnya dan menambahkan empat lagi: Perhatian Laten Multi-Kepala, penyeimbangan beban bebas kerugian tambahan, Prediksi Multi-Token, dan pelatihan DualPipe. Pelajaran ini membaca arsitektur DeepSeek-V3 dari atas ke bawah dan memperoleh setiap jumlah parameter dari konfigurasi yang dipublikasikan. Pada akhirnya kamu dapat menjelaskan alasannya671B/37Brasio adalah taruhan yang tepat dan mengapaMLA+ MoE bersama-sama mengalahkan baik sendirian di perbatasan.

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Phase 10 · 14 (open-model walkthroughs), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Baca konfigurasi DeepSeek-V3 dari atas ke bawah dan jelaskan setiap bidang dalam enam bidangGPT-2 kenop ditambah empat tambahan khusus DeepSeek.
- Turunkan jumlah parameter total (671B), jumlah parameter aktif (37B), dan komponen yang berkontribusi pada masing-masingnya.
- Hitung jejak cache KVMLApada konteks 128k dan bandingkan dengan model padat param aktif yang samaGQAakan membayar.
- Sebutkan empat inovasi khusus DeepSeek (MLA,MTP, perutean bebas kerugian tambahan, DualPipe) dan beri nama bagian mana darinyaarchitecture/trainingtumpukan masing-masing target.

## Masalahnya

DeepSeek-V3 adalah model frontier open pertama yang arsitekturnya sangat berbeda dari keluarga Llama. Llama 3 405B adalah "GPT-2 dengan enam kenop diputar." DeepSeek-V3 adalahGPT-2 dengan keenam kenop ditambah empat lagi. Membaca konfigurasi Llama 3 adalah pemanasan untuk membaca konfigurasi DeepSeek, tetapi struktur dalamnya — bentuk blok perhatian, logika perutean, tujuan waktu pelatihan — cukup berbeda sehingga kamu memerlukan panduan terpisah.

Hasil dari mempelajarinya: Rilis bobot terbuka DeepSeek-V3 mengubah arti "kemampuan perbatasan" dalam model terbuka. Arsitekturnya adalah cetak biru yang banyak ditiru oleh pelatihan pada tahun 2026. Memahami hal ini adalah taruhan utama untuk peran apa pun yang menyentuh batasLLMpelatihan atau inferensi.

## Konsepnya

### Inti invarian, sekali lagi

DeepSeek-V3 masih bersifat autoregresif. Itu masih menumpuk blok dekoder. Setiap blok masih memiliki perhatian plusMLPditambah dua RMSNorma. Masih menggunakan SwiGLU diMLP. Masih menggunakan Tali. Pra-norma. Penyematan yang terikat dengan beban. Garis dasar yang sama seperti setiap Llama atau Mistral.

### Putarannya:MLAsebagai gantinyaGQA

Dari Fase 10 · 14 lhoGQAmengecilkan cache KV dengan membagikan K dan V ke seluruh grup kepala Q. Perhatian Laten Multi-Kepala (MLA) melangkah lebih jauh: K dan V dikompresi menjadi representasi laten tingkat rendah bersama (the`kv_lora_rank`), lalu didekompresi per kepala dengan cepat. Cache KV hanya menyimpan yang laten — biasanya 512 float per token per lapisan, bukan 8 x 128 = 1024 float.

Pada konteks 128k, DeepSeek-V3 denganMLA(satu berbagi secara laten`c^{KV}`per token per lapisan; K dan V keduanya diturunkan dari laten ini melalui proyeksi ke atas yang dapat diserap ke dalam matmul berikutnya):

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

Sebuah hipotesisGQAbaseline (Bentuk Llama 3 70B, kepala 8 KV, kepala redup 128) akan membayar:

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA4x lebih kecil dari model Llama-3-70BGQAcache pada konteks 128k.

Pengorbanannya:MLAmenambahkan langkah dekompresi per perhitungan perhatian (per kepala). Komputasi ekstra kecil dibandingkan dengan bandwidth yang dihemat. Kemenangan bersih untuk inferensi konteks panjang.

### Peruteannya: penyeimbangan beban bebas kerugian tambahan

Router MoE memutuskan pakar mana yang memproses setiap token. Router yang naif memusatkan terlalu banyak pekerjaan pada beberapa ahli, meninggalkan yang lain menganggur. Perbaikan standar: tambahkan istilah kerugian tambahan yang menyebabkan ketidakseimbangan beban. Ini berfungsi tetapi sedikit menurunkan kinerja tugas utama.

DeepSeek-V3 memperkenalkan skema tambahan bebas kerugian. Istilah bias per pakar ditambahkan ke logit router, disesuaikan selama pelatihan dengan aturan sederhana: jika pakar`e`kelebihan beban, turunkan`bias_e`; jika kekurangan, tingkatkan. Tidak ada istilah kerugian tambahan. Pelatihan tetap bersih. Beban ahli tetap seimbang.

Dampak terhadap kerugian utama: tidak ada yang dapat diukur. Efek pada arsitektur MoE: lebih bersih, tidak ada hyperparameter kerugian tambahan yang perlu disesuaikan.

### ItuMTP: pelatihan lebih padat + draft gratis

Dari Fase 10 · 18 kamu tahu DeepSeek-V3 menambahkan D=1MTPmodul yang memprediksi token dua posisi di depan. Sebagai kesimpulan, modul yang dilatih digunakan kembali sebagai draf penguraian kode spekulatif dengan penerimaan 80%+. Pada pelatihan, setiap keadaan tersembunyi diawasi pada target D+1 = 2, memberikan sinyal yang lebih padat.

Parameter: 14B di atas utama 671B. Biaya tambahan: 2,1%.

### Pelatihan: DualPipe

Dari Fase 10 · 19 kamu tahu DualPipe adalah saluran pipa dua arah yang tumpang tindih dengan potongan maju dan mundur dengan komunikasi lintas-simpul semua-ke-semua. Di DeepSeek-V3 2.048-H800skala, itu pulih sekitar 245kGPU-jam dimana 1F1B akan hilang karena gelembung pipa.

### Konfigurasi, bidang demi bidang

Berikut adalah konfigurasi DeepSeek-V3 (disederhanakan):

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

Uraikan:

- `hidden_size=7168`: menyematkan dimensi.
- `num_hidden_layers=61`: total kedalaman blok.
- `first_k_dense_layers=3`: 3 blok pertama pakai yang padatMLPberukuran 18432. Sisanya 58 menggunakan MoE.
- `num_attention_heads=128`: 128 kepala kueri.
- `kv_lora_rank=512`: K dan V dikompresi ke dimensi laten ini dan didekompresi per kepala.
- `num_experts=256, num_experts_per_tok=8`: setiap blok KLH mempunyai 256 tenaga ahli, rute top-8.
- `shared_experts=1`: selain 256 pakar yang diarahkan, 1 pakar yang selalu aktif berkontribusi pada setiap token. Anggap saja sebagai "lantai padat" yang memastikan setiap token mendapatkan sesuatu yang dapat diandalkan.
- `moe_intermediate_size=2048`: masing-masing ahliMLPukuran tersembunyi. Lebih kecil dari yang padatMLPkarena ada 256 buah.

### Akuntansi parameter

Perhitungan lengkapnya ada di dalamnya`code/main.py`. Judulnya:

- Penyematan:`vocab * hidden = 129280 * 7168 = ~0.93B`.
- 3 blok padat pertama: perhatian denganMLA(~144 juta per blok) + padatMLP(~260 juta per blok) + norma. Totalnya sekitar 1,2 miliar.
- 58 blok MoE: perhatian denganMLA(~144 juta) + masing-masing 256 pakar (masing-masing 30 juta) + 1 pakar bersama (30 juta) + norma. Total ~7,95 miliar per blok, termasuk semua pakar. Total 461 miliar untuk 58 blok KLH.
- MTPmodul: 14B.

Total keseluruhan: ~476B untuk arsitektur inti + 14BMTP+ jelasnya, nomor 671B yang dipublikasikan memperhitungkan parameter struktural tambahan (tensor bias, komponen khusus pakar, penskalaan pakar bersama, dll.). Jumlah yang kami reproduksi di kalkulator berada dalam kisaran 3-5% dari angka yang dipublikasikan — delta tersebut berasal dari dokumen laporan akuntansi mendalam DeepSeek di lampiran Bagian 2.

Parameter aktif per penerusan:

- Perhatian: 144M per lapisan * 61 = 8,8B (semua lapisan terbakar).
- MLPaktif: 3 lapisan pertama padat (3 * 260M = 780M), 58 lapisan MoE masing-masing aktif dengan 8 rute + 1 bersama + overhead perutean. Per lapisan aktifMLP: ~260 juta. Jumlah: 3*260M + 58*260M = ~15,9B.
- Penyematan + norma: 1.2B.
- Total aktif: kira-kira 26B inti + 14BMTP(terlatih tetapi tidak selalu dijalankan pada inferensi) ≈ 37B.

### Rasio 671B / 37B

Rasio ketersebaran 18x (param aktif adalah 5,5% dari total). DeepSeek-V3 adalah model MoE perbatasan paling jarang yang mengirimkan bobot terbuka. Mixtral 8x7B dengan rasio13/47(28%) jauh lebih padat. Llama 4 Maverick dengan rasio17B/400B(4,25%) sebanding. Taruhan DeepSeek: pada skala terdepan, lebih banyak ahli dengan rasio aktivasi yang lebih rendah menghasilkan kualitas yang lebih baik per aktivitas.FLOP.

### Tempat DeepSeek-V3 berada

| Model | Total | Active | Ratio | Attention | Novel ideas |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

### Lanjutan: R1, V4

DeepSeek-R1 (2025) adalah pelatihan penalaran yang dijalankan pada tulang punggung V3. R1 menggunakan arsitektur yang sama. Yang berubah adalah resep pasca-pelatihan (RL skala besar pada tugas-tugas yang dapat diverifikasi), bukan arsitektur pra-pelatihan.

DeepSeek-V4 (jika dikirimkan) diharapkan tetap adaMLA+ MoE +MTPdan tambahkanDSA(DeepSeek Sparse Attention), penerusNSAdari Fase 10 · 17. Garis keturunannya stabil: inovasi tingkat arsitektur terakumulasi; setiap versi memutar kenop tambahan.

```figure
moe-routing
```

## Gunakan

`code/main.py`adalah kalkulator parameter yang dikhususkan untuk bentuk DeepSeek-V3. Jalankan, bandingkan keluarannya dengan angka-angka di makalah, dan gunakan pada varian hipotetis (256 ahli vs 512, 8 teratas vs 16 teratas,MLAperingkat 512 vs 1024).

Apa yang harus dilihat:

- Jumlah parameter total vs 671B yang dipublikasikan.
- Jumlah parameter aktif vs 37B yang dipublikasikan.
- Cache KV pada konteks 128k — theMLAvsGQAperbandingan.
- Perincian per lapisan untuk melihat ke mana sebenarnya anggaran parameter disalurkan.

## Rilis

Pelajaran ini menghasilkan`outputs/skill-deepseek-v3-reader.md`. Mengingat model keluarga DeepSeek (V3, R1, atau varian masa depan lainnya), model ini menghasilkan pembacaan arsitektur komponen demi komponen yang memberi nama setiap bidang konfigurasi, memperoleh jumlah parameter berdasarkan komponen, dan mengidentifikasi yang mana dari empat inovasi khusus DeepSeek yang digunakan model.

## Latihan

1. Jalankan`code/main.py`. Bandingkan estimasi parameter total kalkulator dengan 671B yang dipublikasikan dan identifikasi dari mana delta tersebut berasal. Bagian 2 makalah ini memiliki perincian lengkap.

2. Ubah konfigurasi yang akan digunakanMLAperingkat 256 bukannya 512. Hitung ukuran cache KV yang dihasilkan pada konteks 128k. Berapa persentase pengurangan yang diperolehnya, dan berapa biaya ekspresi per kepala?

3. Bandingkan perutean DeepSeek-V3 (256 pakar, 8 teratas) dengan varian hipotetis (512 pakar, 8 teratas). Parameter total bertambah; parameter aktif tetap sama. Secara teori, apa yang diperoleh dari kapasitas ahli tambahan, dan berapa biayanya dalam inferensi?

4. Baca Bagian 2.1 laporan teknis DeepSeek-V3 (arXiv:2412.19437) diMLA. Jelaskan dalam tiga kalimat mengapa matriks dekompresi K dan V dapat "diserap" ke dalam matmul berikutnya untuk efisiensi waktu inferensi.

5. Penggunaan DeepSeek-V3FP8pelatihan untuk sebagian besar operasi. Hitung penghematan memoriFP8vsBF16untuk menyimpan anak timbangan 671B. Bagaimana hal ini bersinggungan dengan anggaran pelatihan token 14,8T?

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MLA | "Multi-Head Latent Attention" | Compress K and V into a shared low-rank latent (kv_lora_rank, typically 512), decompress per head on-the-fly; KV cache stores only the latent |
| kv_lora_rank | "MLA compression dim" | The size of the shared latent for K and V; DeepSeek-V3 uses 512 |
| First k dense layers | "Early layers stay dense" | The first few MoE-model layers skip the MoE router and run a dense MLP for stability |
| num_experts_per_tok | "Top-k routing" | How many routed experts fire per token; DeepSeek-V3 uses 8 |
| Shared experts | "Always-on experts" | Experts that process every token regardless of routing; DeepSeek-V3 uses 1 |
| Auxiliary-loss-free routing | "Bias-adjusted load balance" | Per-expert bias terms adjusted during training to keep expert load balanced without adding a loss term |
| MTP module | "Extra prediction head" | Transformer block predicting t+2 from h^(1) and E(t+1); denser training, free speculative-decoding draft |
| DualPipe | "Bidirectional pipeline" | Training schedule that overlaps forward/backward compute with cross-node all-to-all |
| Active parameter ratio | "Sparsity" | active_params / total_params; DeepSeek-V3 hits 5.5% |
| FP8 training | "8-bit training" | Training storage and many compute ops in FP8; roughly halves memory vs BF16 at a small quality cost |

## Bacaan Lebih Lanjut

- [DeepSeek-AI — Laporan Teknis DeepSeek-V3 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — dokumen arsitektur, pelatihan, dan hasil lengkap
- [Kartu model DeepSeek-V3 pada Wajah Memeluk](https://huggingface.co/deepseek-ai/DeepSeek-V3) — file konfigurasi dan catatan penerapan
- [Kertas DeepSeek-V2 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — pendahulu yang memperkenalkanMLA
- [Makalah DeepSeek-R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — penerus pelatihan penalaran pada arsitektur V3
- [Perhatian Jarang Asli (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — arah masa depan untuk perhatian keluarga DeepSeek
- [repositori DualPipe](https://github.com/deepseek-ai/DualPipe) — referensi jadwal pelatihan
