# Ekonomi Agen, Insentif Token, Reputasi

> Agen otonom horizon panjang (kurva kerja 1-jam ke 8-jam dari METR) butuh agensi ekonomi. **5-layer stack** yang muncul: **DePIN** (komputasi fisik) → **Identity** (W3C DIDs + modal reputasi) → **Cognition** (RAG + MCP) → **Settlement** (account abstraction) → **Governance** (Agentic DAO). Jaringan insentif agen produksi mencakup **Bittensor** (subnet TAO memberi reward pada model spesifik-tugas), **Fetch.ai / ASI Alliance** (LLM ASI-1 Mini + token FET), dan **Gonka** (proof-of-work berbasis transformer yang mengalokasikan ulang komputasi ke tugas AI produktif). Karya akademik: LaMAS terdesentralisasi di AAMAS 2025 memakai **Shapley-value credit attribution** untuk memberi reward adil pada agen kontributor; riset Google "Mechanism design for large language models" mengusulkan **token auctions** dengan pembayaran second-price di bawah agregasi monoton. Pelajaran ini membangun marketplace agen minimal, menerapkan Shapley-value credit attribution pada pipeline multi-agent, dan menjalankan token auction second-price supaya mesin teori permainannya mendarat secara konkret.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 16 (Negotiation and Bargaining), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## Masalah

Sistem multi-agent jadi rumit ketika agen menghasilkan nilai secara bersama tapi harus diberi reward secara individual. Mekanisme klasik — pembagian rata, last-contributor-takes-all — tidak adil atau bisa dimanfaatkan. Pemberian reward berbasis koalisi lewat Shapley value adil secara konstruksi tapi mahal dihitung. Literatur 2025-2026 mendorong aproksimasi yang berguna: Shapley sampling, auction agregasi monoton, dan reputasi on-chain yang menumpuk dari kontribusi terkonfirmasi.

Selain credit attribution, bidang ini beralih ke agen ekonomi sungguhan: TAO Bittensor memberi reward pada komputasi mining untuk fine-tune model spesifik-subnet, Fetch.ai/ASI memberi reward pada pemakaian LLM ASI-1 Mini dengan token FET, Gonka mengalokasikan ulang transformer proof-of-work ke tugas AI produktif. Agen yang bertransaksi secara otonom sudah ada hari ini; pertanyaannya adalah bagaimana menyelaraskan insentif.

Pelajaran ini memperlakukan ekonomi agen sebagai keluarga masalah spesifik — credit attribution, mechanism design, dan reputasi — dan membangun semuanya dengan matematika minimal supaya idenya melekat.

## Konsep

### Stack ekonomi-agen 5 lapis

1. **DePIN (komputasi fisik).** Infrastruktur terdesentralisasi yang menyewakan GPU, penyimpanan, bandwidth. Subnet Bittensor, Render Network, Akash. Bukan khusus-agen; agen yang memakainya.
2. **Identity.** W3C Decentralized Identifiers (DIDs) memberi tiap agen ID yang tahan lama dan independen dari platform mana pun. Reputasi menumpuk pada DID. Agent Network Protocol (ANP) memakai DID sebagai lapisan discovery.
3. **Cognition.** Loop penalaran agen: LLM + RAG + MCP. Inilah yang dibangun oleh phase lainnya.
4. **Settlement.** Account abstraction (ERC-4337) memungkinkan agen membayar gas dari saldo mereka sendiri tanpa memegang ETH. Agen bisa membayar layanan, sesama agen, atau komputasi.
5. **Governance.** Agentic DAO: struktur tata kelola tempat manusia *dan* agen memberi suara pada perubahan protokol, dengan kekuatan suara terikat pada reputasi.

Tidak semua sistem produksi memakai kelima lapisan itu. Bittensor memakai 1, 2, sebagian 3, sebagian 4, tidak ada dari 5. Agen OpenAI memakai tidak ada kecuali 3. Stack ini peta referensi, bukan persyaratan.

### Bittensor, Fetch.ai, Gonka — apa yang berjalan

**Bittensor (TAO).** Subnet adalah tugas terspesialisasi (language modeling, image generation, forecasting). Miner mengirim output model. Validator memberi peringkat; skoring berbobot stake mendistribusikan reward TAO. Tiap subnet punya evaluasi sendiri. Pelajaran ekonominya: bayar untuk kualitas output spesifik-tugas, bukan komputasi yang dipakai.

**Fetch.ai / ASI Alliance.** LLM ASI-1 Mini berjalan di jaringan Fetch.ai; pengguna membayar token FET untuk inference. Narasi agents-as-peers lebih kuat di sini: agen di Fetch bisa memanggil agen lain untuk sebuah tugas dan membayar dalam FET.

**Gonka.** Transformer proof-of-work: "pekerjaannya" adalah forward pass sebuah transformer. Miner memperoleh penghasilan dengan menjalankan tugas inference yang punya output benar yang diketahui (dari data training). PoW yang produktif-terhadap-sumber daya, bukan PoW berbasis hash.

Ketiganya berkelas produksi per April 2026. Distribusi payoff berbeda. Bittensor memberi reward kualitas relatif terhadap validator subnet; Fetch memberi reward utilitas yang diukur oleh pengguna yang membayar; Gonka memberi reward kerja inference yang dapat diverifikasi.

### Shapley-value credit attribution

Tiga agen berkolaborasi pada satu tugas. Outputnya skor 0.8. Siapa berkontribusi apa?

Shapley value: alokasi kredit unik yang memenuhi empat aksioma (efficiency, symmetry, linearity, null). Untuk agen `i`:

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

di mana `S_i_O` adalah himpunan agen sebelum `i` dalam ordering `O`. Dalam praktik: enumerasi semua permutasi, catat kontribusi marjinal tiap agen di tiap permutasi, ambil rata-rata.

Untuk N=3 agen, ada 6 permutasi. Untuk N=10, 3,6 juta — jadi dalam praktiknya kamu sampling ordering alih-alih mengenumerasi.

### Auction second-price untuk agregasi

Google Research ("Mechanism design for large language models") mengusulkan token auction second-price untuk mengagregasi output LLM. Setup-nya: N agen masing-masing mengusulkan satu completion; masing-masing punya nilai privat untuk terpilih. Lelang memilih proposal bernilai tertinggi dan membayar nilai *tertinggi kedua*. Di bawah agregasi monoton (nilai bergantung pada proposal mana yang dipilih, bukan berapa banyak yang diajukan), ini truthful — agen menawar dengan nilai sebenarnya.

Kenapa ini penting bagi sistem LLM: kamu bisa meng-outsourc-kan tugas completion ke beberapa agen dengan harga berbeda; lelang memilih yang terbaik + membayar secara adil, dan agen tidak punya motif untuk melaporkan nilai palsu.

### Modal reputasi

Skor reputasi terikat DID menumpuk dari kontribusi terkonfirmasi. Aturan update sederhana:

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

Dengan faktor decay `alpha` yang mendekati 1. Reputasi:

- Murah dibaca untuk keputusan routing ("kirim tugas sulit ke agen ber-rep tinggi").
- Mahal dipalsukan (menumpuk seiring waktu, terikat pada DID).
- Bisa di-slash: kontribusi yang gagal verifikasi dikurangi.

### LaMAS terdesentralisasi AAMAS 2025

Proposal LaMAS (AAMAS 2025) menggabungkan: identitas DID, Shapley-value credit attribution, dan mekanisme lelang sederhana. Klaim kuncinya: mendesentralisasi langkah credit attribution membuat sistem dapat diaudit dan kebal terhadap manipulasi titik-tunggal.

### Di mana ekonominya runtuh

- **Manipulasi price oracle.** Jika fungsi kredit bisa dimainkan, agen akan memainkannya. Setiap mekanisme butuh tes advesarial.
- **Serangan Sybil.** Satu operator memutar N agen palsu untuk membesar-besarkan kontribusinya sendiri. DIDs memperlambat tapi tidak menghentikan ini; biaya-reputasi-untuk-memalsukan adalah mitigasinya.
- **Biaya verifikasi.** Credit attribution hanya se-adil verifiernya. Jika verifikasi murah (LLM kecil), bisa dimainkan; jika mahal (panel manusia), sistemnya tidak skalabel.
- **Regulatory overhang.** Ekonomi agen beririsan dengan regulasi keuangan. Bittensor, Fetch, dan Gonka semuanya beroperasi di area abu-abu hukum di beberapa yurisdiksi per 2026.

### Kapan ekonomi agen masuk akal

- **Jaringan terbuka dengan operator heterogen.** Tidak ada satu tim yang mengendalikan semua agen.
- **Output yang dapat diverifikasi.** Tanpa verifikasi, credit attribution hanya tebakan.
- **Workflow horizon panjang.** Tugas sekali-jalan tidak diuntungkan dari penumpukan reputasi.
- **Pembayaran berbasis token legal** di yurisdiksimu.

Dalam sistem korporat tertutup, ekonomi mengalah pada alokasi yang lebih sederhana (manajer menugaskan pekerjaan, metrik bersifat internal). Literatur ekonomi lebih banyak berlaku untuk jaringan terbuka.

```figure
swarm-auction
```

## Bangun Sendiri

`code/main.py` mengimplementasikan:

- `shapley(value_fn, agents)` — perhitungan Shapley eksak lewat enumerasi untuk N kecil.
- `second_price_auction(bids)` — mekanisme truthful; pemenang membayar tertinggi kedua.
- `Reputation` — reputasi terikat DID dengan decay eksponensial dan slashing.
- Demo 1: tiga agen berkolaborasi, Shapley eksak mengatribusikan kredit.
- Demo 2: lima agen menawar untuk satu slot tugas; auction second-price memilih pemenang + pembayaran.
- Demo 3: 100 ronde penugasan tugas ke agen dengan rep heterogen; routing berbobot-rep mengalahkan acak.

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan: Shapley value tiap agen; hasil lelang yang menunjukkan equilibrium penawaran truthful; routing berbobot-rep yang menunjukkan peningkatan kualitas 10-20% di atas acak setelah warmup.

## Gunakan

`outputs/skill-economy-designer.md` merancang ekonomi agen minimal: pilihan lapisan identitas, mekanisme credit attribution, mekanisme pembayaran, aturan reputasi.

## Rilis

Menjalankan ekonomi agen pada 2026:

- **Mulai dari reputasi, bukan token.** Reputasi murah diimplementasikan dan berharga sendirian; token menambah kompleksitas hukum dan ekonomi.
- **Verifikasi sebelum memberi reward.** Jangan pernah mendistribusikan kredit tanpa langkah verifikasi independen. Kualitas yang dilaporkan sendiri melahirkan permainan sybil.
- **Shapley-sample, bukan Shapley-exact.** Sampling 100-1000 ordering; enumerasi eksak tidak skalabel.
- **Batasi faktor decay dan beri floor pada reputasi.** Decay tak-berbatas menghapus kontributor sah; decay terlalu lambat memberi reward pada agen ber-rep tinggi yang basi.
- **Audit mekanisme secara advesarial.** Jalankan skenario red-team sebelum membuka jaringan. Setiap mekanisme punya teori permainan; kamu ingin menemukan celahnya, bukan para penyerangnya.

## Latihan

1. Jalankan `code/main.py`. Pastikan jumlah Shapley value sama dengan total nilai (aksioma efficiency). Ubah value function-nya; apakah alokasi Shapley berubah ke arah yang diharapkan?
2. Implementasikan Shapley *sampling* (Monte Carlo atas K ordering). Bagaimana K memengaruhi akurasi aproksimasi? Bandingkan dengan eksak untuk N=4.
3. Implementasikan langkah pembentukan koalisi sebelum lelang: agen boleh bergabung menjadi tim dan menawar sebagai satu unit. Koalisi mana yang terbentuk? Apakah hasilnya Pareto-lebih-baik daripada penawaran individual?
4. Baca post mechanism-design Google Research. Identifikasi satu asumsi yang, jika dilanggar, merusak truthfulness. Seperti apa mode gagal itu dalam setting LLM?
5. Baca paper LaMAS terdesentralisasi AAMAS 2025. Implementasikan langkah Shapley mereka atas 10 agen pada tugas sintetis. Berapa lama perhitungan eksak? Seberapa dekat sampling dengan 100 draw?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| DePIN | "Decentralized physical infrastructure" | Token-incentivized compute/storage/bandwidth. Bittensor, Akash, Render. |
| DID | "Decentralized identifier" | W3C spec for portable IDs. Agent reputation binds to DID, not to a platform. |
| ERC-4337 | "Account abstraction" | Contract accounts that can sponsor gas, enabling agent payments. |
| Shapley value | "Fair credit attribution" | Unique allocation satisfying efficiency, symmetry, linearity, null. |
| Second-price auction | "Vickrey auction" | Truthful mechanism: winner pays second-highest bid. Monotone aggregation compatible. |
| Reputation capital | "Accumulated quality score" | DID-bound score from confirmed contributions; decays over time. |
| Agentic DAO | "Agents + humans govern" | DAO with agent voters as first-class, voting power tied to reputation. |
| TAO / FET / GPU credits | "Token denominations" | Bittensor TAO, Fetch.ai FET, various DePIN tokens. |

## Bacaan Lanjutan

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 2026 survey of the 5-layer agent-economy stack
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — token auctions with monotone aggregation
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Shapley-value credit attribution
- [Bittensor TAO documentation](https://docs.bittensor.com/) — subnet structure and reward distribution
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM and FET token
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — identity foundation
