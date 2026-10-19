# CAIS, CAISI, dan Risiko Skala-Sosietal

> Center for AI Safety (CAIS, San Francisco, didirikan 2022 oleh Hendrycks dan Zhang) menerbitkan kerangka empat-risiko — penyalahgunaan jahat, balapan AI, risiko organisasional, AI nakal — dan pernyataan Mei 2023 tentang risiko kepunahan yang ditandatangani ratusan profesor dan pemimpin perusahaan. Rilisan 2026 dari CAIS: AI Dashboard untuk evaluasi model-frontier, Remote Labor Index (bersama Scale AI), Superintelligence Strategy Paper, newsletter AI Frontiers. Entitas yang berbeda: NIST Center for AI Standards and Innovation (CAISI) — kesepakatan sukarela yang menghadap-pemerintah AS dan evaluasi kapabilitas tak-terklasifikasi yang berfokus pada risiko siber, bio, dan senjata-kimia. CAIS menandai risiko organisasional sebagai salah satu dari empat risiko tingkat-atas: budaya keamanan, audit yang ketat, pertahanan berlapis, dan keamanan informasi bersifat foundational tetapi rutin dikorbankan demi kecepatan deployment. California SB-53, jika ditandatangani, akan menjadi regulasi risiko-katastropik tingkat-negara-bagian AS pertama.

**Type:** Learn
**Languages:** Python (stdlib, four-risk inventory and mitigation matcher)
**Prerequisites:** Phase 15 · 19 (RSP), Phase 15 · 20 (PF + FSF)
**Time:** ~45 minutes

## Masalahnya

Pelajaran 19 dan 20 mencakup kebijakan scaling internal-lab. Pelajaran 21 mencakup evaluasi kapabilitas independen. Pelajaran ini mencakup perspektif ketiga: organisasi masyarakat sipil dan pemerintah yang membentuk diskusi publik dan baseline regulasi untuk risiko AI katastropik.

Dua entitas berbeda penting. CAIS adalah org riset nirlaba yang menerbitkan kerangka berpikir tentang risiko AI dan mengoordinasikan pernyataan publik. CAISI adalah pusat pemerintah AS dalam NIST yang menjalankan kesepakatan sukarela dengan lab dan evaluasi kapabilitas tak-terklasifikasi. Namanya berima; misinya tidak tumpang tindih. Praktisi harus mengenal keduanya.

Konten praktisnya: kerangka empat-risiko CAIS adalah taksonomi risiko skala-sosietal yang paling banyak disitasi dalam literatur. Budaya keamanan dan risiko organisasional adalah salah satu dari keempat itu, dan inilah yang paling langsung di bawah kendali praktisi. SB-53 (California) akan menjadi regulasi risiko-katastropik tingkat-negara-bagian AS pertama jika ditandatangani; bingkai RUU penting karena regulasi tingkat-negara-bagian secara historis memimpin aksi federal dalam kebijakan teknologi AS.

## Konsepnya

### CAIS — Center for AI Safety

- Didirikan: 2022 di San Francisco, oleh Dan Hendrycks dan kolega (nama "Zhang" merujuk pada kolaborator awal, bukan co-founder saat ini; lihat situs CAIS untuk kepemimpinan saat ini).
- Status: nirlaba 501(c)(3).
- Keluaran penting 2023: pernyataan tentang risiko kepunahan, ditandatangani bersama ratusan peneliti dan CEO. Menyatakan: "Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as pandemics and nuclear war."
- Keluaran 2026: AI Dashboard untuk evaluasi model-frontier, Remote Labor Index (bersama Scale AI), Superintelligence Strategy Paper, newsletter AI Frontiers.

### Kerangka empat-risiko

Kerangka CAIS mengelompokkan risiko AI katastropik ke empat kategori tingkat-atas:

1. **Penyalahgunaan jahat**: aktor buruk memakai AI untuk menyebabkan bahaya (sintesis senjata-bio, disinformasi, serangan siber).
2. **Balapan AI**: tekanan kompetitif antara lab, perusahaan, atau negara mendorong deployment melewati titik amannya.
3. **Risiko organisasional**: dinamika internal lab (kegagalan budaya-keamanan, audit yang tidak cukup, keamanan yang kurang-resource) menghasilkan deployment yang buruk.
4. **AI nakal**: AI yang cukup kapabel mengejar tujuan yang berkonflik dengan kesejahteraan manusia.

Ini bukan satu-satunya taksonomi; ini yang paling disitasi. Kategorinya tidak saling eksklusif — AI nakal yang dihasilkan organisasi yang menukar audit demi kecepatan dalam balapan adalah keempatnya sekaligus.

### Di mana risiko organisasional berada

Dari keempat kategori, risiko organisasional paling actionable bagi praktisi. Budaya keamanan lab, ketelitian audit, pelapisan pertahanan, dan keamanan informasi menentukan apakah model mereka dikirim dengan kontrol Pelajaran 10–18 benar-benar terpasang, atau apakah kontrol itu sekadar item checklist yang tidak diverifikasi siapa pun.

Tuas risiko-organisasional yang konkret:

- **Budaya keamanan**: apakah anggota tim merasa bisa mengeskalasi masalah tanpa biaya karier? Survei CAIS menemukan ini prediktor kuat bagi tuas lainnya.
- **Audit yang ketat**: eksternal dan internal. Audit hanya-internal menghasilkan laporan yang optimistis.
- **Pertahanan berlapis**: tidak ada lapisan tunggal yang cukup (tema berjalan Fase 15).
- **Keamanan informasi**: bobot model bocor, data eval bocor, teknik bypass-monitor bocor. RAND SL-4 dalam Pelajaran 19 adalah standar spesifik.

### CAISI — Center for AI Standards and Innovation

- Beroperasi dalam NIST.
- Menjalankan kesepakatan sukarela dengan lab frontier.
- Menerbitkan evaluasi kapabilitas tak-terklasifikasi yang berfokus pada risiko siber, bio, dan senjata-kimia.
- Berbeda dari CAIS; akronimnya bertabrakan; cek URL (nist.gov) untuk mengonfirmasi mana yang kamu baca.

Peran CAISI adalah padanan publik yang menghadap-pemerintah terhadap engagement lab privat METR (Pelajaran 21). Laporan CAISI tak-terklasifikasi; laporan METR sering tergerbang-NDA. Praktisi yang membaca keduanya mendapat gambaran lebih penuh.

### California SB-53

RUU Senat California (sesi 2025–2026) menangani risiko katastropik dari model frontier. Ketentuan kunci sebagaimana dirancang:

- Ambang kapabilitas spesifik yang memicu kewajiban tingkat-negara-bagian.
- Proteksi whistleblower bagi karyawan lab AI.
- Syarat pelaporan insiden untuk kegagalan katastropik.

Jika ditandatangani, ia akan menjadi regulasi risiko-katastropik tingkat-negara-bagian AS pertama. Terlepas dari status penandatanganan, bingkai RUU membentuk bagaimana legislatur negara-bagian lain mendekati soal ini. Praktisi di California harus melacak status RUU; praktisi di tempat lain harus membacanya untuk memahami seperti apa regulasi tingkat-negara-bagian AS nantinya.

### Risiko skala-sosietal bukan soal satu-lapisan

Tema berjalan Fase 15 — defense in depth — berlaku di lapisan societal juga. Tidak ada organisasi, regulasi, atau kerangka tunggal yang menutup risiko katastropik. Ekosistem berfungsi hanya ketika:

- Lab mengirim kebijakan scaling (Pelajaran 19, 20).
- Evaluator eksternal menghasilkan pengukuran (Pelajaran 21).
- Masyarakat sipil melacak dan mempublikasikan (CAIS).
- Pemerintah menjalankan program sukarela dan regulasi baseline (CAISI, SB-53).
- Praktisi membangun kontrol berlapis (Pelajaran 10–18).

Ini sintesis akhir bagi fase ini: setiap pelajaran sebelumnya adalah satu lapisan dalam stack yang kelengkapannya lebih penting dibanding kekuatan satu lapisan.

```figure
a5-four-risks
```

## Gunakan

`code/main.py` mengimplementasikan tool inventaris-risiko kecil. Dengan deployment yang diusulkan, ia menandai deployment terhadap kategori empat-risiko dan mengembalikan checklist mitigasi. Ia alat bantu baca bagi kerangka, bukan pengganti judgement manusia.

## Rilis

`outputs/skill-societal-risk-review.md` mereviu deployment untuk postur risiko skala-sosietal: kategori mana dari keempatnya yang ia sentuh, mitigasi apa yang ada, bagaimana eksposur risiko-organisasionalnya.

## Latihan

1. Jalankan `code/main.py`. Masukkan tiga deployment sintetis pada skala berbeda. Konfirmasi tag empat-risiko cocok dengan yang kamu harapkan; identifikasi satu kasus tempat tool under- atau over-tag.

2. Baca makalah empat-risiko CAIS penuh. Pilih satu kategori risiko dan tulis dua paragraf tentang apa yang menurutmu perkembangan 2026 terpenting dalam kategori itu.

3. Baca draf terkini California SB-53. Identifikasi satu ketentuan yang menurutmu memperkuat postur risiko-katastropik dan satu yang menurutmu melemahkannya. Justifikasi keduanya.

4. Pilih deployment AI produksi yang kamu kenal (milikmu atau yang dipublikasikan). Nilai terhadap sub-tuas risiko-organisasional: budaya keamanan, ketelitian audit, pertahanan berlapis, keamanan informasi. Mana yang terlemah? Berapa biayanya untuk membawanya ke par?

5. Sketsa versi 2028 kerangka empat-risiko yang mencerminkan satu tahun kapabilitas tambahan dan satu tahun pengalaman deployment tambahan. Apa yang akan kamu tambah, hapus, atau kelompokkan ulang?

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| CAIS | "Center for AI Safety" | Non-profit; four-risk framework; 2023 extinction statement |
| CAISI | "US government AI safety" | NIST Center; voluntary agreements; unclassified evals |
| Four-risk framework | "CAIS's taxonomy" | malicious use, AI races, organizational risks, rogue AIs |
| Malicious use | "Bad actor uses AI" | Bioweapons, disinformation, cyberattacks |
| AI races | "Competitive pressure" | Labs/companies/nations push deployment past safety |
| Organizational risk | "Lab internal failure" | Safety culture, audit, defenses, infosec |
| Rogue AI | "Misaligned agent" | Capable AI pursuing goals conflicting with human welfare |
| California SB-53 | "State-level regulation" | 2025–2026 bill; first US state catastrophic-risk regulation if signed |

## Further Reading

- [Center for AI Safety](https://safe.ai/) — institutional home of the four-risk framework.
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) — the four-risk paper.
- [CAIS — May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk) — short joint statement.
- [NIST CAISI](https://www.nist.gov/caisi) — government-facing AI standards and innovation center.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — connects lab-level commitments to societal-scale framing.
