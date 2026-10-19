# Mode Gagal — MAST, Groupthink, Monoculture, Kesalahan Berantai

> Taksonomi referensi untuk 2026 adalah **MAST** (Cemri et al., NeurIPS 2025, arXiv:2503.13657), diturunkan dari 1642 jejak eksekusi pada 7 MAS open-source state-of-the-art yang menunjukkan **tingkat kegagalan 41–86.7%**. Tiga kategori akar: **Specification Problems** (41.77%) — ambiguitas peran, definisi tugas yang tidak jelas; **Coordination Failures** (36.94%) — komunikasi putus, desync state; **Verification Gaps** (21.30%) — validasi hilang, pemeriksaan kualitas absen. Keluarga **Groupthink** (arXiv:2508.05687) menambahkan: collapse monoculture (base model sama → kegagalan berkorelasi), conformity bias (agen saling memperkuat kesalahan satu sama lain), theory of mind yang kurang, dinamika mixed-motive, kegagalan keandalan berantai. Contoh berantai: retry storm di mana kegagalan pembayaran memicu retry pesanan, yang memicu retry inventaris, yang membanjiri service inventoris (beban 10x dalam hitungan detik — butuh circuit breaker). Memory poisoning: halusinasi satu agen masuk ke memori bersama, agen hulu menganggapnya sebagai fakta; akurasi merosot perlahan, membuat diagnosis akar masalah menyakitkan. **STRATUS** (NeurIPS 2025) melaporkan peningkatan sukses-mitigasi 1,5x lewat agen deteksi / diagnosis / validasi yang terspesialisasi. Pelajaran ini memperlakukan mode gagal sebagai target rekayasa kelas satu.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 13 (Shared Memory), Phase 16 · 14 (Consensus and BFT), Phase 16 · 15 (Voting and Debate Topology)
**Time:** ~75 minutes

## Masalah

Sistem multi-agent gagal 41-86.7% dari waktu pada tugas nyata (Cemri et al. 2025 mengukur ini pada 7 MAS open-source). Itu tidak bisa didebug dengan "tambah saja agennya." Kegagalannya punya sebab struktural. Taksonomi MAST memberimu kategorinya. Pelajaran ini memetakan tiap kategori ke pola deteksi, diagnosis, dan mitigasi yang konkret supaya angka-angka itu berhenti terlihat sewenang-wenang.

Praktik produksi 2026 adalah memperlakukan mode gagal sebagai masukan desain. Arsitekturmu belum "cukup baik" sampai kamu bisa menunjuk tiap kategori MAST dan menyebut mitigasi yang kamu terapkan.

## Konsep

### Kategori MAST

**Specification Problems (41.77% dari kegagalan).** Tugas agen tidak didefinisikan dengan cukup ketat. Contoh:

- Ambiguitas peran: dua agen sama-sama mengira dirinya reviewer.
- Tugas kurang spesifik: "ringkas ini" padahal pengguna menginginkan sudut pandang tertentu.
- Kriteria sukses tersirat: agen tidak bisa menilai apakah ia berhasil.

Mitigasi:
- Tulis kontrak peran yang eksplisit. Prompt tiap agen menyatakan apa yang ia lakukan *dan apa yang tidak ia lakukan*.
- Acceptance test per tugas. Sebelum agen mulai, definisikan "selesai itu terlihat seperti X."
- Pre-flight spec check: agen terpisah meninjau definisi tugas sebelum dispatch.

**Coordination Failures (36.94%).** Kegagalan komunikasi atau state.

Contoh:
- Dua agen memperbarui state bersama tanpa sinkronisasi.
- Pesan hilang antar agen (queue failure, timeout).
- State drift: agen A mengira tugas selesai; agen B masih mengeksekusi.

Mitigasi:
- State bersama yang ber-versi dengan optimistic concurrency.
- Acknowledgment eksplisit untuk pesan kritis (retry sampai di-ack).
- Checkpoint sinkronisasi state berkala; deteksi drift sedini mungkin.

**Verification Gaps (21.30%).** Tidak ada pemeriksaan independen atas output.

Contoh:
- Satu agen mengklaim berhasil; tidak ada yang memverifikasi.
- Rantai agen yang semuanya mempercayai output pendahulunya.
- Test coverage hilang pada perilaku komposit yang muncul.

Mitigasi:
- Agen verifier independen (Pelajaran 13). Hanya-baca, akses sumber independen.
- Kontrak handoff eksplisit: "output A harus lolos pemeriksaan C sebelum B mulai."
- Log hasil untuk analisis pasca-peristiwa.

### Keluarga Groupthink (arXiv:2508.05687)

Lima kegagalan terkait saat agen dihomogenkan atau meniru satu sama lain:

**Monoculture collapse.** Base model atau data training sama → kesalahan berkorelasi. Ketika tiga agen berbagi satu LLM, mereka juga berbagi halusinasinya.

**Conformity bias.** Agen menyesuaikan diri ke peer paling keras atau paling percaya diri, meski salah.

**Deficient ToM.** Agen gagal memodelkan keyakinan satu sama lain; koordinasi runtuh (Pelajaran 18).

**Mixed-motive dynamics.** Agen dengan insentif yang sebagian selaras bergeser ke kompromi-tengah yang memuaskan tidak ada siapa pun.

**Cascading reliability failures.** Pola kesalahan satu komponen memicu pola kesalahan pada komponen yang bergantung padanya.

### Contoh berantai — retry storm

Pola insiden klasik 2026:

```
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load
   ↓
inventory service starts timing out
   ↓
every order retries inventory check
   ↓
inventory service sees 10x normal load
   ↓
cluster goes down
```

Perbaikannya klasik: **circuit breaker**. Saat tingkat kesalahan downstream melewati ambang, short-circuit dengan hasil cache atau default. Ditambah anggaran retry berbatas per permintaan.

Circuit breaker adalah salah satu dari sedikit mitigasi kegagalan multi-agent yang langsung kamu pinjam dari distributed systems tanpa modifikasi.

### Memory poisoning (dibahas ulang)

Dari Pelajaran 13: halusinasi satu agen menjadi fakta memori-bersama; agen hulu menalar di atas fakta yang tercemar itu. Dalam istilah MAST, ini adalah verification gap di lapisan memori bersama.

Merosotnya akurasi secara bertahap adalah gejalanya. Kamu tidak mendapat crash; kamu mendapat drift lambat yang sulit ditelusuri akar masalahnya.

Mitigasi: log append-only, provenance, verifier yang tidak bisa ditulisi. Sudah dibahas di Pelajaran 13.

### STRATUS — agen terspesialisasi untuk deteksi kegagalan

STRATUS (NeurIPS 2025) melaporkan peningkatan sukses-mitigasi 1,5x ketika kamu mendeploy:

- **Detection agent.** Mengawasi pola gejala (perbedaan pendapat tinggi, lonjakan retry, drift akurasi).
- **Diagnosis agent.** Diberi gejala, menyimpulkan akar masalah yang mungkin dari taksonomi MAST.
- **Validation agent.** Setelah mitigasi diterapkan, memeriksa apakah gejala hilang.

Ini adalah incident response gaya SRE, diterapkan pada sistem agen. Ketiga peran bisa semuanya agen LLM dengan prompt terspesialisasi.

### Audit mode-gagal

Best practice 2026 adalah audit mode-gagal tahunan (atau per rilis mayor):

1. **Sampel jejak.** Kumpulkan ~1000 jejak eksekusi nyata.
2. **Kategorikan.** Untuk tiap kegagalan jejak, petakan ke kategori MAST + Groupthink.
3. **Hitung tingkat kegagalan-per-kategori.** Kategori mana yang mendominasi sistemmu?
4. **Peringkat mitigasi.** Perbaikan mana yang akan mengeliminasi kegagalan terbanyak?
5. **Pilih 2-3 mitigasi.** Implementasikan; audit ulang kuartal berikutnya.

Disiplinnya lebih penting daripada pilihan spesifiknya. Tanpa audit, kegagalan menyatu ke dalam noise dan tidak pernah ditangani secara sistematis.

### Saat sistem gagal diam-diam

Kategori kegagalan paling berbahaya adalah kegagalan koreksi yang diam. Sistem yang gagal dengan keras (crash, exception, alert) bisa dipantau. Sistem yang menghasilkan output masuk-akal-tapi-salah tidak bisa dideteksi oleh log exception. Inilah mengapa verification gap adalah kategori termahal per-kegagalan meski hanya 21.30% dari jumlah kejadian.

Investasikan pada:
- Review manusia berbasis sampel.
- Regression test golden-dataset.
- Cross-checking lintas-agen pada output penting.

### Kegagalan vs kegagalan lambat

Sebagian kegagalan langsung; sebagian lambat. Kegagalan langsung (timeout, schema mismatch, error auth) murah dideteksi. Kegagalan lambat (memory poisoning, drift monoculture, ambiguitas peran) mahal dideteksi dan dicegah.

Langkah engineering 2026: instrumentasikan proxy kegagalan-lambat supaya kamu bisa menangkap drift sebelum menjadi kesalahan yang terlihat. Tingkat persetujuan, tingkat retry, distribusi panjang output, dan edit-distance antara versi agen berurutan semuanya proxy yang berguna.

```figure
a5-retry-cascade
```

## Bangun Sendiri

`code/main.py` mengimplementasikan:

- `FailureTaxonomy` — mengkategorikan insiden yang disimulasikan ke kategori MAST + Groupthink.
- `CircuitBreaker` — pola klasik; terbuka saat tingkat kesalahan melewati ambang.
- `RetryStormSimulator` — menampilkan kegagalan berantai; menyalakan/mematikan circuit breaker.
- `DetectionAgent` — pematch gejala bergaya STRATUS ber-skrip.

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan:
- retry storm tanpa circuit breaker: kesalahan inventaris meledak (disimulasikan).
- dengan circuit breaker: dibatasi pada ambang; respons mode-terdegradasi dilayani.
- agen deteksi menandai pola itu dan menyebut kategori MAST-nya.

## Gunakan

`outputs/skill-mast-auditor.md` menjalankan audit mode-gagal bergaya MAST pada sistem multi-agent. Jejak → kategorisasi → peringkat mitigasi.

## Rilis

Disiplin mode-gagal di produksi:

- **Audit MAST per kuartal.** Bukan tahunan. Kategori bergeser seiring sistemmu tumbuh.
- **Circuit breaker di mana-mana.** Tiap panggilan keluar ke service dependen mana pun. Ambang default terbuka pada tingkat kesalahan 5-10%.
- **Golden dataset.** Kecil, berkualitas tinggi, diaudit manual. Regression-test terhadapnya tiap minggu.
- **Trio STRATUS.** Agen Detection + Diagnosis + Validation memantau produksi. Mulai hanya dengan detection agent; tambahkan diagnosis ketika gejala bising.
- **Failure budget.** SLO eksplisit untuk tingkat kegagalan per kategori. Melewati anggaran memicu percakapan stop-shipping.

## Latihan

1. Jalankan `code/main.py`. Pastikan circuit breaker membatasi retry storm. Ubah ambang kegagalan dan amati tradeoff-nya.
2. Implementasikan **proxy kegagalan-lambat**: tingkat persetujuan di antara 3 agen paralel. Saat turun tajam, picu alert. Simulasikan drift monoculture dengan mengorelasikan output agen secara bertahap.
3. Baca Cemri et al. (arXiv:2503.13657). Pilih satu dari 7 sistem MAS mereka dan petakan 3 kategori kegagalan teratasnya. Bagaimana ini dibandingkan dengan prediksi MAST?
4. Baca paper Groupthink (arXiv:2508.05687). Identifikasi pola mana dari kelima pola yang paling sulit dideteksi di produksi. Usulkan satu metrik proxy.
5. Rancang trio deteksi-diagnosis-validasi bergaya STRATUS untuk sistem multi-agent spesifik yang kamu kenal. Gejala apa yang diawasi detection? Mitigasi apa yang direkomendasikan diagnosis? Bagaimana validasi memastikan semuanya bekerja?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MAST | "The 2026 taxonomy" | Cemri 2025; 3 root categories + 14 sub-types of failures. |
| Specification Problem | "Role ambiguity" | Task or role under-defined; agents do not know what to do. |
| Coordination Failure | "State drift" | Communication or sync breakdown between agents. |
| Verification Gap | "No one checked" | Outputs accepted without independent validation. |
| Groupthink family | "Homogeneity failures" | Monoculture, conformity, deficient ToM, mixed-motive, cascading. |
| Monoculture collapse | "Same model, same hallucinations" | Correlated errors from shared base model or training data. |
| Retry storm | "Cascading error amplification" | One failure triggers retries which amplify load downstream. |
| Circuit breaker | "Fail fast on error rate" | Open when error rate exceeds threshold; short-circuit with default. |
| STRATUS | "Incident response trio" | Detection + diagnosis + validation agents. 1.5x mitigation success. |
| Memory poisoning | "Hallucinations propagate" | Shared-memory fact tainted; downstream agents reason on poison. |

## Bacaan Lanjutan

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy, NeurIPS 2025
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687) — monoculture, conformity, and the five-family taxonomy
- [STRATUS — specialized agents for MAS incident response](https://neurips.cc/) — NeurIPS 2025 proceedings entry (detection + diagnosis + validation)
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) — the canonical circuit-breaker reference
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — production failure-mode notes
