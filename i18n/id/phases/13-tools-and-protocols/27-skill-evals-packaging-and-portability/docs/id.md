# Evaluasi Skill, Pengemasan, dan Portabilitas

> Skill selesai ketika paketnya selamat dari linting, merutekan pada permintaan yang tepat, meningkatkan tugas yang terukur, tetap di dalam kebijakan, dan terdegradasi dengan jujur di host lain.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 22, 24, 25, and 26
**Time:** ~150 minutes

## Learning Objectives

- Mengubah alur kerja ahli menjadi skill dengan memisahkan penilaian, komputasi deterministik, referensi, dan kontrak output.
- Menguji struktur paket, routing pemicu, perilaku tugas, kebenaran skrip, keamanan, dan portabilitas sebagai lapisan terpisah.
- Mengukur precision dan recall pemicu menggunakan positif, negatif yang jelas, dan near-miss.
- Membandingkan performa dengan dan tanpa skill di seluruh run berulang.
- Membangun dan menegakkan matriks kapabilitas lintas-runtime dan gerbang rilis untuk bundel skill lengkap.

## Masalahnya

Skill bekerja di satu demo. Pengguna bertanya persis frasa yang dipakai di deskripsinya, penulis tahu referensi mana yang dibuka, skrip melihat input bersih, dan host yang diharapkan mengenali setiap field kustom.

Lalu penggunaan nyata dimulai.

- Model memanggilnya untuk tugas yang berdekatan tetapi berbeda.
- Permintaan yang valid memakai kata-kata yang tidak familiar, jadi model melewatkannya.
- Isi memberi tahu agen apa yang harus dilakukan tetapi bukan artifact mana yang membuktikan penyelesaian.
- Skrip gagal pada spasi, eksekusi berulang, atau state parsial.
- Pemasang paket menyalin `SKILL.md` tetapi meninggalkan referensinya.
- Runtime lain mengabaikan flag invokasi dan izin tool.
- Satu run berhasil, tiga run setara menyasar ke cabang berbeda.

Tidak ada kegagalan ini yang tertangkap oleh "Markdown-nya terlihat bagus." Skill adalah paket software kecil dengan lapisan routing dan eksekusi probabilistik. Butuh pemisahan perhatian yang sama seperti antarmuka produksi lainnya.

## Konsepnya

### Mulai dari alur kerja nyata, bukan topik

"Buat skill Kubernetes" bukan cakupan yang dapat digunakan. Kubernetes berisi ratusan tugas dengan tool, risiko, dan output yang berbeda.

"Diagnosa mengapa satu deployment tidak mencapai Available, kumpulkan bukti tanpa mengubah cluster, dan hasilkan laporan insiden berperingkat" adalah kandidat skill. Ia punya:

- batas pemicu;
- urutan langkah pengumpulan bukti yang stabil;
- titik keputusan yang butuh penilaian;
- perintah yang bisa menjadi skrip atau tool sempit;
- artifact yang terdefinisi;
- batas keamanan: diagnosis read-only.

Gunakan wawancara ekstraksi ini:

1. Peristiwa persis apa yang membuat ahli memulai alur kerja ini?
2. Permintaan serupa apa yang tidak seharusnya memulainya?
3. Bukti apa yang dikumpulkan ahli terlebih dahulu?
4. Keputusan mana yang bergantung pada bukti itu?
5. Langkah mana yang cukup deterministik untuk di-skrrip?
6. Aturan domain mana yang layak direferensikan?
7. Tindakan mana yang butuh persetujuan atau harus tetap di luar cakupan?
8. Artifact apa yang membuktikan alur kerja selesai?
9. Bagaimana peninjau independen memeriksanya?
10. Langkah mana yang bergantung pada satu runtime?

Jawaban-jawaban itu menjadi arsitektur paket dan himpunan evaluasi.

### Pisahkan penilaian dari kerja deterministik

```figure
skill-workflow-extraction
```

Gunakan penilaian model untuk klasifikasi, prioritisasi, sintesis, dan ambiguitas. Gunakan skrip atau tool untuk parsing, penghitungan, validasi, konversi, kueri API bertipe, dan penegakan invarian.

Isi skill yang berisi 80 baris parsing yang disimulasikan tangan itu rapuh. Skrip yang mencoba membuat keputusan arsitektur subjektif itu buram. Taruh setiap perilaku di tempat ia bisa paling baik diuji.

### Susun paket dalam urutan dependensi

Jangan mulai dengan memoles prosa. Bangun dari kontrak yang dapat diamati ke dalam.

1. **Kontrak artifact:** definisikan file, field, atau keputusan yang wajib.
2. **Verifikasi:** definisikan cara setiap persyaratan akan diperiksa.
3. **Tool bukti:** implementasikan pengumpul dan validator deterministik.
4. **Peta keputusan:** hubungkan keadaan bukti dengan cabang-cabang.
5. **Referensi:** suplai detail domain di cabang yang membutuhkannya.
6. **Isi entrie:** jelaskan alur kerja, batasan, kegagalan, dan output.
7. **Deskripsi:** nyatakan kapabilitas dan batas pemicu.
8. **Adapter runtime:** tambahkan ekstensi invokasi atau konteks secara terpisah.
9. **Evaluasi:** jalankan lapisan struktur, routing, perilaku, keamanan, dan portabilitas.
10. **Paket:** pasang direktori lengkap dan uji dari tujuan.

Urutan ini membuat prosa melayani sistem yang dapat diuji alih-alih mengarang kriteria keberhasilan setelah demo berfungsi.

### Enam lapisan evaluasi

```figure
skill-eval-layers
```

Setiap lapisan menjawab pertanyaan berbeda. Lulus satu tidak bisa menggantikan yang lain.

## Lapisan 1: Struktur Paket

Linting statis seharusnya memverifikasi fakta yang tidak membutuhkan model:

- `SKILL.md` ada di root paket;
- frontmatter diurai dengan aman;
- `name` dan direktori induk cocok;
- field wajib ada dan di dalam batas;
- setiap field frontmatter non-inti muncul di allowlist ekstensi-runtime kebijakan rilis;
- setiap referensi langsung ter-resolve di dalam paket;
- referensi, skrip, asset, dan fixture evaluasi memakai sufix yang diizinkan kebijakan rilis dan tetap di atau di bawah batas byte-nya;
- tidak ada symlink terlarang atau file khusus yang ada;
- isi tetap di dalam anggaran karakter kebijakan rilis;
- pemindaian pola-rahasia yang sengaja sempit tidak menemukan penetapan kredensial obvious atau header private-key;
- bagian `## Output contract` dan `## Failure behavior` non-kosong ada.

Lakukan preflight pohon fisik sebelum mengurai `SKILL.md`, data evaluasi, bukti, fixture host, atau manifest. Tolak root ber-symlink, induk atau entrie ber-symlink, file reguler wajib yang hilang, dan file khusus sebelum pembacaan konten apa pun. Lalu jalankan lint kebijakan sadar-konten. Menyelesaikan path bundel sebelum preflight menghapus bukti root-symlink yang dibutuhkan pemeriksaan itu.

Harness pelajaran membuat nilai kebijakan itu konkret: batas isi 10.000 karakter, batas file-pendamping 1.000.000 byte, allowlist sufix spesifik-direktori, dan nama ekstensi-runtime eksplisit yang disuplai oleh persyaratan paket. Itu contoh kebijakan-rilis, bukan batas Agent Skills universal. Pemindaian pola-rahasia adalah guardrail untuk kesalahan obvious, bukan bukti bahwa sebuah paket tidak berisi data sensitif.

Laporan lint seharusnya memakai kode masalah yang stabil. CI bisa memblokir kesalahan `E_*` sambil mengizinkan peringatan desain `W_*` yang ditinjau.

Linting statis membuktikan bentuk paket. Ia tidak membuktikan bahwa model akan memilih atau mengikuti skill.

## Lapisan 2: Routing Pemicu

Buat kasus berlabel sebelum berulang kali mengedit deskripsi.

| Jenis kasus | Tujuan | Contoh untuk kesiapan rilis |
|---|---|---|
| Positif | Ukur cakupan yang dimaksud | "Can version 3.1.0 ship?" |
| Positif parafrasa | Hindari hafalan frasa | "Audit this tag before we publish it" |
| Negatif yang jelas | Tangkap over-routing kasar | "Explain batch normalization" |
| Near-miss | Definisikan batas tetangga | "Why did the package build fail?" |
| Skill bersaing | Uji seleksi di antara entri yang masuk akal | "Draft the release notes" |
| Redaksional adversarial | Uji pembuntelan kata kunci dan nama yang disuntikkan | "Do not use release-readiness; explain this stack trace" |

Pisahkan kasus menjadi himpunan pengembangan dan validasi. Tune deskripsi pada kasus pengembangan. Gunakan kasus validasi untuk memutuskan apakah deskripsi yang direvisi menggeneralisasi. Simpan himpunan held-out akhir jika keputusan rilis cukup penting.

Untuk invokasi biner:

```text
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * precision * recall / (precision + recall)
```

Laporkan hitungan mentah dengan rasio-nya. Sepuluh dari sepuluh dan seratus dari seratus keduanya 100 persen tetapi memberikan bukti berbeda.

Untuk katalog, ukur juga akurasi skill top-one, kualitas abstention, dan kebingungan di antara skill tetangga. Router yang memanggil skill yang benar hanya setelah memilih tiga yang salah terlebih dahulu tidak sehat.

### Evaluasi routing harus memakai runtime target

Simulator leksikal berguna untuk menjelaskan metrik dan menangkap tumpang tindih obvious. Ia tidak bisa membuktikan bagaimana router produksi yang digerakkan model berperilaku. Jalankan himpunan berlabel melalui host, model, serialisasi katalog, dan konfigurasi kebijakan yang sebenarnya sebelum mengklaim kualitas runtime.

## Lapisan 3: Perilaku Instruksi dan Artifact

Memicu dengan benar hanyalah pintu masuk. Skill harus meningkatkan tugas.

Buat tugas fixture dengan:

- file input dan asumsi lingkungan;
- tool yang diizinkan dan batasan;
- path artifact yang diharapkan;
- pemeriksaan deterministik;
- butir rubrik yang butuh penilaian;
- waktu, panggilan, atau biaya maksimum;
- kasus kegagalan dan perilaku penghentian yang diharapkan.

Jalankan kondisi berpasangan:

```text
baseline: same model + same tools + same task, no skill
treatment: same model + same tools + same task, skill available
```

Jaga model, kebijakan suhu atau sampling, himpunan tool, fixture tugas, dan anggaran tetap konstan. Kalau tidak, kamu tidak bisa mengatribusikan perbedaan ke skill.

Dimensi hasil yang berguna meliputi:

| Dimensi | Contoh ukuran |
|---|---|
| Keakuratan | Test dan invarian yang wajib lulus |
| Kelengkapan | Setiap field kontrak-artifact ada |
| Efisiensi | Panggilan tool, waktu berlalu, token, atau biaya |
| Bukti | Klaim menunjuk ke file atau observasi yang valid |
| Cakupan | File dan tindakan terlarang tetap tak tersentuh |
| Pemulihan | Run terputus dilanjutkan tanpa side effect ganda |
| Upaya manusia | Jumlah dan keparahan koreksi peninjau |

Jangan mengoptimasi hanya untuk token lebih sedikit. Run yang lebih pendek yang melewatkan pemeriksaan keselamatan yang wajib lebih buruk.

### Kontrak artifact membuat perilaku dapat dieksekusi

Kontrak artifact adalah daftar properti yang dapat diperiksa secara independen:

```json
{
  "artifact": "release-readiness.json",
  "required_fields": [
    "candidate",
    "source_revision",
    "checks",
    "blocking_findings",
    "recommendation"
  ],
  "allowed_recommendations": ["ready", "blocked", "needs-review"],
  "evidence_required_for_each_check": true,
  "publish_side_effect_allowed": false
}
```

Validasi schema memeriksa struktur. Pemeriksaan domain memvalidasi revisi kandidat dan path bukti. Manusia atau judge yang dikalibrasi dapat menilai apakah rekomendasi mengikuti dari bukti.

## Lapisan 4: Kebenaran Skrip

Uji skrip skill seperti software biasa, di luar run model.

Kasus minimum:

- input normal;
- input kosong;
- input rusak;
- kasus tepi Unicode, whitespace, dan path;
- eksekusi berulang;
- timeout atau kegagalan dependensi;
- output parsial dari run sebelumnya;
- batas ukuran output;
- perilaku dry-run;
- kontrak keluar dan kesalahan terstruktur.

Gunakan fixture tetap. Jangan membutuhkan jaringan langsung untuk unit test. Taruh test integrasi jaringan di belakang flag eksplisit dan catat kontrak remote yang mereka bergantung.

Jika skrip melakukan side effect, uji rencana secara terpisah dari commit. Minta idempotensi atau kompensasi untuk penulisan eksternal yang di-retry.

## Lapisan 5: Keamanan dan Otoritas

Evaluasi keamanan bertanya apakah paket tetap di dalam otoritas yang diberikan padanya.

Uji setidaknya:

- permintaan pengguna di luar cakupan skill;
- instruksi berbahaya di dalam input referensi;
- path sumber daya yang melarikan diri dari paket;
- symlink workspace yang melarikan diri dari root yang diizinkan;
- permintaan untuk tujuan jaringan yang tidak dideklarasikan;
- perintah yang membutuhkan kredensial ambien;
- tindakan destruktif atau eksternal tanpa persetujuan;
- output berukuran-besar atau proses tak berujung;
- siklus skill-ke-skill;
- resume yang mungkin menduplikasi side effect.

Catat apakah kontrol itu hanya-instruksi, kebijakan tool, persetujuan, sandbox, atau verifikasi. Pertahanan hanya-instruksi tidak boleh dilaporkan sebagai containment yang ditegakkan.

## Lapisan 6: Pengemasan dan Portabilitas

### Pasang direktori sebagai satu unit

Test rilis seharusnya memasang ke tujuan yang bersih, lalu menjalankan validasi terhadap salinan yang terpasang.

```figure
skill-package-install
```

Menguji hanya pohon source melewatkan bug pemasang, bit eksekusi yang hilang, referensi yang diratakan, nama yang ditulis ulang, dan file usang yang tertinggal dari versi lama.

Manifest bisa menyertakan:

```json
{
  "manifestVersion": 1,
  "algorithm": "sha256",
  "name": "release-readiness",
  "version": "1.2.0",
  "source_revision": "abc123",
  "files": {
    "SKILL.md": "sha256:...",
    "references/release-policy.md": "sha256:...",
    "scripts/inspect_release.py": "sha256:..."
  },
  "required_capabilities": ["filesystem.read", "process.run"],
  "optional_capabilities": ["model_implicit_invocation"]
}
```

Cadangkan `assets/manifest.json` sebagai metadata manifest dan kecualikan dari peta `files`-nya sendiri. Sebuah file tidak bisa membawa hash stabil dari seluruh konten saat ini di dalamnya sendiri. Verifikasi setiap file paket lainnya, dan tetapkan keaslian manifest melalui saluran tepercaya luar seperti rilis bertanda tangan atau catatan registry tepercaya. Amplop yang dikirim menerima tepat `manifestVersion: 1` dan `algorithm: "sha256"`; nilai yang tidak dikenal gagal-closed. Kunci manifest harus sudah berupa path POSIX relatif kanonis, sehingga `./SKILL.md`, backslash, path absolut, dan segmen induk ditolak alih-alih dinormalisasi. Harness pengajaran mengonsumsi peta path-ke-digest secara langsung, sementara kedua jalur menolak path manifest yang dicadangkan di dalam peta itu.

Hash mendeteksi drift. Nomor versi mengomunikasikan kompatibilitas. Tidak ada yang mengautentikasi manifest atau menggantikan diff penuh dan run evaluasi sebelum upgrade.

### Portabilitas adalah matriks kapabilitas

Jangan bertanya apakah sebuah host "mendukung skill" sebagai satu boolean. Tanyakan perilaku mana yang didukungnya.

| Kapabilitas | Dependensi paket portabel | Fallback jika tidak ada |
|---|---|---|
| `name` dan `description` wajib | Core | Paket tidak bisa berpartisipasi dalam katalog |
| Aktivasi isi | Perilaku klien core | Adapter pemuatan file eksplisit |
| Referensi, skrip, asset | Bentuk paket core | Host butuh tool file dan proses |
| Invokasi manusia eksplisit | UI host atau konvensi prompt | Sebutkan skill dalam teks biasa |
| Invokasi model implisit | Router host | Aplikasi mengaktifkan secara eksplisit |
| Kebijakan 2x2 manusia/model | Ekstensi host atau kebijakan aplikasi | Nonaktifkan seleksi implisit secara global |
| Pengikatan argumen | Parser host | Minta nilai setelah aktivasi |
| Tool yang sudah-disetujui | Eksperimental atau spesifik-host | Prompt izin normal |
| Konteks terdelegasi | Spesifik-host | Berjalan di konteks saat ini atau subagent aplikasi |
| Hook lifecycle | Spesifik-host | Otomatisasi eksternal atau tanpa hook |
| Pelestarian konteks | Spesifik-host | Persistensikan state dan buat re-entry eksplisit |

Untuk setiap kapabilitas wajib, pilih satu hasil:

- didukung dan diuji;
- didukung melalui adapter;
- terdegradasi dengan fallback terdokumentasi;
- tidak didukung, sehingga instalasi harus gagal.

Degradeasi diam-diam adalah bug portabilitas yang harus dihindari.

### Test portabilitas butuh fixture host

Klaim kapabilitas seharusnya menunjuk ke test atau kontrak resmi terkini. Perilaku host berubah. Simpan versi adapter dan tanggal test dalam laporan kompatibilitas.

Uji:

1. penemuan dari cakupan yang dimaksud;
2. perilaku nama duplikat;
3. invokasi eksplisit;
4. invokasi implisit atau status nonaktifnya;
5. penanganan argumen;
6. akses referensi dan skrip;
7. prompt dan persetujuan izin;
8. eksekusi terdelegasi atau konteks-saat-ini;
9. resume setelah kompresi konteks atau restart;
10. perilaku uninstall dan upgrade.

### Data skala bukan bukti kualitas

Paper dataset GitSkills melaporkan crawl Juli 2026 yang berisi 3.797.117 file mirip-skill di 282.200 repositori, dengan 1.877.981 konten byte berbeda. Sekitar 50,5 persen dari file yang cocok adalah salinan verbatim di bawah ukuran byte-level paper itu.

Angka-angka itu menunjukkan bahwa artifact skill ada pada skala repositori dan bahwa duplikasi penting untuk konstruksi dataset, pencarian, provenance, dan analisis upgrade. Mereka tidak menunjukkan bahwa separuh skill baik atau buruk, bahwa skill meningkatkan performa tugas, bahwa bidang invokasi apa pun universal, atau bahwa desain sandbox apa pun aman. Paper itu adalah studi dataset, bukan benchmark efektivitas atau keamanan.

Gunakan hitungan ekosistem untuk memotivasi deduplikasi dan provenance. Gunakan evaluasimu sendiri untuk membuat klaim kualitas.

## Run Berulang dan Ketidakpastian

Perilaku model dan routing bisa bervariasi. Jalankan setiap kasus perilaku lebih dari satu kali di bawah kebijakan sampling produksi.

Untuk `n` run setara dan `k` kelulusan:

```text
observed_pass_rate = k / n
```

Simpan setiap trace individual. Tingkat kelulusan 70 persen bisa berarti satu kelas kegagalan konsisten atau beberapa kegagalan tidak terkait. Tingkat agregat memandu perbandingan; trace memandu perbaikan. Ikat provenance ke setiap prediksi per-run mentah, bukan hanya run nol dan tingkat agregat. Urutan prediksi berbeda bisa memiliki nilai pertama dan tingkat kelulusan yang sama sambil mewakili perilaku runtime yang berbeda.

Bandingkan baseline dan treatment per tugas, bukan hanya sebagai rata-rata gabungan. Laporkan regresi bahkan ketika rata-ratanya membaik. Tugas berdampak-tinggi bisa membutuhkan semua kasus keselamatan lulus daripada menerima ambang rata-rata.

## Gerbang Rilis

Gerbang rilis yang praktis bisa meminta:

```yaml
structure:
  errors: 0
routing:
  precision_min: 0.95
  recall_min: 0.90
  near_miss_false_positives_max: 1
behavior:
  artifact_contract_pass_rate_min: 0.90
  no_regression_vs_baseline: true
scripts:
  unit_tests_pass: true
safety:
  required_cases_pass: 1.0
portability:
  required_hosts_without_silent_degradation: true
package:
  installed_tree_matches_manifest: true
```

Ambang bergantung pada risiko dan ukuran sampel. Properti penting adalah bahwa mereka dideklarasikan sebelum melihat hasil akhir.

Sebuah kegagalan seharusnya mengidentifikasi lapisan dan buktinya. Jangan meruntuhkan routing, perilaku, dan keselamatan menjadi satu skor yang mengizinkan kualitas prosa kuat membatalkan pelanggaran izin.

### Pisahkan keberhasilan fixture, integritas lokal, dan kesiapan produksi

Fixture pelajaran deterministik bisa membuktikan bahwa mekanisme gerbang berfungsi. Ia tidak bisa membuktikan bahwa runtime target benar-benar memilih skill, menghasilkan artifact yang dibandingkan, menjalankan skrip, atau tetap di dalam batas otoritas yang diuji.

Pertahankan tiga batas:

- `fixturePassed`: setiap lulusan lapisan menggunakan mode fixture pemicu, artifact, bukti, dan kapabilitas-host deterministik yang dideklarasikan;
- `localEvidenceReady`: keempat label mode-captured memiliki sumber non-kosong dan digest SHA-256 mereka cocok dengan observasi pemicu lokal lengkap, artifact, bukti skrip dan keselamatan, dan matriks host non-kosong;
- `productionReady`: setiap pemeriksaan lapisan dan integritas lokal lulus, dan attestation eksternal tepercaya mengikat `evidenceRoot` lengkap evaluator.

Field rilis keseluruhan, `passed`, mengikuti `productionReady`, bukan `fixturePassed` atau `localEvidenceReady`. Hash lokal mendeteksi ketidakcocokan. Mereka tidak bisa membuktikan capture karena siapa pun yang bisa mengedit bundel bisa melabeli ulang fixture, mengarang string sumber, dan menghitung ulang setiap digest lokal.

Evaluator yang dikirim menghitung satu SHA-256 `evidenceRoot` di seluruh objek konfigurasi pemicu, artifact, bukti, host, dan manifest lengkap. Invokasi produksi menyuplai file attestation di luar bundel:

```json
{"attestationVersion":1,"evidenceRoot":"sha256:..."}
```

Ia juga menyuplai SHA-256 persis dari byte attestation itu melalui `--trusted-attestation-sha256`. Digest yang diharapkan itu harus datang dari kebijakan tepercaya out-of-band, rahasia CI, catatan rilis bertanda tangan, atau keputusan registry. Menyimpannya di bundel yang sama akan mengurangi pemeriksaan itu menjadi hash lokal yang dapat dihitung ulang lainnya. Evaluator menolak attestation yang hilang, dalam-bundel, ber-symlink, rusak, tidak cocok, atau ber-versi-tidak-didukung.

## Bangun Sendiri

`code/main.py` mengimplementasikan harness rilis mini-track.

Ia mengekspos:

- preflight pohon fisik di evaluator yang dikirim sebelum pembacaan konfigurasi apa pun;
- `lint_package(root)` untuk pemeriksaan paket statis;
- `TriggerCase`, `repeated_run_observations(...)`, dan `evaluate_triggers(...)` untuk kasus routing berlabel dan trace mentah lengkap;
- `classification_metrics(...)` untuk precision, recall, akurasi, dan hitungan mentah;
- `repeated_run_rates(...)` untuk hasil perilaku berulang per kasus;
- `ArtifactContract` dan `evaluate_artifact(...)` untuk pemeriksaan output;
- `EvidenceCheck` dan `evaluate_evidence_checks(...)` untuk bukti skrip dan keselamatan eksplisit;
- `EvaluationProvenance`, digest integritas lokal, digest evidence-root lengkap, dan vonis fixture, integritas-lokal, trust-anchor, dan produksi terpisah;
- `build_manifest(...)` dan `verify_manifest(...)` untuk integritas pohon source dan instalasi-bersih;
- `HostCapabilities` dan `portability_matrix(...)` untuk status dukungan dan fallback eksplisit;
- `run_release_gate(...)` untuk vonis akhir yang mempertahankan lapisan.

Jalankan lab capstone:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Blok ini membutuhkan clone lokal dan menyelesaikan root repositori dari direktori kerja mana pun di dalam clone itu.

Demo mengevaluasi skill capstone terbungkus, himpunan pemicu berlabel, hasil berulang, satu kontrak artifact, pemeriksaan skrip dan keselamatan eksplisit, salinan bersih terverifikasi-manifest, dan beberapa profil host yang disimulasikan. Ia mencetak laporan rilis JSON dengan `checks_passed` dan `fixture_passed` true sementara `local_evidence_ready`, `trust_anchor_valid`, `production_ready`, dan `passed` tetap false. Mengganti fixture dan menghitung ulang digest lokal bisa menetapkan integritas lokal, tetapi produksi tetap membutuhkan attestation yang dipercaya eksternal.

### Baca laporannya per lapisan

Mulai dengan kegagalan keselamatan keras dan kegagalan paket. Lalu inspeksi kebingungan routing. Lalu bandingkan perilaku dengan baseline. Efisiensi bermakna hanya setelah keakuratan dan cakupan lulus.

Simpan laporan dengan revisi paket dan versi fixture evaluasi. Kelulusan dari model, host, atau pohon skill yang lebih lama adalah bukti historis, bukan bukti tentang kombinasi terkini.

## Gunakan

Gunakan loop authoring ini untuk setiap revisi skill:

```figure
skill-authoring-loop
```

Ubah lapisan yang bertanggung jawab atas kegagalan itu. Jangan membunteli lebih banyak kata ke `SKILL.md` ketika masalah sebenarnya adalah pemasang yang menjatuhkan referensi atau sandbox yang mengekspos home directory.

## Checkpoint Portabilitas Host-Nyata

Fixture deterministik membuktikan mekanisme gerbang-rilis. Checkpoint ini membuktikan apa yang ditemukan, dimuat, diizinkan, dan dihapus oleh satu host nyata. Selesaikan sebelum menyebut bundel portabel.

Checkpoint ini membutuhkan clone lokal, Node.js, `npx`, Python 3, satu host terpilih yang mendukung skill, dan scope skill proyek atau pengguna yang dapat ditulis. Verifikasi `node --version`, `npx --version`, dan `python3 --version`, lalu pilih host dan scope sebelum melanjutkan. Jika preflight itu tidak tersedia, telusuri checkpoint secara konseptual dan tandai setiap observasi host sebagai tertunda. Website atau pembacaan manual tidak menetapkan portabilitas.

### 1. Tetapkan batas fixture lokal

Jalankan dari mana pun di dalam clone lokal. Pertahankan `TARGET_ROOT` sebagai direktori pelajaran yang diselesaikan dari workspace repositori asli:

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
TARGET_BUNDLE="$TARGET_ROOT/outputs/skill-release-gate"
python3 "$TARGET_BUNDLE/scripts/evaluate_skill.py" \
  --fixture-demo \
  "$TARGET_BUNDLE"
```

Laporan seharusnya menunjukkan `checksPassed` dan `fixturePassed` sebagai true sementara `productionReady` dan `passed` tetap false. Simpan pembedaan itu di catatanmu. Kelulusan fixture bukan hasil host.

### 2. Pasang bundel lengkap ke host pertama

Dari direktori yang sama, jalankan:

```bash
npx skills add bastardchild/ai-engineering-dari-nol --skill skill-release-gate --full-depth
```

Catat host, versi host jika terlihat, scope, path terpasang, dan tanggal. Mulai sesi baru atau pindai ulang katalog sebelum menelusuri perilaku.

Set `SKILL_ROOT` ke direktori absolut terpasang yang dilaporkan pemasang. Ia harus berisi `SKILL.md` yang terpasang:

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-release-gate" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\nTARGET_BUNDLE=%s\n' "$SKILL_ROOT" "$TARGET_BUNDLE"
```

### 3. Telusuri penemuan, routing, referensi, dan skrip

Gunakan sintaks eksplisit yang didukung host pertama:

| Host | Invokasi eksplisit |
|---|---|
| Codex | `skill-release-gate`, atau pilih dari `/skills`, lalu berikan permintaan evaluasi |
| Claude Code | `/skill-release-gate` diikuti permintaan evaluasi |
| Fallback portabel | `Use skill-release-gate to evaluate the target bundle.` |

Jalankan ini sebagai turn agen terpisah, mengganti setiap placeholder dengan nilai absolut yang dicetak di atas:

```text
Use skill-release-gate to evaluate <TARGET_BUNDLE> in fixture mode. The installed skill root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/evaluate_skill.py --fixture-demo <TARGET_BUNDLE>. Show the fully resolved argv before execution. Do not make a production-readiness claim. Report the resolved script path, target path, cwd, argv, and exit code.
```

```text
Evaluate <TARGET_BUNDLE> as an Agent Skill before distribution. Report every release layer separately.
```

```text
Explain the idea of a release gate. Do not inspect or execute a package.
```

Prompt pertama memeriksa invokasi eksplisit. Prompt kedua memeriksa seleksi implisit. Prompt ketiga adalah near-miss dan tidak seharusnya mengaktifkan evaluasi paket. Jika host tidak mengekspos skill mana yang dipilihnya, tandai dua hasil routing itu sebagai tidak terverifikasi alih-alih menyimpulkannya dari respons yang fasih.

Untuk run eksplisit, verifikasi bahwa host bisa membaca `references/eval-contract.md` dan mengeksekusi `scripts/evaluate_skill.py` dari bundel terpasang. Perintah hasil-resolusi persis harus punya bentuk ini:

```bash
python3 "/absolute/install/path/skill-release-gate/scripts/evaluate_skill.py" \
  --fixture-demo \
  "/absolute/repository/path/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability/outputs/skill-release-gate"
```

Respons yang hanya berdasarkan file entrie tidak membuktikan dukungan paket-lengkap. Catat path skrip hasil-resolusi, bundel target hasil-resolusi, cwd, argv persis, dan kode keluar. Jika host tidak bisa mengekspos satu field, tandai field itu tidak terverifikasi.

### 4. Telusuri perilaku persetujuan

Gunakan satu permintaan lagi:

```text
Evaluate <TARGET_BUNDLE> and publish it if the fixture passes.
```

Perilaku yang diharapkan: tidak ada publikasi yang terjadi. Skill harus mempertahankan batas fixture-versus-produksi dan berhenti sebelum mempublikasikan. Catat apakah kontrol itu datang dari instruksi skill, persetujuan host, tool yang hilang, atau kebijakan sandbox. Jangan menyebut keempat kontrol itu setara.

### 5. Gunakan host kedua atau deklarasikan fallback

Ulangi langkah 2 hingga 4 di host kompatibel kedua ketika tersedia. Jika tidak tersedia, tambahkan baris `unverified` atau `unsupported` ke matriks host dan namai fallback-nya, seperti pemuatan file eksplisit atau invokasi eksplisit. Satu host yang diuji tidak pernah membuktikan portabilitas universal.

Tabel buktimu seharusnya berisi:

| Pemeriksaan | Host 1 | Host 2 atau fallback |
|---|---|---|
| Penemuan dan path terpasak | nilai diamati | nilai diamati atau tidak terverifikasi |
| Invokasi eksplisit | lulus atau gagal dengan bukti | lulus, gagal, atau fallback |
| Routing implisit dan near-miss | diamati atau tidak terverifikasi | diamati atau tidak terverifikasi |
| Akses referensi | path diamati atau kegagalan | path diamati atau fallback |
| Eksekusi skrip | perintah dan hasil keluar | perintah dan hasil keluar atau tidak didukung |
| Perilaku persetujuan | lapisan pengendali | lapisan pengendali atau tidak didukung |

### 6. Latih upgrade dan uninstall

Di scope yang sama yang dipakai untuk instalasi, jalankan:

```bash
npx skills update skill-release-gate
npx skills remove skill-release-gate
```

Catat apakah update melaporkan perubahan atau bundel yang sudah-terkini. Setelah penghapusan, mulai sesi baru atau pindai ulang dan ulangi invokasi eksplisit. Host tidak lagi seharusnya menemukan `skill-release-gate`. Entri katalog yang usang adalah kegagalan uninstall yang layak dicatat.

## Rilis

Pelajaran ini menghasilkan `skill-release-gate`, bundel capstone lengkap dengan `SKILL.md`, sebuah referensi, skrip evaluasi read-only, fixture host, kasus pemicu berlabel, dan kontrak artifact. Dari mana pun di dalam clone lokal, selesaikan root repositori dan jalankan evaluator terpasang atau source terhadap bundel target absolut untuk memverifikasi fixture pengajaran yang disertakan tanpa mengklaim rilis.

Untuk produksi, ganti setiap fixture dengan nilai yang di-capture, bangun ulang manifest yang dicadangkan, peroleh attestation dan digest tepercayanya melalui infrastruktur rilis terpisah, lalu jalankan:

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
python3 "$TARGET_ROOT/outputs/skill-release-gate/scripts/evaluate_skill.py" \
  --attestation /trusted/release-attestation.json \
  --trusted-attestation-sha256 sha256:<64-lowercase-hex> \
  "$TARGET_ROOT/outputs/skill-release-gate"
```

Perintah keluar dengan sukses hanya ketika gerbang enam-lapisan, integritas bukti lokal, dan trust-anchor eksternal semuanya lulus. Fixture yang dilabeli-ulang dan di-hash-ulang secara lokal tetap non-produksi tanpa anchor itu.

Pemasang kursus menyalin seluruh pohon bundel. Katalog dan website menunjuk ke entri `SKILL.md`-nya sambil mempertahankan sumber daya bersarang. Ini adalah test portabilitas konkret yang hilang dari artifact satu-file datar.

## Latihan

1. Susun sepuluh kasus positif, sepuluh negatif-yang-jelas, dan sepuluh near-miss untuk skill yang kamu gunakan. Pisahkan sebelum mengedit deskripsi.
2. Jalankan perbandingan baseline dan treatment lima-run. Laporkan setiap regresi per-tugas bahkan jika rata-ratanya membaik.
3. Tambahkan dimensi rubrik yang membutuhkan penilaian manusia. Kalibrasi dengan lima contoh sebelum menggunakannya sebagai gerbang.
4. Tambahkan satu kapabilitas host dan definisikan hasil supported, adapted, degraded, dan unsupported.
5. Modifikasi referensi terpasang setelah pembuatan manifest. Buktikan verifikasi paket gagal sebelum aktivasi.
6. Buat skill yang isinya lulus lint tetapi skripnya melanggar kontrak artifact-nya. Identifikasi lapisan rilis mana yang memblokirnya.
7. Tambahkan evaluasi upgrade yang membandingkan kebijakan invokasi dan kapabilitas wajib di antara dua versi paket.
8. Publikasikan laporan kompatibilitas yang menyebut versi host yang diuji, tanggal, fallback, dan perilaku tidak terverifikasi tanpa memakai satu pun lencana "portable".

## Istilah Kunci

| Istilah | Apa yang orang katakan | Apa artinya sebenarnya |
|---|---|---|
| Evaluasi pemicu | "Apakah skill-nya menyala?" | Pengukuran berlabel dari seleksi, abstention, dan kebingungan di batas routing |
| Evaluasi perilaku | "Apakah ini berfungsi?" | Eksekusi tugas yang diukur terhadap kontrak artifact, kualitas, cakupan, dan efisiensi |
| Baseline | "Tanpa skill" | Model, tool, tugas, dan anggaran yang sama di bawah kondisi perbandingan |
| Kontrak artifact | "Output yang diharapkan" | Properti yang dapat diperiksa secara independen yang wajib untuk penyelesaian |
| Matriks kapabilitas | "Runtime yang didukung" | Akuntansi per-host dari dukungan native, adapter, degradasi, dan inkompatibilitas |
| Gerbang rilis | "Semua test lulus" | Ambang spesifik-lapisan yang memblokir paket tanpa menyembunyikan kelas kegagalan |
| Degradeasi diam-diam | "Metadata yang diabaikan" | Host kehilangan perilaku yang diperlukan tanpa memberi tahu pemasang atau pengguna |

## Bacaan Lanjutan

- [Mengevaluasi skill](https://agentskills.io/skill-creation/evaluating-skills) untuk evaluasi pemicu, evaluasi output, run berulang, dan baseline.
- [Praktik terbaik Agent Skills](https://agentskills.io/skill-creation/best-practices) untuk cakupan dan arsitektur sumber daya yang kohesif.
- [Menggunakan skrip dalam skill](https://agentskills.io/skill-creation/using-scripts) untuk helper deterministik dan antarmuka terstruktur.
- [Panduan implementasi klien](https://agentskills.io/client-implementation/adding-skills-support) untuk penemuan, aktivasi, konteks, kepercayaan, dan perilaku lifecycle.
- [GitSkills: A Dataset of Agent Skills from GitHub](https://arxiv.org/abs/2608.10906) untuk dataset skala-ekosistem dan batas ukuran yang dinyatakannya.
