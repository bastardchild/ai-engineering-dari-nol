# Penemuan Skill dan Progressive Disclosure

> Skill menjadi berguna sebelum isinya dimuat. Namanya dan deskripsinya mendapat tempat di katalog; file-file yang lebih dalam mendapat konteks hanya ketika tugas mencapainya.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 22 (Agent Skills: Portable Contract and Runtime Boundary)
**Time:** ~105 minutes

## Learning Objectives

- Membangun pipeline penemuan sistem berkas yang memisahkan scope, validasi, kebijakan tabrakan, dan publikasi katalog.
- Menjelaskan tiga tingkat disclosure: metadata katalog, instruksi aktif, dan sumber daya spesifik tugas.
- Mendesain referensi agar agen dapat mencapai detail yang diperlukan secara langsung tanpa memuat seluruh paket.
- Menganggarkan ruang katalog secara independen dari konteks skill aktif.
- Menolak path traversal dan pelarian symlink saat skill membaca sumber dayanya sendiri.

## Masalahnya

Agensimu punya 200 skill terpasang. Memuat setiap `SKILL.md`, file referensi, skrip, dan template di awal sesi akan mengubur tugas saat ini dalam prosedur yang tidak relevan. Tidak memuat apa pun akan memaksa pengguna mengingat path sistem berkas yang tepat.

Kompromi yang biasa adalah katalog: tampilkan identitas ringkas dan deskripsi routing untuk setiap skill yang memenuhi syarat, lalu muat isi penuh hanya setelah seleksi. Itu menciptakan dua masalah rekayasa baru.

Pertama, penemuan bukan sekadar pencarian file rekursif. Skill bisa ada di scope proyek, pengguna, administrator, plugin, atau bawaan. Dua paket bisa berbagi nama. Sebuah symlink bisa menunjuk ke luar root tepercaya. Paket yang rusak bisa mengonsumsi ruang katalog atau menjadi mustahil dipanggil.

Kedua, progressive disclosure bisa berubah menjadi progressive confusion. Jika `SKILL.md` bilang "baca panduan yang relevan" dan paket berisi dua belas panduan, model harus menebak. Jika setiap panduan menunjuk ke tiga file lagi, pemuatan menjadi graph walk tanpa batas.

Runtime yang baik membuat penemuan deterministik dan disclosure disengaja.

## Konsepnya

### Penemuan adalah pipeline compiler

Perlakukan sistem berkas sebagai input source. Jangan publikasikan path mentah langsung ke model.

```figure
skill-discovery-pipeline
```

Setiap tahap harus menghasilkan data terstruktur dan kegagalan terstruktur. Log penemuan seharusnya menjawab:

- Root mana yang ditelusuri?
- Kandidat mana yang ditemukan?
- Kandidat mana yang ditolak, dan mengapa?
- Paket mana yang memenangkan tabrakan?
- Entri katalog mana yang dipotong atau dihilangkan karena anggaran?

Tanpa bukti itu, "model tidak menggunakan skill-ku" hampir mustahil didiagnosis.

### Scope adalah kebijakan runtime

Spesifikasi portabel mendefinisikan paket skill, bukan satu path instalasi universal atau urutan precedensi. Host yang memutuskan di mana ia menelusuri.

Runtime generik mungkin memakai scope ini:

| Scope | Contoh root | Kepemilikan yang dimaksud |
|---|---|---|
| Workspace | `<repo>/.agents/skills/` | Maintainer proyek |
| User | `<user-data>/skills/` | Satu developer |
| Administrator | `<system>/skills/` | Kebijakan mesin atau organisasi |
| Plugin | Bundel plugin bertanda tangan | Penerbit dan pemasang plugin |
| Built-in | Paket runtime | Vendor runtime |

Pada Agustus 2026, Codex mendokumentasikan penemuan proyek dari `$CWD/.agents/skills` melalui direktori leluhur hingga root repositori, ditambah lokasi user, administrator, dan built-in. Ia mendukung direktori skill ber-symlink. Nama duplikat bisa keduanya muncul alih-alih digabung. Itu perilaku Codex, bukan persyaratan `SKILL.md`; verifikasi [dokumentasi skill Codex](https://learn.chatgpt.com/docs/build-skills) terkini saat menulis adapter.

Jangan pernah mengarang precedensi dari nama direktori. Deklarasikan sebagai kebijakan dan uji. Lab pelajaran memakai rank integer eksplisit untuk setiap `Scope` sehingga himpunan kandidat yang sama selalu teresolusi dengan cara yang sama.

### Tabrakan butuh identitas di luar `name`

Dua paket bernama `release-readiness` bisa sah. Satu bisa override workspace dan satu lagi default pengguna. Karena itu entri katalog butuh setidaknya:

```json
{
  "name": "release-readiness",
  "description": "Inspect a release candidate for this repository.",
  "scope": "workspace",
  "source": "/repo/.agents/skills/release-readiness",
  "selected": true
}
```

Kebijakan tabrakan umum meliputi:

| Kebijakan | Manfaat | Risiko |
|---|---|---|
| Simpan semua kandidat | Tidak ada yang disembunyikan | Model melihat nama ambigu |
| Scope precedensi tertinggi menang | Invokasi sederhana | Paket lokal bisa menaungi yang tepercaya |
| Tolak duplikat | Tidak ada penaungan diam-diam | Override yang sah berhenti berfungsi |
| Kualifikasi nama berdasarkan sumber | Identitas eksplisit | Nama yang terlihat pengguna menjadi lebih panjang |

Pilih satu kebijakan untuk host. Pertahankan kandidat yang ditolak atau dinaungi dalam diagnostik bahkan ketika tidak ada di katalog model.

### Tiga tingkat disclosure

Spesifikasi Agent Skills menggambarkan pemuatan bertingkat. Kuncinya adalah setiap tingkat punya tujuan berbeda.

```figure
skill-disclosure-levels
```

#### Tingkat 1: metadata katalog

Model butuh informasi yang cukup untuk membedakan skill dari tetangganya. Spesifikasi memperkirakan sekitar 100 token per entri katalog, tetapi serialisasi dan tokenisasi aktual milik host.

Deskripsi yang berguna punya dua klausa:

```yaml
description: Validate a release candidate and produce a readiness report. Use when the user asks whether a version, tag, or package is ready to publish.
```

Klausa pertama menyatakan kapabilitas. Klausa kedua menyatakan batas pemicu. Pelajaran 25 mengevaluasi batas ini dengan prompt positif dan near-miss.

#### Tingkat 2: instruksi aktif

Setelah aktivasi, isi seharusnya berfungsi sebagai peta dan prosedur. Spesifikasi merekomendasikan menjaga `SKILL.md` di bawah 500 baris. Itu sinyal desain, bukan target untuk diisi.

Isi seharusnya berisi:

- batas tugas;
- alur kerja default;
- kondisi percabangan;
- referensi langsung ke file yang lebih dalam;
- kontrak tool dan skrip;
- perilaku kegagalan dan penghentian;
- output yang diharapkan dan verifikasinya.

Jangan memindahkan alur kerja pusat ke referensi hanya untuk membuat file entrie pendek. Aktivasi harus memberi model konteks yang cukup untuk mulai dengan benar.

#### Tingkat 3: sumber daya pendukung

Referensi menyediakan prosa atau data. Skrip menyediakan komputasi deterministik. Aset disalin, diisi, atau diubah menjadi deliverable alih-alih diperlakukan sebagai instruksi.

| Direktori | Model membacanya? | Model mengeksekusinya? | Konten tipikal |
|---|:---:|:---:|---|
| `references/` | Ya, saat diperlukan | Tidak | schema, kebijakan, panduan domain |
| `scripts/` | Bisa memeriksanya | Melalui tool yang diizinkan | validator, konverter, pengumpul |
| `assets/` | Hanya jika berguna | Tidak | template, fixture, gambar, file awal |

Nama-nama ini adalah konvensi, bukan kapabilitas ajaib. Host tetap butuh akses file dan tool eksekusi.

### Referensi spesifik-cabang mengalahkan tumpukan topik

Tulis file entrie sebagai peta keputusan:

```markdown
## Choose the path

- For a Python package, read `references/python-release.md`.
- For a container image, read `references/container-release.md`.
- For a documentation-only release, read `references/docs-release.md`.
- If the release combines artifact types, read only the guides for those artifacts.
```

Ini memberi setiap referensi kondisi pemuatan yang dapat diamati. "Baca `references/` untuk lebih lanjut" tidak.

Jaga graph referensi tetap dangkal. Panduan resmi merekomendasikan tautan langsung dari `SKILL.md` dan menghindari rantai panjang. Satu hop membuat keterjangkauan dapat diuji dan mengurangi peluang batasan yang diperlukan tidak pernah masuk konteks.

```figure
skill-reference-map
```

### Anggaran katalog dan konteks aktif adalah anggaran berbeda

Misalkan `c_i` adalah biaya katalog terserialisasi dari skill `i`, `B_c` anggaran katalog, `b_j` biaya isi aktif, dan `r_k` sumber daya yang benar-benar dimuat.

```text
catalog_cost = sum(c_i for every published skill)
active_cost = sum(b_j for every activated skill) + sum(r_k for every disclosed resource)
```

Mengurangi satu anggaran tidak otomatis mengurangi yang lain. Deskripsi pendek bisa menghemat ruang katalog sementara isi 900 baris yang diaktifkan tetap menenggelamkan tugas. Memecah isi menjadi referensi hanya mengurangi biaya aktif ketika runtime dan instruksi benar-benar menghindari memuat cabang yang tidak relevan.

Saat ini Codex menganggarkan daftar skill awal sebesar 2 persen dari jendela konteks ketika ukuran jendela konteks diketahui. Nilai 8.000 karakter hanyalah fallback ketika ukuran itu tidak diketahui; bukan batas kedua yang dikombinasikan dengan aturan 2 persen. Ketika katalog melebihi anggaran yang berlaku, deskripsi bisa dipotong atau dihilangkan. Perlakukan angka-angka itu sebagai kebijakan Codex terkini, bukan properti standar Agent Skills.

### Path sumber daya adalah batas kepercayaan

Skill seharusnya hanya membaca file di dalam paketnya. Pemeriksaan string prefix literal tidak cukup:

```text
references/../../../../.ssh/config
references/external-link -> /private/company-secrets
```

Resolusi root paket dan kandidat dengan semantik sistem berkas, tolak input absolut, dan verifikasi bahwa kandidat hasil-resolusi tetap berada di bawah root hasil-resolusi. Putuskan apakah symlink diizinkan sebelum penemuan. Jika diizinkan, periksa target hasil-resolusi setiap kali.

```figure
skill-resource-containment
```

Containment path tidak menetapkan kepercayaan konten. Referensi dalam-paket yang valid tetap bisa berisi instruksi berbahaya. Pelajaran 26 menangani ancaman itu.

### Pemuatan harus dapat diamati

Catat event disclosure tanpa mencatat rahasia:

```json
{
  "event": "skill.resource.loaded",
  "skill": "release-readiness",
  "resource": "references/python-release.md",
  "reason": "candidate contains pyproject.toml",
  "bytes": 2840
}
```

Alasan itu mengubah pilihan konteks menjadi bukti yang dapat ditinjau. Ia juga membantu mengidentifikasi instruksi yang menyebabkan agen memuat setiap file "jaga-jaga."

## Bangun Sendiri

`code/main.py` membangun mesin penemuan dan disclosure yang deterministik.

Permukaan penemuan meliputi:

- `Scope` untuk metadata sumber dan precedensi;
- `SkillCandidate` untuk kandidat sistem berkas yang belum divalidasi;
- `discover_scope(scope)` untuk mengenumerasi direktori skill langsung;
- `resolve_collisions(candidates, precedence)` untuk menerapkan satu kebijakan yang dideklarasikan;
- `CatalogEntry` dan `build_catalog(...)` untuk memublikasikan metadata berbatas;
- `CatalogBudget` untuk memperhitungkan entri terserialisasi tanpa berpura-pura karakter adalah token universal.

Permukaan disclosure meliputi:

- `load_skill_body(entry, ...)` untuk aktivasi Tingkat 2;
- `validate_reference(skill_dir, reference)` untuk containment path;
- `load_reference(...)` untuk pembacaan Tingkat 3 yang berbatas.

Jalankan lab:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Blok ini membutuhkan clone lokal dan menyelesaikan root repositori dari direktori kerja mana pun di dalam clone itu.

Demo membuat scope proyek dan pengguna sementara, menyisipkan sebuah tabrakan, membangun katalog di bawah anggaran yang sengaja kecil, mengaktifkan satu skill, dan mencoba baik pembacaan referensi valid maupun pelarian traversal. Tidak ada file permanen yang dipasang.

### Mengapa penemuan dangkal

`discover_scope` memeriksa direktori anak langsung untuk `SKILL.md`. Ia tidak secara rekursif memperlakukan setiap `SKILL.md bersarang sebagai paket terpisah. Ini mempertahankan batas paket dan menghindari tidak sengaja mempublikasikan contoh atau fixture di dalam skill terpasang.

### Mengapa lab tidak mengurai YAML sembarang

Lab mendukung frontmatter skalar yang dibutuhkan katalognya. Runtime produksi seharusnya memakai parser YAML aman dengan schema eksplisit, batas ukuran, dan konstruksi objek kustom dinonaktifkan. "Stdlib-only" adalah batasan pengajaran, bukan izin untuk diam-diam mengarang dialaek YAML parsial.

## Gunakan

Terapkan checklist ini ke adapter penemuan apa pun:

1. Daftar setiap root yang dikonfigurasi dan siapa yang bisa menulis ke sana.
2. Nyatakan apakah paket ber-symlink diizinkan.
3. Validasi nama paket, nama direktori, metadata wajib, dan ukuran isi-entrie.
4. Pertahankan sumber dan scope dalam identitas internal.
5. Deklarasikan dan uji perilaku nama duplikat.
6. Ukur katalog terserialisasi persis yang dikirim ke model.
7. Catat mengapa isi atau sumber daya dimuat.
8. Jaga pembacaan sumber daya di dalam root paket hasil-resolusi.
9. Gagalkan dengan jelas ketika file yang dirujuk hilang.
10. Bangun ulang katalog ketika instalasi atau kebijakan berubah.

## Rilis

Pelajaran ini menghasilkan bundel `skill-catalog-builder`. Ia memindai root berurutan secara eksplisit, menolak file entrie ber-symlink dan ketidakcocokan nama-direktori, menyelesaikan tabrakan lintas-scope, menolak duplikat precedensi-equal, dan menyesuaikan metadata terpilih ke dalam anggaran entri, deskripsi, dan karakter terserialisasi yang dideklarasikan.

Laporan JSON-nya berisi entri terpilih, kandidat yang dinaungi, entri yang dihilangkan, kesalahan validasi, precedensi, dan penggunaan anggaran. Pemuatan isi dan referensi tetap operasi runtime terpisah, sehingga pembuat katalog tidak mengeksekusi skrip atau mengakui seluruh paket ke dalam konteks.

## Latihan

1. Tambahkan scope plugin dan letakkan di antara precedensi pengguna dan built-in. Buktikan hasil tabrakan dengan sebuah test.
2. Ubah kebijakan tabrakan dari precedensi tertinggi ke nama berkualifikasi. Pertahankan kedua entri di katalog.
3. Tambahkan batas ukuran byte ke `load_reference`. Uji file yang persis di batas dan satu byte di atasnya.
4. Buat dua deskripsi yang terdengar hampir identik. Tulis ulang agar batas pemicunya tidak tumpang tindih.
5. Tambahkan manifest berisi hash untuk setiap referensi dan skrip. Deteksi sumber daya yang dimodifikasi sebelum memuatnya.
6. Instrumen demo untuk melaporkan jumlah byte Tingkat 1, Tingkat 2, dan Tingkat 3 secara terpisah.

## Istilah Kunci

| Istilah | Apa yang orang katakan | Apa artinya sebenarnya |
|---|---|---|
| Penemuan skill | "Temukan setiap SKILL.md" | Telusuri scope terkonfigurasi, validasi paket, lampirkan provenance, dan terapkan kebijakan |
| Katalog skill | "Daftar skill terpasang" | Metadata routing ringkas yang terlihat model untuk paket yang memenuhi syarat |
| Kebijakan tabrakan | "Mana yang menang" | Aturan yang dideklarasikan untuk kandidat bernama-sama dari sumber berbeda |
| Progressive disclosure | "Lazy loading" | Admisi konteks bertingkat dari katalog ke isi ke sumber daya spesifik-cabang |
| Graph referensi | "File yang ditautkan skill" | Struktur sumber daya yang terjangkau dan kondisi pemuatannya |
| Containment path | "Tetap di folder" | Verifikasi target sumber daya hasil-resolusi tetap di dalam root paket hasil-resolusi |

## Bacaan Lanjutan

- [Spesifikasi Agent Skills](https://agentskills.io/specification) untuk bentuk paket dan tingkat progressive disclosure.
- [Mengoptimalkan deskripsi skill](https://agentskills.io/skill-creation/optimizing-descriptions) untuk metadata routing katalog.
- [Praktik terbaik Agent Skills](https://agentskills.io/skill-creation/best-practices) untuk referensi langsung dan ukuran file entrie.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills) untuk scope penemuan dan batas katalog Codex terkini.
