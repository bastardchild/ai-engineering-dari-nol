# Rantai Pasokan Registry MCP: Admission, Drift, dan Rollback

> Entri registry memberitahumu apa yang dideklarasikan penerbit. Admission produksi membuktikan apa yang kamu fetch, apa yang kamu amati, apa yang kamu setujui, dan apa yang bisa kamu pulihkan dengan aman.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 17 (gateways and registries), Phase 13 · 18 (production authentication)
**Time:** ~90 minutes

## Learning Objectives

- Memisahkan publikasi Registry, provenance paket, penemuan runtime, dan persetujuan lokal.
- Memverifikasi namespace server MCP tanpa mempercayai nama di dalam catatannya sendiri.
- Pin bukti publikasi immutable, sumber-eksekusi, provenance, dan deskriptor langsung.
- Mendeteksi perubahan status registry dan runtime drift setelah admission.
- Rollback routing ke versi yang sebelumnya diadmit tanpa menulis ulang riwayat.
- Mempertahankan ledger admission yang tamper-evident yang menjelaskan setiap keputusan.

## Masalahnya

Kamu menemukan `com.example/inventory` di registry. Deskripsinya terlihat benar. Paketnya ada. Server menjawab `server/discover`.

Itu bukan satu fakta. Itu adalah rantai fakta dari otoritas yang berbeda:

1. Sebuah penerbit yang diautentikasi untuk sebuah namespace mengirimkan sebuah catatan.
2. Sebuah registry paket menyajikan artifact dengan identitas dan digest tertentu.
3. Sebuah endpoint yang berjalan melaporkan versi protokol, kapabilitas, tool, dan informasi server diagnostik.
4. Organisasimu memutuskan bahwa kombinasi persis itu diizinkan.

Meruntuhkan fakta-fakta itu menjadi "ia ada di registry, jadi percayai" menciptakan titik buta rantai pasokan. Publikasi yang valid tetap bisa ditandai usang. Sebuah tag paket bisa menunjuk ke artifact yang tidak terduga jika kamu tidak mem-pin digest-nya. Sebuah server bisa menambahkan tool destruktif setelah tinjauan. Sebuah rollback bisa diam-diam memilih versi yang tidak pernah diadmit.

Perbaikannya adalah admission controller dengan bukti di setiap batas.

## Registry adalah Indeks, bukan Sistem Persetujuanmu

Registry MCP resmi menyimpan metadata server. Catatan `server.json`-nya menyebut versi server dan mendeklarasikan satu atau lebih paket atau endpoint remote. Aturan publikasi menambahkan autentikasi namespace, pemeriksaan kepemilikan paket, aturan registry terbatas, dan lokasi metadata penerbit yang sempit.

Kontrol-kontrol itu menjawab pertanyaan publikasi. Kebijakan produksimu tetap menjawab pertanyaan penempatan:

| Batas | Pertanyaan | Pemilik bukti |
|---|---|---|
| Namespace | Apakah penerbit diizinkan memakai nama ini? | Autentikasi registry plus input namespace yang kamu verifikasi |
| Record | Apa yang dideklarasikan penerbit untuk versi ini? | Digest `server.json` immutable |
| Sumber eksekusi | Paket atau endpoint remote mana yang akan mengeksekusi? | Field sumber yang dideklarasikan, hasil kepemilikan yang diverifikasi, transport, dan digest tepercaya |
| Runtime | Apa yang diekspos endpoint sekarang? | `server/discover` dan deskriptor tool |
| Admission | Apakah kebijakanmu menyetujui himpunan persis ini? | Pin lokal dan entri ledger |
| Operasi | Apakah ini masih aman, dan apa yang bisa menggantikannya | Pemeriksaan drift, sinkronisasi status, kesehatan, dan jalur rollback |

Versi schema Registry dan versi protokol MCP independen. Sebuah catatan bisa memakai schema server `2025-12-11` yang dipublikasikan sementara server langsung mendukung MCP `2026-07-28`. Jangan pernah menyimpulkan satu dari yang lain.

```figure
mcp-registry-admission
```

## Tujuh Kontrol dalam Satu Keputusan Admission

### 1. Verifikasi namespace

Nama Registry resmi memakai namespace yang diautentikasi. Domain terverifikasi bisa memetakan ke prefix domain yang dibalik. Sebagai contoh, kendali atas `example.com` bisa menetapkan `com.example/*`.

Jangan menerima pemeriksaan string prefix:

```python
server_name.startswith("com.example")
```

Itu juga menerima `com.exampleevil/tool`. Bagi nama di `/`, minta slug non-kosong, dan bandingkan segmen namespace secara persis. Yang lebih penting, lewatkan namespace yang diverifikasi ke admission dari hasil autentikasi. Jangan turunkan kepercayaan dari catatan yang tidak tepercaya.

Namespace berbasis GitHub dan namespace berbasis domain memakai jalur autentikasi yang berbeda. Normalisasi kedua jalur itu menjadi satu input admission: string namespace terverifikasi yang persis.

### 2. Join provenance

Untuk catatan paket, deklarasi dan artifact yang di-fetch harus join pada field eksplisit:

- tipe registry paket
- identifier paket
- versi paket
- hasil kepemilikan yang diverifikasi
- digest artifact yang diunduh

Validasi juga transport paket yang dideklarasikan. Catatan yang hanya berisi endpoint remote adalah valid dan tidak boleh ditolak karena kekurangan paket. Untuk sumber remote, join URL yang dideklarasikan dan tipe transport ke kepemilikan endpoint yang diverifikasi secara independen dan digest dari koneksi tepercaya atau bukti penempatan.

Kode pelajaran mendukung kedua jenis sumber dan meng-hash sumber yang dipilih bersama dengan sumber Registry, nama server, versi Registry, digest catatan, dan digest bukti. Digest provenance yang dihasilkan adalah pointer ringkas ke himpunan bukti lengkap. Ia bukan pengganti untuk mempertahankan bukti itu.

Jangan pernah menerima digest yang hanya disuplai oleh artifact yang sedang kamu verifikasi. Hitung di batas fetch tepercaya, atau terimanya dari layanan paket yang hasil verifikasinya kamu validasi.

### 3. Pin keputusannya, bukan hanya versinya

Versi Registry adalah identifier publikasi unik. Metadata yang dipublikasikan immutable. Catatan yang diubah membutuhkan versi baru. Semantic versioning disarankan, tetapi Registry tidak memintanya dan tidak menerima rentang versi.

Ini berarti `^1.4` bukan pin admission. "latest" juga bukan. Pin yang berguna berisi:

```json
{
  "server": "com.example/inventory",
  "version": "1.0.0",
  "recordDigest": "...",
  "source": {"kind": "package", "registryType": "pypi"},
  "sourceDigest": "...",
  "toolsetDigest": "...",
  "provenanceDigest": "...",
  "registryStatus": "active"
}
```

Mem-pin beberapa lapisan memungkinkanmu mengidentifikasi batas mana yang berubah. Perubahan digest catatan di bawah versi Registry yang sama adalah kegagalan integritas Registry. Perubahan digest sumber di bawah koordinat paket yang sama atau penempatan remote adalah kegagalan integritas sumber-eksekusi. Perubahan digest toolset adalah drift runtime.

### 4. Deteksi drift langsung

Admission seharusnya mengamati server yang benar-benar akan menerima lalu lintas. Panggil `server/discover`, daftar atau dengan cara lain peroleh deskriptor tool yang diekspos melalui jalur tepercaya-mu, dan verifikasi:

- `2026-07-28` ada di `supportedVersions`
- semua kapabilitas yang wajib secara lokal ada
- setiap deskriptor tool memiliki identitas dan permukaan schema yang diperlukan
- digest deskriptor ternormalisasi cocok dengan pin yang diadmit pada pemeriksaan berikutnya

Nilai `_meta["io.modelcontextprotocol/serverInfo"]` yang opsional adalah konteks tampilan, log, dan debugging yang dilaporkan diri. Catat sebagai bukti diagnostik, tetapi jangan pernah menggunakannya untuk menetapkan namespace, kepemilikan paket, kepemilikan endpoint, admission, atau keputusan keamanan lainnya. Alias `serverInfo` langsung di luar `_meta` bukan field kontrak dan tidak boleh dipromosikan menjadi bukti diagnostik.

Normalisasi hanya field yang urutannya tidak bermakna. Contoh mengurutkan daftar tool berdasarkan nama yang stabil sebelum hashing, sehingga perubatan urutan daftar yang tidak berbahaya tidak menyebabkan drift. Ia tidak membuang field deskriptor. Tool baru, schema yang berubah, deskripsi yang berubah, atau anotasi baru mengubah pin.

Contoh memperlakukan deskriptor rusak dan setiap perubahan digest deskriptor sebagai drift, mengarantina pin, menghapus route aktifnya, dan memblokir versi itu sebagai target rollback. Kebijakan produksi bisa mengizinkan perubahan editorial hanya melalui tinjauan baru, karena deskripsi memengaruhi pemilihan tool model. Metadata "kosmetik" bisa mengubah perilaku agen.

### 5. Status Registry adalah state langsung

API Registry melampirkan objek `_meta` tingkat-respons di samping setiap catatan server. Field yang dikelola Registry berada di bawah `_meta["io.modelcontextprotocol.registry/official"]`. Lampaui objek `_meta` respons ke admission dan baca `_meta["io.modelcontextprotocol.registry/official"].status`. Nilai `_meta.status` langsung bukan bentuk wire resmi. Jangan mencampur metadata respons dengan `_meta` milik catatan publikasinya sendiri. Status bisa berupa:

- `active`: dikembalikan secara default dan memenuhi syarat untuk admission lokal
- `deprecated`: masih dapat ditemukan dengan peringatan, tetapi tidak lagi pilihan otomatis yang aman
- `deleted`: disembunyikan secara default sementara catatan historisnya tetap tersedia melalui view terhapus atau inkremental

Sinkronkan status setelah admission. Jika versi active menjadi deprecated atau deleted, karantina pin-nya dan hentikan routing pekerjaan baru ke sana. Pertahankan buktinya. Penghapusan dari daftar default bukan izin untuk menghapus jejak audit-mu.

Metadata kustom yang disuplai penerbit hanya termasuk di bawah `_meta.io.modelcontextprotocol.registry/publisher-provided` dalam catatan publikasi. Metadata respons yang dikelola Registry terpisah. Jangan biarkan penerbit menetapkan status resminya sendiri.

### 6. Rollback berarti pemulihan route

Publikasi immutable tidak diedit selama rollback. Rollback memilih pin yang sebelumnya diadmit, saat ini memenuhi syarat, dan mengubah route aktif.

Target yang aman harus:

1. Memiliki catatan admission yang selesai.
2. Masih memiliki status Registry active di bawah kebijakanmu.
3. Tidak dikarantina oleh bukti runtime atau keamanan.
4. Masih ter-resolve ke paket yang di-pin dan himpunan deskriptor langsung.
5. Lulus pemeriksaan kesehatan saat ini.

Contoh berfokus pada tiga kondisi pertama. Sebuah reconciler yang nyata seharusnya meng-fetch ulang paket dan memeriksa ulang endpoint langsung sebelum aktivasi.

### 7. Tambahkan ledger admission

Basis data admission menyatakan apa yang aktif. Sebuah ledger menjelaskan mengapa.

Setiap entri contoh berisi urutan, waktu, event, server, versi, hasil, alasan, bukti, hash entri sebelumnya, dan hash-nya sendiri. Mengubah hasil yang lebih lama merusak verifikasi entri itu dan setiap tautan berikutnya.

Ini tamper-evident, bukan ajaib tamper-proof. Anchor kepala ledger berkala di domain kepercayaan terpisah, seperti metadata rilis bertanda tangan atau penyimpanan write-once. Batasi siapa yang bisa menambahkan. Jaga token otorisasi, kredensial paket, argumen tool, dan data endpoint privat keluar dari bukti.

## Bangun Sendiri

Controller yang dapat dijalankan ada di `code/main.py`. Ia hanya memakai Python standard library.

Mulai dengan demonstrasi berhingga-finit:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
```

Demonstrasi melakukan lima operasi:

1. Admit `1.0.0` dengan namespace, provenance paket, protokol, kapabilitas, dan tool yang cocok.
2. Admit `1.1.0` dan jadikan aktif.
3. Amati sebuah tool delete yang tidak terduga di runtime.
4. Amati status Registry untuk `1.1.0` menjadi `deprecated`.
5. Pulihkan routing ke pin `1.0.0` yang masih diadmit.

Bentuk yang diharapkan:

```json
{
  "admitted": [true, true],
  "driftAllowed": false,
  "rollbackAllowed": true,
  "activeVersion": "1.0.0",
  "ledgerValid": true
}
```

Baca implementasinya dalam urutan ini:

1. `namespace_for_domain()` dan `namespace_matches()` menetapkan otoritas penamaan yang persis.
2. `digest()` dan `normalized_tools()` menghasilkan bukti deterministik.
3. `RegistryAdmissionController.admit()` menggabungkan publikasi, provenance, runtime, dan kebijakan.
4. `check_live()` membandingkan observasi baru dengan pin.
5. `observe_registry_status()` mengarantina versi yang status Registry-nya berubah.
6. `rollback()` mengaktifkan hanya target memenuh-syarat yang sebelumnya diadmit.
7. `AdmissionLedger.verify()` mendeteksi perubahan pada riwayat yang dicatat.

## Gunakan

Letakkan controller di antara discovery dan routing:

```text
Registry sync -> artifact verifier -> live discovery -> admission controller -> route table
                                               |                 |
                                               v                 v
                                          evidence store    admission ledger
```

Gunakan identitas terpisah untuk pekerjaan-pekerjaan ini. Sebuah worker sync Registry butuh akses baca ke metadata. Sebuah verifier artifact butuh akses fetch paket. Sebuah reconciler route butuh izin mengaktifkan pin yang disetujui. Tidak ada yang membutuhkan setiap kredensial.

Buat state rollout eksplisit. "Approved" berarti bukti lulus kebijakan. "Active" berarti route saat ini memilihnya. "Quarantined" berarti ia tidak bisa menerima pekerjaan baru. "Superseded" berarti versi diadmit lain sedang aktif. Jangan meng-encode keempat makna itu dalam satu Boolean.

Jalankan admission sebelum mengekspos server di `tools/list`. Kalau tidak, klien bisa menemukan tool selama celah di antara publikasi dan evaluasi kebijakan.

## Interactive Lab

Kamu akan mengamati satu batas gagal dalam satu waktu.

### Lab A: tabrakan namespace

Buka shell Python dari direktori kode:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/code
python3 -q
```

Lalu jalankan:

```python
from main import namespace_matches
namespace_matches("com.example/inventory", "com.example")
namespace_matches("com.exampleevil/inventory", "com.example")
```

Hasil pertama adalah `True`; hasil kedua adalah `False`. Ganti perbandingan yang persis dengan `startswith` secara lokal dan amati mengapa nama kedua melewati batas. Pulihkan perbandingan yang persis sebelum melanjutkan.

### Lab B: drift deskriptor

```python
from main import *
times = iter(f"2026-08-21T12:00:{n:02d}+00:00" for n in range(10))
c = RegistryAdmissionController(clock=lambda: next(times))
meta = {OFFICIAL_META_KEY: {"status": "active"}}
c.admit(sample_record("1.0.0"), meta, "com.example", evidence_for("1.0.0"), sample_live("1.0.0"))
c.check_live("com.example/inventory", "1.0.0", sample_live("1.0.0", True))
```

Inspeksi alasan dan state route-nya. Paket dan catatan Registry tidak berubah. Permukaan tool runtime berubah, jadi controller mengarantina dan menonaktifkan pin. Inilah mengapa kontrol rantai pasokan harus berlanjut setelah instalasi.

### Lab C: status dan rollback

Admit `1.1.0`, tandai sebagai deprecated, dan coba kedua target rollback:

```python
c.admit(sample_record("1.1.0"), meta, "com.example", evidence_for("1.1.0"), sample_live("1.1.0"))
c.observe_registry_status("com.example/inventory", "1.1.0", "deprecated")
c.rollback("com.example/inventory", "1.1.0", "unsafe retry")
c.rollback("com.example/inventory", "1.0.0", "restore known release")
c.ledger.verify()
```

Target yang dikarantina ditolak. Pin active sebelumnya diterima. Ledger tetap valid.

## Practice Lab

Perluas controller dengan gerbang persetujuan dua-orang.

Persyaratan:

- Simpan persetujuan sebagai referensi bukti bertanda tangan, bukan nama mutable di pin.
- Minta dua identitas peninjau yang berbeda untuk toolset yang berisi tool dengan `destructiveHint: true`.
- Tolak identitas peninjau duplikat.
- Pertahankan percobaan admission asli di ledger ketika persetujuan tidak lengkap.
- Tambahkan test untuk nol, satu, duplikat, dan dua persetujuan berbeda.
- Jangan log tanda tangan, kredensial, atau argumen tool privat lengkap.

Keberhasilan berarti tool destruktif tidak bisa menjadi aktif sampai kedua identitas menyetujui digest catatan, paket, dan toolset yang persis.

## Shipped Artifact

Pelajaran ini mengirimkan `outputs/skill-mcp-registry-admission.md`. Gunakan sebagai runbook datar yang dapat digunakan ulang ketika meninjau versi Registry baru atau menyelidiki drift. Ia mendefinisikan input, aturan penolakan, bundel bukti, rekonsiliasi status, dan bukti rollback tanpa bergantung pada nama kelas contoh.

## Verify It

Jalankan demonstrasi dan suite deterministik:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Verifikasi seharusnya membuktikan:

- batas namespace yang persis menolak prefix yang mirip
- hanya status Registry ber-namespace resmi yang bisa membuat versi memenuhi syarat
- bukti paket dan remote yang tidak terverifikasi atau tidak cocok ditolak
- metadata penerbit tidak bisa menyamar sebagai metadata yang dikelola Registry
- pengurutan tool dinormalisasi tanpa menyembunyikan perubahan deskriptor
- struktur paket dan tool yang rusak menolak dengan aman
- `serverInfo` tetap diagnostik dan tidak pernah menyuplai otoritas admission
- drift deskriptor mengarantina, menonaktifkan, dan memblokir rollback ke pin
- perubahan status mengarantina pin aktif
- rollback tidak bisa memilih versi yang dikarantina atau tidak dikenal
- pemalsuan ledger terdeteksi

## Production Failure Modes

| Kegagalan | Mengapa terjadi | Respons yang diperlukan |
|---|---|---|
| Nama terlihat valid tetapi namespace tidak pernah diautentikasi | Kebijakan mempercayai teks catatan | Tolak sampai verifier namespace tepercaya menyuplai prefix yang persis |
| Koordinat paket yang sama mengembalikan byte baru | Upstream mutable atau distribusi yang disusupi | Hentikan aktivasi, simpan kedua digest, selidiki batas fetch |
| "Latest" berubah tanpa tinjauan | Seleksi melayang lolos dari pin | Resolve hanya versi dan digest yang diadmit persis |
| Tool baru muncul setelah persetujuan | Drift runtime atau penempatan berbeda | Karantina route dan capture observasi deskriptor yang baru |
| Versi deprecated tetap aktif | Sinkronisasi status hilang atau tertunda | Rekonsiliasi status secara berkala dan sebelum aktivasi |
| Catatan yang terhapus hilang dari sinkronisasi default | Klien hanya meminta catatan aktif | Gunakan rekonsiliasi inkremental atau sadar-terhapus dan pertahankan riwayat lokal |
| Target rollback tidak pernah diadmit | Kontrol route dan state persetujuan terputus | Tolak rollback dan jalankan admission baru untuk target |
| Ledger terverifikasi secara lokal setelah penyerang menulis ulang semua entri | Rantai hash tidak memiliki anchor eksternal | Publikasikan kepala ledger bertanda tangan ke domain kepercayaan terpisah |
| Bukti berisi bearer token atau argumen tool | Logging menyalin permintaan utuh | Redaksi saat pengumpulan dan simpan hanya bukti minimum |

## Operational Rule

Publikasi menjawab "apakah identitas ini boleh menerbitkan nama ini?" Admission menjawab "apakah kita akan mengeksekusi artifact persis ini dan mengekspos perilaku persis ini?" Pertahankan keputusan-keputusan itu terpisah, pin setiap join, dan buat rollback memilih bukti alih-alih ingatan.

## Bacaan Lanjutan

- [Persyaratan server.json Registry resmi](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [Kontrak OpenAPI Registry resmi](https://registry.modelcontextprotocol.io/openapi.yaml)
- [Penemuan server MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
