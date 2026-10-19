# Rekayasa Konformansi MCP: Versioning, Bukti, dan Operasi

> Sebuah server tidak konforman hanya karena jalur bahagianya bekerja melalui satu SDK. Konformansi hidup di wire, di batas versi, melalui perantara, dan selama rollback.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 17 (gateways), Phase 13 · 30 (registry admission)
**Time:** ~100 minutes

## Learning Objectives

- Mengubah aturan normatif MCP menjadi transkrip wire golden dan negatif.
- Menjaga perilaku ketat `2026-07-28` terpisah dari fallback legacy yang berbatas.
- Membedakan field tidak dikenal aditif dari `resultType` tidak dikenal yang invalid.
- Membandingkan bukti JSON-RPC mentah dengan tampilan ternormalisasi SDK.
- Membuktikan integritas header dan body melalui batas proxy nyata.
- Menggerbangkan rilis dengan bukti transkrip yang di-redaksi, kesehatan, dan rollback.

## Masalahnya

Klienmu memanggil `tools/list` melalui SDK dan mendapat tool. Test integrasi lulus.

Hasil itu meninggalkan pertanyaan penting yang belum terjawab:

- Apakah permintaan membawa metadata protokol per-permintaan modern?
- Apakah `MCP-Protocol-Version`, `Mcp-Method`, dan `Mcp-Name` cocok dengan body JSON-RPC?
- Apakah respons berisi `resultType` yang valid di wire, atau apakah SDK mensintesis satu?
- Apakah klien akan mempertahankan field aditif masa depan?
- Apakah kesalahan modern yang dikenali tidak sengaja memicu handshake legacy?
- Apakah proxy mempertahankan status origin dan kesalahan JSON-RPC?
- Apakah serializer notifikasi memancarkan respons yang terlarang?
- Bisakah operasi membuktikan mengapa rilis dipromosikan atau di-rollback tanpa menyimpan rahasia?

Konformansi adalah sekumpulan invarian yang dapat diamati. Bangun harness yang menangkap invarian-invarian itu sebelum lalu lintas produksi harus menemukannya.

```figure
mcp-conformance-operations
```

## Mulai dari Era Versi

MCP `2026-07-28` memakai metadata per-permintaan yang mandiri. Permintaan modern membawa `params._meta.io.modelcontextprotocol/protocolVersion` dan `params._meta.io.modelcontextprotocol/clientCapabilities`. Kunci ber-namespace yang persis itu penting; alias `protocolVersion` atau `clientCapabilities` polos adalah rusak. Ketika header routing yang di-mirror ada di batas HTTP, nilainya harus setuju dengan body JSON-RPC. Hasil sukses modern membawa `resultType`.

Versi hingga `2025-11-25` memakai era inisialisasi yang lebih awal. Hasil legacy tanpa `resultType` diinterpretasikan sebagai complete hanya setelah klien memilih era yang lebih awal itu.

Jangan membuat satu validator longgar yang menerima kedua bentuk sekaligus. Gunakan dua cabang:

| Cabang | Bukti masuk | `resultType` hilang | Inisialisasi |
|---|---|---|---|
| Modern | `server/discover` sukses atau respons modern yang dikenali | Invalid | Bukan jalur default |
| Legacy | Allowlist terkonfigurasi plus hasil `initialize` legacy yang valid setelah probe modern yang tidak konklusif | Diinterpretasikan sebagai complete | Diminta oleh era itu |

Pemisahan itu mencegah peer modern yang rusak dihadiahi validasi yang lebih lemah.

### Mode ketat

Mode ketat membutuhkan bukti perilaku modern. `server/discover` yang sukses membuktikan cabang modern. Kesalahan JSON-RPC modern yang dikenali juga membuktikannya. Perbaiki permintaannya atau berhenti. Jangan pernah downgrade karena server mengembalikan `-32020`, `-32021`, atau `-32022`.

### Mode fallback

Mode fallback melakukan satu probe modern yang berbatas. Timeout, balasan kosong, koneksi tertutup, atau respons yang tidak dikenali bersifat tidak konklusif. Itu tidak membuktikan bahwa peer adalah legacy. Hanya endpoint yang secara eksplisit dikonfigurasi atau di-allowlist untuk kompatibilitas yang boleh kemudian menerima probe legacy yang berbatas, dan klien memilih cabang legacy hanya setelah memvalidasi hasil `initialize` probe itu dan revisi legacy yang dinegosiasikan.

Fallback bukan "coba legacy setelah kesalahan apa pun." Kesalahan modern yang dikenali berisi informasi koreksi yang berguna. Downgrade setelahnya bisa menyembunyikan ketidakcocokan header, deklarasi kapabilitas yang hilang, atau versi yang tidak didukung.

Ini mencegah penyerang, outage, atau proxy filtering memaksa downgrade dengan menjatuhkan respons modern. Catat kebijakan endpoint, observasi modern yang tidak konklusif, bukti legacy positif yang persis, dan era yang dipilih bersama.

Catat era yang dipilih di samping setiap transkrip. Tanpa fakta itu, field yang hilang bisa terlihat dapat diterima dalam satu run test dan invalid dalam run lain.

## Bangun Korpus Transkrip

Fixture transkrip mencatat apa yang melewati batas, bukan hanya panggilan SDK:

```json
{
  "name": "golden-modern-list",
  "era": "modern",
  "headers": {
    "MCP-Protocol-Version": "2026-07-28",
    "Mcp-Method": "tools/list"
  },
  "request": {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {
      "_meta": {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {}
      }
    }
  },
  "responseStatus": 200,
  "responseBody": {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
      "resultType": "complete",
      "tools": []
    }
  }
}
```

Pertahankan dua kelas fixture.

### Transkrip golden

Transkrip golden membuktikan perilaku yang diterima:

- permintaan discovery atau method modern dengan metadata dan header yang cocok
- hasil complete dengan field yang wajib
- hasil `input_required` ketika method bisa meminta lebih banyak input
- hasil ekstensi hanya setelah kapabilitas yang sesuai diiklankan
- hasil legacy tanpa `resultType`, tetapi hanya di era legacy yang dipilih
- pemrosesan notifikasi tanpa respons JSON-RPC

Transkrip golden bersifat presisi, bukan besar. Jaga ID dan timestamp yang mudah berubah tetap deterministik atau normalisasi sebelum perbandingan.

### Transkrip negatif

Transkrip negatif membuktikan perilaku penolakan:

- ketidakcocokan header dan body
- kapabilitas per-permintaan yang hilang
- versi protokol yang cocok tetapi tidak didukung
- `resultType` modern yang hilang
- `resultType` yang tidak dikenal atau tidak diiklankan
- respons `jsonrpc` selain `2.0` atau ID yang berbeda dalam nilai atau tipe JSON
- respons yang berisi `result` dan `error` keduanya, atau tidak keduanya
- kesalahan tanpa `code` integer dan `message` string
- kesalahan protokol yang dikenal dipetakan ke status HTTP yang salah
- respons yang dipancarkan untuk notifikasi
- amplop JSON-RPC yang rusak
- runtuhnya protokol error oleh proxy

Untuk setiap kasus negatif, assert batas penolakan dan kode kesalahan yang stabil. "Panggilannya gagal" terlalu lemah. 500 yang dihasilkan proxy dan `-32020` origin bisa keduanya terlihat seperti kegagalan sambil menceritakan kisah yang sama sekali berbeda kepada operator.

Fixture ketidakcocokan-header harus menyertakan respons HTTP 400 JSON-RPC yang sebenarnya dari server dengan ID permintaan dan kode kesalahan `-32020` yang cocok. Tegakkan itu secara otomatis setiap kali validator lokal mengamati `HeaderMismatch`; jangan menjadikan verifikasi respons sebagai flag fixture opsional. Kasus dengan HTTP 500 dan tanpa body gagal bahkan ketika kode penolakan lokal sudah benar. Harness yang berhenti setelah validator permintaannya sendiri melempar hanya menguji dirinya sendiri, bukan perilaku wire server.

Proyek konformansi MCP resmi berguna sebagai suite eksternal dan referensi ber-versi. Pertahankan juga transkrip lokal-mu. Mereka menangkap proxy, SDK, autentikasi, ekstensi, dan jalur rilis-mu, yang tidak bisa diketahui suite umum.

## Nilai Header Harus Cocok dengan Body RPC

Di Streamable HTTP modern, perantara bisa merutekan atau memberlakukan kebijakan menggunakan header yang di-mirror. Body JSON-RPC tetap sumber kebenaran protokol. Ketidakcocokan adalah kegagalan integritas, bukan petunjuk untuk memilih salah satu nilai.

Validasi dalam urutan ini:

1. Parse dan validasi amplop JSON-RPC dan tipe metadata.
2. Bandingkan `MCP-Protocol-Version` dengan `params._meta.io.modelcontextprotocol/protocolVersion`.
3. Bandingkan `Mcp-Method` dengan `method`.
4. Ketika method punya nama routing, bandingkan `Mcp-Name` dengan nilai body yang sesuai.
5. Setelah kesamaan ditetapkan, putuskan apakah versi dan himpunan kapabilitas yang cocok didukung.

Urutan ini membedakan `-32020` ketidakcocokan dari `-32022` versi tidak didukung. Ia juga menghentikan gateway mengotorisasi nama header sementara origin mengeksekusi nama body yang berbeda.

Nama field HTTP case-insensitive, sementara nilainya tetap case-sensitive. Normalisasi nama header sebelum lookup dan tolak duplikat yang konflik. Untuk `Mcp-Name` yang tidak aman, non-ASCII, atau dengan whitespace di awal-atau-akhir, decode sentinel UTF-8 `=?base64?{Base64EncodedValue}?=` yang persis sebelum membandingkannya dengan body. Tolak sentinel yang tidak lengkap, Base64 yang tidak valid, UTF-8 yang tidak valid, atau nilai mentah yang tidak aman dengan `-32020`. Whitespace mentah di sekeliling invalid bahkan ketika body berisi karakter yang sama karena nilai itu membutuhkan encoding sentinel sebelum transport.

Sebuah perantara bisa menolak HTTP yang rusak sebelum permintaan mencapai server MCP, jadi kegagalannya mungkin kesalahan HTTP tanpa JSON-RPC. Capture apakah penolakan datang dari perantara atau origin. Server MCP origin seharusnya memakai kontrak kesalahan protokol ketika ia menangani permintaan JSON-RPC yang valid.

## Field yang Tidak Dikenal Bukan Hasil yang Tidak Dikenal

Kompatibilitas maju membutuhkan dua aturan berbeda.

### Field tidak dikenal aditif

Objek hasil dan peta `_meta` bisa mendapat field. Sebuah validator seharusnya mempertahankan atau mengabaikan field aditif menurut perannya, kecuali field itu melanggar kontrak cadangan. Contoh mempertahankan hasil mentah lengkap dalam bukti dan menerima `futureHint` di samping hasil yang dikenal.

Jika kamu adalah proxy transparan, mempertahankan field tidak dikenal biasanya lebih aman daripada meny-stripnya. Jika kamu adalah klien aplikasi, mengabaikannya bisa valid. Test diferensial-mu tetap seharusnya mengungkap bahwa SDK melewatkannya sehingga perilakunya disengaja.

### `resultType` tidak dikenal

`resultType` adalah diskriminator. Hasil modern inti memakai `complete` atau `input_required`. Sebuah ekstensi bisa menambahkan nilai lain hanya ketika kapabilitasnya diiklankan. Ekstensi Tasks, sebagai contoh, bisa menambahkan `task` di konteks kapabilitas yang dinegosiasikan itu.

Diskriminator yang tidak dikenal atau tidak diiklankan tidak bisa diperlakukan sebagai complete dengan aman. Klien tidak tahu siklus hidup apa yang akan ia buang. Tolaknya.

Respons mentah yang sama karena itu bisa berisi field tidak dikenal yang dapat diterima dan tipe hasil tidak dikenal yang tidak dapat diterima. Uji kedua kasus.

Diskriminator hanyalah lapisan pertama. Validasi payload spesifik-method setelahnya. Hasil `tools/list` yang lengkap membutuhkan array `tools` yang deskriptornya memiliki nama non-kosong unik, deskripsi yang berguna, dan nilai `inputSchema` ber-akar-objek. Hasil `task` valid hanya untuk `tools/call` yang memenuhi syarat dengan kapabilitas Tasks dan membutuhkan `taskId`, status yang dikenal, timestamp pembuatan dan pembaruan, dan `ttlMs`, ditambah interval polling opsional yang valid. Hasil `completion/complete` yang lengkap membutuhkan objek `completion` dengan paling banyak 100 nilai string, `total` integer non-negatif opsional yang tidak lebih kecil dari nilai yang dikembalikan, dan Boolean `hasMore` opsional. `resultType yang dieja dengan baik tidak bisa membuat payload yang rusak menjadi konforman.

## Invarian Notifikasi

Notifikasi JSON-RPC tidak punya `id`. Penerima tidak boleh mengirim respons sukses atau kesalahan JSON-RPC.

Untuk bentuk notifikasi HTTP yang diterima, harness mengharapkan HTTP `202` dengan body kosong. MCP `2026-07-28` tidak mendefinisikan notifikasi inti klien-ke-server melalui Streamable HTTP. Contoh memakai notifikasi ekstensi kursus ber-namespace hanya untuk menguji invarian serializer satu-arah. Jangan menyajikannya sebagai method inti baru.

Uji serializer-nya, bukan hanya handler-nya. Sebuah handler bisa mengembalikan `None` sementara middleware membungkusnya dalam objek sukses JSON. Capture byte egress akhir.

### Tambahkan Diferensial SDK

SDK sering mengubah objek wire menjadi tipe bahasa yang nyaman. Itu berguna, tetapi objek ternormalisasi tidak bisa membuktikan apa yang diterima.

Untuk setiap fixture berisiko-tinggi, capture:

1. Status, header, dan body respons mentah sebelum decoding SDK.
2. Nilai kembalian atau exception ternormalisasi SDK.
3. Proyeksi semantik yang diharapkan untuk era yang dipilih.
4. Field yang diangkat, disintesis, di-strip, atau diubah oleh SDK.

Contoh mengizinkan penghapusan SDK-saja dari bookkeeping wire yang dikenal seperti `resultType`, `_meta`, `ttlMs`, dan `cacheScope` sambil membandingkan payload aplikasi. Ia melaporkan `futureHint` yang dijatuhkan karena field semantik tidak dikenal itu menghilang.

Jangan berasumsi setiap perbedaan adalah bug SDK. Poinnya adalah membuat transformasi terlihat. Putuskan apakah komponenmu adalah endpoint aplikasi, yang boleh mengabaikan field aditif, atau perantara transparan, yang seharusnya mempertahankannya.

Jalankan diferensial terhadap setiap SDK dan versi yang kamu kirim. Jika dua SDK menormalisasi transkrip yang sama secara berbeda, kebijakan rilis seharusnya menyatakan perilaku mana yang dapat diterima alih-alih memilih output yang paling nyaman setelahnya.

## Capture Bukti Proxy

Kegagalan MCP produksi sebagian besar terjadi di lebih dari satu proses. Catat tiga tampilan:

| Tampilan | Bukti minimum |
|---|---|
| Ingress | header permintaan, body JSON-RPC, content type, route yang diautentikasi, waktu penerimaan |
| Origin | header yang di-forward dan digest body, status origin, header dan body respons |
| Egress | status yang terlihat klien, header, body, dan waktu pengiriman |

Contoh mendeteksi dua transformasi umum:

- kesalahan JSON-RPC HTTP 400 atau 404 origin menjadi proxy 500 generik
- body JSON-RPC egress berbeda dari body origin

Tambahkan assert spesifik-penempatan untuk content type, `Accept`, kompresi, SSE ber-scoped permintaan, header cache, dan korelasi trace. Capture kedua sisi terminasi TLS ketika kebijakan mengizinkan. Jangan pernah log kredensial hanya untuk membuktikan jalurnya.

## Redaksi Sebelum Bukti Meninggalkan Memori

Redaksi adalah bagian dari operasi konformansi, bukan tugas pembersihan nanti. Terapkannya sebelum serialisasi, hashing, log, artefak test, atau upload kegagalan.

Contoh melipat-gandakan nama key case-fold dan menghapus pemisah sebelum pencocokan, lalu secara rekursif mengganti nilai di bawah key seperti `Authorization`, `Cookie`, `Set-Cookie`, `X-Api-Key`, `accessToken`, `clientSecret`, `registrationAccessToken`, `token`, `password`, `secret`, dan `api_key`. Kanonisasi dan denylist harus memakai bentuk yang sama sehingga varian camelCase, ber-hyphen, ber-underscore, dan ber-titik tidak bisa saling melewati kebijakan. Pengumpul produksi seharusnya menambahkan kebijakan argumen spesifik-method, karena key yang tidak berbahaya seperti `query` tetap bisa berisi data pribadi atau teregulasi.

Hash bundel bukti yang di-redaksi. Simpan capture mentah hanya dalam sistem jangka-pendek yang disetujui ketika investigasi tertentu membutuhkannya. Sebuah digest membuktikan bundel mana yang menghasilkan keputusan; ia tidak mengungkap nilai yang dihapus.

## Jadikan Kesehatan dan Rollback Bagian dari Gerbang

Konformansi protokol diperlukan tetapi tidak cukup untuk rilis. Sebuah kandidat konforman tetap bisa timeout, membocorkan memori, atau membebani dependensi.

Definisikan jendela kesehatan sebelum rollout:

- jumlah sampel minimum
- tingkat kesalahan maksimum
- persentil latensi maksimum
- batas saturasi atau sumber daya
- durasi observasi
- perbandingan dengan baseline yang diadmit

Definisikan juga bukti rollback sebelum rollout:

- versi sebelumnya yang persis
- digest bukti admission
- pin artifact dan deskriptor SHA-256
- status Registry saat ini
- hasil kesehatan saat ini
- prosedur pemulihan route
- attestation di atas field-field persis itu dari identitas release-controller tepercaya

Minta target rollback itu terverifikasi dan sehat sebelum promosi, bukan hanya setelah kandidat gagal. Rilis sukses tanpa jalur pemulihan yang dapat digunakan tidak siap-produksi.

Jika sebuah kandidat gagal dan target rollback kekurangan bukti itu, tahan lalu lintas alih-alih menebak. "Roll back ke apa pun yang ada di sana" bukan kontrol operasional.

Jangan mengurangi kesiapan menjadi pemeriksaan truthiness seperti versi non-kosong, `healthy: "yes"`, atau string bukti sembarang. Contoh meminta tipe yang persis, status active, tiga digest SHA-256, signer tepercaya, dan attestation HMAC-SHA-256 yang valid di atas payload rollback lengkap. Kunci demo deterministiknya adalah fixture non-rahasia. Suntikkan kunci yang dilindungi, hasil verifikasi KMS, atau verifier attestation kunci-publik di batas rilis dalam produksi.

Gerbang rilis juga menolak transkrip, diferensial SDK, atau bukti proxy yang kosong. Setiap sumber harus membawa digest bukti yang valid. Jendela kesehatan yang hijau tidak bisa mengisi batas yang tidak pernah diamati.

## Bangun Sendiri

Jalankan harness standard-library:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
```

Demo menjalankan persis lima belas transkrip golden dan negatif, termasuk hasil completion yang valid dan rusak, membandingkan hasil mentah dengan tampilan SDK, memeriksa proxy yang meruntuhkan kesalahan origin, mengevaluasi kesehatan, mengautentikasi bukti rollback, dan memilih target itu.

Bentuk yang diharapkan:

```json
{
  "transcriptsPassed": 15,
  "transcriptsTotal": 15,
  "sdkDroppedFields": ["futureHint"],
  "proxyIssues": [
    "proxy collapsed a protocol error into HTTP 500",
    "proxy changed the origin JSON-RPC body"
  ],
  "releaseAction": "rollback",
  "evidenceDigest": "..."
}
```

Baca `code/main.py` dalam urutan ini:

1. `validate_request()` menegakkan aturan permintaan dan header spesifik-era.
2. `validate_result()` memisahkan diskriminator legacy yang hilang, nilai modern yang valid, ekstensi, dan nilai yang tidak dikenal.
3. `select_era()` mengimplementasikan kebijakan ketat dan fallback berbatas.
4. `run_transcript()` mengevaluasi fixture golden dan negatif.
5. `compare_sdk_view()` mengekspos perbedaan normalisasi.
6. `inspect_proxy()` membandingkan bukti ingress, origin, dan egress.
7. `redact()` menghapus rahasia yang obvious sebelum hashing bukti.
8. `rollback_evidence_ready()` memvalidasi field pin yang persis dan attestation rilis tepercaya.
9. `ReleaseGate.evaluate()` menggabungkan bukti konformansi, SDK, proxy, kesehatan, dan rollback yang non-kosong.

## Gunakan

Jalankan harness di empat titik:

1. Pada setiap perubahan implementasi dengan test adapter in-process.
2. Terhadap binary klien dan server yang di-build melalui transport yang sebenarnya.
3. Melalui proxy atau gateway yang di-deploy di lingkungan staging.
4. Selama rollout canary dengan bukti kesehatan dan rollback langsung.

Pertahankan nama kasus yang sama yang stabil di seluruh lapisan. `negative-header-body-mismatch` seharusnya berarti invarian yang sama dalam laporan unit, end-to-end, proxy, dan canary. Digest bukti akan berbeda karena batasnya berubah; persyaratannya tidak seharusnya berubah.

Simpan schema fixture di version control. Simpan bukti run yang di-redaksi di sistem rilis-mu. Simpan capture mentah jangka-pendek hanya di bawah kontrol akses insiden.

## Interactive Lab

### Lab A: buktikan batas era

Dari direktori `code`, buka Python:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/code
python3 -q
```

Jalankan:

```python
from main import *
validate_result({"tools": []}, "legacy")
validate_result({"tools": []}, "modern")
```

Panggilan legacy menyimpulkan `complete`. Panggilan modern melempar `ProtocolViolation`. Sekarang uji fallback:

```python
select_era({"kind": "timeout"}, "fallback")
select_era(
    {"kind": "timeout"},
    "fallback",
    legacy_allowed=True,
    legacy_evidence={"kind": "initialize_success", "protocolVersion": LEGACY_VERSION},
)
select_era({"kind": "jsonrpc_error", "code": -32021}, "fallback")
```

Timeout pertama gagal-closed karena keheningan bukan bukti legacy. Panggilan kedua memilih legacy hanya karena konfigurasi mengizinkannya dan hasil inisialisasi legacy yang valid diamati. Kesalahan kapabilitas-hilang yang dikenali membuktikan cabang modern.

### Lab B: field aditif versus diskriminator

```python
validate_result({"resultType": "complete", "tools": [], "futureHint": True}, "modern")
validate_result({"resultType": "future_mode", "tools": []}, "modern")
```

Hasil pertama mempertahankan `futureHint`. Hasil kedua ditolak karena diskriminator siklus hidup tidak dikenal.

### Lab C: periksa transformasi SDK

```python
compare_sdk_view(
    {"resultType": "complete", "tools": [], "futureHint": {"mode": "new"}},
    {"tools": []},
)
```

Putuskan apakah komponenmu boleh mengabaikan `futureHint` atau harus meneruskannya. Tulis pilihan itu ke dalam kebijakan rilis. Jangan diam-diam menghapus diferensial-nya.

### Lab D: perbaiki proxy

Modifikasi pertukaran demo agar egress mempertahankan status dan body origin. Jalankan `python3 main.py` lagi. Masalah proxy seharusnya menghilang, tetapi diferensial SDK tetap memblokir promosi. Lalu sertakan `futureHint` dalam tampilan SDK dan amati aksinya berubah menjadi `promote` ketika setiap sumber bukti lulus.

## Practice Lab

Tambahkan transkrip SSE ber-scoped permintaan ke harness.

Persyaratan:

- Capture status respons, content type, event SSE berurutan, dan terminasi stream.
- Buktikan setiap event JSON-RPC memiliki hasil atau kesalahan spesifik-era yang valid.
- Tambahkan kasus negatif untuk proxy yang menbuffer seluruh stream sebelum meneruskan.
- Tambahkan kasus negatif untuk event SSE yang id JSON-RPC-nya berbeda dari permintaan.
- Redaksi data event sebelum menulis bukti.
- Sertakan durasi stream, latensi event-pertama, dan jumlah event dalam jendela kesehatan.
- Buat gerbang rilis hanya memilih target rollback yang ber-bukti ketika stream gagal.

Keberhasilan berarti kasus yang sama berjalan langsung dan melalui proxy, dengan laporan yang mengidentifikasi batas persis yang mengubah perilaku.

## Shipped Artifact

Pelajaran ini mengirimkan `outputs/skill-mcp-conformance-release-gate.md`. Gunakan untuk mengubah perubahan server, klien, gateway, atau SDK menjadi matriks konformansi ber-versi dan keputusan rilis. Artifact meminta bukti wire mentah, kasus negatif, pemilihan era eksplisit, diferensial SDK, bukti proxy, redaksi, ambang kesehatan, dan bukti rollback.

## Verify It

Jalankan demo dan suite deterministik:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Verifikasi seharusnya membuktikan:

- setiap transkrip golden dan negatif yang disertakan mencapai hasil yang diharapkan
- permintaan modern membutuhkan kunci metadata ber-namespace yang persis
- nama header HTTP dicocokkan case-insensitive dan nilai `Mcp-Name` yang ter-encode di-decode dengan persis
- ketidakcocokan header dan body mengembalikan kode ketidakcocokan modern
- versi respons, ID, eksklusivitas hasil-atau-kesalahan, bentuk kesalahan, dan pemetaan HTTP tervalidasi
- persyaratan payload spesifik-method untuk daftar-tool, task, dan completion ditegakkan
- setiap `HeaderMismatch` yang diamati membutuhkan respons HTTP 400 JSON-RPC `-32020` yang sebenarnya
- whitespace `Mcp-Name` mentah ditolak sementara whitespace yang di-encode sentinel persis round-trip
- `resultType` yang hilang valid hanya di era legacy yang dipilih
- field aditif selamat dari validasi mentah sementara tipe hasil yang tidak dikenal gagal
- tipe hasil ekstensi membutuhkan kapabilitas yang diiklankan
- kesalahan modern yang dikenali tidak pernah menyebabkan fallback legacy
- notifikasi tidak menghasilkan respons JSON-RPC
- penghapusan bookkeeping SDK dan kehilangan field semantik dibedakan
- runtuhnya kesalahan proxy terdeteksi dan kredensial di-redaksi secara rekursif di seluruh varian camelCase dan pemisah
- promosi membutuhkan bukti transkrip, SDK, proxy, dan operasional yang sehat non-kosong
- promosi dan rollback keduanya membutuhkan target rollback yang diautentikasi, di-pin, aktif, dan sehat

## Production Failure Modes

| Kegagalan | Apa yang dilaporkan test yang lemah | Apa yang harus dibuktikan harness |
|---|---|---|
| SDK mensintesis diskriminator yang hilang | "tools/list lulus" | Hasil modern mentah kekurangan `resultType` dan invalid |
| Klien downgrade setelah `-32021` | "retry legacy berhasil" | Kesalahan modern yang dikenali melarang fallback |
| Tipe hasil yang tidak dikenal diperlakukan complete | "respons diparse" | Diskriminator siklus hidup yang tidak diiklankan ditolak |
| Proxy mengotorisasi satu tool dan origin mengeksekusi yang lain | "permintaan mencapai server" | `Mcp-Name` sama dengan nama routing body di setiap hop |
| Harness melempar sebelum membaca respons server | "test ketidakcocokan header lulus" | Respons HTTP 400 dan JSON-RPC `-32020` di-capture dan divalidasi |
| Proxy mengubah origin 400 menjadi 500 generik | "kesalahan upstream" | Status origin dan egress serta body JSON-RPC dipertahankan |
| Middleware notifikasi memancarkan `{result: null}` | "handler mengembalikan none" | Body egress akhir kosong dan tidak ada respons JSON-RPC |
| SDK men-strip field aditif | "objek bertipe cocok" | Tampilan mentah dan ternormalisasi menunjukkan field yang dijatuhkan persis |
| Artefak kegagalan membocorkan bearer token | "bundle debug diunggah" | Redaksi terjadi sebelum hashing, logging, atau upload |
| Gaya key kredensial melewati redaksi | "denylist berisi api_key" | Varian camelCase dan pemisah berbagi satu bentuk denylist kanonis |
| Canary tanpa sampel tetapi tampak sehat | "nol kesalahan" | Jumlah sampel minimum ditegakkan |
| Rollback memilih build yang tidak dikenal | "deployment sebelumnya dipulihkan" | Versi target, digest admission, pin, status, dan kesehatan ada |

## Operational Rule

Uji byte yang kamu kirim, byte yang setiap perantara teruskan, semantik yang diekspos setiap SDK, dan bukti yang akan digunakan operasi di bawah tekanan. Kompatibilitas adalah cabang eksplisit. Rollback adalah aksi rilis yang ber-bukti. Tidak ada yang seharusnya menjadi efek samping yang tidak disengaja dari parser yang longgar.

## Bacaan Lanjutan

- [Protokol dasar MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [Negosiasi versi MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Proyek konformansi MCP resmi](https://github.com/modelcontextprotocol/conformance)
