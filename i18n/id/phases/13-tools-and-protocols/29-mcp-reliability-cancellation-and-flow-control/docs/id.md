# Keandalan MCP, Pembatalan, dan Kontrol Aliran

> ID permintaan mengorelasikan sebuah pesan. Ia tidak membuat side effect aman, menghentikan worker, atau melindungi stream dari consumer yang lambat.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 09 and 13
**Time:** ~120 minutes

## Learning Objectives

- Mengimplementasikan sinyal pembatalan yang benar untuk stdio dan Streamable HTTP.
- Menyelesaikan race completion dan pembatalan tanpa mengirim pesan setelah pembatalan.
- Memisahkan pembatalan permintaan dari semantik `tasks/cancel` yang tahan lama.
- Membangun keputusan retry dari side effect dan kunci idempotensi eksplisit.
- Membatasi antrean progres sambil mempertahankan respons akhir.
- Memulihkan stream melalui reconnect, refetch, dan backoff ber-jitter.

## Masalahnya

Jalur bahagia menyembunyikan bug sistem-terdistribusi yang paling mahal.

Sebuah klien memanggil tool. Server memulai kerja. Progres tiba. Sebuah proxy menbuffer stream. Klien mencapai timeout-nya dan terputus. Server menyelesaikan satu milidetik kemudian. Klien mencoba ulang dengan id JSON-RPC baru. Mutasi berjalan dua kali.

Setiap komponen berperilaku secara lokal. Sistem gagal secara global.

MCP mendefinisikan perilaku pesan dan transport, tetapi aplikasimu tetap memiliki:

- anggaran waktu;
- idempotensi bisnis;
- antrean berbatas;
- klasifikasi retry;
- state task yang tahan lama;
- kebijakan reconnect dan refetch.

Pelajaran ini membangun keputusan-keputusan itu ke dalam simulator deterministik. Tidak ada sleep, socket, atau kegagalan acak. Kamu mengontrol urutan event pembatalan secara langsung. Satu test thread tersinkron memaksa dua klien ledger bersaing untuk kunci idempotensi yang sama.

## Pembatalan Permintaan Bersifat Spesifik-Transport

Niatnya sama di setiap transport: klien tidak lagi membutuhkan hasil yang sedang berjalan. Sinyal wire-nya berbeda.

### stdio

stdio memakai satu kanal bidireksional bersama. Klien mengirim sebuah notifikasi:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/cancelled",
  "params": {
    "requestId": 41,
    "reason": "User closed the operation"
  }
}
```

Notifikasi itu fire-and-forget. Server tidak memancarkan respons JSON-RPC untuk itu.

Server seharusnya menghentikan kerja, membebaskan sumber daya, dan menghindari mengirim respons untuk permintaan yang dibatalkan. Ia bisa mengabaikan pembatalan ketika permintaan tidak dikenal, sudah selesai, atau tidak bisa dihentikan dengan aman.

Notifikasi pembatalan yang rusak, tidak dikenal, dan sudah selesai diabaikan. Mengubah race-race itu menjadi kesalahan baru akan menciptakan lebih banyak race.

### Streamable HTTP

Streamable HTTP modern memberi setiap permintaan respons HTTP-nya sendiri atau stream respons SSE. Klien membatalkan dengan menutup stream respons permintaan itu.

Jangan POST `notifications/cancelled` untuk permintaan HTTP biasa. Penutupan stream adalah sinyal pembatalan.

Setelah server mengamati disconnect, ia seharusnya menghentikan kerja dan tidak boleh mengirim lebih banyak pesan untuk permintaan itu.

### Pembatalan yang dikirim server bersifat sempit

Sebuah server tidak memakai `notifications/cancelled` untuk membatalkan panggilan klien sembarang. Di stdio, pembatalan yang dikirim server dicadangkan untuk mengakhiri permintaan `subscriptions/listen`. Jaga jalur itu terpisah dari pembatalan permintaan-klien biasa.

## Pembatalan adalah Race

Dua urutan event sama-sama valid.

### Pembatalan menang

```text
request starts
client sends cancellation signal
server marks request cancelled
worker reaches completion
server suppresses the response
```

### Completion menang

```text
request starts
worker commits the result
server sends the response
cancellation arrives late
server ignores the late notification
```

Klien juga harus mengabaikan respons terlambat untuk permintaan yang sudah ia tinggalkan. Latensi jaringan berarti tidak ada pihak yang bisa membuktikan event mana yang diamati pihak lain terlebih dahulu.

```figure
mcp-reliability-race
```

`RequestCoordinator` pelajaran menyimpan satu state terminal. `complete()` tidak mengembalikan respons setelah pembatalan. Pembatalan terlambat tidak bisa mengubah catatan yang sudah selesai.

## Timeout Butuh Dua Jam

Satu timer inaktivitas tidak cukup.

Gunakan dua batas:

1. **Idle timeout.** Berapa lama permintaan boleh tidak menghasilkan aktivitas yang berguna.
2. **Maximum timeout.** Anggaran wall-clock absolut dari awal permintaan.

Progres bisa mengatur ulang jam idle. Ia tidak boleh pernah menghapus deadline maksimum.

```text
start: 0 ms
progress: 400 ms
progress: 800 ms
progress: 1200 ms
idle timeout: 500 ms
maximum timeout: 2000 ms
```

Pada 1500 ms, permintaan masih aktif karena progres terbaru hanya berumur 300 ms. Pada 2000 ms, deadline maksimum membatalkannya bahkan jika event progres lain tiba pada 1999 ms.

Progres bersifat opsional. Sebuah server bisa menerima token progres dan tidak memancarkan pembaruan apa pun. Jangan pernah mengubah keberadaan token menjadi timeout tak berujung.

Nilai progres MCP harus meningkat. Notifikasi berhenti setelah completion atau pembatalan. Rate-limit progres agar worker yang cepat tidak bisa membanjiri transport.

## Pembatalan Permintaan Bukan `tasks/cancel`

Mekanisme-mekanisme ini memecahkan siklus hidup yang berbeda.

| Mekanisme | Target | Sinyal | Apa artinya berhasil |
|-----------|--------|--------|--------------------|
| Pembatalan permintaan di stdio | Satu RPC yang sedang berjalan | `notifications/cancelled` | Klien meninggalkan permintaan; server seharusnya berhenti jika praktis |
| Pembatalan permintaan di HTTP | Satu stream respons yang sedang berjalan | Tutup stream | Klien meninggalkan permintaan; server seharusnya berhenti jika praktis |
| `tasks/cancel` | Satu Task yang tahan lama | Permintaan MCP biasa | Server mengakui niat pembatalan |

Hasil `tasks/cancel` yang sukses tidak membuktikan worker berhenti. Task bisa tetap `working` sampai checkpoint worker mengamati flag-nya. Kerja bisa selesai sebelum checkpoint itu.

Jangan menghapus state task yang tahan lama ketika koneksi HTTP ditutup. Alasan membuat Task adalah karena siklus hidupnya melampaui satu permintaan dan satu koneksi.

## ID JSON-RPC Baru Bukan Idempotensi

ID JSON-RPC mengorelasikan permintaan dan respons. Mereka tidak mengidentifikasi operasi bisnis.

Misalkan klien mengirimkan charge dengan id `41`, kehilangan respons, dan mencoba ulang dengan id `42`. Server melihat dua pesan berbeda. Tanpa kunci aplikasi, ia tidak bisa tahu bahwa mereka mewakili satu checkout.

Kunci idempotensi mengidentifikasi niat bisnis:

```json
{
  "name": "charge_account",
  "arguments": {
    "account": "acct-7",
    "cents": 1200,
    "idempotencyKey": "checkout-7"
  }
}
```

Server menyimpan:

- kunci itu;
- fingerprint dari argumen operasi;
- hasil yang di-commit.

Kunci yang sama dan argumen yang sama mengembalikan hasil yang tersimpan. Kunci yang sama dengan argumen berbeda ditolak. Ini mencegah penggunaan ulang kunci yang tidak sengaja memutasi operasi bisnis yang berbeda.

### Batas ledger harus atomik dan tahan lama

Urutan ini tidak aman:

```text
check key
run mutation
store result
```

Dua worker bisa keduanya mengamati kunci yang hilang dan keduanya menjalankan mutasi. Crash setelah efek tetapi sebelum penyimpanan menciptakan ambiguitas yang sama saat retry.

Pelajaran memakai ledger SQLite berbasis-file. `BEGIN IMMEDIATE` menserialisasi pemeriksaan kunci, efek bisnis yang disimulasikan, penghitung eksekusi, dan hasil tersimpan menjadi satu transaksi. Dua koneksi ledger independen yang balapan dengan kunci yang sama karena itu mengamati satu hasil yang di-commit dan satu eksekusi. Menutup dan membuka ulang ledger mempertahankan catatan itu.

Setiap nilai pengembalian direkonstruksi dari JSON tersimpan. Pemanggil tidak pernah menerima objek yang dapat dimutasi yang dipegang ledger, sehingga mengubah dictionary yang dikembalikan tidak bisa merusak hasil replay berikutnya.

Efek bisnis simulator adalah tanda terima dan penghitung eksekusi di dalam transaksi SQLite yang sama. Pembayaran, deployment, atau panggilan API eksternal yang nyata tidak dibuat atomik hanya dengan menulis tabel lokal. Produksi membutuhkan transaksi database bersama yang tahan lama, transactional outbox, atau penyedia upstream yang menegakkan kunci idempotensi yang sama. Kunci proses saja tidak melindungi banyak replica atau bertahan dari restart.

### Matriks retry

Klasifikasikan retry sebelum mengimplementasikannya.

| Kelas | Contoh | Aturan retry |
|------|---------|------------|
| Aman | Pembacaan deterministik tanpa side effect | Coba ulang dengan id JSON-RPC baru setelah batas kegagalan dipahami |
| Kondisional | Mutasi dengan kunci idempotensi yang tahan lama | Coba ulang dengan kunci yang sama dan argumen identik |
| Tidak aman | Mutasi tanpa deduplikasi bisnis | Jangan coba otomatis; rekonsiliasi terlebih dahulu |

Anotasi tool seperti `readOnlyHint` dan `idempotentHint` tetap petunjuk yang tidak tepercaya. Kontrak aplikasi dan implementasi server yang memutus keamanan retry.

## Backpressure adalah Bagian dari Keakuratan

Sebuah produser SSE bisa menghasilkan progres lebih cepat daripada klien, proxy, atau jaringan bisa mengonsumsinya. Antrean tanpa batas mengubah kelambatan menjadi kehabisan memori.

Gunakan antrean berbatas dan definisikan apa yang bisa hilang.

Progres dapat diganti. Nilai progres yang lebih baru menggantikan yang lebih lama untuk token yang sama. Respons JSON-RPC akhir tidak dapat diganti.

Buffer pelajaran menerapkan kebijakan ini:

1. Gabungkan progres bersebelahan untuk token yang sama.
2. Jatuhkan progres tertua ketika kapasitas tercapai.
3. Tandai stream sebagai membutuhkan refetch otoritatif.
4. Pertahankan respons akhir.
5. Tolak keadaan di mana mempertahankan respons akhir akan membutuhkan menjatuhkan respons akhir lainnya.

Ini adalah kehilangan berbatas dengan pemulihan eksplisit. Kehilangan diam-diam bukan strategi.

### Buffering proxy

Sebuah server bisa streaming dengan benar sementara reverse proxy menahan event dalam buffer.

Untuk respons SSE, kirim:

```http
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

Spesifikasi Streamable HTTP 2026 merekomendasikan `X-Accel-Buffering: no` sehingga proxy yang kompatibel mengantar event segera.

Untuk stream lama yang tenang, pancarkan secara berkala komentar SSE:

```text
:
```

Klien mengabaikan baris komentar. Perantara melihat lalu lintas dan lebih kecil kemungkinannya menutup koneksi yang menganggur.

Keepalive bukan progres. Jangan mengatur ulang timeout idle semantik operasi hanya karena komentar transport tiba.

## Reconnect Berarti Refetch

Streamable HTTP modern tidak mendukung SSE yang dapat dilanjutkan melalui `Last-Event-ID`.

Setelah stream `subscriptions/listen` putus:

1. Buka permintaan listen baru dengan id JSON-RPC baru.
2. Pulihkan filter langganan yang diinginkan.
3. Refetch tool, resource, prompt, atau Task yang terpengaruh melalui metode otoritatif.
4. Deduplikasikan state aplikasi berdasarkan identifier yang stabil.
5. Jangan replay mutasi yang tidak aman hanya karena responsnya hilang.

Rencana pemulihan contoh secara eksplisit menyetel `sendLastEventId` ke false dan mendaftar resource untuk di-refetch.

### Cegah kawanan reconnect

Jika 10.000 klien reconnect tepat pada satu detik, server yang pulih gagal lagi.

Gunakan exponential backoff dengan jitter dan cap. Pelajaran menghitung jitter deterministik dari id klien dan nomor percobaan sehingga test tetap dapat direproduksi:

```text
attempt 0: up to 250 ms
attempt 1: up to 500 ms
attempt 2: up to 1000 ms
...
cap: 8000 ms
```

Produksi bisa memakai randomness aman-kriptografi atau runtime. Invariannya adalah distribusi, bukan formula tertentu.

## Bangun Sendiri

`code/main.py` membangun lima komponen keandalan kecil.

### `RequestCoordinator`

- memulai permintaan yang sedang berjalan dengan deadline idle dan maksimum;
- memancarkan notifikasi progres monotonik;
- menghasilkan sinyal pembatalan stdio atau HTTP yang benar;
- mengabaikan notifikasi pembatalan yang tidak valid;
- membuat race terminal pembatalan dan completion eksplisit;
- mencadangkan pembatalan yang dikirim server untuk langganan stdio.

### `MutationLedger`

- membuktikan bahwa dua id JSON-RPC mengeksekusi dua kali tanpa kunci bisnis;
- memakai transaksi SQLite berbasis-file untuk pemeriksaan kunci, efek yang disimulasikan, penghitung eksekusi, dan commit hasil;
- mendeduplikasikan argumen yang cocok di bawah satu kunci idempotensi di seluruh koneksi ledger independen;
- menolak satu kunci yang digunakan ulang dengan argumen berbeda;
- mengembalikan salinan defensif dan mempertahankan catatan yang di-commit di seluruh reopen.

### `DurableTaskService`

- mengakui permintaan pembatalan;
- menjaga task tetap `working` sampai checkpoint worker;
- mendemonstrasikan mengapa pengakuan bukan status akhir.

### `BoundedSseBuffer`

- menggabungkan atau menjatuhkan progres di bawah tekanan;
- mencatat bahwa refetch otoritatif diperlukan;
- tidak pernah menjatuhkan respons akhir.

### Helper pemulihan

- mengembalikan header SSE aman-proxy dan komentar keepalive;
- membuat rencana reconnect dan refetch;
- menyebarkan retry dengan exponential backoff dan jitter yang deterministik.

## Gunakan

Dari root repositori:

```bash
cd phases/13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/code
python3 main.py
python3 -m unittest discover tests -v
```

Demo menjalankan kedua sisi race sentral, mengeksekusi mutasi yang dideduplikasikan secara transaksional di ledger berbasis-file sementara, membebani buffer progres berbatas, dan menunjukkan Task tahan lama bergerak dari pembatalan yang diakui ke pembatalan yang diamati worker.

## Interactive Lab

Jalankan empat pengurutan event tanpa menambahkan sleep.

1. Mulai permintaan `A`, batalkan, lalu panggil `complete()`.
2. Mulai permintaan `B`, selesaikan, lalu kirimkan pembatalan.
3. Mulai permintaan `C`, pancarkan progres sebelum setiap deadline idle, lalu lampaui deadline maksimum.
4. Mulai permintaan `D` melalui Streamable HTTP dan tutup stream responsnya.

Catat untuk setiap skenario:

- state permintaan terminal;
- apakah respons akhir ada;
- sinyal pembatalan yang diletakkan di wire;
- event mana yang harus diabaikan klien.

Lalu ubah `D` ke stdio. Operasinya identik, tetapi sinyal pembatalan harus berubah.

## Practice Lab

Tambahkan mutasi `reserve_inventory` ke `MutationLedger`.

Persyaratan:

1. Kunci mengikat SKU, quantity, tenant, dan nama operasi.
2. Retry dengan kunci yang sama dan argumen yang sama mengembalikan reservasi pertama.
3. Retry dengan quantity yang diubah gagal tanpa reservasi lain.
4. Sebuah eksekusi yang sudah commit tetapi kehilangan responsnya dapat direkonsiliasi berdasarkan kunci.
5. Hasil tidak mencatat data rahasia atau pembayaran apa pun.
6. Retry otomatis dinonaktifkan ketika klien tidak menyediakan kunci.
7. Tambahkan drop langganan yang disimulasikan dan refetch catatan inventaris sebelum memutuskan apa yang harus dilakukan selanjutnya.
8. Mulai dua koneksi ledger di barrier dan kirimkan kunci yang sama secara bersamaan. Assert satu reservasi sudah di-commit.
9. Mutasi objek reservasi pertama yang dikembalikan. Replay kuncinya dan buktikan hasil tersimpan tidak berubah.
10. Tutup dan buka ulang file ledger, lalu rekonsiliasi reservasi berdasarkan kunci.

Jaga lab tetap jujur: jika inventaris berada di layanan lain, jelaskan apakah layanan itu menerima kunci idempotensi yang sama atau apakah transactional outbox menjembatani commit lokal ke efek remote.

## Shipped Artifact

`outputs/skill-mcp-reliability-reviewer.md` adalah skill review keandalan datar. Berikan padanya operasi MCP, transport, kebijakan timeout, perilaku retry, kebijakan antrean, dan rencana pemulihan. Ia mengembalikan tabel race, klasifikasi retry, batas idempotensi, pemeriksaan kontrol-aliran, dan fixture kegagalan.

## Verify It

Pelajaran ini selesai ketika pernyataan berikut benar:

- Pembatalan stdio mengirim `notifications/cancelled` dan tidak menerima respons apa pun.
- Pembatalan Streamable HTTP menutup stream permintaan dan tidak mengirim POST pembatalan apa pun.
- Cancel-before-complete menekan respons akhir.
- Complete-before-cancel mempertahankan respons dan mengabaikan pembatalan terlambat.
- Progres bisa mengatur ulang timeout idle tetapi tidak pernah timeout maksimum.
- Id JSON-RPC baru saja mengeksekusi mutasi lagi.
- Satu kunci idempotensi dan argumen identik mengeksekusi satu kali di bawah balapan dua-koneksi bersamaan.
- Catatan yang di-commit bertahan dari reopen dan replay mengembalikan salinan defensif.
- Memutasi satu hasil yang dikembalikan tidak bisa mengubah hasil tersimpan.
- Buffer berbatas tetap di dalam kapasitas dan mempertahankan respons akhir.
- Reconnect memakai permintaan baru, tidak mengirim `Last-Event-ID`, dan me-refetch state yang terpengaruh.
- Pengakuan `tasks/cancel` meninggalkan task non-terminal sampai worker mengamatinya.

## Production Failure Modes

| Kegagalan | Gejala yang dapat diamati | Respons yang benar |
|---------|--------------------|------------------|
| Klien HTTP POST notifikasi pembatalan | Server dan klien tidak setuju tentang umur permintaan | Tutup stream respons SSE permintaan |
| Server merespons setelah pembatalan diterima | Klien menerima hasil terlambat yang tidak dapat digunakan | Hentikan kerja dan tekan lebih banyak pesan ketika pembatalan menang |
| Progres mengatur ulang setiap deadline | Kerja yang macet bertahan selamanya | Pertahankan timeout maksimum absolut terpisah |
| Id RPC baru diperlakukan sebagai deduplikasi | Charge, deployment, atau penghapusan berjalan dua kali | Tambahkan kunci idempotensi aplikasi yang tahan lama |
| Pemeriksaan kunci dan efek terpisah | Worker bersamaan keduanya mengamati kunci yang hilang | Commit klaim kunci, catatan efek, dan hasil secara atomik |
| Ledger in-memory dipakai di seluruh replica | Restart atau worker lain melupakan commit sebelumnya | Gunakan penyimpanan tahan lama bersama atau idempotensi upstream |
| Hasil mutable tersimpan dikembalikan langsung | Mutasi pemanggil merusak replay berikutnya | Serialisasi hasil yang di-commit dan kembalikan salinan defensif |
| Kunci digunakan ulang dengan argumen yang diubah | Satu kunci meng-alias dua niat bisnis | Simpan dan bandingkan fingerprint argumen |
| Antrean progres tanpa batas | Memori naik dengan consumer yang lambat | Gabungkan dan jatuhkan progres yang dapat diganti di dalam batas |
| Respons akhir dijatuhkan di bawah tekanan | Klien tidak bisa tahu hasil permintaan | Cadangkan kapasitas atau usir progres, tidak pernah respons akhir |
| Proxy menbuffer SSE | Progres tiba dalam burst atau setelah timeout | Nonaktifkan buffering dan konfigurasikan timeout proxy yang kompatibel |
| `Last-Event-ID` diasumsikan | Klien melanjutkan dari state yang tidak didukung server | Reconnect dengan permintaan baru dan refetch |
| Setiap klien reconnect segera | Pemulihan menciptakan outage lain | Gunakan exponential backoff ber-cap dengan jitter |
| Task ack diperlakukan sebagai pembatalan akhir | Worker terus berjalan setelah UI bilang berhenti | Poll Task sampai status terminal |

## Capstone Connection

Capstone ekosistem-tool seharusnya memperlakukan keandalan sebagai bukti yang dapat dieksekusi, bukan paragraf dalam diagram arsitektur.

Minta artifact-artifact ini:

- satu transkrip race pembatalan untuk setiap transport;
- tabel retry untuk setiap mutasi yang diekspos;
- catatan kunci idempotensi dan fixture ketidakcocokan;
- transkrip kunci-sama bersamaan, pemeriksaan reopen, dan pemeriksaan alias-mutasi;
- hasil overload buffer-berbatas;
- header SSE reverse-proxy dan kebijakan idle;
- rencana reconnect yang menyebut metode refetch otoritatif;
- trace pembatalan Task yang tahan lama ketika capstone memakai Task.

Permintaan hijau dalam proses lokal hanya membuktikan jalur bahagia. Capstone siap-produksi ketika respons yang hilang, pembatalan terlambat, consumer yang lambat, dan kawanan reconnect memiliki hasil yang deterministik.

## Istilah Kunci

| Istilah | Arti |
|------|---------|
| Pembatalan permintaan | Pengabaian satu permintaan MCP yang sedang berjalan |
| Race pembatalan | Kompetisi di antara event completion dan pembatalan terminal |
| Idle timeout | Batas sejak aktivitas permintaan terakhir yang berguna |
| Maximum timeout | Batas absolut dari awal permintaan, tidak dipengaruhi progres |
| Kunci idempotensi | Identifier aplikasi yang mendeduplikasikan satu niat bisnis |
| Ledger atomik | Batas tahan lama yang meng-commit klaim kunci, catatan efek, dan hasil sebagai satu unit |
| Backpressure | Kontrol yang diterapkan ketika produser melampaui consumer |
| Penggabungan progres | Mengganti progres lebih lama dengan nilai otoritatif yang lebih baru |
| Refetch | Membaca state saat ini lagi setelah gap stream |
| Jitter | Variasi disengata yang menyebarkan retry di seluruh waktu |

## Bacaan Lanjutan

- [MCP Cancellation](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/cancellation)
- [MCP Progress](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/progress)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Tasks Extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
