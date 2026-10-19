# MCP Transports: stdio dan Streamable HTTP Tanpa-State

> Transport membawa pesan MCP. Ia tidak menyediakan state protokol yang hilang. Pada `2026-07-28`, stdio lokal dan Streamable HTTP jarak jauh sama-sama membawa request yang mendeskripsikan dirinya sendiri.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 07 and 08
**Time:** ~65 minutes

## Tujuan Pembelajaran

- Pilih stdio untuk subproses lokal dan Streamable HTTP untuk layanan jaringan.
- Implementasikan kontrak Streamable HTTP modern satu-endpoint, POST-saja.
- Cocokkan dan validasi header versi, method, dan name MCP terhadap body JSON-RPC.
- Kirim SSE berlingkup request dan stream `subscriptions/listen` jangka panjang dengan benar.
- Migrasikan deployment berbasis sesi dan HTTP+SSE lama tanpa menyajikan perilaku lama sebagai modern.

## Masalahnya

Revisi Streamable HTTP lebih awal menggabungkan negosiasi protokol dengan perilaku koneksi dan sesi. Sebuah server bisa membuat `Mcp-Session-Id`, mengekspos stream GET mandiri, menerima DELETE untuk mengakhiri sesi, dan melanjutkan SSE dengan `Last-Event-ID`.

MCP `2026-07-28` menghapus mekanisme itu dari wire modern. Setiap request bisa mendarat di worker mana pun yang sehat karena versi protokol dan kapabilitas klien bepergian di dalam body request. Header HTTP mencerminkan field terpilih untuk routing dan kebijakan, tetapi server memvalidasi header itu terhadap body sebelum eksekusi.

Hasilnya lebih mudah diskalakan dan lebih mudah direasoningkan. Artinya juga server yang mengajarkan transport 2025 sebagai sesuatu yang berlaku saat ini sedang mengajarkan model kegagalan dan keamanan yang salah.

## Konsepnya

### stdio

Binding stdio untuk subproses yang diluncurkan klien:

- Klien menulis satu pesan JSON-RPC UTF-8 per baris ke stdin.
- Server menulis satu pesan JSON-RPC UTF-8 per baris ke stdout.
- Server menulis diagnostik ke stderr.
- Server segera keluar pada EOF stdin.
- Setiap request modern membawa versi dan kapabilitas klien di `params._meta`.

Proses bisa hidup untuk banyak panggilan, tetapi itu bukan sesi protokol modern. Jika ia keluar tak terduga, request yang sedang berjalan hilang. Restart proses, lakukan discovery ulang, list ulang, buka ulang subscription, dan ulangi operasi aman dengan id request baru.

### Streamable HTTP pada 2026-07-28

Sebuah server modern mengekspos satu endpoint MCP, misalnya `/mcp`, yang menerima POST.

Setiap request atau notifikasi JSON-RPC adalah HTTP POST baru. Body berisi satu pesan JSON-RPC. Klien tidak mengirim respons JSON-RPC ke server.

Untuk sebuah request, server mengembalikan salah satu:

- `Content-Type: application/json` dengan satu respons JSON-RPC; atau
- `Content-Type: text/event-stream` dengan notifikasi yang terkait dengan request itu, diikuti respons JSON-RPC final.

Untuk notifikasi yang diterima, server mengembalikan `202 Accepted` tanpa body.

Klien mengiklankan kedua tipe respons:

```http
Accept: application/json, text/event-stream
```

### POST-saja berarti POST-saja

Streamable HTTP modern tidak punya stream GET mandiri dan tidak punya endpoint sesi DELETE.

- `GET /mcp` mengembalikan `405 Method Not Allowed`.
- `DELETE /mcp` mengembalikan `405 Method Not Allowed`.
- `Mcp-Session-Id` diabaikan dan tidak pernah dibuat atau dipantulkan.
- `Last-Event-ID` diabaikan karena stream modern tidak bisa dilanjutkan.

Jika stream berlingkup request putus sebelum respons finalnya, klien telah kehilangan request yang sedang berjalan itu. Ia boleh mengirim request baru dengan id JSON-RPC baru saat retry aman. Ia tidak boleh mencoba melanjutkan stream.

### Validasi Origin

Server memvalidasi `Origin` pada koneksi masuk untuk mencegah DNS rebinding. Jika header ada dan tidak diizinkan secara eksplisit, kembalikan `403 Forbidden`. Klien non-browser boleh menghilangkan `Origin`, yang diizinkan oleh aturan transport resmi.

Server lokal harus mengikat ke `127.0.0.1`, bukan semua antarmuka. Layanan jaringan tetap memerlukan autentikasi dan otorisasi pada setiap request. Validasi Origin bukan autentikasi.

Gunakan pencocokan origin yang tepat setelah konfigurasi kanonis. Pemeriksaan prefix seperti `origin.startswith("https://trusted.example")` tidak aman karena bisa menerima akhiran yang dikendalikan penyerang.

### Header metadata HTTP yang wajib

Setiap request POST modern menyertakan:

```http
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes_search
```

Aturan header:

- `MCP-Protocol-Version` wajib dan harus sama dengan `params._meta.io.modelcontextprotocol/protocolVersion`.
- `Mcp-Method` wajib dan harus sama dengan `method` JSON-RPC.
- `Mcp-Name` wajib untuk `tools/call`, `resources/read`, dan `prompts/get`.
- `Mcp-Name` sama dengan `params.name`, atau `params.uri` untuk `resources/read`.
- Nilai header peka kapitalisasi walaupun nama header tidak.

Nilai `Mcp-Name` yang tidak aman atau non-ASCII memakai sentinel UTF-8 Base64 yang tepat:

```text
=?base64?{Base64EncodedValue}?=
```

Server mendekode nilai itu sebelum membandingkannya dengan body.

Header yang dicerminkan dan hilang, rusak format, atau tidak cocok mengembalikan HTTP `400` dengan kode JSON-RPC `-32020`. Jika header dan body sepakat pada versi yang tidak didukung server, kembalikan HTTP `400` dengan `-32022` dan data error yang tepat seperti `{"supported":["2026-07-28"],"requested":"2027-01-01"}`.

Method modern yang tidak diketahui mengembalikan HTTP `404` dengan JSON-RPC `-32601`. Body JSON-RPC penting karena klien dua-era memakainya untuk membedakan error modern dari endpoint era lama yang salah sasaran.

### SSE berlingkup request

Sebuah server boleh memilih SSE untuk satu request jangka panjang:

```text
POST tools/call id=41
  <- notifications/progress related to id=41
  <- notifications/progress related to id=41
  <- JSON-RPC response id=41
stream closes
```

Server tidak boleh mengirim request JSON-RPC independen pada stream ini. Sampling, elicitation, dan interaksi roots memakai hasil Multi Round-Trip Request. Menutup stream respons membatalkan request itu.

Jangan menambahkan id event SSE untuk replay. Kelanjutan `Last-Event-ID` bukan bagian dari revisi modern.

### Perubahan jangka panjang memakai subscriptions/listen

Notifikasi perubahan memakai request yang dibuka klien, bukan GET mandiri:

```json
{
  "jsonrpc": "2.0",
  "id": "listen-1",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true,
      "resourceSubscriptions": ["notes://note-1"]
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Respons POST adalah stream SSE jangka panjang. Pesan protokol pertamanya adalah `notifications/subscriptions/acknowledged`. Pengakuan itu, setiap notifikasi perubahan, dan hasil final membawa `io.modelcontextprotocol/subscriptionId` di `_meta`, sama dengan id request listen. Server boleh mengirim komentar SSE sebagai keepalive. Saat stream putus, klien mengirim ulang `subscriptions/listen` dengan id request baru dan mengambil ulang data yang terpengaruh.

`resources/subscribe` dan `resources/unsubscribe` milik era lama. Jangan menggunakannya pada koneksi modern.

### State aplikasi eksplisit

Menghapus sesi protokol tidak melarang alur kerja yang memiliki state. Server boleh membuat state handle opak dan mengembalikannya sebagai hasil tool biasa. Klien meneruskan handle itu sebagai argumen eksplisit pada panggilan berikutnya.

Ikat handle ke principal yang diautentikasi, buat agar tidak bisa ditebak, kedaluwarsakan, dan otorisasi setiap penggunaan. Ini membuat state terlihat di lapisan aplikasi alih-alih menyembunyikannya di affinity transport.

Kegagalan yang disebabkan oleh state replica tersembunyi bersifat mekanis:

1. Request A mencapai replica 1 dan membuat sebuah draf di memori proses itu.
2. Respons tidak mengembalikan handle draf karena implementasi mengasumsikan koneksi mengidentifikasi draf itu.
3. Request B adalah POST baru dan mencapai replica 2.
4. Replica 2 memiliki metadata protokol yang valid tetapi tidak punya cara untuk menamai atau memuat draf itu, sehingga alur kerja gagal atau membaca objek lokal yang salah.
5. Routing sticky tampak memperbaiki gejala sampai restart, rollout, reschedule, atau failover memindahkan request berikutnya.

Batas yang benar memiliki dua bagian. Konteks protokol tetap di setiap request. State aplikasi yang tahan lama hidup di shared store di bawah handle yang dibuat server dan dikembalikan ke klien. Panggilan berikutnya menyediakan handle itu, replica mana pun memuat record yang sama, dan otorisasi mengikat record ke principal dan tenant yang diautentikasi. Memori replica boleh meng-cache record, tetapi tidak boleh menjadi satu-satunya salinan yang diperlukan untuk kebenaran.

Pilih mekanisme state berdasarkan lifetime. Variabel berlingkup request bisa melayani satu panggilan. Lanjutan MRTR pendek bisa memakai `requestState` yang dilindungi integritas. Sebuah draf atau task tahan lama memerlukan handle eksplisit plus persistensi bersama, kedaluwarsa, kontrol konkurensi, dan idempotensi. Tidak ada objek itu yang merupakan sesi protokol MCP.

### Kompatibilitas HTTP dua-era

Klien yang mendukung server modern dan lama mencoba POST modern terlebih dulu. Jika menerima HTTP `400`, `404`, atau `405`, ia memeriksa body:

- Error JSON-RPC modern yang dikenali membuktikan server itu modern. Perbaiki request atau ulangi versi yang diiklankan. Jangan menurunkan versi.
- Body kosong atau respons yang tidak dikenali bisa menandakan server HTTP+SSE lama. Baru setelah itu coba endpoint GET lama dan harapkan event `endpoint` lama-nya.

Sebuah server bisa mendukung kedua era selama migrasi dengan me-routing metadata modern ke implementasi POST-saja modern dan mempertahankan endpoint lama terpisah untuk klien lama. Jangan pernah menyebut perilaku GET, DELETE, session id, atau replay lama sebagai bagian dari `2026-07-28`.

```figure
tp-transport-handshake
```

## Gunakan

`code/main.py` mengimplementasikan server Streamable HTTP modern yang terbatas dengan standard library Python. Ia memvalidasi Origin dan header yang dicerminkan, mengabaikan header sesi yang dihapus, mengembalikan JSON untuk panggilan biasa, dan mendemonstrasikan stream SSE `subscriptions/listen` yang terbatas.

```bash
cd code
python3 main.py --probe
python3 -m unittest discover tests -v
```

Probe memeriksa:

- Origin yang tidak valid ditolak;
- discovery berhasil tanpa session id;
- `Mcp-Session-Id` dan `Last-Event-ID` diabaikan;
- ketidakcocokan header mengembalikan `-32020`;
- versi yang tidak didukung mengembalikan `-32022` dengan data `supported` dan `requested` yang tepat;
- notifikasi tanpa id yang diterima mengembalikan HTTP `202` tanpa body;
- GET dan DELETE mengembalikan `405`;
- `subscriptions/listen` adalah stream respons POST yang pengakuannya, notifikasi, dan hasil finalnya membawa subscription id-nya.

## Rilis

Pelajaran ini mengirimkan `outputs/skill-mcp-transport-migrator.md`. Ia menghapus sesi protokol modern, menambahkan validasi header-body, menggantikan GET mandiri dengan `subscriptions/listen`, dan menjaga bridge lama apa pun tetap terpisah secara terlihat.

## Latihan

1. Hapus `Mcp-Method` dari sebuah POST. Konfirmasikan HTTP `400` dan error `-32020`.
2. Kirim header dan body dengan versi yang cocok `2027-01-01`. Konfirmasikan HTTP `400`, error `-32022`, dan data tepat `{"supported":["2026-07-28"],"requested":"2027-01-01"}`.
3. Kirim `Mcp-Name` sentinel Base64 untuk URI resource non-ASCII. Konfirmasikan nilai yang didekode dibandingkan dengan `params.uri`.
4. Putuskan stream listen yang terbatas sebelum respons finalnya. Kirim ulang dengan id JSON-RPC baru dan ambil ulang daftar tools.
5. Tambahkan handle alur kerja eksplisit ke tool ping. Ikat ke subjek otorisasi tanpa memakai affinity koneksi.

## Istilah Kunci

| Term | Meaning |
|------|---------|
| stdio | Newline-delimited JSON-RPC over a client-launched subprocess |
| Streamable HTTP | Single endpoint where each modern message is a new POST |
| Request-scoped SSE | POST response stream containing related notifications and final response |
| `subscriptions/listen` | Long-lived POST request for opted-in change notifications |
| Header mismatch | HTTP `400` and JSON-RPC `-32020` when mirrored headers disagree with body |
| Origin validation | DNS-rebinding defense for incoming connections, not authentication |
| Explicit state handle | Application token passed as an ordinary argument instead of hidden session state |
| Legacy bridge | Separate earlier-era behavior kept only for compatibility |

## Bacaan Lanjutan

- [MCP Transport Overview](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
