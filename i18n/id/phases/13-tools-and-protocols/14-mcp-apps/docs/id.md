# MCP Apps di Protokol Tanpa-State

> Sebuah hasil interaktif tetap merupakan pertukaran tool dan resource MCP. Inti 2026-07-28 membuat pertukaran itu self-contained, sementara ekstensi Apps menambahkan permukaan browser yang ter-sandbox.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Iklankan MCP Apps melalui `server/discover` dan kapabilitas ekstensi per-request.
- Deklarasikan resource `ui://` pada sebuah tool sebelum tool dipanggil.
- Kembalikan hasil tool dan resource yang lengkap pada wire tanpa-state 2026-07-28.
- Pisahkan pesan bridge `ui/initialize` Apps dari handshake inti MCP yang dihapus.
- Terapkan validasi origin, sandboxing, CSP, dan izin least-privilege.

## Masalahnya

Sebuah hasil teks bisa menjelaskan sebuah timeline. Ia tidak bisa memberi pengguna timeline yang bisa mereka saring, periksa, atau tindak lanjuti.

MCP Apps memecahkan masalah presentasi dengan ekstensi opsional. Sebuah definisi tool menunjuk ke resource `ui://`. Host bisa mengambil dan meninjau resource itu sebelum tool berjalan, merendernya di iframe yang ter-sandbox, dan memediasi semua aksi aplikasi melalui bridge JSON-RPC.

Protokol inti berubah di 2026-07-28. Jangan membungkus sebuah App dalam siklus hidup koneksi lama:

- Tidak ada request inti `initialize` maupun notifikasi `notifications/initialized`.
- Tidak ada header `Mcp-Session-Id`.
- Setiap request membawa versi protokol dan kapabilitas klien di `params._meta`.
- Sebuah server mengimplementasikan `server/discover` sehingga klien bisa memeriksa versi, kapabilitas inti, dan ekstensi.
- Setiap hasil sukses memiliki diskriminator `resultType`.
- Streamable HTTP memakai satu POST per request. Entrypoint GET dan DELETE modern mengembalikan 405.

Bridge Apps tetap memiliki method bernama `ui/initialize`. Itu milik dialek postMessage iframe. Ia tidak menciptakan kembali sesi MCP inti.

## Konsepnya

### Dua protokol, satu fitur

Pertahankan lapisannya tetap eksplisit:

1. Inti MCP membawa `server/discover`, `tools/list`, `tools/call`, `resources/list`, dan `resources/read`.
2. Ekstensi MCP Apps mendeklarasikan UI dan mendefinisikan bridge iframe-ke-host.
3. Aturan sandbox browser membatasi apa yang bisa dijangkau UI.

Pengenal ekstensinya adalah `io.modelcontextprotocol/ui`. Kedua peer memilih ikut. Sebuah klien mengirim dukungan ekstensi di dalam objek capabilities pada setiap request:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "server/discover",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/ui": {}
        }
      },
      "io.modelcontextprotocol/clientInfo": {
        "name": "timeline-host",
        "version": "1.0.0"
      }
    }
  }
}
```

`clientInfo` disarankan untuk diagnostik. Itu data yang dilaporkan sendiri, bukan identitas otorisasi.

### Discovery sebelum render

Hasil discovery server mengiklankan ekstensinya:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {},
    "resources": {},
    "extensions": {
      "io.modelcontextprotocol/ui": {}
    }
  },
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "timeline-app-server",
      "version": "2.0.0"
    }
  }
}
```

Server harus mendukung discovery. Sebuah klien tidak dipaksa memanggil discovery sebelum setiap aksi karena setiap aksi membawa kapabilitasnya sendiri.

### Deklarasikan UI di definisi tool

Kontrak Apps modern mengikat UI ke tool di `tools/list`:

```json
{
  "name": "notes_timeline",
  "description": "Render a timeline of notes.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  },
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline.html"
    }
  }
}
```

Ini secara sengaja merupakan metadata pra-panggilan. Host bisa memuat lebih dulu, meng-cache, dan meninjau keamanan HTML sebelum sebuah hasil memintanya untuk menampilkan. Kunci metadata datar yang lebih lama mungkin diterima oleh kode kompatibilitas, tetapi server baru seharusnya memancarkan bentuk bersarang `_meta.ui.resourceUri`.

`tools/list` bisa di-cache di inti saat ini. Sertakan pengurutan deterministik, `ttlMs`, dan `cacheScope`. Gunakan `private` ketika tool yang terlihat bervariasi per pengguna atau token.

### Kembalikan data, lalu biarkan host mengikat tampilannya

Panggilan tool mengembalikan konten biasa plus data terstruktur:

```json
{
  "resultType": "complete",
  "content": [
    {"type": "text", "text": "Timeline ready."}
  ],
  "structuredContent": {
    "notes": [
      {"id": "note-1", "title": "Discover", "created": "2026-07-28"}
    ]
  },
  "isError": false
}
```

Host sudah tahu tampilan mana yang milik tool. Hindari menciptakan blok konten baru hanya untuk mengulang URI itu.

### Sajikan aplikasi sebagai resource

Server mengiklankan `resources` di discovery, sehingga ia juga mengimplementasikan operasi `resources/list` yang wajib. Entri daftar deterministik-nya mencakup URI kanonis, nama yang stabil, deskripsi, dan MIME type. Hasil daftar menyertakan `resultType`, metadata identitas server, `ttlMs`, dan `cacheScope`, persis seperti daftar tool yang deterministik.

Host mengirim `resources/read`. Pada Streamable HTTP, request-nya memiliki:

```text
POST /mcp
MCP-Protocol-Version: 2026-07-28
Mcp-Method: resources/read
Mcp-Name: ui://notes/timeline.html
```

Nilai header dan body JSON-RPC harus cocok. Ketidakcocokan adalah error protokol `-32020`.

Hasilnya berisi resource HTML dan petunjuk cache:

```json
{
  "resultType": "complete",
  "contents": [
    {
      "uri": "ui://notes/timeline.html",
      "mimeType": "text/html;profile=mcp-app",
      "text": "<!doctype html>...",
      "_meta": {
        "ui": {
          "csp": {
            "connectDomains": [],
            "resourceDomains": [],
            "frameDomains": [],
            "baseUriDomains": []
          },
          "permissions": {}
        }
      }
    }
  ],
  "ttlMs": 60000,
  "cacheScope": "public"
}
```

### Cache resource UI sebagai konten yang bisa dieksekusi

Sebuah resource App bukan sesuatu yang bisa dipertukarkan dengan prosa biasa. Entry cache-nya bisa mengeksekusi kode bridge, merender data tool, dan meminta aksi yang dimediasi host. Kuncinya dengan URI `ui://` kanonis, identitas dan versi server yang diakui, digest konten resource, dan konteks otorisasi ketika `cacheScope` bersifat privat. Jangan pernah memakai ulang resource App privat antar principal karena HTML atau metadata kebijakannya bisa berbeda meski URI-nya identik.

Invalidate entry ketika `ttlMs`-nya kedaluwarsa, pengikatan `_meta.ui.resourceUri` tool berubah, versi server atau pin deskriptor yang diakui berubah, atau sebuah subscription perubahan resource yang diakui menamai URI itu. Ambil ulang dan terapkan ulang tinjauan CSP dan izin sebelum memasang kembali. Sebuah iframe yang basi tidak boleh mempertahankan izin yang lebih luas hanya karena versi resource yang baru belum dimuat.

### Tolak ambiguitas wire sebelum kebijakan fitur

Validasi memiliki urutan yang disengaja. Terlebih dulu validasi bentuk JSON-RPC dan minta metadata protokol berbentuk string serta peta kapabilitas klien berbentuk objek. Selanjutnya bandingkan header routing dengan body. Baru setelah itu putuskan apakah versi protokol yang cocok didukung. Urutan ini mencegah proxy dan server menafsirkan request yang berbeda.

| Condition | HTTP | JSON-RPC error |
|-----------|------|----------------|
| Header and body version, method, or name disagree | 400 | `-32020` |
| Header and body agree on an unsupported version | 400 | `-32022`, with `data` exactly `{"supported":["2026-07-28"],"requested":"<actual>"}` |
| `resources/read` lacks the Apps extension capability | 400 | `-32021`, with `data.requiredCapabilities.extensions.io.modelcontextprotocol/ui` |
| Method is unknown | 404 | `-32601` |

Sebuah notifikasi JSON-RPC tidak memiliki `id`, sehingga server tidak pernah mengeluarkan respons JSON-RPC untuknya. Sebuah notifikasi HTTP yang diterima mengembalikan 202 dengan body kosong. Sebuah error bisa mengubah status HTTP, tetapi tetap tidak bisa membuat body error JSON-RPC untuk sebuah notifikasi.

### Sandbox adalah batas, bukan vonis kepercayaan

Sebuah host mengendalikan iframe. App tidak bisa langsung membaca cookie, local storage, atau DOM halaman host. Semua pekerjaan berivilej harus melewati bridge.

Gunakan default berikut:

- Biarkan semua daftar domain CSP kosong, lalu tambahkan hanya origin yang dibutuhkan App. Gunakan `connectDomains` untuk fetch, XHR, dan WebSocket; gunakan `resourceDomains` untuk skrip, gaya, gambar, dan font.
- Bundle kode dan data bila memungkinkan.
- Jangan minta izin kamera, mikrofon, atau lokasi kecuali sebuah fitur yang terlihat membutuhkannya.
- Pin `postMessage` ke origin peer yang tepat dan tolak event dari origin lain mana pun.
- Perlakukan argumen tool, hasil tool, teks resource, dan pesan bridge sebagai input yang tidak tepercaya.
- Pertahankan persetujuan pengguna di host. iframe tidak bisa menyetujui aksi konsekuensialnya sendiri.

Jangan menyalin atribut `sandbox` yang sudah diatur dari sebuah tutorial ke setiap host. Host harus memilih flag berdasarkan model origin App dan desain isolasinya sendiri.

Sebuah domain yang diizinkan tetap merupakan jalur ekstrfiltrasi. `connectDomains: ["https://api.example.com"]` berarti skrip mana pun yang dieksekusi di dalam App bisa mengirim data yang diizinkan ke sana. Pencocokan origin yang tepat mencegah kebingungan tujuan, tetapi tidak memutuskan apakah payload itu pantas. Biarkan akses connect kosong secara default, hindari menaruh bearer token di dalam iframe, proxy operasi sempit melalui host bila memungkinkan, batasi ukuran respons dan request, dan audit aksi pengguna mana yang memicu setiap request keluar. Perlakukan `resourceDomains` terpisah dari `connectDomains`; izin memuat font atau skrip sehariknya tidak memberikan izin unggah data sembarangan.

### Bridge Apps memiliki siklus hidupnya sendiri

Bridge Apps adalah dialek JSON-RPC di atas `postMessage`. Ia bisa menukar `ui/initialize` dan notifikasi `ui/*` dan bisa mem-proxy method yang tampak-inti seperti `tools/call`.

View mengirim `ui/initialize` dengan `appInfo` dan objek `appCapabilities`. Host mengembalikan kapabilitas dan konteks host-nya. Baru setelah respons itu View mengirim `ui/notifications/initialized`. Host harus menunggu notifikasi Apps ini sebelum mengirim pesan ke View.

Handshake lokal itu menciptakan bridge di antara satu iframe dan satu frame host. Ia tidak menegosiasikan versi protokol MCP, tidak membuat state server, dan tidak membuat sesi transport. Perhatikan prefiks yang tepat: `notifications/initialized` inti telah dihapus, sementara `ui/notifications/initialized` Apps tetap ada. Sebuah request inti yang dihasilkan oleh panggilan tool yang di-bridge adalah request self-contained baru dengan id JSON-RPC baru dan metadata request yang lengkap.

### Konteks host, aksi, dan pencabutan

Host tetap menjadi otoritas setelah inisialisasi bridge. Sebuah View boleh meminta aksi tool, navigasi, penggunaan clipboard, atau efek berivilej lain hanya melalui kapabilitas yang diiklankan host. Host memvalidasi request bertipe, pengguna saat ini, target, dan argumen, menerapkan kebijakan persetujuan, dan boleh menolaknya. Klik tombol dan pesan bridge yang valid mengekspresikan niat; keduanya tidak memberikan otoritas.

Perlakukan tema, ukuran, dan aksesibilitas sebagai konteks host yang berubah, bukan input render satu-kali:

- Terapkan token warna dan tipografi yang disediakan host, lalu bereaksi ketika tema atau preferensi kontras berubah.
- Biarkan View melaporkan dimensi yang diinginkan, tetapi biarkan host membatasi dan menerapkan ukuran iframe agar konten tidak bisa keluar dari layout-nya atau membuat overlay yang menyesatkan.
- Pertahankan urutan keyboard, fokus yang terlihat, nama yang dapat diakses, status screen-reader, kontras yang cukup, zoom, dan perilaku reduced-motion di dalam iframe.
- Uji ulang transfer fokus di antara kontrol host dan kontrol View setelah resize dan render ulang.

Kapabilitas bisa dicabut selama App terbuka karena pengguna mengganti akun, kebijakan berubah, server dikarantina, atau host menyempitkan persetujuan. Periksa kapabilitas dan otorisasi pada saat aksi, bukan hanya selama `ui/initialize`. Saat pencabutan, tolak panggilan berivilej yang tertunda, hentikan aktivitas jaringan yang tidak lagi sesuai kebijakan, bersihkan state render yang sensitif, dan pasang kembali atau jatuhkan ke teks ketika resource UI itu sendiri tidak lagi diakui. Sebuah View harus menangani penolakan sebagai hasil biasa, bukan mencoba ulang sampai host menyerah.

### Fallback adalah bagian dari kontrak

Sebuah server yang sadar-Apps tetap bisa melayani host yang tidak mengiklankan ekstensi UI:

- Kembalikan tool yang sama tanpa `_meta.ui` di `tools/list`.
- Pertahankan hasil teks yang berguna untuk `tools/call`.
- Tolak `resources/read` untuk UI dengan error kapabilitas yang hilang.
- Jangan pernah berasumsi sebuah iframe ada ketika memutuskan apakah tool telah selesai.

```figure
t3-ui-sandbox
```

## Bangun Sendiri

`code/main.py` membangun model protokol in-process kecil tanpa SDK. Ia memvalidasi envelope request saat ini dan nilai routing Streamable HTTP, mengiklankan Apps melalui `server/discover`, mendaftar tools dan resources, mengeksekusi tool, dan menyajikan resource HTML yang self-contained.

Model menerima body dan header routing yang sudah di-parse. Ia bukan adaptor HTTP yang lengkap dan tidak mem-parse `Content-Type` atau `Accept`. Gunakan Lesson 09 untuk adaptor Streamable HTTP lengkap yang mewajibkan `Content-Type: application/json` dan nilai `Accept` yang berisi `application/json` dan `text/event-stream`.

Jalankan:

```bash
cd phases/13-tools-and-protocols/14-mcp-apps
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Periksa empat hal di keluaran:

1. Setiap panggilan independen.
2. Setiap request memiliki `_meta` capabilities.
3. `resources/list` mengembalikan deskriptor stabil sebelum pembacaan resource mana pun.
4. Setiap hasil memiliki `resultType` dan metadata identitas server.
5. Tidak ada pengenal sesi inti yang muncul.

## Gunakan

Mulai dari `server/discover`. Konfirmasikan `io.modelcontextprotocol/ui` muncul di peta ekstensi server. Lalu panggil `tools/list` dua kali, sekali dengan kapabilitas Apps dan sekali tanpanya. Respons pertama mendeklarasikan resource-nya. Respons kedua tetap merupakan tool teks-saja yang bisa dipakai.

Baca `ui://notes/timeline.html`. Cari di HTML `hostOrigin` dan guard `event.origin`. Dua baris itu adalah bukti minimum yang terlihat bahwa bridge tidak memakai target wildcard.

## Rilis

Pelajaran ini mengirimkan `outputs/skill-mcp-apps-spec.md`. Gunakan itu untuk meninjau kontrak App sebelum menulis kode framework. Ia memaksa penulis menyatakan envelope inti saat ini, negosiasi ekstensi, fallback, resource UI, kebijakan cache, CSP, izin, method bridge, dan batas persetujuan.

## Latihan

1. Ubah kapabilitas klien menjadi peta ekstensi kosong. Konfirmasikan `tools/list` mempertahankan tool tetapi menghapus pengikatan UI-nya.
2. Kirim `Mcp-Name: ui://notes/other.html` dengan body yang membaca timeline-nya. Konfirmasikan error `-32020`.
3. Ubah resource menjadi `cacheScope: private`. Jelaskan kondisi spesifik-pengguna yang membenarkannya.
4. Pindahkan skrip ke `https://static.example.com/app.js`. Tambahkan origin itu ke `resourceDomains` dan jelaskan risiko rantai-pasok yang baru.
5. Tambahkan tool `notes_open` dan route klik tombol melalui host. Pertahankan persetujuan pengguna di host.

## Istilah Kunci

| Term | Meaning |
|------|---------|
| MCP Apps | Optional extension for interactive HTML rendered by an MCP host |
| `io.modelcontextprotocol/ui` | Extension identifier advertised by both peers |
| `ui://` | Resource scheme for an App's UI template |
| `text/html;profile=mcp-app` | MIME type for MCP App HTML |
| `server/discover` | Current RPC for protocol and capability discovery |
| `resources/list` | Mandatory resource listing method when the server advertises resources |
| `resultType` | Required discriminator for modern successful results |
| `ui/initialize` | First Apps bridge request, separate from removed core initialization |
| `ui/notifications/initialized` | Apps View readiness notification sent after the host responds |
| CSP | Browser policy that restricts scripts, styles, images, and network origins |
| Text fallback | Tool behavior retained for a host without Apps support |

## Bacaan Lanjutan

- [MCP 2026-07-28 base protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Apps overview](https://modelcontextprotocol.io/extensions/apps/overview)
- [MCP Apps build guide](https://modelcontextprotocol.io/extensions/apps/build)
- [Official extension support matrix](https://modelcontextprotocol.io/extensions/client-matrix)
