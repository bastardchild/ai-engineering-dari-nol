# Dasar-dasar MCP: Stateless Requests dan JSON-RPC

> MCP modern tidak memiliki handshake dan tidak memiliki sesi protokol. Setiap request harus membawa metadata yang cukup untuk dipahami, diotorisasi, dirutekan, dan dicoba ulang secara mandiri.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 01 through 05
**Time:** ~55 minutes

## Tujuan Pembelajaran

- Membedakan primitif server MCP dari fitur sisi klien-nya.
- Membangun request dan response JSON-RPC 2.0 yang valid untuk MCP `2026-07-28`.
- Melampirkan versi protokol, capabilities klien, dan identitas klien ke setiap request.
- Menggunakan `server/discover` dan menangani `UnsupportedProtocolVersionError` tanpa handshake.
- Menelusuri satu request mandiri dari validasi hingga hasil lengkap.

## Masalahnya

Sebuah server MCP dapat menerima dua request berurutan dari klien berbeda, dengan capabilities berbeda, pada proses atau worker HTTP yang sama. Jika server mengingat apa yang dideklarasikan request sebelumnya, ia dapat menerapkan izin yang salah atau mengembalikan bentuk wire yang salah.

MCP `2026-07-28` menghilangkan ambiguitas itu. Inti protokolnya stateless. Sebuah server harus memutuskan bagaimana menangani request saat ini dari request saat ini, bukan dari riwayat koneksi.

Ini mengubah model mental. Urutan lama adalah koneksi dulu, kedua handshake, ketiga operasi. Urutan modern lebih sederhana:

1. Klien mengirim request yang mendeskripsikan dirinya sendiri.
2. Server memvalidasi versi dan capabilities request itu.
3. Server menangani metode tersebut.
4. Server mengembalikan hasil bertipe atau error JSON-RPC.

Request berikutnya mengulang proses yang sama dari awal.

## Konsepnya

### Primitif server

Server MCP mengekspos tiga primitif utama:

1. **Tools** adalah tindakan yang dikendalikan model, ditemukan dengan `tools/list` dan dipanggil dengan `tools/call`.
2. **Resources** adalah data yang dialamatkan URI, ditemukan dengan `resources/list` dan diambil dengan `resources/read`.
3. **Prompts** adalah template yang dapat digunakan ulang, ditemukan dengan `prompts/list` dan dirender dengan `prompts/get`.

Roots, sampling, dan logging tetap ada di schema `2026-07-28` untuk kompatibilitas, tetapi sudah deprecated. Implementasi baru harus menggunakan input tool atau resource eksplisit untuk roots, API model-provider langsung untuk sampling, dan stderr atau OpenTelemetry untuk logging. Elicitation tetap tersedia melalui Multi Round-Trip Requests, di mana server mengembalikan input request dan klien mencoba ulang operasi aslinya. Server modern tidak pernah memulai request JSON-RPC yang mandiri.

### Envelope JSON-RPC

MCP menggunakan JSON-RPC 2.0:

- Request: `{jsonrpc, id, method, params}`
- Response: `{jsonrpc, id, result}` atau `{jsonrpc, id, error}`
- Notification: `{jsonrpc, method, params}` tanpa `id`

`id` request mengorelasikan satu response. Ia tidak membuat sesi protokol.

### Metadata request yang wajib

Setiap request modern membawa objek `_meta` di dalam `params`:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/list",
  "params": {
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

Versi protokol dan capabilities klien bersifat wajib. Identitas klien disarankan. Itu adalah data display dan debugging yang dilaporkan sendiri, bukan kredensial keamanan.

Server tidak boleh menyimpulkan nilai-nilai ini dari request sebelumnya, proses stdio, koneksi HTTP, atau header transport saja.

### Hasil lengkap dan identitas server

Setiap hasil modern yang sukses menyertakan `resultType`. Hasil akhir normal menggunakan `"complete"`. Server juga seharusnya mengidentifikasi dirinya di metadata hasil:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "resultType": "complete",
    "tools": [],
    "ttlMs": 30000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "notes-server",
        "version": "1.0.0"
      }
    }
  }
}
```

`tools/list`, `resources/list`, `prompts/list`, `resources/templates/list`, `resources/read`, dan `server/discover` adalah hasil yang dapat di-cache. Mereka menyertakan `ttlMs` dan `cacheScope`. Default yang aman adalah `ttlMs: 0` dan `cacheScope: "private"`. Item daftar seharusnya memiliki urutan deterministik sehingga response yang setara menghasilkan cache key yang stabil dan konteks model yang stabil.

### Discovery tanpa handshake

Setiap server modern harus mengimplementasikan `server/discover`. Klien dapat memanggilnya sebelum metode lain untuk mengambil:

- `supportedVersions`
- `capabilities` server
- `instructions` penggunaan opsional
- identitas server di `_meta` hasil
- petunjuk cache

Discovery berguna, tetapi bukan gerbang. Klien dapat mengirim `tools/list` lebih dulu karena request itu sudah membawa versi protokol dan capabilities-nya.

Jika versi yang diminta tidak didukung, server mengembalikan kode JSON-RPC `-32022` dengan:

```json
{
  "requested": "2027-01-01",
  "supported": ["2026-07-28"]
}
```

Klien memilih versi modern yang didukung bersama dan mencoba ulang dengan id request JSON-RPC baru.

### Satu siklus hidup request

Telusuri request modern dengan urutan ini:

1. Satu envelope JSON-RPC diurai.
2. Konfirmasikan `jsonrpc` adalah `"2.0"`, ada `id`, `method` adalah string, dan `params` adalah objek.
3. Wajibkan string versi dan objek capability di `params._meta`; metadata yang rusak atau hilang adalah `-32602`.
4. Di batas HTTP, bandingkan versi, method, dan header nama yang berlaku dengan body. Ketidakcocokan adalah `-32020` bahkan ketika salah satu dari dua nilai versi tidak didukung.
5. Setelah kesetaraan ditetapkan, tolak versi yang cocok tetapi tidak didukung dengan `-32022`.
6. Periksa capabilities yang wajib, lalu rutekan berdasarkan `method` dan validasi argumen spesifik-method.
7. Autentikasi dan otorisasi operasi konkret sebelum handler-nya berjalan.
8. Kembalikan hasil lengkap dengan identitas server.
9. Lupakan metadata protokol yang berlaku-request.

Urutan itu mencegah dua komponen menafsirkan panggilan yang berbeda. Sebuah gateway tidak boleh mengotorisasi `Mcp-Name: notes.read` sementara origin mengeksekusi `params.name: notes.delete`. Ia juga menjaga input yang rusak, kebingungan header, negosiasi versi, kegagalan capability, otorisasi, dan kegagalan handler sebagai bukti yang terpisah.

Menutup stdin atau response HTTP mengakhiri aktivitas transport. Ia tidak mengakhiri sesi protokol karena MCP modern tidak memiliki sesi protokol.

### Kompatibilitas legacy yang eksplisit

Versi sampai `2025-11-25` menggunakan `initialize`, `notifications/initialized`, capabilities berlingkup-koneksi, dan, pada Streamable HTTP yang lebih awal, sesi protokol opsional. Perilaku itu masih relevan ketika klien dua-era berbicara dengan server lama.

Pisahkan era-nya. Request modern diidentifikasi oleh metadata wajib per-request. Koneksi legacy dipilih hanya melalui jalur fallback yang terdokumentasi. Jangan mengirim `initialize` sebagai default untuk server `2026-07-28`.

"Stateless" oleh karena itu memiliki makna spesifik-era. Di `2026-07-28`, ia adalah invarian protokol: setiap request biasa dapat ditafsirkan secara mandiri dan tidak ada sesi MCP yang ada. Pada versi sampai `2025-11-25`, inisialisasi dan capabilities yang dinegosiasikan termasuk milik sebuah koneksi, sehingga adapter kompatibilitas dapat mempertahankan state koneksi legacy itu. Implementasi dua-era bukanlah satu state machine yang permisif. Ia adalah inti modern stateless di samping adapter legacy yang terisolasi, dengan keputusan pemilihan eksplisit sebelum parser mana pun berjalan.

Kedua makna tidak melarang state aplikasi yang tahan lama. Sebuah workflow, task, atau draf dapat hidup di balik handle opak dalam store bersama. Klien mengirim handle itu sebagai input biasa, dan setiap replica mengautentikasi dan mengotorisasi penggunaannya. Konteks protokol tidak boleh bocor ke store itu sebagai pengganti sesi yang dihapus.

```figure
mcp-tool-call
```

## Gunakan

`code/main.py` membangun, memvalidasi, menelusuri, dan mendispatch pesan MCP modern tanpa framework. Jalankan:

```bash
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Perhatikan tiga invarian dalam output:

- Setiap request mengulang field `_meta`-nya.
- Setiap hasil sukses adalah `resultType: "complete"` dan menyertakan identitas server.
- Hasil daftar berurutan deterministik dan memiliki petunjuk cache eksplisit.

## Rilis

Pelajaran ini mengirimkan `outputs/skill-mcp-handshake-tracer.md`. Nama file historis tetap stabil, tetapi artefaknya kini adalah tracer request stateless. Ia mengaudit setiap pesan secara mandiri dan hanya melabeli lalu lintas handshake legacy ketika itu benar-benar ada.

## Latihan

1. Ubah versi protokol satu request ke `2027-01-01`. Konfirmasikan kode error adalah `-32022` dan data mengiklankan versi yang didukung.
2. Hapus `io.modelcontextprotocol/clientCapabilities` dari request kedua. Konfirmasikan server tidak menggunakan ulang capabilities dari request pertama.
3. Balikkan registry tool in-memory. Konfirmasikan `tools/list` tetap mengembalikan urutan deterministik yang sama.
4. Ubah `cacheScope` dari `public` ke `private`. Jelaskan konteks otorisasi mana yang boleh menggunakan ulang response pada masing-masing kasus.
5. Tambahkan test opsional untuk penghapusan `clientInfo`. Request seharusnya tetap valid karena identitas klien disarankan, bukan wajib.

## Istilah Kunci

| Term | Meaning |
|------|---------|
| Stateless protocol | Every request supplies the metadata needed to interpret it |
| Request metadata | Version, client capabilities, and recommended client identity in `params._meta` |
| `server/discover` | Mandatory server method for versions, capabilities, instructions, and identity |
| `resultType` | Discriminator on every successful modern result |
| Cacheable result | Result that includes required `ttlMs` and `cacheScope` hints |
| Protocol era | Modern per-request metadata or legacy connection-scoped initialization |
| Transport lifetime | Process, connection, or response-stream lifetime, not protocol session state |
| `-32022` | Unsupported protocol version error with requested and supported versions |

## Bacaan Lanjutan

- [MCP Architecture](https://modelcontextprotocol.io/specification/2026-07-28/architecture)
- [MCP Base Protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
