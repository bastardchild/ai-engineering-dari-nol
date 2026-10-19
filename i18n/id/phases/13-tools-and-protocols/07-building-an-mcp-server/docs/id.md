# Membangun Server MCP: Stateless di Python dan TypeScript

> Sebuah server MCP modern tidak mengingat sebuah handshake. Ia memvalidasi metadata pada setiap request, menjalankan satu handler, dan mengembalikan satu hasil bertipe.

**Type:** Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 13, Lesson 06
**Time:** ~85 minutes

## Tujuan Pembelajaran

- Mengimplementasikan `server/discover` yang wajib untuk MCP `2026-07-28`.
- Memvalidasi versi protokol dan capabilities klien pada setiap request.
- Mengekspos tools, resources, dan prompts dengan urutan daftar deterministik.
- Mengembalikan `resultType`, identitas server, dan petunjuk cache pada hasil yang tepat.
- Menyajikan kontrak stateless yang sama melalui stdio ber-delimited baris baru di Python dan TypeScript.

## Masalahnya

Sebuah server yang menyimpan capabilities klien setelah pesan pertama mudah dibangun tetapi sulit dioperasikan. Proses yang sama dapat melayani klien sekuensial. Sebuah request remote dapat mendarat di worker lain. Deklarasi capability yang basi dapat membocorkan perilaku melintasi batas otorisasi.

MCP `2026-07-28` menyelesaikan bagian protokol dari masalah itu dengan membuat setiap request mendeskripsikan dirinya sendiri. Aplikasimu masih dapat menyimpan catatan, job, atau handle state eksplisit yang tahan lama. Yang tidak dapat disimpannya adalah state protokol tersembunyi yang mengubah cara request kemudian didekode.

Pelajaran ini membangun server catatan dua kali. Versi Python dan TypeScript hanya memakai library standar mereka untuk inti protokolnya. Keduanya mengekspos metode yang sama dan menegakkan kontrak wire yang sama.

## Konsepnya

### Loop dispatch modern

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

Tiga aturan stdio masih penting:

- Tulis hanya pesan JSON-RPC ke stdout. Kirim diagnostik ke stderr.
- Batasi pesan dengan baris baru dan flush setiap response.
- Keluar dengan cepat ketika stdin mencapai EOF.

Umur proses adalah umur transport. Itu bukan sesi MCP modern.

### Validasi request

Setiap request harus memiliki:

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Dua field pertama bersifat wajib. `clientInfo` disarankan. Validasi bentuk identitas yang ada, tetapi jangan perlakukan itu sebagai autentikasi.

Jika versi tidak didukung, kembalikan kode `-32022` dengan `requested` dan `supported`. Metadata request yang hilang adalah params tidak valid, kode `-32602`. Jangan pernah mengisi field yang hilang dari panggilan sebelumnya.

### Discovery yang wajib

Server modern harus mengimplementasikan `server/discover`. Hasil discovery yang lengkap menyertakan versi modern yang didukung, capabilities, instruksi opsional, petunjuk cache, dan identitas server di `_meta` hasil:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

Discovery tidak membuka kunci server. Klien dapat memanggil `tools/list` tanpa memanggil discovery karena `tools/list` sudah membawa metadata request yang sama.

### Tools

`tools/list` mengembalikan daftar deterministik dari tool descriptor. Urutan stabil meningkatkan caching response dan menjaga konteks model tetap stabil. Hasil juga mensyaratkan `ttlMs` dan `cacheScope`.

`tools/call` mengembalikan content blocks dan `isError`. Gunakan error JSON-RPC ketika envelope protokol atau parameter metode tidak valid. Gunakan `isError: true` ketika invokasi tool yang valid berjalan tetapi tool-nya sendiri gagal.

Tool annotations tetap bersifat petunjuk, bukan penegakan:

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

Host seharusnya memakainya untuk konfirmasi dan presentasi. Server tetap harus menegakkan otorisasi yang sesungguhnya.

### Resources

`resources/list` mengembalikan URI descriptor yang stabil. `resources/read` mengembalikan contents bertipe. Keduanya dapat di-cache di `2026-07-28`, jadi keduanya menyertakan `ttlMs` dan `cacheScope`.

Gunakan `cacheScope: "private"` untuk data catatan spesifik-pengguna. Cache bersama tidak boleh menggunakan ulang response private di antara konteks otorisasi.

Pengiriman perubahan modern tidak memakai `resources/subscribe`. Klien membuka `subscriptions/listen` dan meminta `resourceSubscriptions` atau kategori list-change. Pelajaran 10 membangun alur itu.

### Prompts

`prompts/list` dapat di-cache dan deterministik. `prompts/get` merender prompt bernama dengan argumen. Hasil prompt yang di-render lengkap, tetapi itu bukan salah satu dari hasil list atau read yang dapat di-cache yang mensyaratkan petunjuk cache.

### Setiap hasil sukses bertipe

Contoh-contoh memakai satu pembungkus untuk setiap sukses:

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

Handler list, read, dan discovery menambahkan `ttlMs` plus `cacheScope`. Menyentralisasi pembungkus ini mencegah satu handler mengabaikan field hasil modern secara diam-diam.

### Tanpa request yang dimulai server

Sebuah server modern dapat mengirim notifikasi yang terkait dengan request klien, atau notifikasi pada stream `subscriptions/listen` yang dibuka klien. Ia tidak boleh mengirim request JSON-RPC-nya sendiri.

Ketika sebuah handler membutuhkan sampling, elicitation, atau input roots, ia mengembalikan hasil `input_required`. Klien memenuhi input request yang tertanam dan mencoba ulang metode asli dengan id request baru. Pelajaran 11 membahas pola Multi Round-Trip Request itu.

### Kompatibilitas legacy yang eksplisit

Sebuah server dua-era juga dapat mengimplementasikan handshake `2025-11-25` pada cabang legacy yang terpisah dengan jelas. Ia memilih perilaku modern ketika field `_meta` modern wajib hadir dan perilaku legacy ketika ia menerima `initialize`.

Jangan meneruskan request `2026-07-28` melalui jalur handshake legacy. Jangan menempelkan field `resultType` modern ke hasil inisialisasi legacy. Kode di pelajaran ini secara disengaja hanya-modern sehingga invariannya tetap terlihat.

```figure
t3-dispatch-loop
```

## Gunakan

Jalankan demo finit dan test server Python:

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

Jalankan port TypeScript dengan TypeScript runner:

```bash
npx tsx main.ts --demo
```

Demo mengirim `server/discover`, mendaftar setiap primitif, memanggil tools, dan menampilkan error versi yang tidak didukung. Setiap request modern mengulang metadata. Setiap sukses menyertakan identitas server.

## Rilis

Pelajaran ini mengirimkan `outputs/skill-mcp-server-scaffolder.md`. Ia menghasilkan rencana server modern dengan kontrak discovery, validasi per-request, daftar deterministik yang dapat di-cache, dan adapter legacy opsional yang terisolasi.

## Latihan

1. Hapus capabilities dari satu request dan buktikan server tidak menggunakan ulang deklarasi request sebelumnya.
2. Balikkan urutan sisipan `TOOLS`, `PROMPTS`, dan catatan. Konfirmasikan semua hasil daftar tetap stabil.
3. Tambahkan tool `notes_delete` yang destruktif dan wajibkan pemeriksaan otorisasi di dalam eksekutor. Pertahankan `destructiveHint` hanya sebagai petunjuk UX.
4. Tambahkan `resources/templates/list` dengan `ttlMs`, `cacheScope`, dan urutan deterministik.
5. Bangun adapter legacy terpisah untuk `2025-11-25`. Tambahkan test yang membuktikan request modern tidak pernah masuk ke dalamnya.

## Istilah Kunci

| Term | Meaning |
|------|---------|
| Stateless server | Handles each request from its own metadata without protocol-session memory |
| `server/discover` | Mandatory modern method that advertises versions and capabilities |
| Complete result | Successful modern result with `resultType: "complete"` |
| Cacheable result | Discovery, list, or resource-read result with `ttlMs` and `cacheScope` |
| Deterministic list | Same logical registry produces the same item order |
| Server identity | Recommended `io.modelcontextprotocol/serverInfo` in result `_meta` |
| Tool error | Valid tool call that returns content with `isError: true` |
| Protocol error | Invalid JSON-RPC or MCP request returned through `error` |

## Bacaan Lanjutan

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
