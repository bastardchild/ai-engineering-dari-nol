# MCP Model Input: Migrasi Sampling dan MRTR Tanpa-State

> MCP 2026-07-28 men-deprecate Sampling untuk desain baru dan menghapus kanal request server-ke-klien. Jika sebuah alur kerja yang ada masih membutuhkan model klien, server mengembalikan hasil `input_required` dan klien mencoba ulang request asli dengan output model. Loop penalaran menjadi eksplisit, berbatas, dan tanpa-state di lapisan protokol.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan mengapa Sampling dideprecate di MCP 2026-07-28 dan pilih default integrasi model langsung untuk server baru.
- Implementasikan alur kerja kompatibilitas yang membawa `sampling/createMessage` melalui Multi Round-Trip Requests (MRTR).
- Letakkan revisi protokol dan kapabilitas klien di setiap objek `_meta` request.
- Kembalikan `resultType: "input_required"` dan ulangi method asli dengan id JSON-RPC baru.
- Lindungi integritas `requestState` dan ikatkan ke principal, method, argumen, dan kedaluwarsa.
- Batasi loop berbantuan model dengan pemeriksaan kapabilitas, persetujuan, validasi respons, dan batas ronde.

## Keputusan Sebelum Protokol

Sebuah tool seperti `summarize_repo` memerlukan dua jenis pekerjaan:

1. Pekerjaan deterministik: daftar berkas, baca berkas yang diizinkan, validasi path, dan rangkai konten.
2. Pekerjaan model: pilih berkas representatif dan sintesis ringkasannya.

Kini kamu punya dua arsitektur yang valid.

### Server baru: integrasikan dengan penyedia model secara langsung

Ini adalah default saat ini. Server memiliki pemilihan model, kredensial, anggaran, retry, dan observability. Ia mengembalikan satu hasil `tools/call` biasa ke MCP client.

Pilih ini ketika server sudah menjadi layanan ter-host atau ketika perilaku model yang dapat diprediksi lebih penting daripada memakai model host.

### Alur kerja Sampling yang ada: migrasikan ke MRTR

Sampling masih ada selama jendela deprecasinya. Sebuah server yang menyasar 2026-07-28 tidak bisa mengirim request `sampling/createMessage` hidup kembali ke klien. Sebagai gantinya ia menanamkan request itu dalam sebuah `InputRequiredResult`.

Pilih jalur kompatibilitas ini hanya ketika memakai model dan kredensial klien merupakan kebutuhan produk yang nyata. Catat rencana penghapusan karena implementasi baru seharusnya tidak mengadopsi Sampling yang sudah dideprecate.

## Kontrak Tanpa-State

Protokol Juli 2026 tidak memiliki pertukaran `initialize`, tidak ada `notifications/initialized`, dan tidak ada `Mcp-Session-Id`. Setiap request membawa informasi yang dulu hidup di handshake:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Server memvalidasi revisi pada setiap request. Versi yang hilang atau bukan string adalah parameter tidak valid, `-32602`. String yang tidak didukung mengembalikan `-32022` dengan data tepat `{"supported":["2026-07-28"],"requested":"<client version>"}`. Kapabilitas Sampling yang hilang mengembalikan `-32021` dengan `data.requiredCapabilities` diatur ke `{"sampling":{}}`.

Envelope tanpa `id` JSON-RPC adalah notifikasi. Penerima boleh memprosesnya, tetapi ia tidak mengeluarkan respons sukses maupun respons error. Adaptor Streamable HTTP mengembalikan `202 Accepted` tanpa body untuk notifikasi yang diterima.

Server juga mengimplementasikan `server/discover` dengan kunci `supportedVersions` yang tepat, kapabilitas, `ttlMs`, dan `cacheScope` sehingga klien bisa mempelajari dan meng-cache kontrak server sebelum memanggil sebuah tool. Karena discovery mengiklankan `tools`, server juga mengimplementasikan `tools/list` yang wajib. Deskriptor `summarize_repo` deterministik-nya mencakup `inputSchema` objek yang valid, `resultType: "complete"`, metadata identitas server, dan petunjuk cache publik.

Setiap hasil modern yang sukses memiliki diskriminator:

- `resultType: "complete"` berarti operasi selesai.
- `resultType: "input_required"` berarti klien harus memenuhi request yang tertanam dan mencoba ulang.
- Ekstensi boleh mendefinisikan tipe hasil tambahan. Ekstensi Tasks menambahkan `"task"` di Lesson 13.

## Satu Ronde MRTR

Server tidak bisa memanggil klien saat menangani request. Sebagai gantinya ia mengembalikan hasil ini:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "pick_files": {
        "method": "sampling/createMessage",
        "params": {
          "messages": [
            {
              "role": "user",
              "content": {
                "type": "text",
                "text": "Choose three representative files and return a JSON array."
              }
            }
          ],
          "systemPrompt": "Return only the requested value.",
          "modelPreferences": {
            "costPriority": 0.8,
            "intelligencePriority": 0.2
          },
          "maxTokens": 400
        }
      }
    },
    "requestState": "opaque-integrity-protected-value"
  }
}
```

Klien memverifikasi bahwa ia mendukung Sampling, menerapkan kebijakan persetujuan dan modelnya, dan memperoleh respons model. Lalu ia mengirim request baru dengan id JSON-RPC yang berbeda:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "inputResponses": {
      "pick_files": {
        "role": "assistant",
        "content": {
          "type": "text",
          "text": "[\"README.md\", \"server.py\", \"docs/intro.md\"]"
        },
        "model": "host-model",
        "stopReason": "endTurn"
      }
    },
    "requestState": "opaque-integrity-protected-value",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}}
    }
  }
}
```

Percobaan ulang itu bukan kelanjutan dari sesi protokol. Itu adalah request baru yang mengulang method dan argumen asli, hanya menambahkan `inputResponses` ronde saat ini, dan memantulkan `requestState` byte demi byte.

MRTR hanya diizinkan pada `tools/call`, `prompts/get`, dan `resources/read`. Sebuah server tidak boleh mengembalikan `input_required` dari method yang tidak terkait.

## State Multi-Ronde

Pelajaran ini memerlukan dua panggilan model:

1. `pick_files` mengembalikan sebuah array JSON.
2. `summary` mengembalikan prosa final.

Setiap percobaan ulang hanya membawa respons untuk ronde itu. Karena itu server menaruh fase dan data antara yang tervalidasi ke `requestState` berikutnya.

Perlakukan nilai itu sebagai sesuatu yang dikendalikan penyerang. Menandatangani nama fase mentah tidak cukup. Ikat state ke:

- principal yang diautentikasi, bukan `clientInfo` yang dilaporkan sendiri;
- method asal;
- digest dari argumen asli;
- kedaluwarsa yang singkat;
- fase saat ini dan nilai antara yang tervalidasi.

Gunakan HMAC ketika kerahasiaan tidak diperlukan. Gunakan enkripsi terotentikasi ketika klien tidak boleh membaca state. Tolak tanda tangan yang buruk, nilai yang kedaluwarsa, principal yang berubah, atau argumen yang berubah dengan `-32602`.

Klien tidak boleh mem-parse atau memodifikasi `requestState`. Satu-satunya tugasnya adalah memantulkan string yang persis pada percobaan ulang.

## Model Preferences adalah Petunjuk

`costPriority`, `speedPriority`, dan `intelligencePriority` adalah preferensi yang independen. Itu bukan distribusi probabilitas dan tidak perlu berjumlah satu. Klien boleh mengabaikannya karena klien yang memiliki kebijakan model.

Pertahankan `includeContext` di `"none"` jika kamu mempertahankan alur Sampling lama. Mode konteks lain meningkatkan risiko kebocoran dan itu sendiri sudah dideprecate. Kirim konteks eksplisit minimum dalam request.

## Invarian Keselamatan

Klien adalah batas kepercayaan untuk request Sampling yang tertanam.

- Tampilkan ke pengguna apa yang diminta server terhadap model ketika kebijakan memerlukan persetujuan.
- Batasi ronde MRTR. Sebuah server berbahaya sebaliknya bisa membuat loop pengeluaran model.
- Validasi setiap respons sampling sebelum memakainya sebagai nama berkas, URL, atau input tool.
- Batasi byte dan token per ronde.
- Tolak input request yang tidak dideklarasikan di kapabilitas klien saat ini.
- Jauhkan output model dari keputusan otorisasi.
- Catat method asal dan kunci input-request tanpa mencatat isi prompt yang sensitif.

`clientInfo` dan `serverInfo` adalah metadata tampilan dan diagnostik. Jangan pernah memakai keduanya sebagai identitas yang diautentikasi.

```figure
t3-sampling-flip
```

## Bangun Sendiri

`code/main.py` mengimplementasikan alur dua-ronde lengkap tanpa paket pihak ketiga:

- `server/discover` mengembalikan `supportedVersions`, mengiklankan dukungan tool, dan mengembalikan petunjuk cache.
- `tools/list` mengembalikan deskriptor `summarize_repo` deterministik dan bisa di-cache dengan skema input berbentuk objek.
- `tools/call` memvalidasi metadata per-request.
- Hasil pertama menanamkan `sampling/createMessage` untuk pemilihan berkas.
- Percobaan ulang pertama memvalidasi hasil model dan menanamkan request kedua.
- `requestState` yang dilindungi HMAC membawa antar-fase di antara request yang independen.
- Hasil akhir memakai `resultType: "complete"`.

Model host palsu membuat contoh ini deterministik. Ganti hanya `fake_host_model` saat menyambungkan host sungguhan. State machine sisi server seharusnya tetap deterministik dan bisa diuji.

## Gunakan

Dari root repositori:

```bash
cd phases/13-tools-and-protocols/11-mcp-sampling/code
python3 main.py
python3 -m unittest discover tests -v
```

Checkpoint yang diharapkan:

- Discovery mengembalikan hasil lengkap dengan `ttlMs` dan `cacheScope`.
- Discovery tool mengembalikan deskriptor terurut yang sama dengan `resultType`, identitas server, dan petunjuk cache.
- Kapabilitas yang hilang dan versi yang tidak didukung memakai data error `-32021` dan `-32022` yang tepat.
- Notifikasi tanpa id tidak menghasilkan respons JSON-RPC.
- Id request adalah `[1, 2, 3]`, membuktikan setiap ronde MRTR independen.
- Dua hasil pertama adalah `input_required`.
- Hasil akhir adalah `complete` dan berisi berkas terpilih plus ringkasan.
- Mengubah argumen asli pada percobaan ulang membuat pemeriksaan state-request gagal.

## Rilis

`outputs/skill-sampling-loop-designer.md` kini adalah perencana migrasi. Ia terlebih dulu memutuskan apakah Sampling harus dihapus demi integrasi model langsung. Jika kompatibilitas diperlukan, ia menghasilkan ronde MRTR, pengikatan state, gerbang kapabilitas, anggaran, validasi, dan rencana penghapusan.

## Latihan

1. Ubah respons pemilihan berkas menjadi JSON yang tidak valid. Konfirmasikan server mengembalikan `-32602` alih-alih mempercayai output model.
2. Ubah `audience` di antara panggilan pertama dan percobaan ulang. Jelaskan mengapa state yang disegel memblokir penggunaan ulang lintas-request.
3. Tambahkan ronde ketiga yang meminta host mengkritik ringkasan. Bawa ringkasan sebelumnya di dalam state yang ditandatangani dan batasi seluruh alur pada tiga ronde.
4. Hapus Sampling dengan mengganti callback host palsu dengan adapter model yang dimiliki server. Daftarkan tanggung jawab persetujuan, penagihan, dan observability apa saja yang berpindah ke server.
5. Tambahkan pengujian kedaluwarsa memakai nilai state yang sudah lewat satu detik dari tenggatnya.

## Istilah Kunci

| Term | Meaning in 2026-07-28 |
|------|------------------------|
| Sampling | Deprecated feature that asks the client's model for a completion |
| MRTR | Stateless retry pattern for client input required during a request |
| `InputRequiredResult` | Result with `resultType: "input_required"` |
| `inputRequests` | Server-assigned map of embedded elicitation, sampling, or roots requests |
| `inputResponses` | Current round's client results keyed like `inputRequests` |
| `requestState` | Opaque server state echoed exactly by the client and verified by the server |
| `resultType` | Required discriminator for modern MCP results |
| Direct model integration | Recommended replacement for new servers that need model inference |
| Capability gate | Rule that prevents sending an embedded request the client did not advertise |
| Loop budget | Maximum rounds, tokens, bytes, time, and spend allowed for the operation |

## Kompatibilitas Era Lama

Sebuah klien yang di-pin ke 2025-11-25 mungkin masih memakai alur `sampling/createMessage` yang diinisialisasi server lama lewat koneksi hidup. Pertahankan perilaku itu hanya di dalam adaptor khusus-versi. Jangan menjadikan jalur ber-sesi sebagai arsitektur untuk server 2026-07-28.

SDK resmi bisa menerjemahkan penangan `input_required` modern untuk peer yang lebih lama. Shim itu adalah batas kompatibilitas, bukan izin untuk menambahkan logika baru yang bergantung sesi.

## Bacaan Lanjutan

- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP Sampling deprecation](https://modelcontextprotocol.io/seps/2577-deprecate-roots-sampling-and-logging)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
