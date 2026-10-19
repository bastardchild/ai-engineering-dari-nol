# MCP Tasks Extension: Kerja Tahan Lama di Inti Tanpa-State

> MCP tanpa-state tidak berarti setiap operasi harus selesai dalam satu request. Ekstensi Tasks resmi memberikan pekerjaan jangka panjang sebuah handle tahan lama yang eksplisit. Sebuah server bisa mengembalikan handle itu dari `tools/call`, instance mana pun bisa menjawab `tasks/get`, dan input klien datang melalui `tasks/update` tanpa menghidupkan kembali sesi protokol.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 11 (stateless MRTR), Phase 13 · 12 (elicitation)
**Time:** ~90 minutes

## Tujuan Pembelajaran

- Bedakan transport protokol tanpa-state dari state task aplikasi yang tahan lama.
- Negosiasikan ekstensi `io.modelcontextprotocol/tasks` di kapabilitas per-request dan `server/discover`.
- Kembalikan `CreateTaskResult` yang diarahkan server dengan `resultType: "task"` hanya setelah pembuatan yang tahan lama.
- Poll dengan `tasks/get`, penuhi input task dengan `tasks/update`, dan minta pembatalan kooperatif dengan `tasks/cancel`.
- Hapus asumsi `tasks/status`, `tasks/result`, dan `tasks/list` yang lebih lama.
- Subscribe ke notifikasi task opsional melalui `subscriptions/listen` pada stream SSE respons POST.
- Modelkan kedaluwarsa task, pemulihan setelah restart, deduplikasi kunci input, dan kesalahan eksekusi dengan benar.

## Mengapa Tasks adalah Ekstensi

Tasks pertama kali muncul sebagai fitur inti eksperimental di 2025-11-25. Redesain Juli 2026 memindahkannya ke ekstensi resmi `io.modelcontextprotocol/tasks` sehingga klien dan server bisa memilih siklus hidup ekstra ini tanpa memperluas protokol inti untuk semua orang.

Spesifikasi ekstensi tetap berupa permukaan draft walaupun itu adalah rumah resmi saat ini untuk Tasks. Pin versi ekstensi yang didukung SDK-mu, jalankan skenario konformansi, dan isolasi adaptor wire dari domain worker dan penyimpananmu.

Gunakan sebuah task ketika operasi memiliki satu atau lebih dari sifat berikut:

- Ia bisa melewati batas waktu request biasa.
- Sebuah antrean worker atau sistem job eksternal sudah memiliki eksekusinya.
- Klien perlu pulih setelah restart-nya sendiri.
- Operasi berhenti untuk input pengguna atau model selama eksekusi.
- Pembatalan dan pengambilan hasil tahan lama adalah kebutuhan produk.

Jangan membuat task untuk lookup deterministik yang murah. Handle, persistensi, polling, kedaluwarsa, dan pembatalan adalah kompleksitas nyata.

## Inti Tanpa-State, Aplikasi Ber-State

MCP 2026-07-28 menghapus `initialize`, `notifications/initialized`, sesi protokol, dan `Mcp-Session-Id`. Itu tidak melarang produk yang ber-state.

Sebuah task id adalah state aplikasi eksplisit:

- Server menyimpannya sebelum mengembalikannya.
- Klien bisa menyimpannya dan melakukan poll lagi setelah restart.
- Id itu bisa di-route ke replica mana pun yang ditopang store tahan lama yang sama.
- Otorisasi diperiksa pada setiap method task.
- Kedaluwarsa dan penghapusan didefinisikan oleh field task, bukan oleh lifetime transport.

Ini berbeda secara operasional dari state tersembunyi yang menempel pada sebuah koneksi.

Pertahankan empat lifetime secara terpisah:

| State | Lifetime | Where it belongs |
|---|---|---|
| Protocol metadata | One request | `params._meta`, validated again on every call |
| Transport work | One stdio request or HTTP response | In-flight coordinator with a bounded deadline |
| MRTR continuation | One retry sequence | Integrity-protected `requestState`, plus replay controls when needed |
| Durable task | Across requests, replicas, restarts, and reconnects | Shared application store keyed by an authorized `taskId` |

Memindahkan record task ke dalam memori proses tidak membuat MCP menjadi ber-state. Itu membuat aplikasinya tidak andal. Protokolnya tetap tanpa-state, tetapi `tasks/get` berikutnya yang di-route ke replica lain tidak bisa memulihkan record itu. Persistensikan sebelum mengembalikan handle, lalu buat setiap method task me-resolve record bersama yang sama di bawah pemeriksaan tenant dan principal.

## Negosiasi Kapabilitas

Klien mengiklankan dukungan pada setiap request yang memenuhi syarat:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "extensions": {
        "io.modelcontextprotocol/tasks": {}
      }
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "lesson-client",
      "version": "1.0.0"
    }
  }
}
```

Server mengembalikan `supportedVersions`, kapabilitas, `ttlMs`, dan `cacheScope` yang tepat dari `server/discover`, dengan ekstensi yang sama di bawah capabilities. Karena ia mengiklankan tools, ia juga mengimplementasikan `tools/list` yang wajib. Hasil itu mengembalikan deskriptor `generate_report` deterministik, `inputSchema` objek yang valid, `resultType: "complete"`, metadata identitas server, dan petunjuk cache publik.

Method task dari klien yang tidak mendeklarasikan ekstensi mengembalikan `-32021`, Missing Required Client Capability, dengan `data.requiredCapabilities` diatur ke `{"extensions":{"io.modelcontextprotocol/tasks":{}}}`. String protokol yang tidak didukung mengembalikan `-32022` dengan data `supported` dan `requested` yang tepat; versi yang hilang atau bukan string mengembalikan `-32602`.

Envelope tanpa `id` JSON-RPC adalah notifikasi. Penerima boleh memprosesnya, tetapi ia tidak mengeluarkan hasil atau error JSON-RPC. Adaptor Streamable HTTP mengembalikan `202 Accepted` tanpa body untuk notifikasi yang diterima.

Saat ini, hanya `tools/call` yang mendukung eksekusi yang diperkaya task. Rancang abstraksi internalmu agar tipe request mendatang tidak memerlukan penulisan ulang penyimpanan.

## Pembuatan Task yang Diarahkan Server

Flag klien lama `params._meta.task.required` sudah hilang. Klien mendeklarasikan dukungan ekstensi, lalu server memutuskan apakah sebuah `tools/call` tertentu menjadi task.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "generate_report",
    "arguments": {"size": "large"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Respons:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "task",
    "taskId": "tsk_786512e29e0d",
    "status": "working",
    "statusMessage": "Preparing report outline.",
    "createdAt": "2026-08-21T10:30:00Z",
    "lastUpdatedAt": "2026-08-21T10:30:00Z",
    "ttlMs": 900000,
    "pollIntervalMs": 1000
  }
}
```

Server tidak boleh mengembalikan handle ini sebelum `tasks/get` untuk id itu bisa di-resolve. Pada store yang konsisten secara akhir, tunggu visibilitas baca sebelum menjawab. Jika tidak, sebuah klien bisa menerima id yang tampak valid lalu langsung mendapat "not found."

Respons task bersifat unsolicited dalam arti bahwa klien tidak meminta mode task. Itu bukan tanpa-negosiasi: request saat ini tetap harus mengiklankan ekstensi.

## Bentuk Task

Setiap task membawa:

- `taskId`: pengenal stabil yang dibuat server;
- `status`: `working`, `input_required`, `completed`, `cancelled`, atau `failed`;
- `createdAt` dan `lastUpdatedAt`: timestamp ISO 8601;
- `ttlMs`: durasi kedaluwarsa sejak pembuatan, atau `null` untuk tanpa batas yang diiklankan;
- `pollIntervalMs` opsional: cadangan poll minimum yang saat ini disarankan server;
- `statusMessage` opsional: konteks untuk pengguna atau model.

Field spesifik-status muncul hanya bila relevan:

- `input_required` menyertakan `inputRequests`.
- `completed` menyertakan bentuk `result` dari request asli.
- `failed` menyertakan objek `error` JSON-RPC.

Klien seharusnya menghormati `pollIntervalMs`. Sebuah server boleh membatasi rate polling yang terlalu agresif dan boleh mengubah interval sepanjang lifetime task.

## Poll dengan `tasks/get`

Klien meminta cuplikan saat ini:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/get
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tasks/get",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

`tasks/get` sendiri telah selesai, sehingga hasilnya selalu memiliki `resultType: "complete"`. Task bersarang di dalamnya masih bisa memiliki `status: "working"` atau `status: "input_required"`.

Pembedaan ini mencegah bug parser yang umum:

```text
result.resultType = complete    means the tasks/get RPC finished
result.status = working        means the represented job is still running
```

Tidak ada panggilan `tasks/result`. Ketika task selesai, respons `tasks/get` berikutnya meng-inline `CallToolResult` asli di bawah `result`:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "completed",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:34:12Z",
  "ttlMs": 900000,
  "result": {
    "resultType": "complete",
    "content": [
      {"type": "text", "text": "Generated large report with approved outline."}
    ],
    "structuredContent": {"size": "large", "approved": true},
    "isError": false,
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "tasks-demo",
        "version": "1.0.0"
      }
    }
  },
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "tasks-demo",
      "version": "1.0.0"
    }
  }
}
```

`resultType` luar mengatakan RPC `tasks/get` telah selesai. `result.resultType` bersarang mengatakan panggilan tool asli telah selesai. Diskriminator bersarang itu wajib. `CallToolResult` bersarang seharusnya juga membawa `io.modelcontextprotocol/serverInfo`-nya sendiri; pelajaran ini menyertakannya alih-alih menyimpan payload tanpa-tipe.

Tidak ada `tasks/list`. Server tanpa sesi tidak bisa menyimpulkan dengan aman task mana yang termasuk dalam daftar berlingkup koneksi. Aplikasi yang memerlukan riwayat seharusnya mengekspos domain tool yang diotorisasi dengan filter dan aturan kepemilikan yang eksplisit.

## Input Selama Eksekusi Task

Input task dan MRTR inti terlihat mirip tetapi memakai kelanjutan yang berbeda.

### Input yang dibutuhkan sebelum pembuatan task

Kembalikan `resultType: "input_required"` inti dari `tools/call` asli. Klien memenuhinya dan mencoba ulang panggilan asli itu. Baru buat task setelah ronde MRTR sinkron itu selesai.

### Input yang dibutuhkan setelah pembuatan task

Atur task ke `input_required`. `tasks/get` mengekspos `inputRequests` yang belum terpenuhi, dan klien mengirim respons melalui `tasks/update`. Klien tidak mencoba ulang `tools/call` asli.

Cuplikan:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "input_required",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:31:00Z",
  "ttlMs": 900000,
  "inputRequests": {
    "approve_outline": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Approve the generated report outline?",
        "requestedSchema": {
          "type": "object",
          "properties": {"approved": {"type": "boolean"}},
          "required": ["approved"]
        }
      }
    }
  }
}
```

Update:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/update
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tasks/update",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "inputResponses": {
      "approve_outline": {
        "action": "accept",
        "content": {"approved": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Respons sukses adalah pengakuan kosong plus `resultType: "complete"`. Perubahan state bisa bersifat konsisten-akhir, sehingga klien terus melakukan poll atau mendengarkan.

Setiap kunci `inputRequests` harus unik untuk seluruh lifetime task. Cuplikan `tasks/get` yang berulang boleh menunjukkan kunci yang sama yang belum terpenuhi; klien mendeduplikasi UI dan server mengabaikan respons untuk kunci yang tidak diketahui, sudah tergantikan, atau sudah dipenuhi. Update sebagian bisa meninggalkan task dalam `input_required` sampai semua kunci yang wajib dijawab.

## Pembatalan Bersifat Kooperatif

`tasks/cancel` menandai niat dan mengembalikan pengakuan lengkap yang kosong. Pengakuan itu tidak menjamin worker berhenti. Pekerjaan bisa selesai lebih dulu, mengabaikan pembatalan, atau bertransisi nanti.

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/cancel
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tasks/cancel",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Untuk ketiga method task, `Mcp-Name` mencerminkan `params.taskId`. Ia tidak mengulang nama method JSON-RPC. `code/main.py` memusatkan aturan ini di `make_http_request`.

Worker pelajaran menghormati pembatalan seketika, sehingga panggilan berulang menjadi idempoten. Sebuah klien produksi tetap harus memperlakukan pembatalan sebagai kooperatif, bukan menyimpulkan status task final dari pengakuan itu.

Jangan memakai `notifications/cancelled` untuk membatalkan sebuah task. Notifikasi itu milik pembatalan request, bukan Tasks yang tahan lama.

Pembedaan ini penting di batas routing. Pembatalan request menyasar satu operasi JSON-RPC yang sedang berjalan atau respons HTTP berlingkup request-nya. Jika `tools/call` sudah mengembalikan `resultType: "task"`, request itu sudah selesai dan menutup transport-nya tidak bisa menamai atau menghentikan pekerjaan yang tahan lama itu. `tasks/cancel` adalah RPC baru yang diotorisasi. Ia membawa `params.taskId`, mencerminkan id itu di `Mcp-Name`, me-resolve backend pemilik task, mencatat niat pembatalan kooperatif, dan mengembalikan pengakuan tanpa mengklaim worker telah berhenti.

Sebuah gateway karena itu harus menyimpan koordinator request dan route task di tabel yang berbeda. Tabel request bisa hilang ketika respons selesai. Route task harus bertahan sampai status terminal dan kedaluwarsa retensi. [Lesson 29: MCP Reliability, Cancellation, and Flow Control](../../29-mcp-reliability-cancellation-and-flow-control/docs/en.md) membangun aturan race, timeout, idempotensi, backpressure, dan retry untuk kedua jalur itu.

## Notifikasi Opsional

Polling adalah baseline. Sebuah klien yang menginginkan update push mengirim `subscriptions/listen` dengan id task. Untuk Streamable HTTP, ini adalah POST yang responsnya adalah stream SSE berlingkup request. Tidak ada stream event GET mandiri dan tidak ada sesi protokol yang dijaga agar tetap hidup.

Server mengakui id yang diterima dengan `notifications/subscriptions/acknowledged` dan kemudian bisa mengirim cuplikan lengkap melalui `notifications/tasks`. Pengakuan dan setiap notifikasi task membawa `io.modelcontextprotocol/subscriptionId` di `_meta`, sama dengan id request `subscriptions/listen`. Setiap notifikasi task selain itu setara dengan apa yang akan dikembalikan `tasks/get` pada saat itu.

Klien tetap harus mendeklarasikan ekstensi Tasks. Mereka seharusnya menyambung ulang dan melanjutkan dari id task yang tahan lama alih-alih bergantung pada replay event atau `Last-Event-ID`.

## Semantik Kegagalan

Gunakan dua lapisan error dengan benar.

### Error protokol

Parameter method yang tidak valid atau task id yang tidak diketahui mengembalikan error JSON-RPC, umumnya `-32602`. Dukungan ekstensi yang hilang mengembalikan `-32021` dengan objek kapabilitas yang diperlukan.

### Hasil eksekusi task

- Hasil tool biasa dengan `isError: true` tetap merupakan task `completed` karena panggilan tool menghasilkan hasil yang didefinisikannya.
- Error JSON-RPC selama eksekusi tertunda membuat task menjadi `failed` dan menyimpan error JSON-RPC itu di bawah `error`.
- Penolakan pengguna bisa menghasilkan `cancelled`, hasil penolakan yang selesai, atau hasil aman spesifik-domain lain. Dokumentasikan pilihannya.

## Ketahanan, Kedaluwarsa, dan Kepemilikan

Persistensikan setidaknya id task, status, timestamp, ttl, interval poll, kepemilikan operasi asli, hasil atau error, input request yang belum terpenuhi, dan semua kunci input yang diterbitkan.

Kunci penyimpanan harus mencakup atau me-resolve tenant dan principal yang otoritatif. Mengetahui sebuah task id tidak boleh memberikan akses. Periksa kepemilikan pada setiap `tasks/get`, `tasks/update`, `tasks/cancel`, dan subscription.

`ttlMs` diukur sejak pembuatan dan bisa berubah. Sebuah klien boleh memperlakukannya sebagai pengaman ketika sebuah task berhenti menghasilkan update yang bisa diamati. Sebuah server boleh gagal lalu menghapus task yang kedaluwarsa nanti. Jangan menggambarkannya sebagai janji untuk mempertahankan hasil yang sudah selesai selama sejumlah milidetik setelah penyelesaian.

Gunakan tulisan atomik atau transaksi. Pelajaran ini menulis sebuah berkas sementara dan meng-renamenya secara atomik. Sebuah layanan multi-replica seharusnya memakai store tahan lama bersama dan lease worker atau kontrol konkurensi yang setara.

```figure
tp-task-lifecycle
```

## Bangun Sendiri

`code/main.py` mengimplementasikan layanan task deterministik:

- `server/discover` mengembalikan `supportedVersions`, petunjuk cache, dan ekstensi Tasks.
- `tools/list` mengembalikan deskriptor `generate_report` deterministik dan bisa di-cache dengan skema input yang valid.
- `tools/call` membuat dan menyimpan task sebelum mengembalikan `resultType: "task"`.
- Instance layanan baru memuat ulang task yang sama, mendemonstrasikan pemulihan setelah restart.
- `tasks/get` mengembalikan cuplikan task yang lengkap.
- Worker bergerak dari `working` ke `input_required`.
- `tasks/update` menerima respons formulir dan mengembalikan pengakuan lengkap yang kosong.
- Worker menyimpan `CallToolResult` bersarang dengan `resultType` dan identitas server-nya sendiri, lalu bertransisi ke `completed`.
- `tasks/cancel` bersifat idempoten dalam implementasi ini.
- Pembangun HTTP mengatur `Mcp-Name` ke `params.taskId` untuk `tasks/get`, `tasks/update`, dan `tasks/cancel`.
- Pembantu notifikasi memakai `notifications/subscriptions/acknowledged` dan `notifications/tasks`, keduanya ditandai dengan id request listen.
- Notifikasi tanpa id tidak menghasilkan respons JSON-RPC.

Worker maju secara eksplisit alih-alih tidur di thread latar belakang. Itu membuat setiap transisi state deterministik dan menjaga contoh protokol tetap terpisah dari mekanisme antrean.

## Gunakan

Dari root repositori:

```bash
cd phases/13-tools-and-protocols/13-mcp-async-tasks/code
python3 main.py
python3 -m unittest discover tests -v
```

Urutan hasil yang diharapkan:

```text
id=0 resultType=complete status=ack
id=1 resultType=task status=working
id=2 resultType=complete status=working
id=3 resultType=complete status=input_required
id=4 resultType=complete status=ack
id=5 resultType=complete status=completed
```

Juga verifikasi bahwa `tasks/status`, `tasks/result`, dan `tasks/list` mengembalikan method-not-found di layanan modern.
Verifikasi bahwa `tools/list` deterministik dan setiap method task HTTP saat ini mencerminkan id task-nya melalui `Mcp-Name`.

## Rilis

`outputs/skill-task-store-designer.md` kini menghasilkan desain yang sadar-ekstensi: negosiasi kapabilitas, pembuatan tahan-lama-sebelum-dikembalikan, method saat ini, alur update input, kepemilikan, kedaluwarsa, pembatalan, subscription, dan migrasi dari method eksperimental yang dihapus.

## Latihan

1. Tambahkan kunci input kedua yang belum terpenuhi. Kirim `tasks/update` sebagian dan buktikan task tetap `input_required` sampai kedua kunci dijawab.
2. Tambahkan kepemilikan tenant ke store dan tolak task id valid yang diajukan oleh principal yang diautentikasi salah.
3. Tambahkan lease worker dengan kedaluwarsa. Demonstrasikan bahwa dua instance layanan tidak bisa menyelesaikan task yang sama secara bersamaan.
4. Implementasikan adaptor SSE POST-respons untuk `subscriptions/listen`. Jangan menambahkan GET, `Last-Event-ID`, atau header sesi.
5. Tambahkan pembersihan kedaluwarsa. Bedakan task yang kedaluwarsa dari task id yang rusak format tanpa membocorkan keberadaan lintas-tenant.

## Istilah Kunci

| Term | Meaning in the current extension |
|------|----------------------------------|
| Tasks extension | Optional `io.modelcontextprotocol/tasks` capability for durable async work |
| `CreateTaskResult` | Server-directed `resultType: "task"` response to an eligible request |
| `tasks/get` | Poll a full current task snapshot, including terminal result or pending input |
| `tasks/update` | Submit responses to a task's outstanding `inputRequests` |
| `tasks/cancel` | Acknowledge cooperative cancellation intent |
| `input_required` | Task status indicating client input is outstanding |
| `pollIntervalMs` | Server-suggested minimum delay before another poll |
| `ttlMs` | Expiry duration measured from task creation |
| Durable-before-return | Rule that the task id must resolve before its handle is sent |
| `notifications/tasks` | Optional full task snapshot delivered on a subscribed SSE response |

## Kompatibilitas Era Lama

Permukaan eksperimental 2025-11-25 memakai augmentasi task yang diminta klien, `tasks/status`, `tasks/result`, dan `tasks/list` opsional. Pertahankan nama-nama itu hanya di dalam adaptor legacy yang di-pin. Sebuah klien saat ini memakai kapabilitas ekstensi, menerima handle yang diarahkan server, melakukan poll `tasks/get`, menyediakan input dengan `tasks/update`, dan membaca hasil akhir dari cuplikan task.

## Bacaan Lanjutan

- [Official MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
