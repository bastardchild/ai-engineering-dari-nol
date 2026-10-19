# Gateway MCP Stateless dan Admisi Registry

> Sebuah gateway harus membuat setiap rute secara eksplisit. Protokol 2026-07-28 memberinya metode, nama, versi, kapabilitas, identitas, cache, dan batas trace tanpa sesi transport.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13 · 15 (security), Phase 13 · 16 (authorization)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Mengagregasi beberapa server MCP di balik satu endpoint 2026-07-28 tanpa afinitas sesi.
- Memvalidasi metadata per-permintaan dan header routing sebelum kebijakan atau penerusan.
- Menggabungkan tool dengan namespace stabil, urutan deterministik, pin deskriptor, RBAC, dan caching privat.
- Memperlakukan catatan registry sebagai bukti discovery yang tetap membutuhkan kebijakan admisi.
- Merutekan SSE berlingkup permintaan, `subscriptions/listen`, retry MRTR, dan panggilan ekstensi Tasks dengan benar.
- Mengisolasi handshake dan dukungan sesi lama dari jalur modern.

## Masalahnya

Menghubungkan satu klien langsung ke satu server itu sederhana. Deployment yang lebih besar membutuhkan jawaban yang konsisten atas pertanyaan yang lebih sulit:

- Server mana saja yang diizinkan?
- Prinsipal mana yang boleh melihat dan memanggil setiap tool?
- Apa yang terjadi ketika dua backend mengekspos nama yang sama?
- Bagaimana perubahan deskriptor ditinjau?
- Di mana batas laju dan peristiwa audit diterapkan?
- Apakah setiap instance dapat menangani permintaan berikutnya?

Sebuah gateway berada di antara klien dan server MCP backend. Ia menyajikan satu endpoint MCP, menerapkan kebijakan lintas-aspek, dan meneruskan permintaan yang telah disetujui.

Desain gateway lama sering memultiplex satu sesi klien menjadi beberapa sesi backend dan menulis ulang `Mcp-Session-Id`. Itu adalah desain kompatibilitas lama. Inti 2026-07-28 tidak memiliki sesi protokol.

## Konsepnya

### Jalur gateway modern

Untuk setiap permintaan:

1. Autentikasi prinsipal dari otorisasi transport.
2. Validasi `MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name`, dan `params._meta`.
3. Otorisasi prinsipal, sumber daya, metode, tool, dan argumen.
4. Terapkan kebijakan deskriptor, registry, laju, dan data.
5. Buat permintaan mandiri yang segar untuk backend terpilih.
6. Validasi hasil backend dan kembalikan hasil gateway.
7. Catat peristiwa audit tanpa mencatat secret.

Tidak ada langkah yang membutuhkan sesi protokol tersembunyi. State aplikasi tetap dapat hidup di database, handle eksplisit, Tasks, atau state MRTR yang dilindungi integritas.

### Kebijakan runtime adalah keputusan gateway yang utama

Admisi memutuskan versi backend mana yang boleh masuk ke gateway. Ia tidak mengesahkan panggilan langsung. Untuk setiap permintaan, gateway menghitung ulang kebijakan dari prinsipal yang diautentikasi, issuer dan sumber daya, tenant, metode dan nama yang cocok, argumen ternormalisasi, pin deskriptor yang diadmisi, kesehatan backend saat ini, irisan kapabilitas, klasifikasi data, status laju, dan persetujuan yang terikat aksi apa pun.

Urutan ini penting. Sebuah catatan **Registry** dapat tetap aktif selama peran seorang pengguna dicabut. Sebuah deskriptor dapat tetap ter-pin sementara argumen tujuan melewati batas tenant. Sebuah backend dapat tetap disetujui selama kebijakan insiden mengkarantina panggilan pengubah-state. Kebijakan runtime karena itu adalah keputusan izin atau tolak yang utama, dengan bukti **Registry** dan deskriptor sebagai masukan.

Jangan cache keputusan allow di bawah pengenal koneksi atau sesi yang sudah dihapus. Jika kebijakan tidak tersedia, ikuti kebijakan kegagalan yang dideklarasikan menurut kelas operasi. Default aman adalah gagal tertutup untuk perubahan state dan pembacaan sensitif, sementara jalur baca publik yang disetujui secara eksplisit hanya boleh memakai kebijakan terakhir-yang-dikenal berumur pendek jika model risikonya mengizinkannya. Catat versi kebijakan dan jalur kegagalan mana yang membuat keputusan itu, lalu validasi hasil backend sebelum mengembalikannya.

### Satu endpoint POST

Streamable HTTP modern mengirim setiap pesan JSON-RPC melalui POST:

```text
POST /mcp
Authorization: Bearer <gateway-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.search
Accept: application/json, text/event-stream
```

Gateway dapat mengembalikan JSON atau SSE berlingkup permintaan untuk POST itu. GET dan DELETE mengembalikan 405 untuk permintaan modern. `Mcp-Session-Id` dan `Last-Event-ID` tidak menciptakan wewenang, afinitas, atau perilaku replay.

Nilai header dan body harus sepakat. Tolak ketidakcocokan dengan `-32020` sebelum mencari backend. Ini memungkinkan load balancer, gateway, dan pembatas laju melakukan routing tanpa mem-parse seluruh body sambil menjaga integritas ujung-ke-ujung.

Validasi dalam satu urutan yang tepat: tipe JSON-RPC dan metadata, kesetaraan header dan body, lalu dukungan untuk versi yang cocok. Ketidakcocokan mengembalikan HTTP 400 dengan `-32020`. Jika header dan body sepakat pada versi yang tidak didukung, kembalikan HTTP 400 dengan `-32022` dan `data` persis `{"supported":["2026-07-28"],"requested":"<actual>"}`. Metode yang tidak dikenal mengembalikan HTTP 404 dengan `-32601`.

`ProtocolError` membawa `data` opsional, dan gateway menerialisasikannya ke dalam objek error JSON-RPC. Sebuah notifikasi tidak punya `id`, sehingga tidak pernah menerima sukses atau error JSON-RPC. Notifikasi HTTP yang diterima mengembalikan 202 dengan body kosong.

### Implementasikan discovery di setiap lapisan

Gateway mengimplementasikan `server/discover` untuk klien. Ia juga menemukan setiap backend sehingga mengetahui versi protokol, kapabilitas, dan ekstensi.

Contoh hasil gateway:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": true}
  },
  "ttlMs": 30000,
  "cacheScope": "private",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "enterprise-gateway",
      "version": "2.0.0"
    }
  }
}
```

Iklankan hanya irisan kapabilitas yang dapat dipenuhi gateway secara utuh. Fitur backend belum tentu aman untuk diekspos. Fitur gateway yang tidak punya jalur backend tidak berguna untuk diiklankan.

`serverInfo` adalah data tampilan dan diagnostik yang dilaporkan sendiri. Jangan memakainya sebagai bukti registry atau penerbit.

### Kapabilitas klien per-permintaan

Setiap permintaan yang diteruskan membutuhkan envelope `_meta` yang mutakhir:

```json
{
  "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientCapabilities": {},
  "io.modelcontextprotocol/clientInfo": {
    "name": "enterprise-gateway",
    "version": "1.0.0"
  }
}
```

Jangan menyalin kapabilitas klien luar ke backend secara membabi buta. Gateway adalah klien bagi backend. Iklankan hanya fitur yang akan dimediasi dengan benar oleh gateway.

### Namespace deterministik

Gabungkan tool backend di bawah nama publik yang stabil:

```text
notes.search
notes.create
issues.list
issues.open
```

Simpan pemetaan dari nama publik ke backend dan nama tool asli. Jangan pernah memilih tabrakan pertama atau terakhir. Nama publik adalah bagian dari kontrak persetujuan dan audit, sehingga mengubahnya adalah sebuah migrasi.

`tools/list` harus deterministik. Ketika visibilitas berbeda per prinsipal, kembalikan `cacheScope: private`. `ttlMs` yang berbatas mengurangi beban discovery backend tanpa membiarkan daftar khusus-pengguna bocor ke lintas konteks otorisasi.

Setiap deskriptor tool yang diekspos menyertakan nama stabil, deskripsi, dan `inputSchema` ber-root objek. Namespace tidak dapat menghapus field deskriptor yang wajib. Hasil daftar lengkap juga menyertakan `resultType`, metadata identitas server, dan petunjuk cache.

### Pin deskriptor yang disetujui

Saat admisi, kanoniskalkan deskriptor lengkap dan simpan digest-nya di bawah nama publik yang berkualifikasi. Saat daftar dan panggilan, bandingkan deskriptor hidup dengan digest yang disetujui.

Jika ia berubah:

- Hapus dari `tools/list`.
- Tolak panggilan langsung.
- Pancarkan peristiwa audit.
- Minta kebijakan atau persetujuan ulang manusia sebelum memperbarui pin.

Sebuah gateway adalah titik penegakan pusat yang berguna, tetapi ia tidak mengubah deskriptor yang dilihat pertama kali menjadi aman. Tinjauan awal tetap diperlukan.

### Registry membantu menemukan, bukan memutuskan

Sebuah `server.json` **Registry** menyediakan metadata publikasi. Catatan berbasis paket dapat terlihat seperti ini:

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "com.example/notes",
  "description": "Example notes MCP server.",
  "version": "1.0.0",
  "packages": [
    {
      "registryType": "npm",
      "identifier": "@example/notes-mcp",
      "version": "1.0.0",
      "transport": {"type": "stdio"}
    }
  ]
}
```

Metadata publikasi tidak membawa keputusan keamanan gateway. Simpan bukti penerbit terverifikasi dan provenance dalam state admisi terpisah:

```json
{
  "registryName": "com.example/notes",
  "registryVersion": "1.0.0",
  "publisher": {"namespace": "com.example", "status": "verified"},
  "provenance": {
    "source": "registry.modelcontextprotocol.io",
    "recordId": "com.example/notes@1.0.0"
  },
  "admission": {"status": "approved", "reviewedBy": "gateway-policy"}
}
```

Gateway memeriksa bentuk `server.json` dan menggabungkannya dengan state eksternal itu. Gateway tetap membutuhkan kebijakan admisi.

Untuk setiap backend yang diadmisi, catat:

- Pengenal registry dan catatan yang persis.
- Bukti namespace atau domain penerbit yang terverifikasi.
- Transport dan endpoint yang diizinkan.
- Versi ter-pin atau kebijakan upgrade yang disetujui.
- Digest artefak atau deskriptor.
- Issuer dan sumber daya otorisasi.
- Peninjau, waktu persetujuan, dan kedaluwarsa.

Jangan menerima server hanya karena nama tampilannya menyerupai produk yang familiar. Jangan memperlakukan keberadaan di registry sebagai tinjauan keamanan operasional. Server privat dapat diadmisi melalui skema bukti yang sama bahkan ketika tidak pernah muncul di registry publik.

Pelajaran ini mengimplementasikan jahitan gateway: gabungkan bukti publikasi ke admisi lokal sebelum sebuah backend menjadi bisa dirutekan. [Lesson 30: MCP Registry Supply Chain, Admission, Drift, and Rollback](../../30-mcp-registry-supply-chain-and-drift/docs/en.md) membangun control plane penuh untuk bukti namespace yang persis, provenance artefak, pin yang tak berubah, drift deskriptor langsung, rekonsiliasi status Registry, ledger admisi yang tamper-evident, dan rollback berbasis bukti. Jaga state rantai pasok itu terpisah dari keputusan runtime per-permintaan di atas.

### Mediasi kredensial

Gateway mengautentikasi para peneleponnya dan secara terpisah mengautentikasi diri ke backend. Kredensial backend tidak pernah sampai ke klien.

Jaga pengikatan ini tetap eksplisit:

```text
outer principal -> gateway role and policy
backend issuer + resource -> backend registration and token
```

Jangan pernah meneruskan token gateway luar ke backend. Jangan pernah menggunakan ulang token backend pada issuer atau sumber daya yang berbeda. Jika sebuah tool bertindak atas nama pengguna akhir, pertahankan delegasi itu dengan pertukaran atau model klaim yang dirancang, bukan dengan menyamar sebagai pengguna memakai kredensial layanan bersama.

### Batas laju tanpa sesi

Kunci batas berdasarkan prinsipal terautentikasi, issuer, sumber daya, tool publik, kelas biaya, dan jendela waktu. Id sesi tidak ada dan akan mudah dirotasi bahkan jika ada.

Terapkan validasi murah sebelum mengonsumsi kerja mahal. Tentukan apakah panggilan yang ditolak dihitung terhadap batas penyalahgunaan, kuota bisnis, atau keduanya.

### Audit rantai keputusan

Catat cukup banyak untuk merekonstruksi sebuah panggilan:

- Pengenal permintaan dan trace.
- Prinsipal dan issuer yang diautentikasi.
- Tool publik dan rute backend.
- Versi pin deskriptor.
- Keputusan kebijakan dan alasannya.
- Latensi dan kelas hasil.
- Round MRTR atau pengenal tugas bila berlaku.

Redaksi token bearer, kode otorisasi, refresh token, secret mentah, dan argumen sensitif yang tidak perlu.

### SSE berlingkup permintaan

POST biasa dapat mengembalikan SSE berlingkup permintaan ketika pekerjaan di-stream selama satu permintaan itu. Menutup stream respons membatalkan permintaan HTTP modern yang sedang berjalan itu.

Jangan membuat stream GET terpisah dan jangan menjanjikan replay Last-Event-ID. Itu adalah asumsi transport lama.

### Notifikasi perubahan jangka panjang

Untuk notifikasi perubahan daftar dan sumber daya, klien yang mutakhir mengirim `subscriptions/listen` melalui POST dan menerima respons SSE. Filter notifikasi memakai field datar yang persis `toolsListChanged`, `promptsListChanged`, `resourcesListChanged`, dan `resourceSubscriptions`:

```json
{
  "jsonrpc": "2.0",
  "id": "listen-tools",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Peristiwa pertama mengakui subset yang didukung. Pengenal langganannya adalah id JSON-RPC dari permintaan yang membuka stream itu:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": "listen-tools"
    },
    "notifications": {
      "toolsListChanged": true
    }
  }
}
```

Gateway lalu hanya meneruskan tipe perubahan yang diakui. Setiap notifikasi pada stream itu membawa `io.modelcontextprotocol/subscriptionId` yang sama di `params._meta`. Tidak ada replay otomatis atau re-listen otomatis. Saat koneksi disambung ulang, klien membuka kembali langganan dan menyegarkan daftar yang ia andalkan. Penutupan elegan yang dimulai server mengembalikan hasil lengkap akhir yang ditandai dengan id langganan yang sama.

Jalur modern menggantikan `resources/subscribe`, `resources/unsubscribe`, dan GET streaming mandiri yang tidak diminta. Simpan itu hanya di jalur lama yang dikunci versi.

### MRTR melalui gateway

Ketika sebuah backend mengembalikan `resultType: input_required`, gateway hanya dapat meneruskan hasil itu jika klien luar mendukung permintaan input yang dibutuhkan. Pertahankan `requestState` byte demi byte kecuali gateway secara sengaja mengakhiri dan menerbitkan ulang interaksi itu.

Klien mencoba ulang tool publik asli dengan id JSON-RPC segar dan `inputResponses`. Gateway meng-otorisasi ulang percobaan itu, memeriksa rute publik yang sama, lalu meneruskan permintaan backend yang segar. Ia tidak boleh berasumsi bahwa round sebelumnya memberikan persetujuan tanpa batas.

### Routing ekstensi Tasks

Tasks adalah ekstensi resmi yang diidentifikasi oleh `io.modelcontextprotocol/tasks`. Tasks bukan pengganti sesi inti.

Klien mendeklarasikan ekstensi di dalam kapabilitas klien per-permintaan, dan gateway mengiklankannya di discovery hanya ketika ia dapat mempertahankan lifecycle secara utuh. Untuk `tools/call` yang didukung, backend sendirilah yang memutuskan apakah mengembalikan hasil biasa atau `resultType: task`. Hasil tugas membawa `taskId`, `status`, timestamp, `ttlMs`, dan `pollIntervalMs` opsional langsung di dalam hasil. Tugas harus sudah dapat dibaca secara durable sebelum hasil itu dikirim.

Gateway mencatat prinsipal terautentikasi dan rute backend untuk pengenal tugas yang opaque. Panggilan `tasks/get`, `tasks/update`, dan `tasks/cancel` berikutnya memakai `params.taskId` sebagai `Mcp-Name`, yang memberi perantara sebuah kunci routing. `tasks/get` mengembalikan `resultType: complete` dengan state tugas saat ini dan menisbikan hasil akhir atau error protokol di state terminal. `tasks/update` mengirim `inputResponses` ber-kunci untuk input tugas yang masih tertunda dan mengembalikan pengakuan complete kosong. `tasks/cancel` adalah niat kooperatif dengan pengakuan complete kosong, bukan jaminan bahwa pekerjaan berhenti.

Jangan mengimplementasikan metode `tasks/list` atau `tasks/result` baru. Metode itu milik model eksperimen lama. Sebuah tugas yang membutuhkan input mengekspos permintaan tertanam yang lengkap melalui `tasks/get`; klien menjawabnya melalui `tasks/update`, bukan dengan mencoba ulang panggilan tool asli. Klien tetap poll pada interval yang disarankan; pembuatan tugas tetap diarahkan server.

State rute tugas durable adalah data aplikasi yang di-kunci oleh handle tugas, bukan sesi protokol.

### Batas kompatibilitas

Jika gateway harus melayani klien atau backend yang lebih lama:

- Deteksi eranya secara eksplisit.
- Simpan inisialisasi, sesi transport, stream GET, langganan sumber daya, dan kosakata tugas lama di dalam adapter lama.
- Jangan pernah membocorkan id sesi lama ke routing atau otorisasi modern.
- Lebih suka probe discovery yang berbatas dan kebijakan fallback eksplisit daripada downgrade diam-diam.

```figure
t3-gateway-funnel
```

## Bangun Sendiri

`code/main.py` mengimplementasikan gateway protokol in-process dan dua server backend. Setiap backend menerima permintaan protokol mutakhir yang segar. Gateway menyediakan discovery, `tools/list` deterministik yang difilter pengguna, routing bernamespace, `server.json` **Registry** plus state admisi eksternal, pin deskriptor, RBAC, batas laju ber-kunci prinsipal, keputusan audit, dan pengakuan SSE `subscriptions/listen` yang dimodelkan.

Model menerima body permintaan yang sudah diparse, header routing, dan identitas bearer yang diautentikasi. Ia bukan adapter HTTP lengkap dan tidak mem-parse `Content-Type` atau kontrak `Accept` penuh. Hubungkannya ke adapter Streamable HTTP dari Pelajaran 09, yang mensyaratkan `Content-Type: application/json` dan nilai `Accept` yang memuat baik `application/json` maupun `text/event-stream`.

Jalankan:

```bash
cd phases/13-tools-and-protocols/17-mcp-gateways-and-registries
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Demo mencetak id permintaan luar dan id permintaan backend yang segar sehingga lompatan stateless terlihat.

## Gunakan

Gantikan objek backend in-process dengan klien protokol mutakhir yang nyata. Pertahankan jahitan yang sama:

- Catatan admisi sebelum koneksi.
- Discovery backend sebelum eksposur kapabilitas.
- Nama publik berkualifikasi sebelum otorisasi.
- Pin deskriptor sebelum daftar atau panggilan.
- Metadata per-permintaan yang segar sebelum penerusan.
- Validasi hasil sebelum mengembalikan.

## Rilis

Pelajaran ini mengirimkan `outputs/skill-gateway-bootstrap.md`. Ia menghasilkan desain gateway modern yang mencakup ingress, discovery, admisi, namespace, otorisasi, caching, streaming, langganan, MRTR, Tasks, observabilitas, dan isolasi lama.

## Latihan

1. Tambahkan konteks trace ke metadata permintaan luar dan yang diteruskan, lalu catat korelasinya di peristiwa audit.
2. Tambahkan backend yang mendukung Tasks dan rutekan `tasks/get` berdasarkan task id di `Mcp-Name`.
3. Ubah satu deskriptor backend dan buktikan bahwa discovery maupun panggilan langsung terblokir.
4. Tambahkan kapabilitas server spesifik prinsipal dan jelaskan mengapa discovery harus tetap di-cache secara privat.
5. Tulis antarmuka adapter lama tanpa menambahkan state lama apa pun ke kelas `Gateway` modern.

## Istilah Kunci

| Term | Meaning |
|------|---------|
| MCP gateway | Policy and routing server between clients and backend MCP servers |
| Admission record | Evidence and policy decision allowing one backend into the gateway |
| Qualified tool name | Stable public route such as `notes.search` |
| Descriptor pin | Approved digest checked during discovery and dispatch |
| Private cache scope | Cached result restricted to one authorization context |
| Request-scoped SSE | Streaming response attached to one POST request |
| `subscriptions/listen` | Client-opened SSE stream for selected long-lived change notifications |
| Task route | Application mapping from an opaque task id to its backend |
| Legacy adapter | Explicit version-gated boundary for old handshake and session behavior |

## Bacaan Lanjutan

- [Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [Official Registry server.json requirements](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
