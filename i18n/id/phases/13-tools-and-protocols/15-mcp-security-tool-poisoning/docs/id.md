# MCP Security: Metadata yang Diracuni, Routing, dan State MRTR

> Tanpa-state tidak berarti tanpa-kepercayaan. Itu berarti setiap request mengekspos bukti yang dibutuhkan server dan gateway untuk memvalidasi panggilan itu secara independen.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Perlakukan deskripsi tool, anotasi, informasi klien, dan informasi server sebagai data yang tidak tepercaya.
- Deteksi keracunan metadata, perubahan deskriptor, dan tabrakan nama lintas-server.
- Validasi metadata request 2026-07-28 dan header routing Streamable HTTP.
- Lindungi `requestState` MRTR dari perubahan dan ikatkan konfirmasi ke argumen yang tepat.
- Terapkan otorisasi dan batas rate ke principal, bukan ke sesi protokol yang dihapus.

## Masalahnya

Sebuah model membaca deskripsi tool untuk memutuskan apa yang akan dipanggil. Sebuah router membaca nama tool untuk memutuskan ke mana mengirim request. Sebuah pengguna membaca label untuk memutuskan apa yang akan disetujui. Satu deskriptor berbahaya bisa menyasar ketiganya.

Panduan keamanan MCP resmi langsung: deskripsi dan anotasi seharusnya diperlakukan sebagai tidak tepercaya kecuali berasal dari server yang tepercaya. Bahkan saat itu, kepercayaan deployment bisa berubah. Sebuah update server, paket yang disusupi, kesalahan registry, atau merge gateway bisa mengubah apa yang dilihat model.

Protokol saat ini juga mengubah batas keamanan. Di 2026-07-28 tidak ada handshake inti dan tidak ada sesi transport. Sebuah desain keamanan yang mengunci persetujuan, batas rate, atau riwayat audit hanya berdasarkan `Mcp-Session-Id` bukanlah desain saat ini.

## Konsepnya

### Tujuh permukaan serangan yang layak diperiksa

Gunakan daftar konkret alih-alih instruksi kabur untuk berhati-hati.

1. **Keracunan metadata.** Sebuah deskripsi berisi instruksi yang tidak terkait dengan perilaku tool yang dideklarasikan.
2. **Rug pull deskriptor.** Sebuah nama, deskripsi, skema, atau anotasi yang sebelumnya disetujui berubah.
3. **Shadowing lintas-server.** Dua backend mengekspos nama tool yang sama tanpa kualifikasi dan routing memilih salah satunya secara diam-diam.
4. **Kebingungan header dan body.** `Mcp-Method` atau `Mcp-Name` tidak sepakat dengan request JSON-RPC.
5. **Eskalasi kapabilitas.** Sebuah peer mengklaim ekstensi atau fitur klien dan server salah mengartikan deklarasi itu sebagai otorisasi.
6. **Perubahan state MRTR.** Sebuah klien mengubah `requestState`, menjawab pertanyaan yang berbeda, atau memakai ulang konfirmasi dengan argumen yang berbeda.
7. **Kebingungan identitas rantai-pasok.** Sebuah nama tampilan yang familiar diperlakukan sebagai bukti identitas penerbit atau server.

Permukaan-permukaan ini saling tumpang tindih. Hash pinning membantu terhadap perubahan deskriptor tetapi tidak membuktikan bahwa deskriptor pertama aman. Pemindaian statis menangkap frasa yang jelas tetapi bukan instruksi halus. Namespacing mencegah satu kelas tabrakan tetapi bukan server ber-namespace yang berbahaya. Tumpuk kontrolnya.

### Envelope request saat ini adalah bukti, bukan identitas

Setiap request 2026-07-28 berisi:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "elicitation": {"form": {}}
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "security-lab",
      "version": "1.0.0"
    }
  }
}
```

Validasi versi dan bentuk kapabilitas pada setiap request. Pakai kapabilitas untuk memilih bentuk respons yang kompatibel. Jangan memakai `clientInfo` sebagai principal yang diautentikasi. Itu dilaporkan sendiri.

Peringatan yang sama berlaku untuk `io.modelcontextprotocol/serverInfo` di metadata hasil. Itu berguna untuk log dan debugging. Itu bukan sertifikat, bukti registry, atau keputusan otorisasi.

### Validasi routing sebelum kebijakan

Untuk `tools/call`, Streamable HTTP menyertakan:

```text
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.export
```

Method header harus sama dengan method body. Nama header harus sama dengan `params.name`. Tolak ketidaksepakatan dengan `-32020` sebelum memilih backend, menerapkan RBAC, atau mengonsumsi token batas-rate.

Urutan ini menutup ambiguitas yang umum: satu komponen mengotorisasi body sementara yang lain me-route berdasarkan header.

Validasi wire mengikuti satu urutan yang tepat. Validasi tipe JSON-RPC dan metadata, bandingkan nilai header dengan body, lalu periksa apakah versi yang cocok didukung. Header yang tidak cocok mengembalikan HTTP 400 dengan `-32020`. Jika header dan body sepakat pada versi yang tidak didukung, kembalikan HTTP 400 dengan `-32022` dan `data` persis `{"supported":["2026-07-28"],"requested":"<actual>"}`. Method yang tidak diketahui mengembalikan HTTP 404 dengan `-32601`.

Setiap objek error menyertakan `data` opsional ketika kontrak memerlukan informasi pemulihan terstruktur. Sebuah notifikasi tidak memiliki `id`, sehingga ia tidak pernah menerima respons sukses atau error JSON-RPC. Sebuah notifikasi HTTP yang diterima mengembalikan 202 dengan body kosong.

### Pin seluruh deskriptor

Hash deskripsi saja melewatkan perubahan skema dan anotasi. Kanonikalisasi dan hash field deskriptor yang disetujui pengguna:

```python
normalized = json.dumps(tool, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(normalized.encode()).hexdigest()
```

Simpan digest di bawah kunci berkualifikasi seperti `notes.export`, bersama dengan bukti penerbit dan waktu persetujuan di luar contoh main-main ini.

Pada setiap refresh:

- Kunci tidak diketahui: karantina sampai ditinjau.
- Kunci sama, digest berbeda: karantina sebagai rug pull sampai disetujui ulang.
- Nama tanpa kualifikasi yang duplikat: wajibkan namespacing deterministik.
- Kena scanner: blokir dan tinjau deskriptor lengkapnya.

Kesetaraan hash membuktikan kestabilan, bukan keamanan. Sebuah deskriptor yang diracuni tetap teracuni saat dipin dengan sempurna.

### Pemindaian statis adalah tripwire

Pola sederhana bisa menandai tag perintah, override instruksi, penyembunan, akses rahasia, dan tujuan jaringan yang tersamar. Biayanya cukup murah untuk waktu instalasi dan CI.

Itu bukan bukti semantik. Sebuah deskripsi aman bisa berisi frasa yang ditandai di dalam peringatan yang sah. Sebuah deskripsi berbahaya bisa menghindari setiap frasa. Perlakukan keluaran scanner sebagai bukti tinjauan, bukan skor kesalahan otomatis.

### Namespace sebelum menggabungkan

Misalkan dua server sama-sama mengekspos `search`. Jangan pernah membiarkan urutan discovery memutuskan mana yang menang.

```text
notes.search
issues.search
```

Nama berkualifikasi adalah nama gateway publik. Catat pemetaan backend secara terpisah. Nama yang stabil membuat persetujuan, audit, hash pin, dan routing `Mcp-Name` merujuk ke objek yang sama.

### Kapabilitas adalah deklarasi kompatibilitas

`clientCapabilities` per-request memberi tahu server fitur protokol apa yang bisa diproses klien. Itu tidak memberikan klien akses ke tools, data, atau aksi.

Otorisasi tetap berasal dari principal yang diautentikasi dan kebijakan resource. Urutannya adalah:

1. Autentikasi kredensial transport.
2. Validasi versi, header, dan bentuk request.
3. Periksa kompatibilitas kapabilitas.
4. Otorisasikan principal, tool, resource, dan argumen.
5. Eksekusi atau minta input pengguna.

### Lindungi konfirmasi MRTR tanpa-state

Sebuah tool yang berkonsekuensi mungkin memerlukan konfirmasi pengguna. MCP saat ini memakai Multi Round-Trip Requests sebagai pengganti callback server-ke-klien.

Respons pertama:

```json
{
  "resultType": "input_required",
  "inputRequests": {
    "confirm": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Export notes to archive?",
        "requestedSchema": {
          "type": "object",
          "properties": {
            "confirm": {"type": "boolean"}
          },
          "required": ["confirm"]
        }
      }
    }
  },
  "requestState": "opaque-integrity-protected-value"
}
```

Klien memperoleh input dan mencoba ulang method asli dengan id JSON-RPC baru:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes.export",
    "arguments": {"query": "private", "destination": "archive"},
    "requestState": "opaque-integrity-protected-value",
    "inputResponses": {
      "confirm": {
        "action": "accept",
        "content": {"confirm": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

Setiap nilai `inputRequests` adalah request tertanam lengkap dengan `method` dan `params`. Kuncinya harus cocok dengan entri yang sesuai di `inputResponses`. Sebuah elicitation form memakai `requestedSchema` berakar-objek, dan klien harus sudah mendeklarasikan kapabilitas elicitation form sebelum server memintanya.

Kapabilitas saat ini memiliki dua deklarasi form yang valid. `{"elicitation":{}}` secara implisit mendukung elicitation form, sementara `{"elicitation":{"form":{}}}` menyatakannya secara eksplisit. Deklarasi URL-saja seperti `{"elicitation":{"url":{}}}` tidak mendukung request form. Server mengembalikan HTTP 400 dengan `-32021` dan `data.requiredCapabilities` sama dengan `{"elicitation":{"form":{}}}`.

Perlakukan `requestState` sebagai input yang bersifat permusuhan. Tandatangani atau enkripsi, validasi, dan ikatkan ke method, tool, argumen yang tepat, tujuan, kedaluwarsa, principal, dan nonce sekali-pakai ketika replay menjadi masalah. Kode pelajaran memakai HMAC dan pencocokan argumen yang tepat untuk membuat batas itu terlihat.

Buku besar nonce tidak boleh hidup di dalam satu objek gateway. Model yang bisa dijalankan menyuntikkan replay store ber-TTL dan berbatas yang bisa dibagikan oleh beberapa instance gateway. Klaim atomiknya adalah batas eksekusi: hanya penerimaan yang tervalidasi atau penolakan terminal yang eksplisit yang mengonsumsi state. Respons yang rusak format atau `cancel` tidak mengeksekusi apa pun dan tetap bisa dicoba ulang sampai kedaluwarsa. Armada produksi memerlukan klaim kondisional yang sama di penyimpanan tahan lama bersama.

Jangan menyimpan konteks konfirmasi tersembunyi di sesi protokol. Instance server mana pun seharusnya bisa memvalidasi percobaan ulang.

### Rule of two untuk panggilan berisiko tinggi

Klasifikasikan sebuah panggilan di sepanjang tiga sumbu:

- Ia mengonsumsi input yang tidak tepercaya.
- Ia bisa mengakses data sensitif.
- Ia menyebabkan aksi eksternal yang berkonsekuensi.

Satu langkah otomatis seharusnya tidak menggabungkan ketiganya. Pecah, kurangi privilige, atau minta input pengguna yang eksplisit melalui MRTR. Ini adalah heuristik desain, bukan kapabilitas protokol.

### Kurangi otoritas sebelum eksekusi

Tanpa-state saja bukan keamanan. Itu menghapus riwayat protokol yang tersembunyi, tetapi sebuah request self-contained masih bisa meminta handler yang terlalu berkuasa untuk membocorkan data atau membuat perubahan tak dapat dipulihkan. Keamanan berasal dari mengurangi otoritas di setiap batas:

1. **Verb bertipe.** Ekspos satu operasi berbatas seperti `archive_note`, bukan tool generik `run` atau `request` yang bisa mengekspresikan kekuatan yang tidak terkait.
2. **Argumen tervalidasi.** Gunakan skema tertutup bila memungkinkan, tolak field yang tidak dikenal, normalisasi pengenal sekali, batasi ukuran, dan validasi tujuan, tenant, dan kepemilikan resource sebelum evaluasi kebijakan.
3. **Otorisasi saat ini.** Ikatkan principal yang diautentikasi ke verb, resource, environment, dan argumen yang ternormalisasi. Anotasi tool dan kapabilitas klien tidak memberikan otoritas ini.
4. **Persetujuan terikat-aksi.** Untuk panggilan yang berkonsekuensi, ikatkan persetujuan ke digest dari verb bertipe dan argumen ternormalisasi, plus principal, kedaluwarsa, dan kebijakan sekali-pakai. Field apa pun yang berubah memerlukan keputusan baru.
5. **Penolakan kelas-utama.** Modelkan deny, persetujuan yang kedaluwarsa, penolakan pengguna, dan tujuan yang tidak aman sebagai hasil biasa yang tidak mengeksekusi efek samping apa pun. Jangan menerjemahkan penolakan menjadi tool fallback yang lebih lemah.
6. **Bukti audit tersensor.** Catat siapa yang meminta, deskriptor dan versi kebijakan mana yang diakui, tujuan ternormalisasi apa yang diotorisasi, mengapa keputusan mengizinkan atau menolak, dan apakah eksekusi dimulai. Simpan digest atau nilai tersensor alih-alih rahasia.

Setiap langkah mempersempit apa yang boleh dilakukan komponen berikutnya. Handler akhir seharusnya menerima perintah domain yang sudah tervalidasi, bukan teks model mentah plus kredensial yang luas. Ulangi seluruh rantai pada percobaan ulang MRTR, update task, atau panggilan yang di-forward gateway. Persetujuan sebelumnya tidak mengubah request berikutnya menjadi lalu lintas sesi yang tepercaya.

### Jalur interaksi saat ini dan era lama

Roots, Sampling, dan Logging dideprecate untuk implementasi 2026-07-28 yang baru. Sebuah gateway boleh mempertahankan kode kanal-request lama hanya sebagai jalur kompatibilitas yang digerbang-versi.

Jangan membangun pertahanan baru di sekitar pembatas sampling per-sesi. Terapkan kuota ke principal yang diautentikasi, issuer, resource, tool, dan jendela waktu. Untuk pekerjaan interaktif saat ini, periksa input request dan respons MRTR.

### Pemeriksaan transport tanpa-state

- Terima pesan MCP modern di endpoint POST tunggal.
- Kembalikan 405 untuk GET dan DELETE modern.
- Jangan membuat atau bergantung pada `Mcp-Session-Id`.
- Abaikan header sesi dan replay lama sebagai input otoritas.
- Kembalikan JSON atau SSE berlingkup request untuk POST itu.
- Gunakan `subscriptions/listen` hanya untuk notifikasi perubahan jangka panjang yang dipilih ikut.

```figure
tp-tool-poisoning
```

## Bangun Sendiri

`code/main.py` mengimplementasikan model gateway keamanan in-process kecil. Ia mengkanonikalisasi dan memin seluruh deskriptor tool, melaporkan keracunan metadata dan shadowing, memvalidasi envelope request modern dan nilai routing, dan melakukan ekspor terkonfirmasi dua-ronde dengan `requestState` yang ditandatangani dan replay store bersama yang disuntikkan.

Model dimulai setelah adaptor HTTP mem-parse body JSON dan header routing. Ia tidak memvalidasi `Content-Type` atau `Accept`. Sambungkan dispatcher yang sama ke adaptor Streamable HTTP lengkap dari Lesson 09, yang mewajibkan `Content-Type: application/json` dan nilai `Accept` berisi `application/json` dan `text/event-stream`.

Jalankan:

```bash
cd phases/13-tools-and-protocols/15-mcp-security-tool-poisoning
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Contoh ini secara sengaja memutasi sebuah deskriptor. Scanner dan perbandingan digest menghasilkan temuan yang independen. Ekspor lalu mendemonstrasikan respons `input_required` dan percobaan ulang tanpa-state.

## Gunakan

Ganti `SAFE_TOOLS` dengan cuplikan ternormalisasi dari server yang sudah kamu setujui. Jauhkan kredensial dan rahasia dari cuplikan itu. Tinjau setiap deskriptor baru atau yang berubah sebelum memperbarui digest-nya.

Di sebuah gateway, jalankan pemeriksaan yang sama saat discovery dan sekali lagi sebelum dispatch. Sebuah cache bisa mengurangi pekerjaan discovery, tetapi persetujuan yang di-cache harus kedaluwarsa atau di-invalidate ketika deskriptornya berubah.

## Rilis

Pelajaran ini mengirimkan `outputs/skill-mcp-threat-model.md`. Ia menghasilkan threat model protokol saat ini di sepanjang batas metadata, routing, kapabilitas, otorisasi, MRTR, caching, registry, dan kompatibilitas.

## Latihan

1. Ikatkan principal yang diautentikasi dan keputusan otorisasi saat ini ke state MRTR yang disegel, lalu tolak percobaan ulang di bawah principal yang berbeda.
2. Ganti replay store di memori dengan insert kondisional yang persisten dan buktikan dua proses tidak bisa sama-sama mengklaim satu nonce.
3. Suntikkan sebuah kegagalan setelah klaim replay tetapi sebelum ekspor yang disimulasikan. Definisikan dan uji aturan transaksi atau idempotensi yang membuat pemulihan aman.
4. Ubah `inputSchema` sebuah tool tanpa mengubah deskripsinya. Konfirmasikan pinning seluruh deskriptor menangkapnya.
5. Tambahkan kebijakan yang menolak caching publik ketika `tools/list` berbeda per principal.
6. Modelkan server yang lebih lama di belakang gateway. Letakkan semua perilaku handshake dan sesi di belakang cabang kompatibilitas `2025-11-25` yang eksplisit.

## Istilah Kunci

| Term | Meaning |
|------|---------|
| Metadata poisoning | Instructions or deceptive claims embedded in a tool descriptor |
| Rug pull | Change to a previously approved descriptor |
| Tool shadowing | Ambiguous routing caused by duplicate unqualified names |
| Header mismatch | Routing header and JSON-RPC body disagreement, error `-32020` |
| Hash pin | Digest of the complete approved descriptor |
| MRTR | Stateless response and retry pattern for server-requested input |
| `requestState` | Opaque round-trip value that must be treated as untrusted input |
| Capability declaration | Statement of protocol compatibility, not authorization |
| Implicit form support | An empty `elicitation` capability object, equivalent to form support |
| Qualified tool name | Stable gateway name such as `notes.search` |

## Bacaan Lanjutan

- [MCP security and trust guidance](https://modelcontextprotocol.io/specification/2026-07-28#security-and-trust--safety)
- [Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Deprecated features](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
