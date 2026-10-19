# Kontrak Tool MCP dan Konten

> Sebuah tool aman untuk diotomasi hanya ketika penemuan, argumen, hasil, paginasi, dan metadata transport setuju pada satu kontrak.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 07, 09, and 10
**Time:** ~120 minutes

## Learning Objectives

- Mendefinisikan input dan output tool dengan JSON Schema 2020-12.
- Memvalidasi hasil terstruktur tanpa berasumsi bahwa mereka adalah objek JSON.
- Memilih di antara teks, gambar, audio, tautan resource, dan resource tertanam.
- Menolak definisi `x-mcp-header` yang tidak aman sebelum tool mencapai model.
- Meng-encode nilai parameter-header dan memverifikasi parity header-ke-body yang persis.
- Menelusuri paginasi cursor tanpa menginterpretasikan nilai cursor.
- Membatasi dan mengotorisasi saran `completion/complete`.

## Masalahnya

Memanggil fungsi Python itu mudah. Memanggil kapabilitas remote melalui AI host itu masalah kontrak.

Server memublikasikan sebuah deskriptor. Klien mengubah deskriptor itu menjadi konteks model dan antarmuka pengguna. Model membuat argumen. Sebuah gateway bisa merutekan permintaan dari header yang di-mirror. Server mengeksekusi tool. Klien kemudian memutuskan apakah hasil itu cukup aman dan valid untuk dikembalikan ke model.

Satu batas yang lemah merusak seluruh rantai.

Pertimbangkan lima kegagalan:

- Deskriptor menyatakan hasil adalah objek, tetapi server mengembalikan array.
- Klien menghentikan paginasi ketika `nextCursor` adalah string kosong.
- Sebuah parameter token di-mirror ke header HTTP dan menjadi terlihat oleh perantara.
- Sebuah nilai routing Unicode dikirim sebagai header mentah, lalu gateway dan origin menginterpretasikan byte yang berbeda.
- Sebuah endpoint completion menyarankan environment produksi kepada pemanggil yang tidak bisa mengaksesnya.

Tidak ada kegagalan ini yang diperbaiki oleh prompting yang lebih baik. Mereka membutuhkan kontrak protokol dan aplikasi yang eksplisit.

## Pipeline Kontrak

Perlakukan setiap panggilan tool sebagai lima gerbang:

1. **Discover.** Baca daftar tool berpaginasi yang deterministik.
2. **Admit.** Validasi setiap deskriptor dan terapkan kebijakan keamanan lokal.
3. **Invoke.** Validasi argumen dan bangun metadata transport.
4. **Execute.** Jalankan handler dan klasifikasikan kegagalan dengan benar.
5. **Consume.** Validasi blok konten dan output terstruktur sebelum penggunaan model.

```figure
mcp-contract-pipeline
```

Host memiliki gerbang admission dan konsumsi. Server tidak bisa memaksa klien mempercayai anotasi, schema, atau output-nya.

## JSON Schema adalah Batas Runtime

Di MCP `2026-07-28`, `inputSchema` dan `outputSchema` memakai JSON Schema. Ketika `$schema` tidak ada, dialect default adalah 2020-12.

Schema input harus berupa objek schema. Sebuah tool tanpa argumen tetap harus menyatakan persis apa yang diterimanya:

```json
{
  "type": "object",
  "additionalProperties": false
}
```

Ini lebih ketat daripada `{ "type": "object" }`, yang menerima properti sembarang.

Schema output bersifat opsional. Setelah server memublikasikannya, setiap hasil tool lengkap berkomitmen untuk mengembalikan `structuredContent` yang patuh, termasuk hasil dengan `isError: true`. Flag error mengklasifikasikan hasil eksekusi; ia tidak memaafkan kontrak output yang dipublikasikan. Klien seharusnya memvalidasi hasil alih-alih mempercayai deskriptor.

### Konten terstruktur adalah nilai JSON apa pun

Jangan hard-code `structuredContent` sebagai dictionary. Ia bisa berupa:

- sebuah objek;
- sebuah array;
- sebuah string;
- sebuah number;
- sebuah boolean;
- `null`.

Tool ini mengembalikan array:

```json
{
  "name": "tag_catalog",
  "inputSchema": {
    "type": "object",
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "array",
    "items": {"type": "string"}
  }
}
```

Hasil suksesnya valid:

```json
{
  "resultType": "complete",
  "content": [
    {
      "type": "text",
      "text": "[\"contracts\", \"mcp\", \"stateless\"]"
    }
  ],
  "structuredContent": ["contracts", "mcp", "stateless"],
  "isError": false
}
```

Untuk kompatibilitas, hasil terstruktur juga seharusnya menyertakan JSON terserialisasi dalam blok teks. Teks itu bukan sumber validasi. `structuredContent` yang menjadi sumber validasi.

### Validator kecil tetap mengajarkan batasnya

Pelajaran memakai subset JSON Schema yang disengaja karena ia tetap di dalam Python standard library. Ia memeriksa mekanisme yang dipakai tool contoh:

- tipe object, array, string, integer, number, boolean, dan null;
- properti yang wajib;
- `additionalProperties: false`;
- item array;
- nilai enum;
- panjang string minimum.

Ini bukan pengganti untuk validator produksi lengkap. Pelajaran yang dapat digunakan ulang adalah di mana validasi terjadi: setelah penemuan untuk deskriptor, sebelum eksekusi untuk argumen, dan sebelum konsumsi untuk hasil terstruktur.

## Blok Konten Membawa Biaya Berbeda

Array `content` bisa menggabungkan beberapa jenis konten.

| Jenis | Gunakan untuk | Batas utama |
|------|------------|---------------|
| `text` | Ringkasan yang dapat dibaca manusia dan model | Perlakukan teks sebagai output tidak tepercaya |
| `image` | Bukti visual yang di-encode base64 | Validasi media type dan ukuran |
| `audio` | Output lisan atau terekam yang di-encode base64 | Validasi media type dan batas durasi |
| `resource_link` | URI yang mungkin di-fetch klien nanti | Otorisasi-ulang pembacaan sumber daya nanti |
| `resource` | Data yang tertanam langsung di hasil | Terapkan batas payload dan konten sekarang |

Tautan resource bukan bukti bahwa resource muncul di `resources/list`. Itu referensi yang dikembalikan oleh panggilan tool ini. Klien tetap menerapkan kebijakan resource-nya ketika mengikuti URI.

Resource tertanam menghindari round trip lain tetapi meningkatkan ukuran respons saat ini. Gunakan tautan untuk artifact besar atau yang berubah secara independen. Gunakan resource tertanam untuk bukti kecil yang harus berpergian secara atomik dengan hasil.

Hasil `evidence_bundle` pelajaran menyertakan semua lima jenis. Klien memvalidasi setiap blok sebelum menerima hasil.

## `x-mcp-header` adalah Metadata Routing

Sebuah properti di dalam `inputSchema` bisa mendeklarasikan `x-mcp-header`. Melalui Streamable HTTP, klien meng-mirror argumen itu ke `Mcp-Param-{name}`.

```json
{
  "region": {
    "type": "string",
    "x-mcp-header": "Region"
  }
}
```

Dengan `region: "eu-west"`, transport bisa memancarkan:

```http
Mcp-Param-Region: eu-west
```

Anotasi itu ada agar load balancer, gateway, atau mesin kebijakan bisa merutekan tanpa mengurai body JSON. Itu bukan tempat untuk menaruh kredensial.

Protokol membatasi anotasi itu:

- nama header non-kosong dan mengikuti sintaks token nama-field HTTP;
- nama header unik tanpa memandang huruf besar-kecil;
- tipe properti adalah string, integer, atau boolean;
- `number` tidak diizinkan;
- anotasi hanya muncul pada anggota langsung dari `inputSchema.properties`;
- nilai integer tetap di dalam `-9007199254740991` hingga `9007199254740991`.

Aturan lokasi bersifat sintaktis dan fail-closed. Telusuri seluruh pohon schema, bukan hanya properti yang kebetulan dipahami validator-mu. Tolak anotasi di bawah `properties` objek bersarang, cabang `oneOf`, `items`, sebuah definisi yang dicapai melalui `$ref`, atau schema output apa pun. Menyelesaikan sebuah referensi tidak mengubah node yang dirujuk menjadi properti top-level langsung.

Pelajaran ini menambahkan kebijakan penempatan: tolak deskriptor yang meng-mirror nama seperti `password`, `secret`, `token`, `api_key`, atau `authorization`. Spesifikasi resmi menyarankan penulis server untuk tidak meng-mirror parameter sensitif. Klien bisa mengubah saran itu menjadi aturan admission keras.

Audit nama header, bukan nilainya. Kode contoh mencatat `Mcp-Param-Region` sambil menjaga `eu-west` keluar dari event audit.

### Encode nilai sebelum membangun header HTTP

Nilai parameter boleh berpergian sebagai teks polos hanya ketika itu string non-kosong dari karakter ASCII terlihat dari `!` hingga `~` dan tidak menyerupai sentinel encoding. Semuanya lagi memakai bentuk persis ini:

```text
=?base64?{Base64UTF8}?=
```

`Base64UTF8` adalah base64 standar di atas byte UTF-8 yang persis. Jangan memangkas, menormalisasi, atau mengganti nilai terlebih dahulu. Encode Unicode, string kosong, spasi, tab, karakter kontrol, CR atau LF, whitespace di awal atau akhir, dan nilai apa pun yang diawali `=?base64?`. Meng-encode nilai yang tampak-sentinel lagi adalah yang memungkinkan penerima memulihkan teks literal asli alih-alih meng-decode-nya sebagai sintaks transport.

Boolean dirender sebagai `true` atau `false` huruf kecil. Integer dirender dalam basis 10 dan harus tetap di dalam rentang integer aman JavaScript. Nilai di luar rentang itu ditolak alih-alih dibulatkan oleh perantara.

### Server memeriksa salinan yang di-mirror

Pembangunan header hanya separuh klien. Di batas Streamable HTTP, server harus:

1. menemukan nama `Mcp-Param-*` yang dikenali tanpa memandang huruf besar-kecil nama header;
2. meng-decode bentuk sentinel base64 persis ketika ada;
3. membandingkan teks yang di-decode dengan argumen body JSON yang sesuai secara persis;
4. menolak header yang dikenali yang hilang, ganda, tidak terduga, rusak, atau tidak cocok sebelum dispatch.

Penolakan itu adalah HTTP `400` dengan kode kesalahan JSON-RPC `-32020`. Baik nilai body maupun bentuk header ter-encode-nya tidak termasuk dalam catatan audit. Catat nama header yang dikenali dan kategori penolakannya saja.

`code/main.py` memodelkan batas ini secara langsung. [Pelajaran 09](../../09-mcp-transports/) membahas urutan validasi Streamable HTTP yang lebih luas, termasuk parity method dan protocol-version.

## Kursor Paginasi Bersifat Opaque

Operasi daftar MCP memakai paginasi cursor. Server memilih ukuran halaman dan format cursor. Klien mendapat satu keputusan:

```python
if result.get("nextCursor") is None:
    break
cursor = result["nextCursor"]
```

Jangan menulis ini:

```python
if not result.get("nextCursor"):
    break
```

String kosong adalah cursor yang valid. Truthiness akan berhenti terlalu awal.

Klien tidak boleh meng-decode cursor, menambahkannya, membandingkannya dengan cursor sebelumnya untuk pengurutan, atau menyimpulkan nomor halaman. Server bisa menandatangani cursor, mengikatnya ke versi katalog, atau memetakannya ke state privat. Itu detail implementasi server.

Server contoh secara sengaja mengembalikan `""` setelah halaman pertama. Klien harus mengirim nilai persis itu pada permintaan kedua. Trace-nya adalah:

```text
<first request with no cursor>
<second request with cursor "">
```

Cursor yang tidak valid menghasilkan JSON-RPC invalid params, kode `-32602`.

## Completion adalah Permukaan Otorisasi

`completion/complete` menyediakan saran untuk argumen prompt dan argumen template-resource. Ia berguna untuk bentuk interaktif, tetapi ia bisa membocorkan nama yang dilindungi metode daftar biasa.

Permintaan completion menyebut sebuah referensi dan argumen yang sedang di-complete:

```json
{
  "method": "completion/complete",
  "params": {
    "ref": {
      "type": "ref/prompt",
      "name": "deployment_review"
    },
    "argument": {
      "name": "environment",
      "value": "st"
    }
  }
}
```

Hasilnya mengembalikan paling banyak 100 nilai dan mungkin melaporkan `total` plus `hasMore`.

Terapkan batas otorisasi yang sama yang dipakai prompt atau resource yang dirujuk. Seorang analis dalam contoh menerima `development` dan `staging`. Hanya operator yang bisa menerima `production`.

Completion produksi juga membutuhkan:

- validasi input;
- filtering sadar-pemanggil;
- request debouncing di klien;
- rate limiting di server;
- jumlah hasil yang berbatas;
- log yang tidak mengekspos nilai saran sensitif.

Completion adalah bantuan, bukan bypass penemuan.

## Dua Lapisan Kesalahan

Pisahkan kesalahan protokol dari kesalahan eksekusi tool.

Gunakan kesalahan JSON-RPC ketika permintaan MCP tidak bisa di-dispatch dengan benar:

- nama tool tidak dikenal;
- bentuk permintaan rusak;
- metadata permintaan hilang;
- cursor tidak valid.

Gunakan hasil tool lengkap dengan `isError: true` ketika invokasi mencapai tool dan tool melaporkan kegagalan yang dapat ditindaklanjuti:

- sebuah sumber laporan tidak tersedia;
- sebuah tanggal di luar rentang yang didukung;
- sebuah aturan bisnis menolak operasi yang diminta.

Model sering bisa memperbaiki kesalahan eksekusi tool. Mereka tidak bisa memperbaiki server yang melanggar schema output-nya sendiri.

Jika tool mendeklarasikan schema output, modelkan kegagalan yang dapat ditindaklanjuti di dalam schema itu. Kegagalan `route_report` contoh mengembalikan region yang dimintanya dengan `accepted: false`, bersama teks kesalahan yang dapat dibaca manusia dan `isError: true`.

## Bangun Sendiri

`code/main.py` membangun kedua sisi batas dengan Python standard library.

Server mengimplementasikan:

- validasi metadata MCP per-permintaan;
- `server/discover` dengan kapabilitas tools dan completions;
- paginasi `tools/list` deterministik;
- empat deskriptor tool, termasuk satu yang harus ditolak;
- output terstruktur array;
- setiap jenis blok konten tool saat ini;
- gerbang parity Streamable HTTP yang meng-decode header parameter yang dikenali dan mengembalikan HTTP `400` plus JSON-RPC `-32020` pada ketidakcocokan;
- completion yang diotorisasi dan ber-rate-limit.

Klien mengimplementasikan:

- admission deskriptor;
- validasi penempatan `x-mcp-header` seluruh-pohon dan kebijakan field sensitif;
- encoding nilai ASCII-terlihat-polas atau base64 UTF-8 yang persis;
- loop cursor opaque yang mengikuti string kosong;
- validasi argumen dan hasil;
- validasi blok-konten;
- event audit header yang berisi nama tetapi bukan nilai.

Deskriptor yang sengaja tidak aman adalah data pengajaran. Ia membuktikan bahwa satu tool yang ditolak tidak mencegah tool yang valid dimuat.

## Gunakan

Dari root repositori:

```bash
cd phases/13-tools-and-protocols/28-mcp-tool-contracts-and-content/code
python3 main.py
python3 -m unittest discover tests -v
```

Demo mencetak tool yang diadmit, deskriptor yang ditolak, kedua permintaan paginasi, konten array terstruktur, jenis blok-konten, nama header yang di-mirror, apakah nilai membutuhkan encoding, status parity HTTP, dan nilai completion yang difilter-pemanggil.

## Interactive Lab

Buka `code/main.py` dan temukan `TOOLS`.

1. Ubah `tag_catalog.outputSchema.type` dari `array` ke `object`.
2. Jalankan demo. Klien seharusnya menolak array yang dikembalikan.
3. Pulihkan schema-nya.
4. Pertahankan `nextCursor` halaman pertama sebagai `""`, lalu buat halaman terakhir mengembalikan `nextCursor: None` alih-alih menghilangkan field-nya.
5. Jalankan test dan bandingkan trace cursor-nya.
6. Tambahkan `x-mcp-header: "Authorization"` ke properti string.
7. Konfirmasi admission deskriptor menolaknya sebelum invokasi.
8. Coba nilai `region` yang berisi Unicode, newline, spasi di sekeliling, dan teks literal `=?base64?SGVsbG8=?=`. Decode setiap header yang dipancarkan dan buktikan nilai aslinya selamat persis.
9. Pindahkan anotasi ke bawah `oneOf`, `items`, atau definisi `$ref`. Konfirmasi setiap deskriptor ditolak bahkan jika cabang itu tidak pernah dipakai demo.
10. Hapus header yang dikenali atau ubah nilai ter-decode-nya. Konfirmasi batas HTTP mengembalikan status `400` dan kode JSON-RPC `-32020`.

Poinnya bukan menghafal bentuk JSON. Poinnya adalah mengamati setiap gerbang gagal di batas yang memilikinya.

## Practice Lab

Perluas lab kontrak dengan tool `search_evidence`.

Persyaratan:

1. Schema input-nya menerima `query`, `limit`, dan field routing `region` yang aman.
2. Schema output-nya adalah array objek dengan `uri`, `title`, dan `score`.
3. Hasilnya menyertakan teks kompatibilitas dan tautan resource per item.
4. Argumen menolak properti yang tidak dikenal.
5. `limit` dibatasi oleh validasi aplikasi.
6. Seorang pemanggil tanpa akses ke satu URI tidak pernah melihat URI itu melalui completion atau output tool.
7. Test menyertakan score yang tidak patuh, anotasi header yang tidak valid, dan daftar dua-halaman.
8. Test nilai-header mencakup ASCII terlihat, Unicode, karakter kontrol, whitespace, teks yang tampak-sentinel, dan kedua batas integer aman-JavaScript.
9. Fixture HTTP menerima nama header case-insensitive tetapi menolak nilai yang dikenali yang hilang atau tidak cocok dengan status `400` dan kode `-32020`.

## Shipped Artifact

`outputs/skill-mcp-contract-reviewer.md` adalah skill review datar yang dapat digunakan ulang. Berikan padanya deskriptor tool, hasil contoh, perilaku paginasi, dan kebijakan completion. Ia mengembalikan keputusan admission, rencana validasi hasil, kebijakan header, dan test kegagalan konkret.

## Verify It

Pelajaran ini selesai ketika pernyataan berikut benar:

- `tools/list` mengembalikan urutan logis yang sama pada panggilan berulang.
- Klien melakukan permintaan kedua ketika `nextCursor` adalah `""`.
- Deskriptor header-sensitif yang tidak aman dikecualikan sementara tool lain tetap tersedia.
- Sebuah array lulus schema output array-nya.
- Sebuah objek gagal schema array yang sama.
- Hasil kesalahan tidak bisa menghilangkan atau melanggar schema output yang dipublikasikan.
- Blok teks, gambar, audio, tautan resource, dan resource tertanam tervalidasi.
- Event audit header berisi nama dan bukan nilai.
- ASCII terlihat polos tetap polos; nilai Unicode, kontrol, ber-padding, kosong, dan yang tampak-sentinel round-trip melalui encoding base64 UTF-8 yang persis.
- Integer yang di-mirror di luar rentang aman JavaScript ditolak.
- Anotasi di bawah `oneOf`, `items`, objek bersarang, definisi `$ref`, atau schema output ditolak saat admission.
- Nama header yang dikenali case-insensitive lulus hanya ketika nilai ter-decode persis cocok dengan body; salinan yang hilang atau tidak cocok menghasilkan HTTP `400` dan JSON-RPC `-32020`.
- Completion analis tidak pernah mengembalikan `production`.
- Kegagalan tool memakai `isError: true`; panggilan protokol yang rusak memakai JSON-RPC `error`.

## Production Failure Modes

| Kegagalan | Apa yang dilihat pembelajar | Respons yang benar |
|---------|-----------------------|------------------|
| Klien berasumsi output objek | Array valid gagal atau dibungkus diam-diam | Validasi terhadap schema yang dipublikasikan tanpa tipe hanya-objek |
| Kursor kosong diperlakukan false | Halaman terakhir menghilang | Lanjutkan setiap kali `nextCursor` ada dan non-null |
| Nilai sensitif di-mirror | Rahasia muncul di proxy, WAF, atau data trace | Tolak deskriptor dan jaga rahasia di data permintaan yang dilindungi |
| Unicode atau whitespace mentah di-mirror | Gateway dan origin tidak setuju atau nilai dinormalisasi | Gunakan encoding sentinel base64 UTF-8 yang persis dan bandingkan setelah decoding |
| Anotasi tersembunyi di cabang schema | Klien melewatkan metadata routing saat admission | Telusuri seluruh pohon schema dan izinkan hanya properti top-level langsung |
| Integer besar di-mirror | Perantara JavaScript membulatkan nilai routing | Tolak nilai di luar rentang integer aman JavaScript |
| Header dan body tidak setuju | Gateway merutekan satu target sementara origin mengeksekusi yang lain | Tolak sebelum dispatch dengan HTTP `400` dan JSON-RPC `-32020` |
| Schema output diabaikan | Kode downstream mengonsumsi struktur yang rusak | Validasi sebelum penggunaan model atau aplikasi |
| Tautan resource dipercaya otomatis | Pemanggil mengikuti URI yang tidak terotorisasi | Otorisasi-ulang setiap pembacaan resource |
| Completion berbagi saran global | Nama tenant tersembunyi bocor | Filter berdasarkan pemanggil, referensi, dan otorisasi |
| Anotasi tool diperlakukan sebagai kebijakan | Operasi destruktif melewati konfirmasi | Tegakkan otorisasi dan persetujuan di luar anotasi |
| Satu tool yang rusak merusak discovery | Seluruh server menjadi tidak tersedia | Tolak deskriptor yang buruk dan admit tool yang valid secara independen |

## Capstone Connection

Capstone Phase 13 membutuhkan gateway yang bisa menggabungkan tool dari beberapa server. Pelajaran ini menyediakan inti admission-nya.

Gunakan artifact untuk menilai empat bagian bukti capstone:

- penemuan berpaginasi yang deterministik dan lengkap;
- validasi deskriptor sebelum paparan model;
- output terstruktur tervalidasi plus blok konten yang berbatas;
- completion dan metadata routing yang mempertahankan batas otorisasi.

Jangan mengklaim kompatibilitas gateway dari `tools/call` yang berhasil saja. Capture deskriptor, trace halaman, himpunan tool yang diadmit, himpunan tool yang ditolak, dan satu hasil tervalidasi.

## Istilah Kunci

| Istilah | Arti |
|------|---------|
| `inputSchema` | Objek JSON Schema yang mendefinisikan argumen tool yang diterima |
| `outputSchema` | JSON Schema opsional yang mendefinisikan `structuredContent` |
| `structuredContent` | Nilai JSON apa pun yang dihasilkan oleh hasil tool |
| Blok konten | Teks, gambar, audio, tautan resource, atau resource tertanam bertipe |
| `x-mcp-header` | Anotasi schema yang meng-mirror argumen primitif ke metadata Streamable HTTP |
| Kursor opaque | Token paginasi yang diterbitkan server yang nilainya tidak diinterpretasikan klien |
| Referensi completion | Nama prompt atau URI/template resource yang argumennya sedang di-complete |
| Admission | Keputusan klien untuk mengekspos atau menolak deskriptor yang ditemukan |

## Bacaan Lanjutan

- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Completion](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/completion)
- [MCP Pagination](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination)
- [MCP Streamable HTTP Parameter Headers](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#custom-headers-from-tool-parameters)
