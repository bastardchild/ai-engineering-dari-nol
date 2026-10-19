# MCP Resources dan Prompts: Konteks yang Bisa Dialamatkan untuk Server Tanpa-State

> Tools menjalankan operasi. Resources mengekspos konten yang bisa dialamatkan. Prompts mengemas template pesan pilihan pengguna. MCP server yang baik menjaga kontrak-kontrak itu tetap terpisah dan dapat diprediksi.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lesson 07 (Building an MCP Server), Phase 13, Lesson 09 (MCP Transports)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Pilih di antara tools, resources, dan prompts berdasarkan maksud konsumennya.
- Iklankan permukaan resource dan prompt melalui `server/discover` yang wajib.
- Bangun hasil `resources/list` dan `prompts/list` yang deterministik.
- Terapkan `ttlMs` dan `cacheScope` tanpa membocorkan data spesifik-pengguna.
- Kembalikan error JSON-RPC `-32602` untuk URI resource yang tidak valid atau tidak diketahui.
- Buka stream POST-respons `subscriptions/listen` dan korelasikan setiap event berdasarkan subscription ID.
- Perlakukan konten resource dan template prompt sebagai output server yang tidak tepercaya.

## Mulai dari Konsumennya

Cara termudah menyalahgunakan MCP adalah dimulai dari kode implementasi. Query basis data menjadi tool karena fungsi terasa familiar. Alur kerja yang bisa dipakai ulang menjadi resource karena disimpan di sebuah berkas. Sebuah prompt menjadi kebijakan tersembunyi karena host bisa menyuntikkannya.

Mulai dari siapa yang memilih dan apa yang mereka harapkan.

| Primitive | Primary intent | Selection owner | Typical result |
|---|---|---|---|
| Tool | Perform an operation | Model or application | Structured action result |
| Resource | Read content at a URI | Host, application, or user | Text or binary content |
| Prompt | Start a reusable message workflow | User through host UI | One or more prompt messages |

Sebuah catatan di `notes://note-1` adalah resource karena itu adalah konten yang bisa dialamatkan. `delete_note` adalah tool karena itu mengubah state. `review_note` adalah prompt karena pengguna memilih alur kerja review yang sudah disiapkan.

Jangan mengekspos satu operasi sebagai ketiganya hanya untuk terlihat lengkap. Setiap permukaan tambahan memerlukan discovery, otorisasi, caching, penanganan error, pengujian, dan dokumentasi.

## Envelope Tanpa-State 2026-07-28

Pelajaran ini menyasarkan revisi protokol MCP `2026-07-28`. Tidak ada handshake inisialisasi maupun sesi protokol dalam profil ini. Setiap request membawa versi protokol dan kapabilitas klinenya di kunci `_meta` yang dipesan.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Sebuah server harus mengimplementasikan `server/discover`. Hasilnya
mengiklankan versi yang didukung, kapabilitas resource dan prompt, identitas
implementasi, dan petunjuk cache. Sebuah klien boleh memanggil method lain
secara langsung, tetapi discovery memberinya satu cuplikan stabil sebelum membangun UI.

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "resources": {"listChanged": true, "subscribe": true},
    "prompts": {"listChanged": true}
  },
  "ttlMs": 3600000,
  "cacheScope": "public"
}
```

Hasil biasa menyatakan `"resultType": "complete"`. `_meta` respons mengidentifikasi implementasi yang melayani melalui `io.modelcontextprotocol/serverInfo`. Informasi ini berguna untuk diagnostik. Itu bukan identitas autentikasi. Sebuah request yang membawa revisi yang tidak didukung mengembalikan `-32022` dengan revisi yang diminta serta revisi yang didukung server.

Kontrak tanpa-state mengubah naluri desainmu. Sebuah daftar tidak boleh bergantung pada panggilan sebelumnya pada satu koneksi. Otorisasi boleh mengubah himpunan yang terlihat karena kredensial adalah input request, tetapi riwayat koneksi tidak boleh.

## Resource adalah Kontrak URI yang Stabil

Sebuah resource adalah konten yang diidentifikasi URI. Rancang URI-nya sebelum handlernya.

Sifat URI yang baik:

- Cukup stabil untuk di-bookmark atau diteruskan antar-request.
- Ber-namespace ke domain server.
- Tidak tergantung pada ID proses atau koneksi.
- Divalidasi sebelum mengakses penyimpanan.
- Diotorisasi pada setiap pembacaan.

`notes://note-1` lebih baik daripada `note-1` karena namespace-nya eksplisit. Sebuah server berkas boleh memakai URI `file://`, tetapi ia tetap harus memeriksa batas direktori yang dikonfigurasi setelah menyelesaikan symlink dan segmen relatif.

`resources/list` mengembalikan resource yang saat ini terlihat oleh pemanggil. Urutkan berdasarkan kunci stabil seperti URI. Urutan deterministik mencegah cache miss yang berisik, cuplikan yang berubah, dan UI host yang melompat antar-refresh.

```json
{
  "resultType": "complete",
  "resources": [
    {
      "uri": "notes://note-1",
      "name": "Architecture decision",
      "description": "Why the service uses a stateless boundary",
      "mimeType": "text/markdown"
    }
  ],
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

`resources/read` mengembalikan satu atau lebih item konten. URI yang tidak diketahui bukanlah pembacaan kosong yang berhasil. Spesifikasi Resources saat ini menetapkan URI resource yang tidak valid atau tidak diketahui sebagai parameter JSON-RPC yang tidak valid, kode `-32602`.

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "error": {
    "code": -32602,
    "message": "Unknown or invalid resource URI",
    "data": {
      "uri": "notes://missing"
    }
  }
}
```

Pembedaan itu memungkinkan klien memisahkan ketiadaan dari dokumen kosong yang valid. Ia juga mencegah fallback yang tidak sengaja ke pencarian yang lebih luas.

### Template resource

Sebuah template resource menjelaskan keluarga URI berparameter. Gunakan satu ketika mendaftar setiap item konkret akan mahal atau tak terbatas. Sebagai contoh, `notes://projects/{project}/decisions/{decision}` memberi tahu klien cara membentuk alamat yang valid tanpa mengembalikan setiap keputusan.

Template tidak melemahkan validasi. Parse variabel, terapkan otorisasi, paksa batas panjang dan karakter, dan bangun query penyimpanan dengan parameter bertipe. Jangan pernah menggabungkan ekor URI sewenang-wenang ke path berkas atau statement basis data.

### Konten bukan instruksi tepercaya

Teks resource bisa berisi prompt injection, rahasia, perintah menyesatkan, atau markup yang rusak format. Host harus mempertahankan provenance dan memperlakukan konten resource sebagai data. Server harus membatasi ukuran konten, mengembalikan MIME type yang akurat, menyensor field yang tidak bisa diakses pemanggil, dan menghindari pengembalian record yang tidak terkait.

## Prompts adalah Template yang Dikendalikan Pengguna

Prompt MCP dirancang untuk pilihan pengguna yang eksplisit. Sebuah host boleh merendernya sebagai slash command, item menu, atau tombol alur kerja. Protokol tidak menuntut satu bentuk UI.

`prompts/list` seharusnya deterministik untuk otorisasi request yang sama. Setiap prompt memerlukan nama yang stabil, deskripsi yang berguna, dan deklarasi argumen yang memungkinkan host mengumpulkan input sebelum `prompts/get`.

```json
{
  "resultType": "complete",
  "prompts": [
    {
      "name": "review_note",
      "title": "Review a note",
      "description": "Review one note for a named concern",
      "arguments": [
        {
          "name": "uri",
          "description": "The note resource URI",
          "required": true
        }
      ]
    }
  ],
  "ttlMs": 600000,
  "cacheScope": "public"
}
```

`prompts/get` me-resolve argumen menjadi pesan. Ia tidak menggantikan instruksi sistem host. Host memutuskan bagaimana pesan yang dikembalikan masuk ke konteks model dan mempertahankan kebijakan tepercayanya sendiri pada prioritas lebih tinggi.

Validasi argumen prompt di batas server. URI prompt seharusnya lulus pemeriksaan otorisasi yang sama dengan pembacaan resource langsung. Jangan menjadikan prompt sebagai saluran sampingan di sekitar akses resource.

## Petunjuk Cache adalah Bagian dari Kebenaran

`ttlMs` memberi tahu klien berapa lama sebuah hasil boleh dipakai ulang. `cacheScope` menjelaskan siapa yang boleh membagikan nilai cache itu.

| Scope | Meaning | Typical use |
|---|---|---|
| `public` | May be reused across users when authorization permits | Public prompt catalog |
| `private` | Bound to the requesting user or credential context | User-owned note content |

Pilih TTL berdasarkan laju perubahan data dan kerusakan akibat basi. Lima menit mungkin cocok untuk katalog prompt publik. Pembacaan catatan privat bisa memakai satu menit.

MCP hanya mendefinisikan `public` dan `private` sebagai nilai `cacheScope`. Untuk hasil yang membawa rahasia atau cepat berubah, kembalikan `cacheScope: "private"` dengan `ttlMs: 0`, lalu terapkan aturan no-store yang lebih ketat apa pun dalam kebijakan cache host. `no-store` sendiri bukan nilai `cacheScope` di MCP.

Petunjuk cache tidak pernah menggantikan otorisasi. Kunci cache harus mencakup setiap dimensi request yang mengubah visibilitas, termasuk tenant, pengguna, scope, locale, dan kursor pagination. Jika cache bersama tidak bisa mengekspresikan dimensi itu dengan aman, gunakan `private` dengan TTL nol dan kebijakan no-store tingkat host.

## Subscription Memakai Stream Respons yang Dibuka Klien

Pola subscription modern menggantikan RPC `resources/subscribe` sebelumnya dan endpoint event HTTP GET lama.

Klien mengirim `subscriptions/listen` sebagai request JSON-RPC biasa. Lewat Streamable HTTP ini adalah POST yang responsnya tetap terbuka sebagai stream SSE. Objek `notifications` adalah allowlist. Sebuah server tidak boleh mengirim tipe notifikasi yang tidak diminta.

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "subscriptions/listen",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    },
    "notifications": {
      "resourcesListChanged": true,
      "promptsListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

ID request adalah subscription ID. Sebelum event mana pun yang diminta, server mengirim `notifications/subscriptions/acknowledged`. Filternya hanya berisi subset yang diterima server.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "notifications": {
      "resourcesListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

Setiap event berikutnya pada stream itu membawa metadata yang sama.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "uri": "notes://note-1"
  }
}
```

Notifikasi itu mengatakan resource telah berubah. Klien membacanya lagi melalui `resources/read`, tunduk pada otorisasi saat ini. Ia tidak berasumsi event itu berisi dokumen yang baru.

Beberapa subscription bisa berbagi satu channel stdio. Subscription ID memungkinkan klien melakukan demultiplex terhadapnya. Lewat HTTP, menutup stream respons membatalkan subscription. Sebuah server yang mengakhiri stream dengan anggun mengembalikan respons `resultType: "complete"` final yang dikorelasikan dengan request asli.

Jangan memakai stream subscription sebagai sesi protokol. Pembacaan berikutnya tetap merupakan request lengkap yang bisa mencapai instance server sehat mana pun.

```figure
t3-primitive-sort
```

## Interactive Lab

Gunakan gambar tersebut untuk mengklasifikasi lima kapabilitas dari sebuah project tracker: detail issue, buat issue, template sprint review, kebijakan proyek, dan tutup issue. Lalu putuskan daftar mana yang boleh di-cache secara publik, pembacaan mana yang harus tetap privat, dan resource mana yang layak mendapat notifikasi pembaruan.

Untuk setiap klasifikasi, sebutkan pemilihnya. Jika model menjalankan sebuah aksi, gunakan tool. Jika host membaca konten yang dialamatkan URI, gunakan resource. Jika pengguna memulai alur kerja pesan yang sudah disiapkan, gunakan prompt.

## Practice Lab

Jalankan simulator dari root repositori:

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

Periksa transkrip dengan urutan ini:

1. Konfirmasikan `server/discover` mengiklankan revisi saat ini dan kedua kapabilitas.
2. Konfirmasikan kedua hasil daftar terurut dan memakai `resultType: "complete"`.
3. Konfirmasikan hasil daftar dan pembacaan membawa petunjuk cache yang disengaja.
4. Ubah URI pembacaan ke `notes://missing` dan amati `-32602`.
5. Konfirmasikan pengakuan subscription mendahului event resource.
6. Konfirmasikan event dan penutupan anggun keduanya membawa subscription ID `5`.

Model Python tidak membuka koneksi HTTP sungguhan. Ia merepresentasikan pesan yang harus diletakkan SDK pada stream respons berlingkup request. Gunakan SDK resmi untuk framing dan transport di produksi.

## Shipped Artifact

`outputs/skill-primitive-splitter.md` adalah design review yang bisa dipakai ulang untuk pemilihan primitive MCP. Kini ia memeriksa discovery deterministik, scope cache, perilaku URI tidak valid, dan filter subscription modern.

Pelajaran ini juga mengirimkan `assets/primitive-split.svg`, versi statis dari batas primitive dan subscription untuk belajar offline.

## Verify It

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

Hasil yang diharapkan: program utama mencetak transkrip JSON dan perintah pengujian melaporkan setidaknya dua belas tes yang lulus.

## Capstone Connection

Gunakan kontrak ini saat server capstone-mu mengekspos pengetahuan yang bisa dialamatkan di samping aksi. Sertakan satu cuplikan katalog deterministik, satu pembacaan resource yang diotorisasi, satu resolusi prompt, satu kasus URI tidak valid, dan satu transkrip subscription.

Buktimu seharusnya menunjukkan bahwa tidak ada daftar yang bergantung pada riwayat koneksi dan bahwa event subscription tidak pernah memberikan akses ke resource dasarnya.

## Latihan

1. Tambahkan template resource `notes://projects/{project}/notes/{id}` dan validasi kedua variabelnya.
2. Tambahkan pagination ke `resources/list` sambil mempertahankan urutan deterministik.
3. Ubah satu resource menjadi `cacheScope: "private"` dengan `ttlMs: 0`, tambahkan kebijakan no-store tingkat host, dan jelaskan ancaman yang membenarkan kedua kontrol itu.
4. Tambahkan subscription perubahan daftar prompt dan buktikan tidak ada event yang dikirim saat filter mengabaikan `promptsListChanged`.
5. Buat dua subscription bersamaan dan buktikan setiap event membawa ID request yang benar.
6. Tambahkan subjek otorisasi ke handler pembacaan dan buktikan entry cache tidak bisa melintasi subjek.

## Istilah Kunci

- **Resource:** Konten yang dialamatkan URI dan diekspos oleh MCP server.
- **Prompt:** Template pesan yang dikendalikan pengguna dan diekspos oleh MCP server.
- **Deterministic list:** Hasil discovery dengan keanggotaan dan pengurutan stabil untuk input request yang sama.
- **`ttlMs`:** Durasi kesegaran cache dalam milidetik.
- **`cacheScope`:** Batas berbagi untuk sebuah hasil cache.
- **`subscriptions/listen`:** Request jangka panjang yang stream responsnya mengirimkan notifikasi yang telah difilter secara eksplisit.
- **Subscription ID:** ID request listen asli, diulang di metadata notifikasi.
- **Invalid parameters:** Error JSON-RPC `-32602`, dipakai untuk URI resource yang tidak valid atau tidak diketahui.
- **Unsupported protocol version:** Error JSON-RPC `-32022`, termasuk revisi `supported` dan `requested`.
- **`server/discover`:** Method server wajib yang mengembalikan revisi yang didukung, kapabilitas, identitas, dan petunjuk cache opsional.

## Bacaan Lanjutan

- [MCP 2026-07-28 Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP 2026-07-28 Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP 2026-07-28 Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Caching](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/caching)
