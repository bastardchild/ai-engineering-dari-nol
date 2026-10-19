# Scope Eksplisit dan Elicitation Tanpa-State

> Roots dideprecate di MCP 2026-07-28 dan memang tidak pernah merupakan sandbox keamanan. Taruh scope di argumen tool yang terlihat atau URI resource, otorisasikan di server, dan pakai MRTR ketika sebuah tool benar-benar membutuhkan input pengguna. Pengguna melihat keputusannya, model melihat handle-nya, dan instance server mana pun bisa memproses percobaan ulangnya.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 11 (stateless MRTR)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Ganti Roots yang dideprecate dengan parameter workspace eksplisit, URI resource, atau konfigurasi server.
- Pisahkan petunjuk scope dari otorisasi, penahanan path, dan sandboxing sistem operasi.
- Kirim `elicitation/create` mode-form melalui hasil `input_required` MRTR.
- Iklankan dukungan elicitation di kapabilitas klien per-request dan tolak mode yang tidak didukung.
- Validasi `accept`, `decline`, dan `cancel` sebagai hasil yang berbeda.
- Ikatkan konfirmasi destruktif ke principal yang diautentikasi, argumen asli, himpunan kandidat, dan kedaluwarsa.

## Dua Masalah yang Terlihat Mirip

Sebuah tool catatan menerima request ini: "Delete the old TPS report."

Server harus menjawab dua pertanyaan yang berbeda.

1. Workspace mana yang boleh disentuh operasi ini?
2. Di antara tiga catatan yang cocok, mana yang dimaksud pengguna?

Yang pertama adalah scope dan otorisasi. Yang kedua adalah disambiguasi interaktif. mencampur keduanya mengarah pada desain berbahaya, seperti memperlakukan folder yang diberikan klien sebagai bukti bahwa pemanggil boleh menghapus semua isinya.

## Roots adalah Permukaan Migrasi

Revisi MCP lebih awal memungkinkan klien mengiklankan Roots dan memberi tahu server saat daftarnya berubah. Roots adalah panduan informasional. Ia tidak membatasi apa yang bisa dibaca proses server, tidak mengotorisasi pemanggil, dan tidak membuat sandbox sistem operasi.

MCP 2026-07-28 men-deprecate `roots/list` dan `notifications/roots/list_changed` untuk desain baru. Lebih suka salah satu pengganti eksplisit berikut:

- Argumen tool `workspaceUri` atau `directory` ketika scope bervariasi per panggilan.
- URI resource ketika operasi sudah menyasar sebuah resource.
- Konfigurasi server ketika satu deployment memiliki satu workspace tetap.
- Sandbox proses atau filesystem terpenjara ketika kode harus secara teknis tidak bisa keluar.

Jika sebuah integrasi 2026-07-28 yang ada masih membutuhkan `roots/list` selama jendela deprecasi, server menanamkannya di MRTR `inputRequests`. Ia tidak boleh mengirim request balik secara langsung. Itu adalah adaptor migrasi; penangan baru seharusnya menerima scope eksplisit sebagai gantinya.

Model bisa melihat dan mengulang handle eksplisit. Scope sesi-transport yang tersembunyi lebih sulit diperiksa, diputar ulang, diaudit, dan di-route.

### Aturan tiga lapis

Sebuah URI eksplisit tetap tidak mengotorisasi dirinya sendiri. Paksa ketiga lapisan:

1. **Otorisasi:** Apakah principal yang diautentikasi ini diizinkan memakai workspace ini?
2. **Penahanan:** Apakah URI target yang dinormalisasi tetap di dalam batas workspace yang diotorisasi?
3. **Sandbox:** Bisakah sistem operasi menghentikan server yang disusupi agar tetap tidak bisa keluar?

Server yang bisa dijalankan menyimpan allowlist URI workspace yang diotorisasi, menormalisasi path yang di-percent-encode, memeriksa batas komponen path yang nyata, dan memeriksa ulang penahanan tepat sebelum penghapusan.

Pemeriksaan prefix string yang naif itu salah:

```text
allowed:   file:///work/notes
attacker:  file:///work/notes-evil/secret.md
traversal: file:///work/notes/%2e%2e/private.md
```

Kedua path berbahaya diawali string yang menyesatkan. Normalisasi dulu, lalu bandingkan komponen path. Sebuah server filesystem produksi juga harus bertahan terhadap race symbolic-link dan semantik path spesifik-platform.

## Elicitation Masih Ada, tetapi Pengirimannya Berubah

Elicitation adalah fitur klien saat ini untuk mengumpulkan input pengguna selama `tools/call`, `prompts/get`, atau `resources/read`. Nama method-nya tetap `elicitation/create`. Yang berubah adalah arah aliran wire-nya.

Sebuah server 2026-07-28 tidak mengirim request JSON-RPC balik. Ia mengembalikan sebuah `InputRequiredResult`:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "delete_choice": {
        "method": "elicitation/create",
        "params": {
          "mode": "form",
          "message": "Choose one matching note and confirm deletion.",
          "requestedSchema": {
            "type": "object",
            "properties": {
              "note_id": {
                "type": "string",
                "enum": ["note-3", "note-7", "note-14"]
              },
              "confirm": {"type": "boolean"}
            },
            "required": ["note_id", "confirm"]
          }
        }
      }
    },
    "requestState": "integrity-protected-delete-state"
  }
}
```

Host merender formulir itu. Pengguna bisa menerima, menolak secara eksplisit, atau menutupnya. Lalu klien mencoba ulang `tools/call` asli dengan id baru:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes_delete",
    "arguments": {
      "workspaceUri": "file:///Users/alice/Documents/Notes",
      "title": "TPS report"
    },
    "inputResponses": {
      "delete_choice": {
        "action": "accept",
        "content": {"note_id": "note-14", "confirm": true}
      }
    },
    "requestState": "integrity-protected-delete-state",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

Tidak ada sesi protokol di antara kedua panggilan itu. Server memverifikasi state yang dipantulkan, memvalidasi respons terhadap skema yang diharapkan, memeriksa bahwa catatan terpilih berada di himpunan kandidat yang ditandatangani, mengotorisasi ulang workspace, memeriksa ulang penahanan, lalu menghapus.

## Negosiasi Kapabilitas per Request

Sebuah klien yang mendukung elicitation mode-form mendeklarasikan:

```json
{
  "io.modelcontextprotocol/clientCapabilities": {
    "elicitation": {"form": {}}
  }
}
```

Kapabilitas elicitation kosong, `"elicitation": {}`, tetap setara dengan dukungan form-saja demi kompatibilitas. `"elicitation": {"form": {}}` yang eksplisit juga mendukung mode form. Deklarasi URL-saja, `"elicitation": {"url": {}}`, tidak. Server tidak boleh menanamkan mode yang absen dari kapabilitas request saat ini, meskipun request sebelumnya mengiklankannya.

Setiap request juga membawa `io.modelcontextprotocol/protocolVersion`. Versi yang hilang atau bukan string mengembalikan `-32602`. String yang tidak didukung mengembalikan `-32022` dengan data `supported` dan `requested` yang tepat. Dukungan elicitation yang hilang atau URL-saja mengembalikan `-32021` dengan `data.requiredCapabilities` diatur ke `{"elicitation":{"form":{}}}`.

Envelope tanpa `id` JSON-RPC adalah notifikasi. Proses itu tanpa mengeluarkan respons sukses atau error JSON-RPC. Pada Streamable HTTP, notifikasi yang diterima menerima `202 Accepted` tanpa body.

`clientInfo` seharusnya disertakan untuk diagnostik, tetapi itu dilaporkan sendiri dan tidak bisa mengidentifikasi pengguna untuk otorisasi.

Server mengimplementasikan `server/discover` dan mengembalikan `supportedVersions`, kapabilitas, `ttlMs`, dan `cacheScope` dengan `resultType: "complete"`. Ia tidak mengiklankan Roots untuk desain modern ini. Karena ia mengiklankan tools, ia juga mengimplementasikan `tools/list` yang wajib. Hasil itu mengembalikan deskriptor `notes_delete` deterministik, `inputSchema` objek yang valid, metadata identitas server, dan petunjuk cache publik.

## Mode Form

Mode form memakai JSON Schema terbatas yang dirancang untuk dialog yang bisa dipakai. Akarnya adalah sebuah objek dan propertinya adalah field primitive datar atau array enum yang didukung. Objek bersarang dalam dan skema dokumen tujuan-umum tidak berada tempatnya di dialog konfirmasi.

Gunakan mode form untuk:

- memilih satu di antara beberapa kandidat;
- mengonfirmasi operasi destruktif;
- mengumpulkan preferensi yang tidak sensitif;
- mengumpulkan sedikit nilai yang harus diputuskan oleh pengguna, bukan model.

Jangan memakai mode form untuk kata sandi, API key, access token, atau kredensial pembayaran. Rahasia-rahasia itu akan melewati MCP client dan bisa mencapai log atau konteks model.

Server memvalidasi konten yang dikembalikan sekali lagi. Validasi formulir sisi klien meningkatkan UX tetapi tidak menciptakan kepercayaan.

## Mode URL

Mode URL mengirim URL web yang aman untuk interaksi luar-jalur:

```json
{
  "method": "elicitation/create",
  "params": {
    "mode": "url",
    "message": "Connect the report service to continue.",
    "url": "https://mcp.example.com/connect/report-service"
  }
}
```

Gunakan ketika informasi sensitif harus dikirim langsung ke alur web yang dikendalikan server, seperti otorisasi pihak ketiga. Klien menampilkan tujuan lengkap dan memperoleh persetujuan sebelum membukanya. Ia tidak boleh melakukan prefetch terhadap URL.

Respons `accept` berarti pengguna setuju membuka URL. Itu tidak membuktikan alur eksternal selesai. Pada percobaan ulang, server memeriksa state-nya sendiri lalu menyelesaikan atau mengembalikan hasil `input_required` lain.

Elicitation URL bukan pengganti otorisasi antara MCP client dan MCP server. Itu untuk interaksi eksternal yang perlu dilakukan MCP server atas nama pengguna. Server harus mengikat pengguna browser ke principal yang sama dengan yang memulai operasi MCP.

## Cabang Respons

Perlakukan aksi-nya sebagai keputusan produk, bukan alias:

| Action | Meaning | Safe server behavior |
|--------|---------|----------------------|
| `accept` | User submitted the interaction | Validate content and continue |
| `decline` | User explicitly refused | Return a complete, non-error refusal outcome |
| `cancel` | User dismissed or could not finish | Stop safely and allow a later retry |

Jangan pernah menafsirkan konten yang hilang sebagai persetujuan. Jangan pernah mengubah penolakan menjadi loop prompt berulang.

## Melindungi State MRTR Destruktif

Daftar kandidat tidak boleh hidup hanya di dalam prompt atau nilai Base64 yang tidak ditandatangani. Sebuah klien mengendalikan semua yang ia kirim balik.

Pelajaran ini menandatangani payload state yang berisi:

- principal yang diautentikasi;
- method asal;
- digest dari `workspaceUri` dan `title`;
- id catatan yang diizinkan dan ditampilkan di formulir;
- fase operasi;
- kedaluwarsa singkat.

Sebelum mutasi, server juga memeriksa record catatan yang hidup. Ini menangkap race penghapusan dan target yang berpindah keluar workspace setelah formulir ditampilkan.

Untuk aksi finansial satu-kali atau tak dapat dipulihkan, HMAC saja tidak mencegah state yang valid diputar ulang di dalam masa berlakunya. Simpan dan konsumsi sebuah nonce tepat sekali di sebuah replay store yang dibagikan oleh setiap instance penangan. Pelajaran ini menyuntikkan store ber-TTL dan berbatas serta menahan klaim atomiknya saat melakukan penghapusan di memori. Sebuah basis data produksi seharusnya menghubungkan klaim nonce dan mutasi dalam satu transaksi atau batas conditional-write yang setara.

Validasi interaksi sebelum mengklaim nonce. Respons yang rusak format atau `cancel` tidak melakukan mutasi apa pun dan meninggalkan state yang masih bisa dicoba ulang sampai kedaluwarsa. Sebuah `decline` eksplisit bersifat terminal, sehingga pelajaran mengonsumsi nonce tanpa menghapus apa pun.

```figure
t3-roots-boundary
```

## Bangun Sendiri

`code/main.py` mendemonstrasikan tool `notes_delete` modern:

- `tools/list` mengembalikan deskriptor deterministik dan bisa di-cache dengan skema workspace dan title yang wajib.
- Scope adalah argumen `workspaceUri` yang eksplisit.
- Konfigurasi server mengotorisasi workspace itu untuk principal pelajaran.
- Normalisasi URI menolak kebingungan prefix dan traversal yang di-encode.
- Setiap penghapusan destruktif memerlukan elicitation mode-form.
- Elicitation bepergian di dalam `resultType: "input_required"`.
- `requestState` yang ditandatangani mengikat daftar kandidat yang tepat dan argumen asli.
- Replay store yang disuntikkan menolak state yang sama yang diterima atau ditolak di antara instance server.
- Percobaan ulang memakai id request baru dan mengembalikan `resultType: "complete"`.

Penyimpanan data berada di memori sehingga perilaku protokol mudah diperiksa. Aturan keamanannya tetap sama dengan basis data.

## Gunakan

Dari root repositori:

```bash
cd phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/code
python3 main.py
python3 -m unittest discover tests -v
```

Checkpoint yang diharapkan:

- Discovery mengiklankan tools tanpa Roots.
- Discovery tool mengembalikan `notes_delete` dengan `resultType`, identitas server, dan petunjuk cache.
- Id request `1` mengembalikan formulir di `inputRequests.delete_choice`.
- Id request `2` memantulkan state yang ditandatangani dan menyelesaikan penghapusan.
- Path prefix dan path traversal yang di-encode sama-sama gagal pada penahanan.
- Title yang berubah tidak bisa memakai ulang state konfirmasi asli.
- Sebuah penolakan meninggalkan catatan tanpa perubahan.
- Dua objek server yang berbagi state catatan dan replay tidak bisa sama-sama mengeksekusi satu konfirmasi.
- Deklarasi formulir kosong dan eksplisit berfungsi, sementara dukungan URL-saja mengembalikan kebutuhan formulir `-32021` yang tepat.
- Kegagalan versi yang tidak didukung memakai bentuk data `-32022` yang tepat.
- Notifikasi tanpa id tidak menghasilkan respons JSON-RPC.

## Rilis

`outputs/skill-elicitation-form-designer.md` merancang scope eksplisit, pemeriksaan otorisasi, formulir MRTR, cabang respons, dan pengikatan state. Ia menolak memperlakukan Roots yang dideprecate sebagai sandbox atau mengumpulkan rahasia melalui mode form.

## Latihan

1. Ganti replay store di memori dengan SQLite. Gunakan satu transaksi untuk mengklaim nonce dan menghapus catatannya, lalu buktikan dua proses tidak bisa sama-sama commit.
2. Tambahkan negosiasi kapabilitas `url` dan alur setup luar-jalur. Jauhkan kredensial pihak ketiga dari `inputResponses`.
3. Ganti map catatan di memori dengan basis data SQLite sementara. Periksa ulang otorisasi dan penahanan di dalam transaksi mutasi.
4. Tambahkan kebijakan symbolic-link untuk implementasi filesystem sungguhan. Jelaskan mengapa penahanan leksikal URI saja tidak bisa menghentikan keberangkatan symlink.
5. Rancang adaptor 2025-11-25 yang memetakan keluaran penangan MRTR modern ke elicitation yang diinisialisasi server era lama. Jaga agar terisolasi dari penangan saat ini.

## Istilah Kunci

| Term | Meaning in 2026-07-28 |
|------|------------------------|
| Roots | Deprecated informational workspace hints, not authorization or sandboxing |
| Explicit scope | Workspace, directory, or resource handle visible in request arguments |
| Containment | Normalized path-component check that keeps a target inside a boundary |
| Elicitation | Client feature for obtaining user input during an MCP operation |
| Form mode | In-band structured user input using a restricted flat schema |
| URL mode | Out-of-band interaction for sensitive or external workflows |
| MRTR | Stateless input-required result followed by a fresh retry |
| `requestState` | Opaque state echoed exactly and integrity-checked by the server |
| Decline | Explicit user refusal |
| Cancel | Dismissal or incomplete interaction without approval |

## Kompatibilitas Era Lama

Untuk peer yang di-pin ke 2025-11-25, `roots/list`, `notifications/roots/list_changed`, dan `elicitation/create` yang diinisialisasi server secara langsung mungkin masih ada. Beri label adaptor itu sebagai legacy. Jangan mengizinkan daftar Root lama melewati otorisasi server, dan jangan membawa asumsi sesi-protokol ke penangan modern.

## Bacaan Lanjutan

- [MCP 2026-07-28 Elicitation](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Roots deprecation](https://modelcontextprotocol.io/specification/2026-07-28/client/roots)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
