# Membangun Klien MCP: Discovery, Routing, dan Fallback Dua-Era

> Sebuah klien MCP modern mengulang kontraknya pada setiap request. Keputusan kompatibilitas tersulitnya adalah mengetahui kapan sebuah server lama benar-benar lama dan kapan sebuah server modern melaporkan error yang dapat diperbaiki.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lesson 07
**Time:** ~85 minutes

## Tujuan Pembelajaran

- Membangun setiap request MCP `2026-07-28` dengan metadata terkini.
- Memburu server stdio dengan `server/discover` dan memilih versi yang didukung bersama.
- Mengotorisasi probe legacy yang terbatas hanya untuk peer yang secara eksplisit di-allowlist.
- Menerima era legacy hanya setelah memvalidasi hasil `initialize` positif untuk revisi yang didukung.
- Menggabungkan daftar tool deterministik tanpa menimpa tabrakan secara diam-diam.
- Merutekan panggilan ke peer yang memiliki setiap tool tanpa mengarang sesi protokol.

## Masalahnya

Sebuah host agen biasanya berbicara dengan lebih dari satu server MCP. Ia harus menemukan setiap server, menggabungkan katalog tool, menyelesaikan nama ganda, merutekan panggilan, dan pulih dari kegagalan transport.

Revisi `2026-07-28` membuat steady state lebih sederhana karena setiap request mandiri. Kompatibilitas membuat startup lebih halus. Sebuah klien dapat menemui:

- server modern yang mendukung versi yang dipilih;
- server modern yang mengembalikan error versi atau header yang dikenali;
- server legacy yang belum pernah mendengar `server/discover`;
- server legacy yang tetap diam sampai menerima `initialize`.

Memperlakukan setiap probe error sebagai legacy berbahaya. Request modern yang rusak, server yang kelebihan beban, proses yang mati, dan server lama dapat semuanya menghasilkan timeout atau penutupan koneksi yang sama. Sinyal-sinyal itu ambigu. Klien harus menggabungkan intensi operator eksplisit dengan bukti protokol positif sebelum memilih era legacy.

## Konsepnya

### Sebuah peer, bukan sesi protokol

Simpan satu record transport peer untuk setiap proses atau endpoint server:

- handle transport atau fungsi kirim;
- era dan versi protokol yang dipilih;
- capabilities server yang terakhir ditemukan;
- daftar tool deterministik terakhir;
- id request yang tertunda untuk korelasi;
- kesehatan transport.

Ini adalah pembukuan klien. Itu bukan state sesi protokol. Pada MCP modern, server tetap menerima versi dan capabilities terkini pada setiap request.

### Membangun setiap request modern dari awal

```python
def modern_request(request_id, method, params, version, capabilities):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {
            **params,
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": version,
                "io.modelcontextprotocol/clientCapabilities": capabilities,
                "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
            },
        },
    }
```

Jangan melampirkan metadata sekali ke objek koneksi dan berasumsi bahwa ia mencapai wire. Cap dan periksa request yang diserialisasi final.

### Discovery modern

`server/discover` mengembalikan versi yang didukung, capabilities server, instruksi, petunjuk cache, dan identitas server yang disarankan. Sebuah klien memilih versi modern tertinggi yang didukung bersama.

Discovery bersifat opsional untuk klien yang hanya-modern, tetapi disarankan di stdio. Beberapa server legacy menerima operasi sebelum inisialisasi, sehingga mengirim `tools/list` lebih dulu dapat menghasilkan sukses yang ambigu. `server/discover` menciptakan batas era yang bersih.

### Probe kompatibilitas stdio

Klien stdio dua-era mengirim `server/discover` dengan metadata modern pilihannya sebelum request lainnya. Ada tiga kelas hasil:

1. **DiscoverResult.** Server adalah modern. Pilih versi yang didukung bersama dan lanjutkan dengan metadata per-request.
2. **Error modern yang dikenali.** Server adalah modern. Untuk `-32022`, pilih dari `data.supported` dan coba ulang dengan id request baru. Untuk error header atau capability, perbaiki request-nya. Jangan kirim `initialize`.
3. **Sinyal ambigu.** Error JSON-RPC yang tidak dikenali, timeout, penutupan koneksi, atau response kosong tidak mengidentifikasi sebuah era. Gagal tertutup kecuali peer persis itu dikonfigurasi untuk kompatibilitas legacy.

Error protokol modern yang dikenali meliputi:

- `-32020` HeaderMismatch
- `-32021` MissingRequiredClientCapability
- `-32022` UnsupportedProtocolVersion

Error modern yang dikenali tetap modern bahkan ketika peer berada di allowlist legacy. Begitu sebuah server membuktikan bahwa ia memahami kosakata error modern, mengirim `initialize` adalah sebuah downgrade.

Jangan memperlakukan `-32601` sebagai bukti legacy positif. Ia hanya membuat peer yang secara eksplisit di-allowlist memenuhi syarat untuk satu probe legacy. Aturan yang sama berlaku untuk timeout, penutupan koneksi, atau response kosong.

### Allowlisting adalah intensi operator, bukan bukti

Kompatibilitas legacy harus menjadi properti eksplisit dari satu konfigurasi peer yang di-pin:

```python
client.add_server("archive", archive_transport, allow_legacy=True)
```

Ikat pilihan itu ke perintah atau endpoint yang dikonfigurasi. Jangan menggunakan wildcard yang memungkinkan server sembarang memilih dirinya sendiri ke semantik yang lebih lemah. Peer tanpa `allow_legacy=True` gagal setelah hasil discovery yang ambigu dan tidak pernah menerima `initialize`.

Allowlist memberikan izin untuk memprobe. Ia tidak memilih era-nya. Klien mengirim satu `initialize` di bawah batas waktu yang ditegakkan transport, lalu mensyaratkan semua hal berikut:

- response JSON-RPC `2.0` dengan id request yang cocok;
- tepat satu `result` dan tanpa `error`;
- sebuah `protocolVersion` di himpunan revisi legacy yang dikonfigurasi klien;
- field `capabilities` bernilai objek;
- objek `serverInfo` dengan field `name` dan `version` berupa string yang tidak kosong.

Timeout, penutupan koneksi, response error, hasil yang rusak, id yang tidak cocok, atau revisi yang tidak didukung gagal tertutup. Hanya hasil positif yang struktur-valid yang memilih era legacy. Kode meneruskan `legacy_probe_timeout_ms` ke adapter transport; adapter stdio atau HTTP sungguhan harus menegakkan batas waktu itu alih-alih sekadar mencatatnya.

Cache era yang dipilih untuk transport peer. Jangan memprobe lagi sebelum setiap panggilan.

### Legacy adalah cabang kompatibilitas

Setelah probe terbatas mengembalikan bukti legacy positif yang valid, klien menggunakan versi legacy yang dipilih persis sebagaimana didefinisikan oleh revisi itu:

1. Verifikasi envelope response dan id korelasi.
2. Verifikasi revisi yang dinegosiasikan berada di himpunan legacy yang dikonfigurasi.
3. Catat capabilities yang tervalidasi dan identitas server.
4. Kirim `notifications/initialized` hanya setelah semua pemeriksaan lulus.
5. Gunakan bentuk request legacy selama umur transport itu.

Cabang ini ada untuk interoperabilitas dengan peer yang diketahui. Itu bukan desain default untuk server atau request baru. Jika transport restart atau endpoint-nya berubah, buang cache peer-era dan negosiasikan ulang.

### Menemukan dan meng-cache tools

Untuk setiap peer aktif, panggil `tools/list`. Hasil modern menyertakan `resultType`, `ttlMs`, dan `cacheScope`. Hormati petunjuk kesegaran di dalam konteks otorisasi yang tepat. Ambil ulang setelah kedaluwarsa atau event list-change yang di-subscribe.

Klien harus memperlakukan `resultType` yang hilang dari server legacy sebagai `"complete"`. Jangan mensyaratkan field cache modern pada response dari era yang dinegosiasikan lebih awal.

Server seharusnya mengembalikan urutan deterministik. Klien juga seharusnya mengurutkan sebelum menggabungkan sehingga urutan registry lokal tidak bergantung pada waktu startup proses.

### Penggabungan namespace yang aman-tabrakan

Dua server dapat sama-sama mengekspos `search`. Pilih kebijakan yang dideklarasikan:

1. **Prefix saat tabrakan.** Pertahankan nama kanonis pertama dan ekspos tabrakan kemudian sebagai `<server>/<tool>`.
2. **Tolak saat tabrakan.** Jangan memuat duplikatnya dan tampilkan error konfigurasi yang jelas.
3. **Timpa diam-diam.** Jangan pernah menggunakan ini. Ia menyembunyikan server mana yang menerima tindakan yang dipilih model.

Simpan nama kanonis dan lokal. Model melihat nama kanonis. `tools/call` yang keluar memakai nama lokal yang dideklarasikan server pemiliknya.

### Merutekan sebuah panggilan

Routing adalah lookup murni:

```text
canonical tool name
  -> peer name + local tool name
  -> new JSON-RPC request id
  -> modern request metadata or explicit legacy shape
  -> matching response id
```

Jangan mengirim panggilan ketika transport pemiliknya tidak tersedia. Sambungkan ulang atau restart transport, lalu jalankan ulang discovery dan `tools/list`. Request modern yang tertunda dan hilang di transport yang rusak dapat dicoba ulang dengan id JSON-RPC baru ketika kebijakan keamanan operasinya mengizinkan.

### Notifikasi dan subscription

Perubahan list dan resource modern tiba hanya pada stream `subscriptions/listen` yang dibuka klien. Klien mengirim filter notifikasi, menunggu `notifications/subscriptions/acknowledged`, dan mengorelasikan event dengan id request listen di metadata notifikasi.

Saat putus koneksi, buka request listen baru dan ambil ulang list atau resource yang relevan. Stream modern tidak dilanjutkan dengan `Last-Event-ID`.

### Tanpa request yang dimulai server

Server modern tidak memanggil klien dengan request JSON-RPC yang mandiri untuk sampling, elicitation, atau roots. Mereka mengembalikan `input_required`, dan klien mencoba ulang request asli setelah memenuhi input request yang tertanam.

Jangan memblokir reader response peer saat memenuhi input. Pertahankan korelasi dan buat id JSON-RPC baru untuk retry.

```figure
tp-client-merge
```

## Gunakan

`code/main.py` memakai fungsi peer in-process sehingga keputusan protokol tetap terlihat. Ia terhubung ke dua peer modern dan satu peer legacy yang sengaja di-allowlist, lalu menggabungkan dan merutekan tools mereka. Callable transport menerima anggaran timeout sehingga cabang kompatibilitas tidak dapat menyembunyikan probe tanpa batas.

```bash
cd code
python3 main.py
python3 -m unittest discover tests -v
```

Test-test membuktikan batas-batas yang dilewati demo normal:

- request modern mengulang metadata;
- `-32022` mencoba ulang discovery modern tanpa inisialisasi;
- error modern yang dikenali tidak pernah menurunkan, bahkan untuk peer yang di-allowlist;
- timeout, penutupan koneksi, response kosong, dan error yang tidak dikenali tidak memicu `initialize` tanpa allowlist;
- peer yang di-allowlist menjadi legacy hanya setelah hasil `initialize` yang valid dan didukung;
- hasil legacy yang rusak dan tidak didukung meninggalkan peer tidak tersedia;
- era yang berhasil dipilih di-cache selama umur transport.

## Rilis

Pelajaran ini mengirimkan `outputs/skill-mcp-client-harness.md`. Ia menycaffold penempelan request modern, negosiasi era stdio, penggabungan namespace deterministik, routing, dan cabang kompatibilitas legacy yang gagal-tertutup.

## Latihan

1. Buat server palsu mengembalikan `-32022` tanpa versi yang didukung bersama. Konfirmasikan klien gagal alih-alih mengirim `initialize`.
2. Allowlist server legacy palsu, buat probe `initialize` terbatasnya timeout, dan buktikan peer tetap `unknown` dan tidak tersedia.
3. Tambahkan daftar tool `cacheScope: "private"` untuk dua konteks otorisasi. Konfirmasikan klien tidak pernah membagikan hasil cache satu konteks ke konteks lainnya.
4. Ubah kebijakan tabrakan menjadi penolakan dan buat startup gagal dengan kedua nama peer di dalam error.
5. Tambahkan simulator `subscriptions/listen` yang finit. Saat stream hilang, listen ulang dengan id request baru dan ambil ulang tools.

## Istilah Kunci

| Term | Meaning |
|------|---------|
| Peer | Client-side record for one server transport and its discovered data |
| Protocol era | Modern per-request metadata or legacy initialization semantics |
| Discovery probe | Initial `server/discover` used to identify the stdio era |
| Recognized modern error | Error that proves modern behavior and forbids legacy fallback |
| Legacy allowlist | Operator configuration permitting one bounded compatibility probe for a pinned peer |
| Positive legacy evidence | Valid, correlated `initialize` result for an explicitly supported legacy revision |
| Merged namespace | Canonical tool names across all active peers |
| Collision policy | Prefix or reject rule for duplicate tool names |
| Era cache | Selected modern or legacy behavior stored for one transport peer |
| Transport recovery | Restart or reconnect, rediscover, relist, and retry safely with a new id |

## Bacaan Lanjutan

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Versioning](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
