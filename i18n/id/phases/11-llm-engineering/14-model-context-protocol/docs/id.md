# Protokol Konteks Model (MCP)

> MCPmemberi host AI satu protokol untuk menemukan dan memanggil alat, sumber daya, dan perintah. Revisi 28-07-2026 membuat protokol tersebut tidak memiliki kewarganegaraan: kemampuan dan konteks versi berjalan seiring dengan setiap permintaan, bukan dalam jabat tangan yang terikat koneksi.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 03 (Structured Outputs)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Membedakan sebuahMCPhost, klien, server, transport, dan server primitif.
- Membangun sebuahJSON-RPCpermintaan dengan metadata yang dibutuhkan olehMCP28-07-2026.
- Gunakan`server/discover`untuk memeriksa versi, identitas, dan kemampuan.
- Mengembalikan hasil yang diketik dan mengetahui cache dari alat, sumber daya, dan perintah.
- Jelaskan betapa modernnya tanpa kewarganegaraanMCPberinteroperasi dengan server era jabat tangan.
- Pilih batas kondisi aman, transportasi, dan persetujuan untuk server.

## Masalahnya

Aplikasi kamu memerlukan kueri database, operasi kalender, dan pembaca file. Tanpa protokol bersama, setiap host AI memerlukan penemuan khusus, pemanggilan, kesalahan, pengangkutan, dan perekat otorisasi untuk kemampuan yang sama.

MCPmengurangi matriks integrasi itu. Server menerbitkan standarJSON-RPCpermukaan. Klien yang patuh dapat menemukan permukaan, menyajikannya kepada model atau pengguna, memanggilnya, dan menafsirkan hasilnya tanpa adaptor khusus server.

Batasan yang penting mudah untuk dilewatkan.MCPmenstandarkan komunikasi. Ini tidak memutuskan alat mana yang harus dipanggil oleh model, membuat konten yang tidak tepercaya menjadi aman, atau mengubah permintaan tanpa kewarganegaraan menjadi status aplikasi yang tahan lama. Host dan server kamu masih memiliki keputusan tersebut.

## Konsepnya

![MCP host, stateless request, and server primitives](../assets/mcp-architecture.svg)

### Tiga server primitif

1. **Alat** adalah tindakan yang dapat dipanggil. Setiap alat memiliki nama, deskripsi,JSONMasukan skema, dan penangan.
2. **Sumber daya** diberi nama,URI-konten beralamat yang dapat dibaca klien.
3. **Permintaan** adalah templat yang dapat digunakan kembali yang dapat diekspos oleh host kepada pengguna.

Tuan rumahnya adalah aplikasi AI. SebuahMCPklien di dalam host itu berbicara ke satu server. Transportasi membawaJSON-RPCpesan di antara mereka.

### Permintaan tanpa kewarganegaraan menggantikan jabat tangan

MCP28-07-2026 dihapus`initialize`dan`notifications/initialized`. Ini juga menghapus sesi tingkat protokol. Setiap permintaan membawa konteks yang diperlukan untuk menafsirkannya`params._meta`:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Versi protokol dan kemampuan klien diperlukan. Identitas klien direkomendasikan. Hilang`_meta`, bidang wajib yang hilang, atau bidang wajib dengan tipe yang salah memiliki format yang salah dan mengembalikan Param yang Tidak Valid (`-32602`). String versi yang terbentuk dengan baik yang tidak didukung oleh server akan dikembalikan`UnsupportedProtocolVersionError`(`-32022`). Server dapat memproses permintaan yang valid tanpa memulihkan catatan negosiasi sebelumnya.

Tanpa kewarganegaraan bukan berarti suatu aplikasi tidak akan pernah dapat mempertahankan statusnya. Artinya negara tidak tersembunyi di balik suatuMCPkoneksi atau`Mcp-Session-Id`. Jika alur kerja memerlukan kesinambungan, server akan mencetak pegangan buram dan klien meneruskan pegangan tersebut sebagai argumen alat biasa pada panggilan berikutnya. Otorisasi tetap harus diperiksa pada setiap permintaan.

### Penemuan dan pemilihan versi

Setiap server modern mengimplementasikannya`server/discover`. Hasilnya mengiklankan versi, kemampuan, dan identitas server yang didukung:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "complete",
    "supportedVersions": ["2026-07-28"],
    "capabilities": {
      "tools": {},
      "resources": {},
      "prompts": {}
    },
    "ttlMs": 3600000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "demo-server",
        "version": "1.0.0"
      }
    }
  }
}
```

Klien dapat memanggil metode lain secara langsung dan menangani kesalahan versi, namun penemuan membuat tampilan kemampuan dan pemilihan versi menjadi eksplisit. Versi yang tidak didukung kembali`UnsupportedProtocolVersionError`dengan kode`-32022`. Datanya berisi`supported`, serangkaian revisi server, dan`requested`, revisi yang ditolak.

Di stdio, klien era ganda menyelidiki dengan`server/discover`. Hasil penemuan atau kesalahan modern yang diketahui seperti`UnsupportedProtocolVersionError`mengidentifikasi server modern. Kesalahan atau batas waktu apa pun yang tidak dikenali sebagai izin modern memungkinkan kembali ke 25-11-2025`initialize`mengalir. Perilaku lama adalah kode kompatibilitas, bukan standar modern.

### Hasilnya eksplisit

Setiap hasil inti 28-07-2026 memiliki`resultType`:

- `complete`berarti operasi selesai.
- `input_required`berarti server memerlukan perjalanan pulang pergi lagi melalui pola Permintaan Multi Pulang Pergi. Server inti hanya dapat mengembalikannya dari`tools/call`,`resources/read`, atau`prompts/get`.

Klien harus memperlakukan hasil warisan yang dihilangkan`resultType`sebagai lengkap.

Server harus menyertakan`io.modelcontextprotocol/serverInfo`dalam setiap hasil`_meta`. Identitas ini dilaporkan sendiri dan untuk tampilan, pencatatan, dan proses debug, bukan untuk keputusan keamanan.

Daftar dan baca hasilnya juga dibawa`ttlMs`dan`cacheScope`. Sebuah deterministik`tools/list`pesanan ditambah petunjuk kesegaran memungkinkan penemuan cache klien dengan aman dan meningkatkan stabilitas cache cepat.`cacheScope: public`mengizinkan cache bersama;`private`membatasi penggunaan kembali pada konteks panggilan.

### Format kawat dan transportasi

MCPkegunaanJSON-RPC2.0 melalui stdio atau StreamableHTTP.

- Sebuah permintaan telah`jsonrpc`,`id`,`method`, dan`params`.
- Sebuah respon mempunyai kecocokan`id`dan keduanya`result`atau`error`.
- Notifikasi tidak ada`id`dan tidak mengharapkan tanggapan.

Modern Dapat DialirkanHTTPmemperlihatkan satu titik akhir yang menerimaPOST. Masing-masingJSON-RPCpesan mendapat miliknya sendiriPOST. Sebuah permintaanPOSTmenerima salah satunyaJSONobjek atau aliran Peristiwa Terkirim Server dengan cakupan permintaan yang diakhiri dengan respons akhir. Pemberitahuan yang diterimaPOSTmenerimaHTTP202 tanpa badan tanggapan; revisi inti ini tidak mendefinisikan pemberitahuan klien-ke-server melalui StreamableHTTP.

Tidak ada yang berdiri sendiriMCP GETaliran,DELETEtitik akhir sesi,`Mcp-Session-Id`, atau`Last-Event-ID`diputar ulang pada 28-07-2026. Pemberitahuan perubahan yang berumur panjang menggunakan a`subscriptions/listen` POSTyang jawabannya tetap terbuka sebagaiSSEaliran.

### Input klien tanpa permintaan yang dimulai oleh server

Revisi yang lebih lama memungkinkan server mengirimkan permintaan seperti`sampling/createMessage`,`roots/list`, atau`elicitation/create`di atas sungai. Protokol saat ini menggunakan Permintaan Multi Pulang Pergi. Panggilan alat yang memenuhi syarat, pembacaan sumber daya, atau pengembalian cepat`resultType: input_required`dengan setidaknya satu dari`inputRequests`atau`requestState`. Klien mengumpulkan masukan apa pun yang diminta, mencoba kembali metode asli dengan yang baruJSON-RPCID dan yang sesuai`inputResponses`, dan menggemakan hal yang sama persis`requestState`ketika satu disediakan. Jika tidak`inputRequests`hadir, percobaan ulang dihilangkan`inputResponses`.

Roots, Sampling, dan Logging tetap berfungsi tetapi tidak digunakan lagi, sehingga implementasi baru tidak boleh mengadopsinya. Permintaan Roots atau Sampling yang ada masuk ke dalamMRTR `inputRequests`, tidak pernah sebagai server-ke-klien yang independenJSON-RPCpermintaan. Lebih memilih parameter file atau direktori eksplisit, URI sumber daya, konfigurasi server, dan integrasi penyedia model langsung. Gunakan stderr untuk diagnostik stdio dan OpenTelemetry untuk telemetri produksi.

```figure
mcp-nxm-collapse
```

## Bangun Sendiri

### Langkah 1: mendaftarkan permukaan server

Pendaftaran tetap sederhana meskipun kontrak permintaan berubah:

```python
server = MCPServer("demo-server")

@server.tool(
    "add",
    "Add two integers.",
    {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    }
)
def add(a: int, b: int) -> dict:
    return {"sum": a + b}
```

Implementasi yang dikirimkan di`code/main.py`juga mendaftarkan sumber daya dan prompt. Ini sengaja menggunakan perpustakaan standar sehingga kamu dapat melihat setiap amplop daripada mendelegasikan protokol ke satuSDK.

### Langkah 2: lampirkan metadata ke setiap permintaan

```python
def request(method, params=None):
    body_params = dict(params or {})
    body_params["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": "demo-client",
            "version": "1.0.0"
        }
    }
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": body_params
    }
```

Jangan menyimpan metadata ini dalam cache hanya di objek koneksi. Server memvalidasinya pada setiap permintaan.

### Langkah 3: secara opsional temukan sebelum mendaftar

Telepon`server/discover`, pilih versi yang didukung, lalu telepon`tools/list`. Langsung`tools/list`juga valid jika kamu sudah mengetahui versinya dan dapat menanganinya`-32022`.

Demo mengembalikan daftar alat dalam urutan nama dan lampirannya`ttlMs`,`cacheScope`,`resultType`, dan identitas server. Pemanggilan alat mengembalikan hasil yang lengkap dan tidak dapat di-cache karena keluarannya dapat bergantung pada keadaan saat ini.

### Langkah 4: memetakan permintaan yang sama keHTTP

Sebuah jarak jauh`tools/call` POSTtermasuk header yang mencerminkanJSON-RPCtubuh:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: application/json, text/event-stream
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: add
```

Itu`MCP-Protocol-Version`header harus sesuai dengan versi di`_meta`.`Mcp-Method`diperlukan pada setiapJSON-RPCpermintaan dan harus cocok`method`.`Mcp-Name`diperlukan hanya untuk`tools/call`,`resources/read`, dan`prompts/get`, yang mana harus sesuai dengan nama alat, sumber dayaURI, atau nama prompt. Header wajib yang hilang atau ketidakcocokan muncul kembaliHTTP400 dengan`HeaderMismatch`kode`-32020`.

### Langkah 5: menegakkan keamanan di luar negara protokol

- Validasi otorisasi dan audiensi pada setiapHTTPpermintaan.
- Ikat server lokal ke localhost dan validasi`Origin`di Dapat DialirkanHTTP.
- Tandai alat yang bermutasi dengan`destructiveHint: true`dan memerlukan persetujuan tuan rumah.
- Lewati direktori dan cakupan file secara eksplisit alih-alih bergantung pada Root yang tidak digunakan lagi.
- Perlakukan sumber daya dan keluaran alat sebagai data yang tidak tepercaya.
- Biarkan stdout dicadangkanJSON-RPCdi bawah stdio; tulis diagnostik ke stderr.

## Gunakan

Jalankan pelajaran dari direktorinya:

```bash
python3 code/main.py
cd code
python3 -m unittest discover tests -v
```

Baris pertama harus melaporkan penemuan`demo-server`di protokol`2026-07-28`. Kemudian periksa`MCPClient.request`: itu merekonstruksi`_meta`untuk setiap panggilan. Hapus metadata dari satu permintaan dan amati server menolaknya.

## Rilis

`outputs/skill-mcp-server-designer.md`mengubah domain menjadi tanpa kewarganegaraanMCPdesain. Gerbang penerimaannya memerlukan hasil penemuan, kebijakan metadata per permintaan, daftar cache-aware deterministik, penanganan status eksplisit, header transport, otorisasi, dan aturan persetujuan.

## LanjutkanMCPMenyelam Lebih Dalam

Pelajaran ini memberi kamu model protokol. Fase 13 mengubah empat batasan produksi menjadi pelajaran membangun dan memverifikasi yang terpisah:

1. [MCPKontrak Alat dan Konten](../../../13-tools-and-protocols/28-mcp-tool-contracts-and-content/docs/en.md) mencakup skema masukan tertutup, konten terstruktur, metadata perutean, penomoran halaman buram, otorisasi penyelesaian, dan perbedaan antara kesalahan protokol dan domain alat.
2. [MCPKeandalan, Pembatalan, dan Kontrol Aliran](../../../13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/docs/en.md) mencakup pembatalan permintaan, pembatalan tugas yang tahan lama, tenggat waktu, idempotensi, tekanan balik, buffering proxy, dan perilaku menyambungkan kembali.
3. [MCPRantai Pasokan Registri, Penerimaan, Drift, dan Rollback](../../../13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/docs/en.md) mencakup bukti namespace, asal artefak, pin yang tidak dapat diubah, live drift, status Registri, bukti penerimaan, dan rollback.
4. [MCPRekayasa Kesesuaian](../../../13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/docs/en.md) mencakup transkrip kawat emas dan negatif, era versi ketat,SDKperbedaan, bukti proksi, redaksi, gerbang kesehatan, dan rollback rilis.

Ikuti mereka secara berurutan ketika server akan melewati batas tim atau kepercayaan. Bersama-sama mereka beralih dari “metode yang berhasil” menjadi “kontrak tetap aman dan dapat didiagnosis melalui penerapan.”

## Latihan

1. Tambahkan sebuah`subtract`alat dan konfirmasi`tools/list`tetap diurutkan berdasarkan abjad.
2. Hapus kunci versi protokol dan verifikasi Params Tidak Valid (`-32602`). Kemudian kirimkan versi yang sudah dibuat dengan baik tetapi tidak didukung`2025-11-25`, verifikasi`-32022`, konfirmasi`requested`menggemakan revisi itu, dan memilih dari`supported`.
3. Tambahkan server-minted`draftId`untuk operasi pembuatan, lalu memerlukannya sebagai argumen untuk memperbarui. Jelaskan mengapa ini merupakan status aplikasi dan bukan sesi protokol.
4. Kembali`input_required`dari alat yang memerlukan konfirmasi pengguna. Coba lagi panggilan asli dengan ID baru, dan`inputResponses`masuk, dan tepatnya`requestState`alih-alih menciptakan server-ke-klienJSON-RPCpermintaan.
5. Buat sketsa klien stdio era ganda. Perlakukan hasil atau kesalahan modern yang diakui sebagai kesalahan modern, dan izinkan fallback ke kesalahan tersebut`initialize`hanya untuk kesalahan atau batas waktu yang tidak dikenali.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| MCP | "Tool protocol for LLMs" | JSON-RPC protocol for server discovery, tools, resources, prompts, and extensions |
| Host | "The AI app" | Owns the model and UI and mounts one or more MCP clients |
| Client | "The connector" | Speaks MCP to one server on behalf of a host |
| Stateless MCP | "No session" | Every request carries version and capabilities; no protocol state is keyed by a connection |
| `server/discover` | "Capability probe" | Required server method advertising versions, capabilities, and identity |
| `resultType` | "Result state" | Marks a result as `complete` or `input_required` |
| State handle | "Workflow id" | Server-minted application identifier passed as an ordinary argument |
| Streamable HTTP | "Remote transport" | One POST endpoint with JSON or request-scoped SSE responses |
| MRTR | "Ask and retry" | Input request embedded in a result, followed by a retry of the original operation |

## Bacaan Lebih Lanjut

- [MCPPerubahan penting pada 28-07-2026](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCPpenemuan server](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCPDapat dialirkanHTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCPPermintaan Multi-Pulang Pergi](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCPfitur yang tidak digunakan lagi](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
