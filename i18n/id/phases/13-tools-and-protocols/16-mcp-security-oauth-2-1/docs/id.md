# MCP Authorization: CIMD, Issuer Binding, PKCE, dan Step-Up

> Sebuah request MCP jarak jauh bersifat tanpa-state, tetapi otorisasinya bukan anonim. Ikat setiap kredensial ke issuer yang membuatnya dan setiap token ke resource yang menerimanya.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 15 (security)
**Time:** ~90 minutes

## Tujuan Pembelajaran

- Temukan authorization server melalui metadata protected-resource.
- Utamakan Client ID Metadata Documents di atas Dynamic Client Registration yang dideprecate.
- Deklarasikan `application_type` yang benar ketika jalur kompatibilitas DCR tidak bisa dihindari.
- Validasi `iss` respons otorisasi dan isolasikan kredensial berdasarkan issuer.
- Gunakan PKCE, resource indicators, validasi audience, dan incremental scopes.
- Kirim request MCP 2026-07-28 yang diotorisasi tanpa sesi protokol.

## Masalahnya

Sebuah MCP server jarak jauh boleh membaca record privat, menulis sistem eksternal, atau memicu pekerjaan yang mahal. Autentikasi memberi tahu siapa yang mengajukan sebuah kredensial. Otorisasi juga harus menjawab:

- Authorization server mana yang menerbitkan kredensial itu?
- MCP resource mana yang dituju token itu?
- Klien dan redirect URI mana yang menyelesaikan alurnya?
- Operasi apa saja yang disetujui pengguna?
- Apakah request yang persis ini masih cocok dengan persetujuan itu?

Profil otorisasi 2026-07-28 memperkeras pendaftaran klien dan penanganan issuer. Ia mengutamakan Client ID Metadata Documents, men-deprecate Dynamic Client Registration, mewajibkan `application_type` yang tepat pada DCR, memvalidasi respons issuer RFC 9207, dan melarang penggunaan ulang kredensial lintas issuer.

Aturan-aturan ini melengkapi inti tanpa-state. Itu tidak memulihkan handshake inti maupun `Mcp-Session-Id`.

## Konsepnya

### Kenali tiga peran

- **MCP client:** mengirim request atas nama resource owner.
- **MCP resource server:** menerima access token dan menyajikan endpoint MCP.
- **Authorization server:** mengautentikasi resource owner, mengumpulkan consent, dan menerbitkan token.

Resource server dan authorization server bisa dioperasikan bersama, tetapi jaga pengenal dan tanggung jawab validasinya tetap terpisah.

### Otorisasi berlaku untuk HTTP

Spesifikasi otorisasi MCP berlaku untuk transport berbasis HTTP. Sebuah server stdio lokal berjalan di bawah batas kepercayaan proses dan sistem operasi. Jangan menambahkan alur OAuth browser palsu ke stdio hanya demi simetri.

Untuk Streamable HTTP jarak jauh, kirim bearer token di header `Authorization` pada setiap request. Jangan pernah menaruhnya di URL.

### Mulai dari metadata protected-resource

Resource server memublikasikan metadata RFC 9728:

```json
{
  "resource": "https://notes.example.com/mcp",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:delete", "notes:read", "notes:write"]
}
```

Klien mulai dari URL resource MCP, mengambil dokumen ini, memilih authorization server yang diiklankan, lalu mengambil metadata OAuth atau OpenID Connect server itu.

Pertahankan path resource saat menyusun URL well-known RFC 9728. Untuk resource `https://notes.example.com/mcp`, pelajaran ini memakai `https://notes.example.com/.well-known/oauth-protected-resource/mcp`. Membuang akhiran `/mcp` bisa memilih metadata untuk protected resource lain di origin yang sama.

Jangan menebak authorization server dari sebuah hostname. Jangan mengikuti issuer yang ditemukan dari body error yang belum divalidasi. Simpan kebijakan untuk issuer mana yang bersedia dipercaya klien.

### Verifikasi metadata authorization server

Metadata seharusnya mengekspos endpoint dan kontrol yang didukung:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "code_challenge_methods_supported": ["S256"],
  "authorization_response_iss_parameter_supported": true,
  "client_id_metadata_document_supported": true
}
```

Wajibkan S256 untuk PKCE. Catat string issuer yang persis. Nilai persis itulah yang menjadi kunci untuk pendaftaran dan penyimpanan token.

### Ikuti prioritas pendaftaran

Gunakan informasi klien yang sudah terdaftar-pre ketika klien sudah memiliki hubungan eksplisit dengan issuer yang dipilih. Jika tidak, utamakan Client ID Metadata Documents ketika authorization server mengiklankan dukungan. Gunakan DCR hanya sebagai fallback kompatibilitas yang dideprecate, lalu minta informasi klien jika tidak ada mekanisme itu yang tersedia.

### Utamakan Client ID Metadata Documents

Sebuah Client ID Metadata Document memberi authorization server sebuah URL HTTPS yang sekaligus merupakan pengenal klien dan lokasi metadata-nya:

```json
{
  "client_id": "https://client.example.com/oauth/metadata.json",
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

Authorization server mengambil dan memvalidasi dokumen itu. `client_id` harus berupa URL HTTPS dengan path, dan nilai di dalam dokumen harus sama persis dengan URL itu. Field dokumen yang wajib adalah `client_id`, `client_name`, dan `redirect_uris`. `application_type` muncul dalam contoh ini tetapi bukan sebuah kebutuhan CIMD. Penggunaan wajibnya yang baru secara khusus adalah jalur DCR.

Perlakukan pengambilan dokumen sebagai operasi yang sensitif-SSRF. Resolve dan validasi tujuannya, tolak alamat loopback, privat, link-local, dan alamat lain yang tidak diizinkan, periksa ulang setelah redirect dan perubahan DNS, batasi redirect, byte, dan waktu, wajibkan JSON, dan cache hanya sesuai kontrol cache HTTP yang tervalidasi. Perlakukan `client_name` dan field tampilan lain sebagai teks yang tidak tepercaya.

CIMD menghilangkan kebutuhan membuat pengenal dinamis baru untuk setiap kontak pertama. Ia tidak menghilangkan validasi redirect URI, kebijakan issuer, atau consent pengguna.

### DCR adalah jalur kompatibilitas

Dynamic Client Registration tetap tersedia untuk authorization server yang lebih lama, tetapi sudah dideprecate untuk implementasi MCP baru.

Saat memakai DCR, deklarasikan `application_type`:

```json
{
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

- Klien desktop, mobile, command-line, dan loopback memakai `native`.
- Aplikasi browser yang di-hosting jarak jauh memakai `web` dan redirect HTTPS jarak jauh.

Membuang field ini bisa ber-default ke `web` dalam implementasi pendaftaran OpenID Connect dan membuat redirect loopback yang sah menjadi gagal.

Simpan kode DCR di belakang keputusan fallback yang eksplisit. Jangan diam-diam jatuh ke fallback setelah kegagalan validasi CIMD yang sewenang-wenang. Itu bisa mengubah sebuah kegagalan keamanan menjadi jalur pendaftaran yang lebih lemah.

### Ikatkan kredensial ke issuer

Simpan materi pendaftaran yang dibuat issuer di bawah issuer yang persis:

```text
issuer_credentials[issuer] = pre_registered_or_dcr_client
tokens[(issuer, resource)] = access_token
```

Jika discovery protected-resource berubah dari `https://auth-one.example` ke `https://auth-two.example`, evaluasi ulang kepercayaan. Jangan pernah mengirim client secret issuer pertama, client id DCR, registration access token, refresh token, atau access token ke yang kedua. Klien terdaftar-pre dan DCR harus memakai kredensial yang diterbitkan untuk issuer yang baru.

Sebuah client id CIMD berbeda karena itu adalah URL HTTPS yang di-hosting-sendiri, bukan kredensial yang dibuat authorization server. URL CIMD yang sama bersifat portabel: issuer tepercaya yang baru mengambil dan memvalidasi dokumen tanpa pendaftaran ulang DCR. Respons otorisasi dan token tetap divalidasi dan disimpan di bawah issuer yang baru.

### Authorization code dengan PKCE

Alur interaktifnya adalah:

1. Buat `code_verifier` berentropi tinggi.
2. Turunkan `code_challenge` S256.
3. Kirim request otorisasi dengan `client_id`, `redirect_uri`, `scope`, `code_challenge`, dan `resource` yang persis.
4. Terima respons otorisasi yang berisi `code` dan, bila disediakan, `iss`.
5. Validasi `iss` terhadap issuer yang persis tercatat sebelum memakai field respons apa pun.
6. Tukar kode dengan `code_verifier`, redirect URI yang sama, dan `resource` yang sama.
7. Simpan token yang dihasilkan di bawah `(issuer, resource)`.

Parameter `resource` dari RFC 8707 muncul di request otorisasi dan token. Ia mengidentifikasi URI MCP server yang kanonis.

### Validasi `iss` dengan tepat

RFC 9207 mencegah respons otorisasi dari satu issuer tertukar dengan respons dari issuer lain.

Ketika `iss` ada, bandingkan dengan issuer yang tercatat tanpa case folding, perubahan slash-trailing, penghapusan port-default, atau normalisasi percent-encoding. Saat tidak cocok, jangan bertindak berdasarkan kode itu atau bahkan menampilkan detail error yang dikendalikan penyerang dari respons itu.

Sebuah authorization server yang menyertakan `iss` mengiklankan `authorization_response_iss_parameter_supported: true`. Klien saat ini tetap memvalidasi `iss` yang ada walaupun iklan itu hilang.

### Validasi audience di MCP server

Resource server hanya menerima token yang diterbitkan untuk dirinya sendiri:

```text
token.issuer == configured_authorization_server
token.audience == canonical_mcp_resource
```

Token yang tidak valid, kedaluwarsa, issuer-nya salah, atau audience-nya salah menerima 401. MCP server tidak boleh menerima atau meneruskan token yang ditujukan untuk layanan lain.

### Minta scope saat ini yang paling kecil

Mulai dari scope yang dibutuhkan sekarang. Jika sebuah tool berikutnya membutuhkan lebih, server mengembalikan 403 dengan tantangan scope yang otoritatif:

```text
WWW-Authenticate: Bearer error="insufficient_scope",
  scope="notes:delete",
  resource_metadata="https://notes.example.com/.well-known/oauth-protected-resource/mcp"
```

Klien menjelaskan izin baru, memperoleh consent, melakukan alur otorisasi baru dengan himpunan scope gabungan, dan mencoba ulang request MCP dengan id JSON-RPC baru.

Jangan berasumsi scope yang ditantang merupakan subset dari `scopes_supported`. Tantangan itu otoritatif untuk operasi saat ini.

### Otorisasi dan wire MCP tanpa-state

Sebuah panggilan tool yang diotorisasi tetap membawa envelope request saat ini yang lengkap:

```text
POST /mcp
Authorization: Bearer <access-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.delete
```

```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "notes.delete",
    "arguments": {"id": "note-7"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "oauth-lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Token mengotorisasi principal. Metadata request menegosiasikan perilaku protokol. Keduanya tidak saling menggantikan.

Validasi wire dalam urutan tetap: tipe JSON-RPC dan metadata, kesetaraan header dan body, lalu dukungan protokol. Ketidakcocokan routing atau header versi mengembalikan HTTP 400 dengan `-32020`. Jika header dan body sepakat pada versi yang tidak didukung, kembalikan HTTP 400 dengan `-32022` dan `data` persis `{"supported":["2026-07-28"],"requested":"<actual>"}`. Method yang tidak diketahui mengembalikan HTTP 404 dengan `-32601`.

Setiap error request, termasuk 401 token tidak valid dan 403 scope tidak cukup, adalah envelope error JSON-RPC dengan `id` request asli. Informasi pemulihan terstruktur berada di `data` error opsional; `WWW-Authenticate` tetap merupakan header respons HTTP. Sebuah notifikasi tidak memiliki `id`, sehingga ia tidak menerima body JSON-RPC. Sebuah notifikasi HTTP yang diterima mengembalikan 202 dengan body kosong.

Server mengimplementasikan `server/discover` dan mengiklankan tools, sehingga ia juga mengimplementasikan method `tools/list` yang wajib. Deskriptor tool-nya memiliki nama, deskripsi, dan nilai `inputSchema` berakar-objek yang stabil. Daftarnya deterministik dan mengembalikan `resultType`, metadata identitas server, `ttlMs` yang berbatas, dan `cacheScope`. Discovery dan daftar tool yang independen-pengguna bisa tersedia sebelum otorisasi. Terapkan kebijakan normal dan cache privat jika keduanya bervariasi per principal.

### Tidak ada token passthrough

Sebuah MCP server tidak boleh meneruskan access token MCP klien ke API hilir. Peroleh token hilir terpisah dengan audience yang tepat atau gunakan desain token-exchange yang eksplisit. Validasi audience hanya bekerja ketika layanan menolak token yang dibuat untuk orang lain.

### Refresh token

Refresh token bersifat opsional. Ketika diterbitkan, simpan secara rahasia dan kuncinya berdasarkan issuer dan resource. Jangan berasumsi bahwa itu ada. Rotasikan ketika authorization server mendukung rotation dan deteksi penggunaan ulang nilai yang sudah dibatalkan.

```figure
t3-scope-stepup
```

## Bangun Sendiri

`code/main.py` adalah simulator protokol dan otorisasi in-process. Ia mengimplementasikan discovery protected-resource, metadata authorization server, pendaftaran CIMD, fallback DCR yang digerbang-versi, pemeriksaan application type, PKCE, validasi issuer, token terikat-resource, scope step-up, `server/discover`, `tools/list`, dan request tool tanpa-state.

Model menerima body request dan header routing yang sudah diparse. Ia bukan adaptor HTTP yang lengkap dan tidak mem-parse `Content-Type` atau `Accept`. Sambungkan ke adaptor Streamable HTTP Lesson 09, yang mewajibkan `Content-Type: application/json` dan nilai `Accept` berisi `application/json` dan `text/event-stream`.

Jalankan:

```bash
cd phases/13-tools-and-protocols/16-mcp-security-oauth-2-1
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Keluarannya menampilkan discovery lebih dulu, pendaftaran CIMD, satu pembacaan biasa, dua scope step-up yang terpisah, dan penyimpanan kredensial berbasis-issuer.

## Gunakan

Petakan objek simulator ke komponen produksi:

- `ResourceServer.protected_resource_metadata` menjadi endpoint RFC 9728.
- `AuthorizationServer.metadata` menjadi discovery RFC 8414 atau OpenID Connect.
- `Client.enroll` menjadi resolusi CIMD plus cabang kompatibilitas DCR yang eksplisit.
- Kredensial klien yang dibuat issuer dan `tokens_by_issuer_resource` menjadi record terenkripsi. Sebuah URL CIMD boleh tetap portabel sementara hasil otorisasinya tetap terikat-issuer.
- `ResourceServer.handle` menjadi middleware yang memvalidasi header MCP, token, dan scope tool saat ini sebelum dispatch sambil menjaga setiap error request tetap dalam envelope JSON-RPC yang cocok.

## Rilis

Pelajaran ini mengirimkan `outputs/skill-oauth-scope-planner.md`. Kini ia merancang prioritas pendaftaran, penyimpanan kredensial terikat-issuer, application type, PKCE, resource indicators, tantangan scope, dan batas request tanpa-state saat ini.

## Latihan

1. Tambahkan rotasi refresh-token dan tolak penggunaan ulang refresh token sebelumnya.
2. Tambahkan allowlist issuer. Saat issuer berubah, gunakan ulang hanya URL CIMD yang portabel; tolak semua kredensial dan token yang dibuat issuer sebelumnya.
3. Tambahkan kedaluwarsa ke authorization code dan konfirmasikan pertukaran yang terlambat gagal.
4. Bangun varian klien web dengan redirect HTTPS jarak jauh dan bandingkan metadata DCR-nya dengan klien native.
5. Tambahkan resource kedua di bawah issuer yang sama. Konfirmasikan access token-nya tidak bisa dipakai di resource pertama.

## Istilah Kunci

| Term | Meaning |
|------|---------|
| Protected-resource metadata | RFC 9728 document that identifies the resource and authorization servers |
| CIMD | HTTPS metadata document whose URL is the OAuth client identifier |
| DCR | Deprecated dynamic client enrollment retained for compatibility |
| `application_type` | `native` or `web`, used to validate redirect URI rules |
| PKCE | Verifier and S256 challenge that protect an intercepted authorization code |
| `iss` | RFC 9207 authorization response issuer identifier |
| Resource indicator | RFC 8707 parameter that binds a token request to an MCP resource |
| Audience | Resource for which a token is valid |
| Step-up | New consent and token issuance for an additional current-operation scope |
| Issuer-bound credentials | Registration and token records isolated by exact authorization server issuer |

## Bacaan Lanjutan

- [MCP 2026-07-28 authorization specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [RFC 9728: OAuth 2.0 Protected Resource Metadata](https://www.rfc-editor.org/rfc/rfc9728)
- [RFC 8707: Resource Indicators for OAuth 2.0](https://www.rfc-editor.org/rfc/rfc8707)
- [RFC 9207: OAuth 2.0 Authorization Server Issuer Identification](https://www.rfc-editor.org/rfc/rfc9207)
- [OAuth Client ID Metadata Document draft](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/)
