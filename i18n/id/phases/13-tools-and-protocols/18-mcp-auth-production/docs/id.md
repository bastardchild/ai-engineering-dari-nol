# MCP Auth di Production: Enrollment dan Token Terikat-Issuer

> Pelajaran 16 membangun state machine OAuth 2.1. Pelajaran ini memperkeras batas production-nya untuk MCP 2026-07-28: Client ID Metadata Documents lebih dulu, dynamic registration yang sudah usang hanya untuk kompatibilitas, validasi issuer authorization-response, kredensial klien ber-kunci issuer, refresh JWKS, dan token terikat-audience pada setiap permintaan stateless.
>
> **Catatan spesifikasi (2026-07-28):** Dynamic Client Registration didepresiasi demi Client ID Metadata Documents. DCR tetap sebagai mekanisme kompatibilitas. Ketika dipakai, klien mendeklarasikan `application_type` yang benar. Sebuah klien memvalidasi nilai RFC 9207 `iss` yang ada dan tidak pernah menggunakan ulang kredensial lintas issuer server-otorisasi.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 16 (OAuth 2.1 state machine), Phase 13 · 17 (gateways)
**Time:** ~90 minutes

## Tujuan Pembelajaran

- Menemukan server otorisasi melalui metadata RFC 8414 dan memverifikasi kontraknya.
- Melakukan enrollment melalui Client ID Metadata Document dan mengisolasi DCR yang sudah usang sebagai fallback.
- Memvalidasi `iss` RFC 9207, pendaftaran kunci berdasarkan issuer server-otorisasi, dan token terikat-sumber daya berbasis issuer plus sumber daya.
- Meng-cache dan menyegarkan kunci JWKS pada jadwal sehingga verifikasi tanda tangan bertahan melewati rotasi kunci.
- Meng-pin token ke satu sumber daya MCP memakai resource indicator RFC 8707 dan menolak penggunaan ulang confused-deputy.
- Memilih validasi JWT atau introspeksi token, menentukan kesegaran pencabutan, dan gagal dengan aman ketika dependensi identitas tidak tersedia.
- Memisahkan server otorisasi, server sumber daya, dan klien sehingga masing-masing hanya menegakkan pemeriksaannya sendiri.
- Mengaudit server otorisasi terhadap checklist deployment dan menolak enrollment atau penggunaan ulang token yang tidak aman.

## Masalahnya

Simulator Pelajaran 16 menjalankan OAuth 2.1 di memori. Production punya tiga celah operasional yang tidak terlihat oleh simulator berbasis memori saja.

Celah pertama adalah isolation enrollment dan kredensial. Organisasi nyata bisa menjalankan ratusan server MCP dan ribuan klien MCP. Revisi 2026-07-28 memilih **Client ID Metadata Document**: klien memakai URL HTTPS dengan path yang ia kendalikan sebagai identitasnya, dan server-otorisasi menarik metadata-nya. Dynamic registration RFC 7591 hanya tersisa sebagai jalur kompatibilitas yang sudah usang. Ketika DCR tak terelakkan, permintaan mendeklarasikan `application_type` yang benar. Klien menyimpan pendaftaran di bawah issuer server-otorisasi dan access token di bawah pasangan `(issuer, resource)`. Issuer yang berarti enrollment baru, dan sumber daya yang berbeda berarti token terikat-audience terpisah.

Celah kedua adalah rotasi kunci. Validasi JWT bergantung pada kunci penandatanganan server-otorisasi, yang dipublikasikan sebagai JSON Web Key Set (JWKS). Server-otorisasi merotasi ini pada jadwal (sering per jam, kadang lebih cepat di bawah respons insiden). Sebuah server MCP yang mengambil JWKS sekali saat boot divalidasi dengan baik sampai jendela rotasi — lalu setiap permintaan gagal sampai restart. Production memasang JWKS sebagai nilai yang di-cache bersama job refresh yang menimpa cache sebelum kunci sebelumnya kedaluwarsa, plus fetch fallback saat cache miss untuk kasus di mana token yang ditandatangani kunci lebih baru dari cache tiba.

Celah ketiga adalah pengikatan audience. Pelajaran 16 memperkenalkan resource indicator RFC 8707. Di production, indikator itu menjadi pemeriksaan klaim yang keras pada setiap permintaan. Server MCP membandingkan `token.aud` dengan URL sumber daya kanoniknya sendiri dan menolak ketidakcocokan dengan HTTP 401. Ini adalah satu-satunya pertahanan terhadap server MCP hulu (atau klien jahat yang memegang token untuk satu server) yang mem-play ulang token itu ke server lain di trust mesh yang sama.

Pelajaran ini memetakan setiap celah ke bagian konkret dari permukaannya. Dokumen metadata adalah sebuah endpoint HTTP. Refresh cache JWKS adalah job terjadwal plus cache key-value. Validasi JWT adalah rutin yang dijalankan server sumber daya sebelum meneruskan tool apa pun. Jaga ketiga peran itu terpisah dan masing-masing hanya menegakkan pemeriksaan yang menjadi miliknya: server-otorisasi menerbitkan dan merotasi kunci, server sumber daya meng-cache dan memvalidasi, klien menemukan dan mendaftar.

## Cakupan: Penegakan Production Setelah Pelajaran 16

[Lesson 16: MCP Security with OAuth 2.1](../../16-mcp-security-oauth-2-1/docs/en.md) memiliki state machine kode-otorisasi, PKCE, discovery resource terproteksi, resource indicator, dan keputusan scope. Pelajaran ini tidak mendefinisikan alur OAuth kedua. Ia dimulai setelah kontrak-kontrak itu ada dan menanyakan bagaimana server sumber daya yang ter-deploy terus menegakkannya selama rotasi kunci, validasi token opaque, pencabutan, kegagalan dependensi, rollout, dan respons insiden.

Batas production lebih sempit dan lebih operasional:

- Jalur JWT memverifikasi issuer ter-pin, algoritma, kunci tanda tangan, audience, klaim waktu, dan scope pada setiap permintaan sambil menyegarkan JWKS dengan aman.
- Jalur token opaque memanggil endpoint introspeksi terautentikasi milik issuer dan memvalidasi status active, audience atau sumber daya, kedaluwarsa, subject, dan scope yang dikembalikan.
- Kebijakan pencabutan menentukan seberapa cepat sebuah kredensial harus berhenti bekerja dan cache mana yang dapat menunda fakta itu.
- Kebijakan kegagalan memutuskan apa yang terjadi ketika infrastruktur discovery, JWKS, introspeksi, atau pencabutan tidak tersedia.
- Bukti mencatat metadata issuer mana, set kunci atau respons introspeksi, klaim token, versi kebijakan, dan alasan penolakan yang menghasilkan keputusan tanpa menyimpan token.

Pembagian ini menjaga pelajaran tetap bisa dikomposisi. Pelajaran 16 membuktikan alurnya. Pelajaran 18 membuktikan bahwa sebuah token tetap tepercaya, atau ditolak, setelah sampai ke jalur permintaan MCP yang nyata.

## Konsepnya

### RFC 8414 — Metadata Server-Otorisasi OAuth

Sebuah dokumen di `/.well-known/oauth-authorization-server` menggambarkan semua yang dibutuhkan klien:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "client_id_metadata_document_supported": true,
  "registration_endpoint": "https://auth.example.com/register",
  "authorization_response_iss_parameter_supported": true,
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

Klien yang diberi URL sumber daya MCP menautkan discovery: `oauth-protected-resource` dari RFC 9728 (dokumen server sumber daya) menamai issuer, lalu `oauth-authorization-server` (RFC ini) menamai setiap endpoint. Klien tidak pernah hard-code URL otorisasi.

Untuk pengenal sumber daya yang punya path, sisipkan segmen well-known sebelum path itu. Sebagai contoh, `https://mcp.example.com/team/server` merepresentasikan metadata protected-resource di `https://mcp.example.com/.well-known/oauth-protected-resource/team/server`. Menambahkan `/.well-known/...` setelah path sumber daya adalah keliru.

Kontrak yang kamu verifikasi sebelum mempercayai IdP untuk MCP:

- `code_challenge_methods_supported` memuat `S256` (PKCE per RFC 7636). Spesifikasi eksplisit: jika field ini **tidak ada**, server-otorisasi tidak mendukung PKCE dan klien **MUST** menolak melanjutkan.
- `grant_types_supported` memuat `authorization_code` dan menolak `password` dan `implicit`.
- Setidaknya satu jalur enrollment tersedia: `client_id_metadata_document_supported: true` (CIMD, yang disukai), klien yang sudah terdaftar, atau `registration_endpoint` (kompatibilitas RFC 7591 yang sudah usang).
- Jika `authorization_response_iss_parameter_supported` bernilai true, klien mewajibkan `iss` RFC 9207 yang dikembalikan dan membandingkannya persis dengan issuer yang direkam sebelum redirect.
- `response_types_supported` persis `["code"]` untuk OAuth 2.1.

Jika `S256` hilang, server MCP menolak ter-deploy terhadap IdP ini — tidak ada mode terdegradasi untuk PKCE. Jika *kedua* jalur enrollment tidak diiklankan dan kamu tidak punya `client_id` terdaftar-pra, kamu juga tidak bisa melakukan enrollment; manifes deployment-nya yang salah, bukan kodenya.

### RFC 9728 (ringkasan) — Metadata Sumber Daya Terproteksi

Pelajaran 16 sudah membahas RFC 9728. Delta di production: dokumen ini adalah satu-satunya tempat klien mencari server-otorisasi yang dipercaya oleh server MCP *ini*. Sebuah server MCP tunggal bisa menerima token dari beberapa IdP (satu untuk staf, satu untuk mitra). RFC 9728 mendeklarasikan himpunan itu; RFC 8414 mendokumentasikan apa yang didukung setiap IdP.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Documents (default yang disarankan)

CIMD membalik pendaftaran dari *push* menjadi *pull*. Alih-alih meminta server-otorisasi membuatkan `client_id`, klien memakai URL HTTPS yang ia kendalikan **sebagai** `client_id`-nya. URL itu me-resolve ke dokumen metadata JSON; server-otorisasi mengambilnya saat diminta selama alur OAuth. Kepercayaan berakar di DNS: jika operator server mempercayai `app.example.com`, ia mempercayai klien yang dilayani dari `https://app.example.com/client.json`. Tidak ada round-trip pendaftaran, tidak ada namespace `client_id` yang bisa habis, tidak ada state per-server yang harus disinkronkan.

Dokumen metadata yang di-host klien:

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

Nilai `client_id` di dokumen **MUST** sama dengan URL tempat dokumen itu dilayani (server-otorisasi memverifikasi ini; ketidakcocokan ditolak). Server-otorisasi mengiklankan dukungan dengan `client_id_metadata_document_supported: true` di metadata RFC 8414-nya.

Untuk kontrak CIMD saat ini, `client_id`, `client_name`, dan array `redirect_uris` yang tidak kosong adalah wajib. Pengenal klien adalah URL HTTPS absolut dengan path. `application_type` boleh disertakan, tetapi itu bukan field CIMD yang wajib. Jangan menyalin persyaratan DCR untuk `application_type` ke jalur CIMD yang disukai.

Dua fakta keamanan yang blak-blakan disebut spesifikasi:

- **SSRF.** Server-otorisasi mengambil URL yang disuplai penyerang. Ia harus membela terhadap server-side request forgery (tanpa fetch ke endpoint internal/admin).
- **Impersonasi localhost.** CIMD saja tidak dapat menghentikan penyerang lokal mengklaim URL metadata klien yang sah dan mengikat redirect `localhost` apa pun. Server-otorisasi **MUST** menampilkan hostname redirect URI secara jelas saat consent dan **SHOULD** memperingatkan untuk redirect yang hanya `localhost`.

Karena CIMD tidak membutuhkan state sisi-server, tidak ada registrar yang harus dibangun sebagaimana DCR mensyaratkan. Sisi klien bersifat read-only: layani dokumen metadata kamu dari endpoint HTTPS statis dan biarkan server-otorisasi menariknya.

Jika operator server-otorisasi sudah menyediakan pengenal klien, gunakan pendaftaran ber-scope issuer itu sebelum mencoba enrollment otomatis. Jika tidak, pilih CIMD. Gunakan DCR yang sudah usang hanya jika issuer tidak dapat memakai pra-pendaftaran maupun CIMD.

### RFC 7591: enrollment kompatibilitas yang sudah usang

DCR didepresiasi pada revisi 2026-07-28. Simpan hanya untuk server-otorisasi yang tidak bisa mengonsumsi CIMD dan yang pra-pendaftarannya tidak praktis. Klien kompatibilitas melakukan POST:

```json
POST /register
Content-Type: application/json

{
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

Server merespons dengan `client_id` dan `registration_access_token` untuk pembaruan berikutnya:

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`application_type` bukan hiasan. Klien desktop loopback mendeklarasikan `native`; klien yang di-host server mendeklarasikan `web` dan memakai redirect URI HTTPS. `token_endpoint_auth_method: none` adalah default yang tepat untuk klien native publik. Ia hanya mendapatkan `client_id`, dengan PKCE yang menyediakan proof-of-possession.

Tiga jebakan production:

- Endpoint pendaftaran harus membatasi laju berdasarkan IP sumber. Tanpa itu, aktor permusuhan menskrip jutaan pendaftaran palsu dan menghabiskan namespace `client_id`. Jalankan pemeriksaan pembatas laju sebelum registrar menangani permintaan.
- `software_statement` (JWT bertanda tangan yang menjamin klien) diwajibkan oleh beberapa IdP enterprise. Mock pelajaran melewatkannya; production memasang langkah verifikasi yang menolak pendaftaran tanpa tanda tangan dari apa pun selain redirect URI localhost.
- `registration_access_token` harus disimpan sebagai hash, bukan plaintext. Pencurian token ini berarti penyerang dapat menulis ulang redirect URI milik klien.

### RFC 8707 (ringkasan) — Resource Indicator

Pelajaran 16 menetapkan bentuknya. Aturan production: setiap permintaan token menyertakan `resource=<canonical-mcp-url>`, dan server MCP memverifikasi `token.aud` cocok dengan URL sumber dayanya sendiri pada setiap panggilan. URI kanonik adalah pengenal yang *paling spesifik* untuk server: memakai scheme dan host huruf kecil, tanpa fragment, dan secara konvensi tanpa slash penutup. Komponen path **tidak** dihapus oleh aturan — spesifikasi mempertahankannya ketika diperlukan untuk mengidentifikasi satu server MCP tertentu. `https://mcp.example.com`, `https://mcp.example.com/mcp`, `https://mcp.example.com:8443`, dan `https://mcp.example.com/server/mcp` semuanya adalah URI kanonik yang valid. Pilih satu per server dan pin `aud` persis ke itu. (Mock pelajaran ini memakai audience host-bare seperti `https://notes.example.com` demi kepadatan; deployment yang meng-co-host beberapa server MCP di satu origin membedakannya lewat path.)

### RFC 7636 (ringkasan) — PKCE

PKCE adalah wajib di OAuth 2.1. Alur kode-otorisasi pelajaran selalu membawa `code_challenge` dan `code_verifier`. Server menolak permintaan token apa pun tanpa verifier atau dengan verifier yang tidak meng-hash ke challenge yang tersimpan.

### Profil otorisasi MCP 2026-07-28

Revisi MCP saat ini mempertahankan batas resource-server OAuth sambil membuat transport MCP stateless. Tidak ada sesi protokol untuk meng-cache keputusan identitas. Lapisan otorisasi karena itu memvalidasi setiap permintaan secara independen:

- Implementasikan metadata protected-resource RFC 9728, dan sediakan lokasinya melalui header `WWW-Authenticate: Bearer resource_metadata="..."` pada 401 **atau** URI well-known `/.well-known/oauth-protected-resource` (SEP-985 membuat header itu opsional dengan fallback well-known). Field metadata `authorization_servers` **MUST** menamai setidaknya satu server.
- Hanya menerima token melalui `Authorization: Bearer ...` pada **setiap** permintaan — tidak pernah di query string, tidak pernah hanya divalidasi saat awal sesi.
- Validasi `aud`, `iss`, `exp`, dan scope yang wajib per permintaan. Server **MUST** memvalidasi bahwa token memang diterbitkan untuknya (audience); ketiadaan atau ketidakcocokan `aud` ditolak, tidak pernah diperlakukan sebagai wildcard.
- Pada 401/403, kembalikan `WWW-Authenticate: Bearer` yang membawa `error=...`, parameter `resource_metadata="<PRM-URL>"` (URL dokumen metadata, *bukan* resource bare), dan `scope="..."` pada `insufficient_scope` (403). Catatan: parameternya adalah `resource_metadata`, sebuah penunjuk discovery — tidak ada parameter `resource` di challenge.
- Discovery server-otorisasi menerima **salah satu** metadata OAuth RFC 8414 **atau** OpenID Connect Discovery 1.0; klien harus mencoba kedua akhiran well-known tersebut secara berurutan prioritas.
- Klien (bukan server) yang membela terhadap **serangan mix-up**: ia merekam `issuer` yang diharapkan sebelum redirect dan memvalidasi nilai `iss` yang dikembalikan di respons otorisasi yang sebenarnya (RFC 9207) sebelum menebus kodenya. PKCE saja tidak menghentikan mix-up, karena klien menyerahkan `code_verifier`-nya ke endpoint token apa pun tempat ia disalurkan.
- Sebuah kredensial klien milik satu issuer server-otorisasi. Jika discovery resolve ke issuer yang berbeda, klien mendaftar ulang alih-alih menyajikan `client_id`, token pendaftaran, atau access token lama.
- CIMD adalah mekanisme enrollment yang disukai. DCR sudah usang; permintaan DCR kompatibilitas tetap mendeklarasikan `application_type` yang benar.

Draf OAuth 2.1 adalah substratnya; RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD adalah permukaannya; spesifikasi MCP adalah profilnya.

### Checklist kapabilitas deployment

Tabel fitur vendor cepat basi. Periksa metadata yang dikembalikan oleh server-otorisasi yang benar-benar akan kamu deploy. Gerbangnya bersifat mekanis:

| Check | Required decision |
|---|---|
| Discovered issuer | Exact HTTPS issuer expected by policy |
| PKCE | `S256` advertised; otherwise stop |
| Enrollment | CIMD preferred, pre-registration accepted, DCR only as deprecated compatibility |
| Authorization response | Validate RFC 9207 `iss` when present or advertised |
| Resource binding | Token request carries `resource`; resource server requires the matching `aud` |
| Credential storage | Key client IDs and registration credentials by issuer; key access tokens by issuer plus resource |
| DCR compatibility | Declare `native` or `web`; reject redirect URIs that do not fit the declared application type |

Jangan menyimpulkan dukungan dari nama produk atau tingkatan harga. Rekam dokumen yang ditemukan ke dalam bukti deployment dan gagal tertutup ketika field yang wajib tidak ada.

### Pola refresh JWKS (rotate di AS, refresh di server sumber daya)

Jaga dua kata kerja itu terpisah, karena mencampurnya adalah bug production yang nyata:

- **Rotate** adalah yang dilakukan *server-otorisasi*: membuat kunci penandatanganan baru, mempublikasikannya di JWKS, lalu memensiarkan yang lama. Server sumber daya tidak punya peran di sini dan tidak bisa melakukannya — ia tidak memegang private key milik IdP.
- **Refresh** adalah yang dilakukan *server sumber daya*: meng-`GET` ulang JWKS yang dipublikasikan ke dalam cache-nya. Itu satu-satunya aksi JWKS yang pernah dilakukan server sumber daya.

Mode kegagalan production adalah cache yang basi. Selesaikan dengan job refresh terjadwal plus cache key-value. Server sumber daya menjalankan sebuah job (cron, timer, apa pun yang ditawarkan runtime-mu) yang, pada interval tetap, mengambil `<issuer>/.well-known/jwks.json` dan menimpa `cache[issuer] = {keys, fetched_at}`. Validator membaca dari cache itu. Token yang `kid`-nya tidak ada di cache memicu **satu** refresh sinkron sebagai fallback, lalu diperiksa ulang. Ini menangani dua kasus sekaligus: refresh terjadwal, dan jendela overlap kunci di mana token yang ditandatangani kunci baru datang sebelum refresh terjadwal berikutnya.

Fallback **harus berupa re-fetch, bukan rotate**. Jika kamu memasang jalur cache-miss ke rotate-and-mint, dua hal rusak: (1) membuat kunci segar menghasilkan `kid` yang *tetap* tidak cocok dengan token, sehingga lookup tetap gagal; dan (2) penyerang yang menyemprotkan token dengan nilai `kid` acak memaksa seri pembuatan kunci tak berbatas — DoS yang didatangkan sendiri. Re-fetch bersifat idempoten, sehingga `kid` palsu paling banyak merugikan satu fetch.

Bentuk cache-nya:

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

Dua kunci sekaligus adalah keadaan steady-state. Server-otorisasi merotasi dengan memperkenalkan kunci berikutnya (`k_2026_04`) sebelum mensiasi kunci sebelumnya (`k_2026_03`), sehingga token yang diterbitkan di bawah kunci lama tetap valid sampai kedaluwarsa. Cache memegang gabungannya; validator memilih berdasarkan `kid`.

### Rutin validasi

Server MCP menjalankan validasi sebelum meneruskan tool apa pun. Bentuk yang dipakai `code/main.py`:

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` mendekode JWT, resolve kunci penandatangannya dari cache JWKS (menyegarkan sekali saat miss), memverifikasi tanda tangannya, lalu memeriksa `iss` terhadap allow-list, `aud` terhadap sumber daya kanonik server ini, `exp`, dan scope yang wajib — mengembalikan challenge `WWW-Authenticate` pada kegagalan pertama. Menjadikannya satu rutin tunggal di server sumber daya berarti setiap titik masuk (setiap panggilan tool, setiap transport) melewati pemeriksaan yang sama; tidak ada jalur yang mencapai tool tanpa memvalidasi lebih dulu.

### Token opaque memakai introspeksi, bukan tebakan

Tidak setiap access token adalah JWT. Jika issuer mendokumentasikan token opaque, server sumber daya tidak dapat mendekodekannya menjadi klaim yang tepercaya. Ia mengirim token ke endpoint introspeksi RFC 7662 milik issuer melalui backchannel terautentikasi dan mewajibkan `active: true`, konteks issuer yang diharapkan, audience atau sumber daya MCP yang persis, klaim waktu yang belum kedaluwarsa, dan scope yang dibutuhkan tool konkret.

Cache introspeksi berdasarkan issuer, digest token satu-arah, dan sumber daya MCP. Jangan pernah memakai token yang jelas sebagai label log atau cache. Batasi entri cache positif dengan yang paling awal antara kedaluwarsa token, panduan cache issuer, dan tujuan kesegaran pencabutan deployment. Jaga negative caching cukup singkat sehingga token yang baru diterbitkan tidak tetap secara keliru inactive. Hasil untuk satu sumber daya tidak dapat mengesahkan sumber daya lain bahkan jika string token opaque-nya identik.

Jangan memilih mode validasi dari isi token yang dikendalikan penyerang. Pin perilaku JWT versus introspeksi ke metadata issuer tervalidasi dan konfigurasi deployment. Di jalur JWT, pin algoritma yang diterima dan `jwks_uri` tepercaya; jangan pernah mengikuti URL kunci atau algoritma yang hanya dipilih oleh header token.

### Pencabutan adalah kontrak kesegaran

RFC 7009 mengizinkan klien meminta server-otorisasi mencabut token. Permintaan itu tidak menghapus salinan yang sudah di-cache oleh setiap server sumber daya. Tentukan penundaan pencabutan maksimum yang dapat diterima dan buat setiap cache mematuhinya.

Deployment token opaque dapat mencapai pencabutan yang lebih ketat dengan meng-introspeksi pada setiap panggilan berisiko tinggi atau memakai cache positif yang pendek. Deployment JWT mandiri biasanya menggabungkan masa hidup access-token pendek dengan pencabutan refresh-token, pensiunan kunci untuk insiden seluruh issuer, dan denylist subject, sesi, atau token-id opsional untuk penolakan darurat lokal. JWT bertanda tangan tetap valid secara kriptografis sampai kedaluwarsa kecuali server sumber daya memiliki bukti pencabutan eksternal yang mutakhir.

Logout, penonaktifan akun, penarikan consent, dan respons insiden adalah pemicu yang berbeda tetapi harus menyatu pada satu pernyataan yang terukur: setelah paling lama jendela pencabutan yang dideklarasikan, setiap replica menolak kredensial itu. Uji pernyataan itu melalui load balancer, bukan hanya terhadap satu proses yang sudah hangat.

### Kegagalan dependensi membutuhkan keputusan yang dideklarasikan

Jangan pernah mengimprovisasi kebijakan ketersediaan di dalam exception handler.

| Failure | Safe production behavior |
|---|---|
| Scheduled JWKS refresh fails, known `kid` remains in a still-valid bounded cache | Continue only within the declared stale-on-error window and emit degraded health evidence |
| Token has an unknown `kid` and the one allowed refresh fails | Reject; never accept an unverifiable signature |
| Introspection is unavailable | Fail closed for protected calls; do not convert network failure into `active: true` |
| Protected-resource or issuer metadata changes unexpectedly | Stop new enrollment and token acquisition; keep only explicitly pinned, unexpired configuration under a bounded incident policy |
| Revocation endpoint is unavailable | Report logout or revocation as incomplete, retain the credential locally as unusable when possible, and do not claim global revocation succeeded |
| Clock source or claim type is invalid | Reject rather than widening skew until the token passes |

Klasifikasikan kegagalan secara terpisah dari kredensial yang tidak valid. Gangguan dependensi adalah kesalahan operasional dengan kebijakan kesehatan dan retry. Tanda tangan, issuer, audience, kedaluwarsa, atau scope yang buruk adalah penolakan otorisasi. Keduanya tidak sampai ke penangan tool, dan keduanya tidak boleh membocorkan isi token ke dalam bukti audit.

### Walkthrough replay-audience (pembatasan hak istimewa access-token)

Server A (`notes.example.com`) dan Server B (`tasks.example.com`) sama-sama mendaftar terhadap server-otorisasi yang sama. Server A disusupi. Penyerang mengambil token catatan milik seorang pengguna dan mem-play ulang ke Server B.

Validator Server B:

1. Dekode JWT, ambil JWKS berdasarkan `kid`, verifikasi tanda tangan.
2. Periksa `iss` terhadap `authorization_servers` milik metadata protected-resource-nya. (Lulus — IdP yang sama.)
3. Periksa `aud == "https://tasks.example.com"`. (Gagal — `aud` token adalah `https://notes.example.com`.)
4. Kembalikan 401 dengan `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`.

Klaim audience adalah satu-satunya pertahanan terhadap serangan ini di lapisan protokol. Melewatkannya demi performa adalah kesalahan production paling umum; validator harus berjalan pada setiap permintaan, bukan hanya saat awal sesi. Spesifikasi menyebut ini **access-token privilege restriction**: server MCP `MUST` menolak token apa pun yang tidak menamainya di audience.

> **Catatan penamaan.** Spesifikasi menyisihkan istilah *confused deputy* untuk masalah yang terkait tetapi berbeda: server MCP yang bertindak sebagai **proxy** OAuth terhadap API pihak ketiga, memakai client ID statis, yang meneruskan token tanpa mendapatkan consent pengguna per-klien. Pengikatan audience memperbaiki replay di atas; perbaikan confused-deputy adalah consent per-klien **plus** tidak pernah meneruskan token masuk ke API hulu (server MCP `MUST` mendapatkan token hulu terpisahnya sendiri).

### Serangan mix-up (pertahanan sisi-klien yang tidak bisa diberikan server)

Sebuah klien berbicara dengan banyak server-otorisasi sepanjang hidupnya. AS jahat dapat mencoba membuat klien menebus kode otorisasi milik AS yang jujur di endpoint token milik penyerang. Pengikatan audience tidak membantu di sini — serangan terjadi sebelum token apa pun ada. Pertahanannya ada di klien (RFC 9207):

1. Sebelum redirect, klien merekam `issuer` yang diharapkan dari metadata AS yang tervalidasi.
2. Pada respons otorisasi, klien membandingkan parameter `iss` yang dikembalikan dengan issuer yang direkam itu (perbandingan string sederhana, tanpa normalisasi) sebelum mengirim kode ke mana pun.
3. Ketidakcocokan (atau `iss` tidak ada saat AS mengiklankan `authorization_response_iss_parameter_supported`) → tolak, dan jangan tampilkan field `error` pun.

PKCE saja tidak menghentikan mix-up, karena klien menyerahkan `code_verifier`-nya ke endpoint token apa pun tempat ia disalurkan. Inilah sebabnya spesifikasi merekam issuer per-permintaan bersama verifier PKCE dan `state`.

### Mode kegagalan

- **JWKS basi.** Validator menolak token yang valid setelah AS merotasi kunci. Perbaikannya adalah pola cron-refresh + cache-miss-refetch di atas. Jangan pernah meng-cache JWKS tanpa job refresh.
- **Rotate sebagai fallback.** Memasang jalur cache-miss ke rotate-and-mint alih-alih re-fetch adalah bug nyata: ia tidak pernah menghasilkan `kid` yang hilang, dan mengubah nilai `kid` yang dikendalikan penyerang menjadi DoS pembuatan kunci. Fallback-nya harus berupa `refresh-jwks` yang idempoten.
- **Klaim `aud` hilang.** Beberapa IdP default meng-`aud`-skip kecuali `resource` hadir dalam permintaan token. Validator harus menolak token tanpa `aud`, bukan memperlakukan ketiadaan sebagai wildcard.
- **Mix-up lewat pemeriksaan `iss` yang hilang.** Klien yang tidak memvalidasi parameter respons-otorisasi `iss` RFC 9207 terhadap issuer yang ia rekam sebelum redirect dapat disalurkan untuk menebus kode AS yang jujur di endpoint token penyerang. Ini adalah kegagalan sisi-klien; server sumber daya tidak dapat mengkompensasinya.
- **Balapan peningkatan scope.** Dua alur step-up paralel untuk pengguna yang sama dapat sama-sama berhasil dan menghasilkan dua access token dengan scope berbeda. Validator harus memakai token yang disajikan pada permintaan, bukan mencari "scope pengguna saat ini" — itu menciptakan jendela TOCTOU.
- **Pencurian token pendaftaran.** `registration_access_token` yang bocor memungkinkan penyerang menulis ulang redirect URI. Hash ini saat disimpan; minta klien menyajikannya dalam plaintext pada setiap pembaruan; rotasi saat ada kecurigaan.
- **`iss` tidak di-pin.** Validator yang menerima `iss` apa pun membiarkan penyerang mendirikan server-otorisasi miliknya sendiri, mendaftarkan klien untuk audience target, dan menerbitkan token. Daftar `authorization_servers` di metadata protected-resource adalah allow-list-nya; tegakkan.
- **Bentrok cache kredensial atau token.** Klien yang hanya meng-kunci pendaftaran berdasarkan sumber daya dapat menyajikan identitas satu server-otorisasi ke server lain. Klien yang hanya meng-kunci access token berdasarkan issuer dapat mem-play token ke audience yang salah. Kunci pendaftaran berdasarkan issuer tervalidasi, kunci access token berdasarkan `(issuer, resource)`, dan daftar ulang setiap kali issuer berubah.

```figure
t3-jwks-rotate
```

## Gunakan

`code/main.py` menelusuri alur production lengkap dengan Python stdlib dan tiga peran: `AuthorizationServer`, `ResourceServer`, dan `Client`. Alurnya:

Dari root repository, jalankan:

```bash
cd phases/13-tools-and-protocols/18-mcp-auth-production
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Perintah pertama mencetak transkrip enrollment ber-issuer dan validasi token. Perintah kedua melaporkan delapan belas pemeriksaan yang lulus. Tidak ada perintah yang membuka listener jaringan atau menulis kredensial.

1. Server-otorisasi mempublikasikan metadata RFC 8414 di `/.well-known/oauth-authorization-server`.
2. Klien MCP memanggil endpoint metadata dan memeriksa opsi enrollment-nya (`client_id_metadata_document_supported` untuk CIMD, `registration_endpoint` untuk DCR) serta dukungan PKCE `S256`.
3. Klien memeriksa keberadaan pra-pendaftaran ber-scope issuer, jika tidak ada maka ia mendaftar dengan Client ID Metadata Document HTTPS-nya. DCR yang sudah usang tetap menjadi metode kompatibilitas yang dapat diuji terpisah.
4. Klien merekam issuer yang tervalidasi, membuat challenge S256, menerima kode otorisasi sekali-pakai plus `iss`, memvalidasi issuer yang dikembalikan itu, lalu menebus kode dengan verifier asli dan indikator `resource` RFC 8707.
5. Klien MCP memanggil tool pada server MCP dengan `Authorization: Bearer ...`.
6. Server MCP menjalankan `validate`, meresolve kunci penandatanganan dari cache JWKS.
7. IdP merotasi sebuah kunci; refresh terjadwal menarik ulang JWKS ke dalam cache.
8. Panggilan berikutnya divalidasi terhadap kunci yang sudah disegar tanpa restart, dan token sebelumnya tetap tervalidasi selama jendela overlap.
9. Percobaan replay-audience terhadap sumber daya MCP lain mendapat 401 dengan `audience mismatch` dan penunjuk `resource_metadata`.

JWT di sini memakai HS256 dengan secret bersama (sehingga pelajaran berjalan hanya dengan stdlib). Production memakai RS256 atau EdDSA dengan pola JWKS di atas; logika validasinya selain itu identik. Karena IdP dan server sumber daya hidup dalam satu proses, `refresh_jwks` membaca daftar kunci server-otorisasi secara langsung; di atas wire itu adalah HTTP `GET` ke `jwks_uri`.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-mcp-auth.md`. Diberi konfigurasi server MCP dan himpunan kapabilitas IdP, skill ini mengeluarkan permukaan auth yang harus dibangun — metadata protected-resource, jalur enrollment yang harus dipakai (CIMD, pra-pendaftaran, atau fallback DCR), jadwal refresh JWKS, pemetaan scope, dan aturan penolakan yang harus diterapkan ketika IdP tidak mendukung profil RFC lengkap.

## Latihan

1. Jalankan `code/main.py`. Telusuri alurnya. Perhatikan bagaimana IdP merotasi kunci di langkah 6, `refresh_jwks` yang terjadwal menarik ulang set yang dipublikasikan, dan baik token lama (jendela overlap) maupun token segar tervalidasi tanpa restart.

2. Tambahkan IdP baru ke daftar `authorization_servers` pada metadata protected-resource. Terbitkan token yang ditandatangani IdP baru dan konfirmasikan validator menerimanya. Terbitkan token yang ditandatangani IdP yang tidak terdaftar dan konfirmasikan validator menolak dengan `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`.

3. Tambahkan pemeriksaan pembatas laju ke `register_client` yang berjalan sebelum registrar menerima permintaan. Gunakan token-bucket per IP sumber yang disimpan di dict kecil ber-kunci IP.

4. Baca RFC 7591 dan identifikasi dua field yang tidak divalidasi oleh handler `/register` di pelajaran. Tambahkan validasinya. (Petunjuk: `software_statement` dan skema URI `redirect_uris`.)

5. Tambahkan server-otorisasi kedua. Konfirmasikan klien menyimpan enrollment terpisah ber-kunci issuer dan menolak menggunakan ulang token atau `client_id` milik issuer pertama.

6. Buktikan perbaikan DoS-nya. Kirim validator sebuah token dengan `kid` acak dan konfirmasikan `refresh_jwks` berjalan paling banyak sekali dan jumlah kunci server-otorisasi tidak bertambah. Lalu pasang ulang secara disengaja fallback-nya ke rotate-and-mint dan amati jumlah kunci naik per token palsu — kembalikan re-fetch setelahnya.

7. Uji DCR yang sudah usang dengan klien `native` maupun `web`. Konfirmasikan klien web dengan redirect URI HTTP dan klien native tanpa redirect loopback yang persis ditolak.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| ASM | "OAuth metadata document" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "Client metadata URL" | Client ID Metadata Document: an HTTPS URL used as the `client_id`; the AS pulls the JSON. Preferred enrollment in MCP 2026-07-28 |
| DCR | "Self-service client registration" | RFC 7591 `POST /register`; deprecated for current MCP and retained only for compatibility |
| JWKS | "Public keys for JWT validation" | JSON Web Key Set, fetched from `jwks_uri`, indexed by `kid` |
| Rotate vs refresh | "Updating the keys" | *Rotate* = AS mints/retires signing keys; *refresh* = resource server re-fetches the published set. Resource servers only ever refresh |
| Resource indicator | "Audience parameter" | RFC 8707 `resource` parameter pinning the token to one server |
| `aud` claim | "Audience" | JWT claim the validator compares against the canonical resource URL |
| Audience replay | "Token replay" | Token issued for Server A presented to Server B; defended by audience validation (spec: access-token privilege restriction) |
| Confused deputy | "Proxy token misuse" | An MCP proxy with a static client ID forwarding a token without per-client consent; distinct from audience replay |
| Mix-up attack | "Wrong token endpoint" | Client steered to redeem an honest AS's code at an attacker's endpoint; defended client-side via RFC 9207 `iss` |
| `iss` allow-list | "Trusted authorization servers" | The set named in protected-resource metadata's `authorization_servers` |
| `resource_metadata` | "Where to find the PRM doc" | `WWW-Authenticate` parameter naming the RFC 9728 metadata URL on a 401/403 |
| Public client | "Native or browser client" | OAuth client with no `client_secret`; PKCE compensates |
| `WWW-Authenticate` | "401/403 response header" | Carries `Bearer error=...` directives that drive client recovery |

## Bacaan Lanjutan

- [MCP authorization specification (2026-07-28)](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization) - profil otorisasi MCP saat ini
- [MCP 2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog) - CIMD, validasi issuer, deprecation DCR, dan perubahan kredensial ber-kunci issuer
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — kontrak discovery
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR (jalur fallback)
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — proof-of-possession klien-publik
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — pengikatan audience
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — discovery server sumber daya
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — parameter `iss` yang membela terhadap serangan mix-up
- [RFC 7662: OAuth 2.0 Token Introspection](https://datatracker.ietf.org/doc/html/rfc7662)
- [RFC 7009: OAuth 2.0 Token Revocation](https://datatracker.ietf.org/doc/html/rfc7009)
