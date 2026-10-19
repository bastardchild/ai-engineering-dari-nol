# Antarmuka Tool — Mengapa Agen Membutuhkan I/O Terstruktur

> Model bahasa menghasilkan token. Sebuah program mengambil tindakan. Jarak di antara keduanya adalah antarmuka tool: kontrak yang memungkinkan model meminta sebuah tindakan dan host mengeksekusinya. Setiap stack tahun 2026 — function calling di OpenAI, Anthropic, dan Gemini; `tools/call` milik MCP; task parts milik A2A — adalah pengkodean yang berbeda dari loop empat langkah yang sama. Pelajaran ini menamai loop tersebut dan menunjukkan mesin minimum untuk menjalankannya.

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 minutes

## Tujuan Pembelajaran

- Menjelaskan mengapa LLM yang hanya bisa menghasilkan teks tidak dapat, dengan sendirinya, mengambil tindakan terhadap dunia nyata.
- Menggambar loop tool-call empat langkah (describe → decide → execute → observe) dan menyebutkan siapa yang memiliki setiap langkah.
- Menulis deskripsi tool sebagai tiga bagian: nama, input JSON Schema, dan fungsi eksekutor deterministik.
- Membedakan tool murni dan tool yang memiliki side effect serta menyatakan mengapa pemisahan itu penting untuk keamanan.

## Masalahnya

LLM memancarkan distribusi probabilitas atas token berikutnya. Itulah keseluruhan permukaan output. Jika kamu menanyakan sebuah model chat "what is the weather in Bengaluru right now," ia dapat menulis kalimat yang masuk akal, tetapi ia tidak dapat menelpon ke sebuah weather API. Kalimat itu mungkin benar secara kebetulan atau sudah basi tiga hari.

Menutup jarak itulah tujuan dari antarmuka tool. Program host — runtime agenmu, Claude Desktop, ChatGPT, Cursor, atau skrip kustom — mengiklankan daftar tool yang dapat dipanggil ke model. Model, ketika ia memutuskan bahwa tindakan diperlukan, memancarkan payload terstruktur yang menyebut sebuah tool beserta argumennya. Host mengurai payload itu, menjalankan tool yang sesungguhnya, dan menyuplai hasilnya kembali. Loop berlanjut sampai model memutuskan tidak ada lagi panggilan yang diperlukan.

Versi pertama dari kontrak ini dikirimkan pada Juni 2023 sebagai parameter "functions" milik OpenAI. Anthropic menyusul dengan blok `tool_use` di Claude 2.1. Gemini menambahkan `functionDeclarations` beberapa bulan kemudian. Setiap provider kini mengekspos bentuk yang sama: daftar tool bertipe JSON-Schema masuk, tool call berbentuk JSON payload keluar. Model Context Protocol (November 2024) menggeneralisasi kontrak sehingga satu registry tool melayani setiap model. A2A (April 2026, v1.0) melapis primitif yang sama untuk delegasi agen-ke-agen.

Loop empat langkah adalah invarian di balik semua ini. Segala sesuatu yang lain di Phase 13 adalah sebuah elaborasi.

## Konsepnya

### Langkah satu: describe

Host mendeklarasikan setiap tool dengan tiga field.

- **Name.** Pengenal yang stabil dan dapat dibaca mesin. `get_weather`, bukan "weather thing".
- **Description.** Ringkasan natural-language satu paragraf. "Use when the user asks about current conditions for a specific city. Do not use for historical data."
- **Input schema.** Objek JSON Schema (draft 2020-12) yang menjelaskan argumen tool.

Model menerima daftar itu. Provider modern menerialisasi deklarasi ini ke dalam system prompt menggunakan template khusus provider, sehingga kamu sebagai pemanggil hanya berurusan dengan bentuk terstruktur.

### Langkah dua: decide

Diberikan pesan pengguna dan tool yang tersedia, model memilih satu dari tiga perilaku.

1. **Menjawab langsung** dalam teks. Tanpa tool call.
2. **Memanggil satu atau lebih tool.** Memancarkan objek call terstruktur. Di bawah `parallel_tool_calls: true` (default di OpenAI dan Gemini, opt-in di Anthropic) model dapat memancarkan beberapa panggilan dalam satu giliran.
3. **Menolak.** Structured outputs mode ketat dapat menghasilkan blok `refusal` bertipe alih-alih sebuah call.

Payload tool call memiliki tiga field yang stabil: `id` call, `name` tool, dan objek `arguments` JSON. Id ada agar host dapat mengorelasikan hasil kemudian dengan panggilan spesifik tersebut, yang penting ketika panggilan paralel kembali tidak berurutan.

### Langkah tiga: execute

Host menerima panggilan, memvalidasi argumen terhadap schema yang dideklarasikan, dan menjalankan eksekutor. Argumen yang tidak valid berarti model berhalusinasi sebuah field atau menggunakan tipe yang salah — mode kegagalan yang sangat umum pada model lemah. Host produksi melakukan salah satu dari tiga hal pada argumen yang tidak valid: gagal cepat dan menampilkan error ke model, memperbaiki JSON dengan parser berbatasan, atau mencoba ulang model dengan error validasi yang disertakan dalam prompt.

Eksekutor itu sendiri adalah kode biasa. Python, TypeScript, perintah shell, query database. Ia menghasilkan sebuah hasil, yang biasanya berupa string tetapi bisa berupa nilai JSON apa pun atau blok konten terstruktur (teks, gambar, atau referensi resource di MCP). Hasil harus dapat diserialisasi.

### Langkah empat: observe

Host menambahkan hasil tool ke percakapan (sebagai pesan role `tool` dengan `id` yang cocok) dan memanggil ulang model. Kini model memiliki output tool dalam konteks dan dapat menghasilkan jawaban akhir atau meminta lebih banyak panggilan. Ini berlanjut sampai model berhenti memancarkan panggilan atau host mencapai batas keamanan pada jumlah iterasi.

### Pemisahan kepercayaan

Tool hadir dalam dua varian yang penting untuk keamanan.

- **Pure.** Hanya-baca, deterministik, tanpa side effect. `get_weather`, `search_docs`, `get_current_time`. Aman dipanggil secara spekulatif.
- **Consequential.** Mengubah state, membelanjakan uang, menyentuh data pengguna. `send_email`, `delete_file`, `execute_trade`. Harus di-gate.

"Rule of Two" milik Meta tahun 2026 untuk keamanan agen mengatakan satu giliran dapat menggabungkan setidaknya dua dari: input tidak tepercaya, data sensitif, tindakan konsekuensial. Antarmuka tool adalah tempat kamu menegakkan aturan itu — dengan menolak panggilan, mewajibkan konfirmasi pengguna, atau menaikkan scope. Lihat Phase 13 · 15 untuk bab keamanan lengkap dan Phase 14 · 09 untuk kebijakan izin level agen.

### Di mana loop berada

| Context | Who describes | Who decides | Who executes |
|---------|---------------|-------------|--------------|
| Single-turn function calling (OpenAI/Anthropic/Gemini) | App developer | LLM | App developer |
| MCP | MCP server | LLM via MCP client | MCP server |
| A2A | Agent Card publisher | Calling agent | Called agent |
| Web browser (function-calling agent) | Browser extension / WebMCP | LLM | Browser runtime |

Di mana pun, empat langkah yang sama. Nama kolomnya berubah; strukturnya tidak.

### Mengapa tidak sekadar meminta model memancarkan JSON?

"Ask the model to reply in JSON" adalah pola pra-function-calling. Ia gagal sekitar 5 sampai 15 persen pada model frontier, dan jauh lebih sering pada model yang lebih kecil. Mode kegagalan meliputi kurung kurawal yang hilang, koma ekor, field yang berhalusinasi, dan tipe yang salah. Kamu kemudian membutuhkan langkah perbaikan JSON, retry, atau decoder berbatasan.

Function calling native lebih baik karena tiga alasan. Pertama, provider melatih model end-to-end pada bentuk call yang persis, sehingga laju JSON valid naik ke 98 sampai 99 persen pada mode strict. Kedua, payload call berada di slot protokolnya sendiri, bukan di dalam teks bebas — sehingga tool call tidak pernah bocor ke balasan yang terlihat oleh pengguna. Ketiga, provider menegakkan kepatuhan schema dengan constrained decoding (strict mode milik OpenAI, `tool_use` milik Anthropic, `responseSchema` milik Gemini). Output dijamin tervalidasi.

Phase 13 · 02 membedah ketiga API provider berdampingan. Phase 13 · 04 masuk mendalam ke structured outputs.

### Pemutus sirkuit

Loop berakhir ketika model berhenti memancarkan panggilan atau host mencapai jumlah giliran maksimum. Host produksi menetapkan ini antara 5 dan 20 giliran. Di luar itu, kamu hampir pasti berada dalam loop yang tidak dapat dikeluari oleh model. Claude Code defaultnya 20; OpenAI Assistants 10; mode agen Cursor 25.

Alternatifnya — loop tanpa batas — muncul setiap enam bulan sebagai post-mortem "agent spent $400 in API calls overnight". Jangan rilis tanpa batas.

Phase 14 · 12 membahas pemulihan error dan self-healing secara mendalam; Phase 17 membahas rate limit produksi.

### Ke mana Phase 13 beranjak dari sini

- Pelajaran 02 sampai 05 menghaluskan permukaan tool-call level provider.
- Pelajaran 06 sampai 14 menggeneralisasi loop ke MCP.
- Pelajaran 15 sampai 18 membela loop dari server yang bermusuhan, pengguna adversarial, dan permukaan autentikasi remote yang tidak terautentikasi.
- Pelajaran 19 sampai 22 memperluas pola ke kolaborasi agen-ke-agen, observabilitas, routing, dan packaging.
- Pelajaran 23 mengirimkan ekosistem lengkap yang memakai setiap primitif.

Setiap pelajaran yang tersisa adalah elaborasi dari loop empat langkah ini. Pegang sebagai invarian.

```figure
tp-tool-loop
```

## Gunakan

`code/main.py` menjalankan loop empat langkah tanpa LLM. Sebuah fungsi "decider" palsu meniru model dengan pattern-matching pada pesan pengguna; eksekutor, validator schema, dan harness langkah observe adalah yang sesungguhnya. Jalankan untuk melihat koreografi request/response lengkap dengan state antara yang dapat dicetak, lalu ganti decider palsu dengan provider sungguhan mana pun di pelajaran kemudian.

Yang perlu diperhatikan:

- Registry tool menyimpan tiga field per tool: name, description, schema, dan referensi eksekutor.
- Validator adalah subset JSON Schema minimal (types, required, enum, min/max) yang ditulis hanya dengan stdlib. Phase 13 · 04 mengirimkan versi yang lebih lengkap.
- Loop membatasi jumlah iterasi pada lima. Agen produksi membutuhkan persis jenis pemutus sirkuit ini.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-tool-interface-reviewer.md`. Diberikan draf definisi tool (name + description + schema + outline eksekutor), skill mengauditnya untuk kecocokan loop: apakah namanya stabil-secara-mesin, apakah deskripsinya ringkasan penggunaan yang lengkap, apakah schema menggunakan JSON Schema 2020-12 dengan benar, dan apakah klasifikasi pure-vs-consekuensial eksplisit.

## Latihan

1. Tambahkan tool keempat ke `code/main.py` bernama `get_stock_price(ticker)`. Tulis deskripsinya sebagai "Use when the user asks for a current stock price by ticker. Do not use for historical prices or market summaries." Jalankan harness dan konfirmasikan bahwa decider palsu mengarahkan kueri yang menyebut ticker ke tool baru itu.

2. Rusak validator schema. Berikan sebuah call yang objek `arguments`-nya kehilangan field yang wajib, dan konfirmasikan host menolaknya sebelum eksekusi. Lalu berikan call dengan satu field tak dikenal tambahan. Putuskan: haruskah host menolak atau mengabaikan? Justifikasi pilihanmu dengan argumen keamanan.

3. Klasifikasikan setiap tool dalam harness sebagai pure atau consequential. Tambahkan flag `consequential: true` ke entri registry yang membutuhkannya, dan ubah loop agar mencetak baris "would confirm with user" setiap kali tool consequential dipilih. Inilah bentuk confirmation gate yang dibutuhkan setiap host produksi.

4. Gambar loop empat langkah di kertas dengan tabel kolom provider di atas terisi untuk klien favoritmu (Claude Desktop, Cursor, ChatGPT, atau stack kustom). Silang-referensikan dengan varian khusus MCP di Phase 13 · 06.

5. Baca panduan function-calling OpenAI dari awal sampai akhir. Identifikasi satu field yang ada di request tetapi tidak ada di loop empat langkah sebagaimana disajikan di sini. Jelaskan apa yang ditambahkannya dan mengapa itu nyaman alih-alih esensial.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tool | "A thing the model can call" | A triple of name + JSON-Schema-typed input + executor function |
| Function calling | "Native tool use" | Provider-level API support for emitting structured tool calls instead of prose |
| Tool call | "The model's request to act" | A JSON payload with `id`, `name`, `arguments` emitted by the model |
| Tool result | "What the tool returned" | The executor's output, wrapped in a `tool` role message with matching id |
| Parallel tool calls | "Many calls at once" | Multiple call objects in one model turn, independent and orderable by id |
| Strict mode | "Guaranteed JSON" | Constrained decoding that forces the model's output to validate against the declared schema |
| Pure tool | "Read-only tool" | No side effects; safe to re-run |
| Consequential tool | "Action tool" | Mutates external state; requires gate, audit, or user confirmation |
| Four-step loop | "The tool-call cycle" | describe → decide → execute → observe |
| Host | "Agent runtime" | The program that holds the tool registry, calls the model, and runs the executor |

## Bacaan Lanjutan

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — canonical reference for OpenAI-style tool declarations and call shapes
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — Claude's `tool_use` / `tool_result` block format
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — `functionDeclarations` and parallel-call semantics in Gemini
- [Model Context Protocol — Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) — the current stateless, provider-agnostic generalization of the tool interface
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — the schema dialect every modern tool API speaks
