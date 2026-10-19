# Function Calling Deep Dive — OpenAI, Anthropic, Gemini

> Tiga provider frontier bertemu pada loop tool call yang sama di tahun 2024 lalu berpisah pada semua hal lain. OpenAI memakai `tools` dan `tool_calls`. Anthropic memakai blok `tool_use` dan `tool_result`. Gemini memakai `functionDeclarations` dan korelasi unique-id. Pelajaran ini membedakan ketiganya berdampingan sehingga kode yang berjalan di satu provider tidak rusak ketika kamu memindahkannya.

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01 (the tool interface)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Menyatakan tiga perbedaan bentuk antara payload function-calling OpenAI, Anthropic, dan Gemini (declaration, call, result).
- Menerjemahkan satu deklarasi tool ke seluruh format ketiga provider dan memprediksi di mana batasan strict-mode akan berbeda.
- Menggunakan `tool_choice` di setiap provider untuk memaksa, melarang, atau memilih otomatis tool call.
- Mengetahui batas keras per-provider (jumlah tool, kedalaman schema, panjang argumen) dan tanda error yang masing-masing pancarkan ketika batas dilanggar.

## Masalahnya

Bentuk permintaan function-calling berbeda per provider. Tiga contoh konkret dari stack produksi 2026:

**OpenAI Chat Completions / Responses API.** Kamu mengirim `tools: [{type: "function", function: {name, description, parameters, strict}}]`. Respons model berisi `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]` di mana `arguments` adalah string JSON yang harus kamu urai. Strict mode (`strict: true`) menegakkan kepatuhan schema melalui constrained decoding.

**Anthropic Messages API.** Kamu mengirim `tools: [{name, description, input_schema}]`. Respons kembali sebagai `content: [{type: "text"}, {type: "tool_use", id, name, input}]`. `input` sudah diurai (sebuah objek, bukan string). Kamu membalas dengan pesan `user` baru yang berisi blok `{type: "tool_result", tool_use_id, content}`.

**Google Gemini API.** Kamu mengirim `tools: [{functionDeclarations: [{name, description, parameters}]}]` (bersarang di bawah `functionDeclarations`). Respons tiba sebagai `candidates[0].content.parts: [{functionCall: {name, args, id}}]` di mana `id` unik di Gemini 3 ke atas untuk korelasi parallel-call. Kamu membalas dengan `{functionResponse: {name, id, response}}`.

Loop yang sama. Nama field berbeda, nesting berbeda, konvensi string-vs-objek berbeda, mekanisme korelasi berbeda. Sebuah tim yang menulis agen cuaca di OpenAI membayar biaya port dua hari ke Anthropic dan satu hari lagi ke Gemini hanya untuk plumbing-nya.

Pelajaran ini membangun penerjemah yang menyatukan ketiga format menjadi satu deklarasi tool kanonis dan merutekan di tepi. Phase 13 · 17 menggeneralisasi pola yang sama menjadi sebuah LLM gateway.

## Konsepnya

### Struktur yang umum

Setiap provider membutuhkan lima hal:

1. **Tool list.** Nama, deskripsi, dan input schema per tool.
2. **Tool choice.** Memaksa tool tertentu, melarang tool, atau membiarkan model memutuskan.
3. **Call emission.** Output terstruktur yang menyebut tool dan argumen.
4. **Call id.** Mengorelasikan respons ke panggilan yang tepat (penting untuk paralel).
5. **Result injection.** Pesan atau blok yang mengikat hasil kembali ke panggilan.

### Perbedaan bentuk, field per field

| Aspect | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| Declaration envelope | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema field | `parameters` | `input_schema` | `parameters` |
| Response container | `tool_calls[]` on assistant message | `content[]` of type `tool_use` | `parts[]` of type `functionCall` |
| Arguments type | stringified JSON | parsed object | parsed object |
| Id format | `call_...` (OpenAI generates) | `toolu_...` (Anthropic) | UUID (Gemini 3+) |
| Result block | role `tool`, `tool_call_id` | `user` with `tool_result`, `tool_use_id` | `functionResponse` with matching `id` |
| Force-a-tool | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Forbid tools | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Strict schema | `strict: true` | schema-is-schema (always enforced) | `responseSchema` at request level |

### Batas yang benar-benar akan kamu temui

- **OpenAI.** 128 tool per request. Kedalaman schema 5. String argumen <= 8192 byte. Strict mode mensyaratkan tanpa `$ref`, tanpa `oneOf`/`anyOf`/`allOf` yang tumpang tindih, setiap properti terdaftar di `required`.
- **Anthropic.** 64 tool per request. Kedalaman schema secara efektif tak terbatas tetapi batas praktis 10. Tanpa flag strict-mode; schema adalah kontrak dan model cenderung mematuhinya.
- **Gemini.** 64 fungsi per request. Tipe schema adalah subset OpenAPI 3.0 (sedikit berbeda dari JSON Schema 2020-12). Parallel call unique-id sejak Gemini 3.

### Perilaku `tool_choice`

Tiga mode yang didukung semua orang, dengan nama berbeda.

- **Auto.** Model memilih tool atau teks. Default.
- **Required / Any.** Model harus memanggil setidaknya satu tool.
- **None.** Model tidak boleh memanggil tool.

Ditambah satu mode yang unik bagi tiap provider:

- **OpenAI.** Memaksa tool tertentu berdasarkan nama.
- **Anthropic.** Memaksa tool tertentu berdasarkan nama; flag `disable_parallel_tool_use` memisahkan single vs multi.
- **Gemini.** `mode: "VALIDATED"` merutekan setiap respons melalui validator schema terlepas dari intensi model.

### Parallel calls

`parallel_tool_calls: true` milik OpenAI (default) memancarkan beberapa panggilan dalam satu pesan assistant. Kamu menjalankan semuanya dan membalas dengan pesan role tool berkelompok yang berisi satu entri per `tool_call_id`. Anthropic secara historis melakukan single-call; `disable_parallel_tool_use: false` (default sejak Claude 3.5) mengaktifkan multi. Gemini 2 mengizinkan parallel call tetapi tidak memberikan id yang stabil; Gemini 3 menambahkan UUID sehingga respons yang tidak berurutan terkorelasi dengan bersih.

### Streaming

Ketiganya mendukung streamed tool call. Format wire-nya berbeda:

- **OpenAI.** Delta chunk dari `tool_calls[i].function.arguments` tiba secara bertahap. Kamu akumulasikan sampai `finish_reason: "tool_calls"`.
- **Anthropic.** Event block-start / block-delta / block-stop. Chunk `input_json_delta` membawa argumen sebagian.
- **Gemini.** `streamFunctionCallArguments` (baru di Gemini 3) memancarkan chunk dengan `functionCallId` sehingga beberapa parallel call dapat berinterleaving.

Phase 13 · 03 masuk mendalam ke perakitan ulang parallel + streaming. Pelajaran ini berfokus pada bentuk deklarasi dan single-call.

### Errors dan perbaikan

Error argumen tidak valid juga terlihat berbeda.

- **OpenAI (non-strict).** Model mengembalikan `arguments: "{bad json}"`, penguraian JSON-mu gagal, kamu menyisipkan pesan error dan memanggil ulang.
- **OpenAI (strict).** Validasi terjadi selama decoding; JSON tidak valid mustahil tetapi `refusal` dapat muncul.
- **Anthropic.** `input` dapat berisi field yang tidak terduga; schema bersifat advisory. Validasi di sisi server.
- **Gemini.** Keanehan OpenAPI 3.0: `enum` pada field objek diabaikan diam-diam; validasi sendiri.

### Pola penerjemah

Deklarasi tool kanonis di kodenmu terlihat seperti ini (kamu memilih bentuknya):

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

Tiga fungsi kecil menerjemahkannya ke ketiga bentuk provider. Harness di `code/main.py` melakukan persis ini, lalu melakukan round-trip sebuah tool call palsu melalui bentuk respons masing-masing provider. Tanpa jaringan — pelajaran ini mengajarkan bentuknya, bukan HTTP-nya.

Tim produksi membungkus penerjemah ini dalam `AbstractToolset` (Pydantic AI), `UniversalToolNode` (LangGraph), atau `BaseTool` (LlamaIndex). Phase 13 · 17 mengirimkan gateway yang mengekspos API berbentuk OpenAI di depan salah satu dari ketiganya.

```figure
function-call-args
```

## Gunakan

`code/main.py` mendefinisikan satu dataclass `Tool` kanonis dan tiga penerjemah yang memancarkan JSON deklarasi OpenAI, Anthropic, dan Gemini. Lalu ia mengurai respons provider buatan tangan untuk masing-masing bentuk ke dalam objek call kanonis yang sama, mendemonstrasikan bahwa semantiknya identik di balik kulitnya. Jalankan dan bandingkan ketiga deklarasi berdampingan.

Yang perlu diperhatikan:

- Ketiga blok deklarasi hanya berbeda pada envelope dan nama field.
- Ketiga blok respons berbeda pada di mana call berada (top-level `tool_calls`, blok `content[]`, entri `parts[]`).
- Satu fungsi `canonical_call()` mengekstrak `{id, name, args}` dari ketiga bentuk respons.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-provider-portability-audit.md`. Diberikan integrasi function-calling terhadap satu provider, skill menghasilkan audit portabilitas: batas provider mana yang diandalkannya, field mana yang perlu diganti nama, dan apa yang rusak ketika dipindahkan ke masing-masing provider lain.

## Latihan

1. Jalankan `code/main.py` dan verifikasi bahwa ketiga JSON deklarasi provider menerialisasi objek `Tool` yang sama di baliknya. Ubah tool kanonis untuk menambahkan parameter enum dan konfirmasikan hanya penerjemah Gemini yang perlu menangani keanehan OpenAPI.

2. Tambahkan parser `ListToolsResponse` untuk setiap provider yang mengekstrak daftar tool yang dikembalikan model setelah panggilan `list_tools` atau discovery. OpenAI tidak memilikinya secara native; catat asimetri ini.

3. Implementasikan konversi `tool_choice`: petakan `ToolChoice(mode="force", tool_name="x")` kanonis ke ketiga bentuk provider. Lalu petakan `mode="any"` dan `mode="none"`. Periksa tabel diff pelajaran ini.

4. Pilih salah satu dari ketiga provider dan baca panduan function-calling-nya dari awal sampai akhir. Temukan satu field dalam spesifikasi schema-nya yang tidak didukung dua lainnya. Kandidat: `strict` OpenAI, `disable_parallel_tool_use` Anthropic, `function_calling_config.allowed_function_names` Gemini.

5. Tulis sebuah test vector: tool call yang argumennya melanggar schema yang dideklarasikan. Jalankan melalui validator masing-masing provider (yang stdlib di Pelajaran 01 akan cukup sebagai proksi) dan catat error mana yang menyala. Dokumentasikan provider mana yang akan kamu gunakan di produksi untuk ketegasan.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Function calling | "Tool use" | Provider-level API for structured tool-call emission |
| Tool declaration | "Tool spec" | Name + description + JSON Schema input payload |
| `tool_choice` | "Force / forbid" | Auto / required / none / specific-name modes |
| Strict mode | "Schema enforcement" | OpenAI flag that constrains decoding to match schema |
| `tool_use` block | "Anthropic's call shape" | Inline content block with id, name, input |
| `functionCall` part | "Gemini's call shape" | A `parts[]` entry containing name, args, and id |
| Arguments-as-string | "Stringified JSON" | OpenAI returns args as a JSON string, not an object |
| Parallel tool calls | "Fan-out in one turn" | Multiple tool calls in one assistant message |
| Refusal | "Model declines" | Strict-mode-only refusal block instead of a call |
| OpenAPI 3.0 subset | "Gemini schema quirk" | Gemini uses a JSON-Schema-like dialect with minor differences |

## Bacaan Lanjutan

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — canonical reference including strict mode and parallel calls
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` and `tool_result` block semantics
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — parallel calls, unique ids, and OpenAPI subset
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini's enterprise surface
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict-mode schema enforcement details
