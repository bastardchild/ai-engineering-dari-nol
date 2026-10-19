# Parallel Tool Calls dan Streaming dengan Tools

> Tiga lookup cuaca independen yang diserialisasi adalah tiga round trip. Jalankan secara paralel dan total waktu runtuh menjadi panggilan tunggal yang paling lambat. Setiap provider frontier kini memancarkan beberapa tool call dalam satu giliran. Imbalannya nyata; plumbing-nya halus. Pelajaran ini membedah kedua bagiannya: fan-out paralel dan perakitan ulang argumen yang di-stream, dengan penekaan pada jebakan korelasi id.

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Menjelaskan mengapa `parallel_tool_calls: true` ada dan kapan menonaktifkannya.
- Mengorelasikan chunk argumen yang di-stream ke tool-call id yang tepat selama fan-out paralel.
- Merakit ulang string `arguments` sebagian menjadi JSON lengkap tanpa mengurai terlalu awal.
- Menjalankan benchmark cuaca tiga kota yang mendemonstrasikan latensi sekuensial vs paralel.

## Masalahnya

Tanpa parallel call, sebuah agen yang menjawab "what is the weather in Bengaluru, Tokyo, and Zurich" melakukan ini:

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

Tiga round trip LLM, masing-masing juga membayar latensi eksekutor. Sekitar 4x waktu wall-clock ideal.

Dengan parallel call:

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

Satu round trip LLM. Waktu eksekutor adalah maksimum dari ketiganya, bukan jumlahnya. Benchmark produksi di OpenAI, Anthropic, dan Gemini menunjukkan pengurangan wall-clock 60 sampai 70 persen pada beban fan-out.

Harganya adalah kompleksitas korelasi. Ketika ketiga panggilan selesai tidak berurutan, hasil-hasmu harus membawa `tool_call_id` yang cocok agar model dapat menyusunnya berurutan. Ketika hasil di-stream, kamu harus merakit fragmen argumen sebagian menjadi JSON lengkap sebelum mengeksekusi. Gemini 3 menambahkan id unik sebagian untuk menyelesaikan masalah dunia nyata di mana dua parallel call ke tool yang sama tak terbedakan.

## Konsepnya

### Mengaktifkan paralel

- **OpenAI.** `parallel_tool_calls: true` aktif secara default. Setel `false` untuk memaksa serial.
- **Anthropic.** Paralel melalui `disable_parallel_tool_use: false` (default di Claude 3.5 ke atas). Setel `true` untuk serial.
- **Gemini.** Selalu mampu paralel; `tool_config.function_calling_config.mode = "AUTO"` membiarkan model memutuskan.

Nonaktifkan paralel ketika tool memiliki ketergantungan urutan (`create_file` lalu `write_file`), ketika output satu panggilan memberi informasi pada input lainnya, atau ketika rate limiter tidak dapat menangani fan-out.

### Korelasi id

Setiap panggilan yang dipancarkan model memiliki `id`. Setiap hasil yang dikembalikan host harus menyertakan id yang sama. Tanpa ini, hasil-hasil itu ambigu.

- **OpenAI.** `tool_call_id` pada setiap pesan role tool.
- **Anthropic.** `tool_use_id` pada setiap blok `tool_result`.
- **Gemini.** `id` pada setiap `functionResponse` (Gemini 3 ke atas; Gemini 2 dicocokkan berdasarkan nama yang rusak untuk parallel call bernama sama).

### Menjalankan panggilan secara bersamaan

Host menjalankan eksekutor setiap panggilan di thread, coroutine, atau worker remote-nya sendiri. Harness paling sederhana menggunakan thread pool; produksi menggunakan asyncio dengan `asyncio.gather` atau structured concurrency. Urutan penyelesaian tidak dapat diprediksi — id-nya adalah pengenal.

Satu bug umum: membalas dengan hasil dalam urutan daftar panggilan alih-alih urutan penyelesaian. Ini biasanya berfungsi karena model hanya peduli pada `tool_call_id`, tetapi jika sebuah hasil dijatuhkan atau diduplikasi, pengiriman yang tidak berurutan membuat debugging lebih sulit. Lebih suka membalas dalam urutan penyelesaian dengan id eksplisit.

### Streaming tool call

Ketika model melakukan stream, `arguments` tiba dalam potongan. Tiga stream terpisah dari chunk untuk tiga parallel call saling berinterleaving di wire. Kamu membutuhkan satu akumulator per id.

Bentuk per provider:

- **OpenAI.** Setiap chunk adalah `choices[0].delta.tool_calls[i].function.arguments` (string sebagian). Chunk membawa `index` (posisi dalam daftar panggilan). Kamu akumulasikan per-index, baca `id` saat pertama kali muncul, dan urai JSON saat `finish_reason = "tool_calls"`.
- **Anthropic.** Event stream adalah `message_start`, lalu satu `content_block_start` per blok dengan tipe `tool_use` (berisi id, name, input kosong). Event `content_block_delta` membawa chunk `input_json_delta`. `content_block_stop` menutup setiap blok.
- **Gemini.** `streamFunctionCallArguments` (Gemini 3 ke atas) memancarkan chunk dengan `functionCallId` sehingga panggilan saling berinterleaving dengan bersih. Sebelum Gemini 3, streaming mengembalikan satu panggilan lengkap pada satu waktu.

### JSON sebagian dan jebakan parse-early

Kamu tidak dapat mengurai `arguments` sampai ia lengkap. JSON sebagian seperti `{"city": "Beng` tidak valid dan akan memunculkan error. Gate yang benar adalah sinyal akhir-panggilan dari provider: `finish_reason = "tool_calls"` milik OpenAI, `content_block_stop` milik Anthropic, atau event stream-end milik Gemini. Baru setelah itu coba `json.loads`. Pendekatan yang lebih tangguh menggunakan parser JSON inkremental yang menghasilkan event saat struktur selesai; panduan streaming OpenAI merekomendasikan ini untuk UX yang menampilkan indikator "thinking" langsung. Menghitung kurawal tidak andal sebagai tes kelengkapan (kurawal di dalam string yang di-quote atau konten yang di-escape menyebabkan false positive) dan hanya boleh digunakan sebagai heuristik debug informal.

### Penyelesaian tidak berurutan

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

Balasan host tetap harus menyebut id-idnya:

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

Urutan dalam balasan tidak berpengaruh pada kebenaran di OpenAI atau Anthropic. Gemini menerima urutan apa pun selama id cocok.

### Benchmark: sekuensial vs paralel

Harness di `code/main.py` mensimulasikan tiga eksekutor dengan latensi 400, 600, dan 800 ms. Sekuensial menjalankannya dalam 1800 ms total. Paralel menjalankannya dalam max(400, 600, 800) = 800 ms. Perbedaannya konstan, tidak proporsional, sehingga penghematan tumbuh seiring jumlah tool.

Peringatan dunia nyata: parallel call menekan API hilir. Fan-out 10 arah ke layanan ber-rate-limit akan gagal. Phase 13 · 17 membahas backpressure level gateway; semantik retry direncanakan untuk phase mendatang.

### Wall-clock fan-out streaming

Jika model itu sendiri melakukan stream, kamu dapat mulai mengeksekusi begitu argumen satu panggilan lengkap, alih-alih menunggu semua panggilan finalisasi. Ini adalah optimasi yang didokumentasikan OpenAI tetapi tidak semua SDK mengeksposnya. Harness di pelajaran ini melakukannya: begitu stream simulasi menghasilkan objek argumen yang lengkap, host memulai panggilan itu.

```figure
tp-parallel-fanout
```

## Gunakan

`code/main.py` memiliki dua bagian. Bagian pertama menjalankan tiga panggilan cuaca simulasi secara sekuensial dan paralel menggunakan `concurrent.futures.ThreadPoolExecutor` dan mencetak waktu wall-clock. Bagian kedua memutar ulang respons streaming palsu — chunk `arguments` untuk tiga parallel call yang saling berinterleaving di satu stream — dan merakitnya per-id dengan `StreamAccumulator`. Tanpa LLM, tanpa jaringan, hanya logika perakitan ulang.

Yang perlu diperhatikan:

- Timer sekuensial mencapai 1,8 detik. Timer paralel mencapai 0,8 detik pada latensi palsu yang sama.
- Akumulator menangani chunk yang tiba tidak berurutan dengan buffering per-id dan mengurai hanya ketika JSON tiap panggilan lengkap.
- Eksekutor dimulai begitu argumen sebuah id finalisasi, bukan setelah semua stream selesai.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-parallel-call-safety-check.md`. Diberikan registry tool, skill mengaudit tool mana yang aman diparalelkan, mana yang memiliki ketergantungan urutan, dan mana yang akan menenggelamkan rate limit hilir — mengembalikan registry terrevisi dengan flag `parallel_safe` per tool.

## Latihan

1. Jalankan `code/main.py` dan variasikan latensi simulasi. Konfirmasikan bahwa rasio paralel-ke-sekuensial kira-kira `max/sum` (run nyata sedikit menyimpang dari ideal karena penjadwalan thread, serialisasi, dan overhead harness). Pada distribusi latensi apa paralel berhenti penting?

2. Perluas akumulator untuk menangani kasus "call dibatalkan di tengah stream" dengan menjatuhkan buffernya dan memancarkan event `cancelled`. Provider mana yang mendokumentasikan kasus ini secara eksplisit? Periksa semantik `content_block_stop` Anthropic dan perilaku `finish_reason: "length"` OpenAI.

3. Ganti thread pool dengan `asyncio.gather`. Benchmark keduanya. Kamu seharusnya melihat kemenangan kecil pada async karena biaya context-switch yang lebih rendah, tetapi hanya jika eksekutor melakukan I/O nyata.

4. Pilih dua tool yang TIDAK boleh diparalelkan (mis. `create_file` lalu `write_file`). Tambahkan graph `ordering_dependency` ke registry dan gate fan-out paralel pada graph itu. Ini adalah mesin minimum untuk penjadwalan sadar-ketergantungan, yang diformalkan oleh phase agent-engineering mendatang.

5. Baca bagian parallel-function-calling OpenAI dan dokumen `disable_parallel_tool_use` Anthropic. Identifikasi satu jenis tool dunia nyata di mana Anthropic merekomendasikan menonaktifkan paralelisme. (Petunjuk: mutasi konsekuensial pada resource yang sama.)

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Parallel tool calls | "Fan-out in one turn" | Model emits multiple tool calls in a single assistant message |
| `parallel_tool_calls` | "OpenAI's flag" | Enable or disable multi-call emission |
| `disable_parallel_tool_use` | "Anthropic's inverse" | Opt-out flag; default is parallel enabled |
| Tool call id | "Correlation handle" | Per-call identifier the result message must echo |
| Accumulator | "Stream buffer" | Per-id string buffer for partial `arguments` chunks |
| Out-of-order completion | "Fastest first" | Parallel calls finish in unpredictable order; ids are the glue |
| Dependency graph | "Ordering constraints" | Tools whose outputs feed into inputs of other tools; cannot parallelize |
| Parse-early trap | "JSON.parse exploded" | Attempting to parse an incomplete `arguments` string |
| `streamFunctionCallArguments` | "Gemini 3 feature" | Streamed argument chunks with unique id per call |
| Completion-order reply | "Don't wait for all" | Reply with results as they arrive, keyed by id |

## Bacaan Lanjutan

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — default behavior and the opt-out flag
- [Anthropic — Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use) — `disable_parallel_tool_use` and result batching
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — id-correlated parallel calls from Gemini 3
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — chunked argument reassembly for OpenAI streams
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — `content_block_delta` with `input_json_delta`
