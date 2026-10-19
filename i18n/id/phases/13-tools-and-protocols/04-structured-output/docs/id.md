# Structured Output — JSON Schema, Pydantic, Zod, Constrained Decoding

> "Ask the model nicely to return JSON" gagal 5 sampai 15 persen waktunya, bahkan pada model frontier. Structured output menutup jarak itu dengan constrained decoding: model secara harfiah dicegah memancarkan token yang akan melanggar schema. Strict mode milik OpenAI, tool use bertipe schema milik Anthropic, `responseSchema` milik Gemini, `output_type` milik Pydantic AI, dan `.parse` milik Zod adalah lima bentuk permukaan dari ide yang sama. Pelajaran ini membangun validator schema dan kontrak strict-mode yang akan dipakai learner untuk setiap pipeline ekstraksi produksi.

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Menulis JSON Schema 2020-12 untuk target ekstraksi dengan batasan yang tepat (enum, min/max, required, pattern).
- Menjelaskan mengapa strict mode dan constrained decoding memberikan jaminan yang berbeda dari "validasi setelah generasi".
- Membedakan tiga mode kegagalan: parse error, pelanggaran schema, penolakan model.
- Mengirimkan pipeline ekstraksi dengan perbaikan bertipe dan penanganan penolakan bertipe.

## Masalahnya

Sebuah agen yang membaca email purchase-order perlu mengubah teks bebas menjadi `{customer, line_items, total_usd}`. Tiga pendekatan.

**Pendekatan satu: prompt untuk JSON.** "Reply in JSON with fields customer, line_items, total_usd." Berhasil 85 sampai 95 persen pada model frontier. Gagal dengan enam cara: kurawal hilang, koma ekor, tipe salah, field berhalusinasi, terpotong di batas token, teks prosa yang bocor seperti "Here is your JSON:".

**Pendekatan dua: validasi setelah generasi.** Hasilkan bebas, urai, validasi terhadap schema, coba ulang saat gagal. Andal tetapi mahal — kamu membayar untuk setiap retry, dan bug truncation berbiaya satu giliran ekstra per kemunculan.

**Pendekatan tiga: constrained decoding.** Provider menegakkan schema pada waktu decode. Token yang tidak valid dimatikan dari distribusi sampling. Output dijamin terurai dan dijamin tervalidasi. Kegagalan runtuh menjadi satu mode: penolakan (model memutuskan input tidak cocok dengan schema).

Setiap provider frontier 2026 mengirimkan bentuk pendekatan tiga.

- **OpenAI.** `response_format: {type: "json_schema", strict: true}` ditambah `refusal` dalam respons jika model menolak.
- **Anthropic.** Penegakan schema pada input `tool_use`; `stop_reason: "refusal"` bukan hal yang ada, tetapi `end_turn` tanpa tool call adalah sinyalnya.
- **Gemini.** `responseSchema` di level request; pada 2026 Gemini mengirimkan batasan grammar level-token untuk tipe terpilih.
- **Pydantic AI.** `output_type=InvoiceModel` memancarkan `RunResult` terstruktur bertipe ke `InvoiceModel`.
- **Zod (TypeScript).** Parser runtime yang memvalidasi output provider terhadap Zod schema; berpasangan dengan `beta.chat.completions.parse` milik OpenAI.

Benang merahnya: deklarasikan schema sekali, tegakkan end to end.

## Konsepnya

### JSON Schema 2020-12 — lingua franca

Setiap provider menerima JSON Schema 2020-12. Konstruk yang paling sering kamu pakai:

- `type`: salah satu dari `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`.
- `properties`: pemetaan nama field ke subschema.
- `required`: daftar nama field yang harus muncul.
- `enum`: himpunan tertutup dari nilai yang diizinkan.
- `minimum` / `maximum` (angka), `minLength` / `maxLength` / `pattern` (string).
- `items`: subschema yang diterapkan pada setiap elemen array.
- `additionalProperties`: `false` melarang field ekstra (default bervariasi per mode).

Strict mode OpenAI menambahkan tiga persyaratan: setiap properti harus terdaftar di `required`, `additionalProperties: false` di mana-mana, dan tanpa `$ref` yang belum terselesaikan. Jika kamu melanggar ini, API mengembalikan 400 pada waktu request.

### Pydantic, binding Python

Pydantic v2 menghasilkan JSON Schema dari model berbentuk dataclass melalui `model_json_schema()`. Pydantic AI membungkus ini sehingga kamu menulis:

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

dan framework agen menerjemahkan schema ke strict mode OpenAI, `input_schema` Anthropic, atau `responseSchema` Gemini di tepi. Output model kembali sebagai instance `Invoice` bertipe. Error validasi memunculkan `ValidationError` dengan path error bertipe.

### Zod, binding TypeScript

Zod (`z.object({customer: z.string(), ...})`) adalah padanan TS-nya. Node SDK milik OpenAI mengekspos `zodResponseFormat(Invoice)` yang diterjemahkan ke payload JSON Schema API.

### Penolakan

Strict mode tidak dapat memaksa model menjawab. Jika input tidak dapat muat dalam schema ("the email was a poem, not an invoice"), model memancarkan field `refusal` yang berisi alasannya. Kodenmu harus menangani ini sebagai hasil primer, bukan sebagai kegagalan. Penolakan juga berguna sebagai sinyal keamanan: model yang diminta mengekstrak nomor kartu kredit dari email berkonten terlindungi mengembalikan penolakan dengan alasan keamanan terlampir.

### Constrained decoding di terbuka

Implementasi open-weights menggunakan tiga teknik.

1. **Grammar-based decoding** (`outlines`, `guidance`, `lm-format-enforcer`): bangun sebuah deterministik finite automaton dari schema; pada setiap langkah, matikan logits dari token yang akan melanggar FSM-nya.
2. **Logit masking dengan parser JSON**: jalankan parser JSON streaming selang-seling dengan model; pada setiap langkah, hitung himpunan token-berikutnya yang valid.
3. **Speculative decoding dengan verifier**: model draf murah mengusulkan token, verifier menegakkan schema.

Provider komersial memilih salah satu dari ini di balik layar. State of the art 2026 lebih cepat daripada generasi biasa untuk output terstruktur pendek dan kecepatannya kira-kira sama untuk yang panjang.

### Tiga mode kegagalan

1. **Parse error.** Output bukan JSON yang valid. Tidak dapat terjadi di bawah strict mode. Masih dapat terjadi pada provider non-strict.
2. **Pelanggaran schema.** Output terurai tetapi melanggar schema. Tidak dapat terjadi di bawah strict mode. Umum di luarnya.
3. **Penolakan.** Model menolak. Harus ditangani sebagai hasil bertipe.

### Strategi retry

Ketika kamu berada di luar strict mode (tool use Anthropic, OpenAI non-strict, Gemini lama), pola pemulihannya adalah:

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

Satu retry biasanya cukup. Tiga retry menangkap flake model lemah. Di luar tiga adalah tanda schema yang buruk: model tidak dapat memenuhinya untuk beberapa input, dan prompt atau schema-nya perlu diperbaiki.

### Dukungan model kecil

Constrained decoding bekerja pada model kecil. Model terbuka 3B parameter dengan penegakan grammar mengalahkan model 70B parameter dengan prompt mentah pada tugas terstruktur. Ini alasan utama structured output penting untuk produksi: ia memutuskan keandalan dari ukuran model.

```figure
constrained-decoding
```

## Gunakan

`code/main.py` mengirimkan validator JSON Schema 2020-12 minimal dalam stdlib (types, required, enum, min/max, pattern, items, additionalProperties). Ia membungkus schema `Invoice` dan menjalankan output LLM palsu melalui validator, mendemonstrasikan jalur parse error, pelanggaran schema, dan penolakan. Ganti output palsu dengan respons sungguhan provider mana pun di produksi.

Yang perlu diperhatikan:

- Validator mengembalikan daftar `[ValidationError]` bertipe dengan path dan pesan. Itulah bentuk yang ingin kamu tampilkan ke prompt retry.
- Cabang penolakan TIDAK melakukan retry. Ia mencatat log dan mengembalikan penolakan bertipe. Phase 14 · 09 memakai penolakan sebagai sinyal keamanan.
- Pemeriksaan `additionalProperties: false` menyala pada input test adversarial, menunjukkan mengapa strict mode menutup pintu terhadap field yang berhalusinasi.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-structured-output-designer.md`. Diberikan target ekstraksi teks bebas (invoice, support ticket, resume, dll.), skill menghasilkan JSON Schema 2020-12 yang kompatibel strict-mode dan model Pydantic yang mencerminkannya, dengan penolakan bertipe dan penanganan retry yang di-stub.

## Latihan

1. Jalankan `code/main.py`. Tambahkan test case keempat yang `total_usd`-nya berjumlah negatif. Konfirmasikan validator menolaknya dengan path batasan `minimum`.

2. Perluas validator untuk mendukung `oneOf` dengan discriminator. Kasus umumnya: `line_item` entah produk atau layanan, ditandai dengan `kind`. Strict mode memiliki aturan halus di sini; periksa panduan structured outputs OpenAI.

3. Tulis schema Invoice yang sama sebagai BaseModel Pydantic dan bandingkan output `model_json_schema()` dengan schema buatan tanganmu. Identifikasi satu field yang di-set Pydantic secara default tetapi diabaikan oleh versi buatan tangan.

4. Ukur tingkat penolakan. Konstruksikan sepuluh input yang seharusnya tidak dapat diekstraksi (lirik lagu, bukti matematika, email kosong) dan jalankan melalui provider sungguhan dengan strict mode. Hitung penolakan vs output berhalusinasi. Ini ground truth-mu untuk retry sadar-penolakan.

5. Baca panduan structured outputs OpenAI dari awal sampai akhir. Identifikasi satu konstruk yang secara eksplisit dilarangnya di strict mode tetapi diizinkan JSON Schema biasa. Lalu rancang schema yang memakai konstruk terlarang itu secara non-esensial dan refactor agar kompatibel strict.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "The schema spec" | IETF-draft schema dialect every modern provider speaks |
| Strict mode | "Guaranteed schema" | OpenAI flag that enforces schema via constrained decoding |
| Constrained decoding | "Logit masking" | Decode-time enforcement that masks invalid next-tokens |
| Refusal | "Model declines" | Typed outcome when input cannot fit the schema |
| Parse error | "Invalid JSON" | Output did not parse as JSON; impossible under strict |
| Schema violation | "Wrong shape" | Parsed but violated types / required / enum / range |
| `additionalProperties: false` | "No extras allowed" | Forbids unknown fields; required in OpenAI strict |
| Pydantic BaseModel | "Typed output" | Python class that emits and validates JSON Schema |
| Zod schema | "TypeScript output type" | TS runtime schema for provider output validation |
| Grammar enforcement | "Open-weights constrained decode" | FSM-based logit masking, as in outlines / guidance |

## Bacaan Lanjutan

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict mode, refusals, and schema requirements
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — August 2024 launch post explaining the decoding guarantee
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — typed output_type bindings that serialize to each provider
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — the canonical spec
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — enterprise deployment notes and strict-mode caveats
