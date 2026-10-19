# Output Terstruktur & Decoding Terkendali

> Minta JSON pada LLM. Dapat JSON kebanyakan waktu. Di produksi, "kebanyakan" adalah masalahnya. Decoding terkendali mengubah "kebanyakan" menjadi "selalu" dengan mengedit logit sebelum sampling.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## Masalahnya

Sebuah pengklasifikasi mem-prompt LLM: "Return one of {positive, negative, neutral}." Model mengembalikan "The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ...". Parsermu crash. F1 pengklasifikasimu 0.0.

Generasi bebas-bentuk bukan kontrak. Ia saran. Sistem produksi butuh kontrak.

Tiga lapisan ada di 2026.

1. **Prompting.** Minta baik-baik. "Return only the JSON object." Berhasil ~80% pada model frontier, kurang pada yang kecil.
2. **API output terstruktur native.** `response_format` OpenAI, tool use Anthropic, JSON mode Gemini. Andal pada skema yang didukung. Terkunci vendor.
3. **Decoding terkendali.** Modifikasi logit pada setiap langkah generasi sehingga model *tidak bisa* mengeluarkan token invalid. 100% valid by construction. Berhasil pada model lokal mana pun.

Pelajaran ini membangun intuisi untuk ketiganya dan menyebutkan kapan meraih mana.

## Konsepnya

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**Cara kerja decoding terkendali.** Pada tiap langkah generasi, LLM menghasilkan vektor logit atas seluruh vocabulary (~100k token). *Prosesor logit* duduk antara model dan sampler. Ia menghitung token mana yang valid diberi posisi kini dalam tata bahasa target — JSON Schema, regex, context-free grammar — dan menyetel logit semua token invalid ke negatif infinit. Softmax atas logit tersisa menaruh massa probabilitas hanya pada lanjutan valid.

Implementasi di 2026:

- **Outlines.** Mengompilasi JSON Schema atau regex menjadi finite-state machine. Setiap token mendapat lookup valid-next-token O(1). Berbasis FSM, sehingga skema rekursif perlu di-flatten.
- **XGrammar / llguidance.** Engine tata bahasa context-free. Menangani JSON Schema rekursif. Overhead decoding nyaris-nol. OpenAI mengkredit llguidance dalam implementasi output terstruktur 2025 mereka.
- **Guided decoding vLLM.** `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` bawaan via backend Outlines, XGrammar, atau lm-format-enforcer.
- **Instructor.** Wrapper berbasis Pydantic atas LLM apa pun. Retry saat validasi gagal. Lintas-provider, tetapi tidak memodifikasi logit — ia mengandalkan retry + prompt sadar-output-terstruktur.

### Hasil yang kontraintuitif

Decoding terkendali sering *lebih cepat* daripada generasi tak terkendali. Dua alasan. Pertama, ia menyusutkan ruang pencarian token-berikutnya. Kedua, implementasi cerdas melewatkan generasi token sepenuhnya untuk token paksa (scaffolding seperti `{"name": "` — setiap byte sudah ditentukan).

### Jebakan yang merugikanmu

Urutan field penting. Taruh `answer` sebelum `reasoning`, dan model berkomitmen pada jawaban sebelum berpikir. JSON valid. Jawaban salah. Tak ada validasi yang menangkapnya.

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Urutan field skema adalah logika, bukan formatting.

```figure
constrained-decoder
```

## Bangun Sendiri

### Langkah 1: generasi terkendali-regex dari nol

Lihat `code/main.py` untuk implementasi FSM mandiri. Ide intinya dalam 30 baris:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM melacak bagian tata bahasa mana yang sudah kita penuhi sejauh ini. `valid_tokens(state, tokenizer)` menghitung token vocabulary mana yang bisa memajukan FSM tanpa meninggalkan jalur accepting.

### Langkah 2: Outlines untuk JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Nol error validasi. Selamanya. FSM membuat output invalid tak terjangkau.

### Langkah 3: Instructor untuk Pydantic agnostik-provider

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

Mekanisme berbeda. Instructor tidak menyentuh logit. Ia memformat skema ke prompt, mem-parse output, dan retry saat validasi gagal (default 3 kali). Berhasil dengan provider mana pun. Retry menambah latensi dan biaya. Portabilitas lintas-provider adalah nilai jualnya.

### Langkah 4: API vendor native

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Decoding terkendali sisi-server. Paritas reliabilitas dengan Outlines untuk skema yang didukung. Tanpa manajemen model lokal. Menguncimu ke vendor.

## Jebakan

- **Skema rekursif.** Outlines me-flatten rekursi ke kedalaman tetap. Output berstruktur-pohon (komentar bersarang, AST) butuh XGrammar atau llguidance (berbasis CFG).
- **Enum besar.** Enum 10.000-opsi terkompilasi lambat atau timeout. Beralih ke retriever: prediksi kandidat top-k dulu, kendalikan ke itu.
- **Tata bahasa terlalu ketat.** Paksa regex `date: "YYYY-MM-DD"` dan model tidak bisa mengeluarkan `"unknown"` untuk tanggal hilang. Model mengkompensasi dengan mengarang tanggal. Izinkan `null` atau sentinel.
- **Komitmen prematur.** Lihat jebakan urutan-field di atas. Selalu taruh reasoning dulu.
- **JSON mode vendor tanpa skema.** JSON mode murni hanya menjamin JSON valid, bukan valid *untuk use case-mu*. Selalu berikan skema penuh.

## Gunakan

Stack 2026:

| Situation | Pick |
|-----------|------|
| OpenAI/Anthropic/Google model, simple schema | Native vendor structured output |
| Any provider, Pydantic workflow, can tolerate retries | Instructor |
| Local model, need 100% validity, flat schema | Outlines (FSM) |
| Local model, recursive schema | XGrammar or llguidance |
| Self-hosted inference server | vLLM guided decoding |
| Batch processing with retries acceptable | Instructor + cheapest model |

## Rilis

Simpan sebagai `outputs/skill-structured-output-picker.md`:

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## Latihan

1. **Mudah.** Prompt model open-weights kecil (mis. Llama-3.2-3B) tanpa decoding terkendali untuk `Review(sentiment, confidence, evidence_span)`. Ukur fraksi yang ter-parse sebagai JSON valid pada 100 ulasan.
2. **Sedang.** Korpus sama dengan JSON mode Outlines. Bandingkan tingkat compliance, latensi, dan akurasi semantik.
3. **Sulit.** Implementasikan decoder terkendali-regex dari nol untuk nomor telepon (`\d{3}-\d{3}-\d{4}`). Verifikasi 0 output invalid pada 1000 sampel.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Constrained decoding | Force valid output | Mask invalid-token logits at every generation step. |
| Logit processor | The thing that constrains | Function: `(logits, state) -> masked_logits`. |
| FSM | Finite-state machine | Compiled grammar representation; O(1) valid-next-token lookup. |
| CFG | Context-free grammar | Grammar that handles recursion; slower but more expressive than FSM. |
| Schema field order | Does it matter? | Yes — first field commits; always put reasoning before answer. |
| Guided decoding | vLLM's name for it | Same concept, integrated into the inference server. |
| JSON mode | OpenAI's early version | Guarantees JSON syntax; does NOT guarantee schema match. |

## Bacaan Lanjutan

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — the Outlines paper.
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — fast CFG-based constrained decoding.
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — inference server integration.
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API reference + gotchas.
- [Instructor library](https://python.useinstructor.com/) — Pydantic + retries across providers.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — benchmarking 6 constrained decoding frameworks.
