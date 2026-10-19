# Chatbot — Berbasis Aturan ke Neural ke Agen LLM

> ELIZA menjawab dengan pencocokan pola. DialogFlow memetakan intent. GPT menjawab dari bobot. Claude menjalankan tool dan memverifikasi. Setiap era memecahkan kegagalan terburuk era sebelumnya.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## Masalahnya

Seorang pengguna berkata "I want to change my flight." Sistem harus mencari tahu apa maunya, informasi apa yang kurang, bagaimana mendapatkannya, dan bagaimana menyelesaikan aksinya. Lalu pengguna berkata "wait, what if I cancel instead?" dan sistem harus mengingat konteks, beralih tugas, dan menjaga state.

Percakapan itu sulit bagi sistem ML. Input terbuka. Output harus koheren selama banyak turn. Sistem mungkin perlu bertindak pada dunia (mengubah penerbangan, menagih kartu). Setiap langkah salah terlihat oleh pengguna.

Arsitektur chatbot telah berputar melalui empat paradigma, masing-masing diperkenalkan karena yang sebelumnya gagal terlalu terlihat. Pelajaran ini menelusurinya berurutan. Lanskap produksi 2026 adalah hibrida dari dua terakhir.

## Konsepnya

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

### Setengah abad terskrip, 1950-2001

Paradigma pertama tidak bertahan lima tahun. Ia bertahan lima puluh. Mengetahui busurnya penting karena setiap sistem di dalamnya adalah mesin yang sama — cocokkan input, keluarkan respons kalengan, perbarui sedikit state — dan lima puluh tahun menambahkan aturan ke mesin itu tak pernah menghasilkan kasus umum. Langit-langit itulah alasan paradigma dua sampai empat ada.

**1950.** Turing menghindari "can machines think?" dengan mengusulkan pengganti operasional: jika interogator tak bisa membedakan mesin dari manusia lewat teletype, pertanyaan filosofisnya moot. Percakapan menjadi benchmark bidang ini sebelum bidang ini punya nama.

**1956.** Namanya tiba — workshop musim panas di Dartmouth menciptakan "artificial intelligence" atas konjektur bahwa setiap fitur kecerdasan "pada prinsipnya bisa dideskripsikan begitu presisi sehingga mesin bisa dibuat mensimulasikannya." Proposal itu menganggarkan dua bulan untuk progres substansial.

**1966.** ELIZA menghadirkan trik refleksi yang kamu bangun di Langkah 1: aturan dekomposisi menarik fragmen dari input, aturan reassembly menggemakannya kembali sebagai pertanyaan. Sekitar 200 pola total, nol state, nol pemahaman — dan pengguna tetap curhat padanya. Weizenbaum menghabiskan sisa kariernya cemas betapa sedikit mesin yang dibutuhkan.

**1972.** PARRY, dibangun di Stanford untuk memodelkan paranoia, menambahkan bagian yang tak dimiliki ELIZA: state internal. Variabel numerik untuk fear, anger, dan mistrust diperbarui setiap turn dan menggerbang script mana yang menyala berikutnya, sehingga input identik menghasilkan respons berbeda tergantung percakapan sejauh ini. Dalam uji transkrip buta, psikiater membedakan PARRY dari pasien manusia pada level chance. Ia adalah leluhur langsung persona conditioning — system prompt yang diimplementasikan sebagai tiga float. Tahun yang sama, kedua bot diarahkan satu sama lain lewat ARPANET: script terapis mewawancarai mesin state paranoia, percakapan bot-ke-bot pertama di jaringan.

**1995.** ALICE menskalakan resep ELIZA dengan AIML, dialek XML untuk pasangan pola-template. Kira-kira 40.000 kategori tulisan tangan, tiga kemenangan Loebner Prize. Ia membuktikan hukum scaling sistem berbasis aturan: lebih banyak aturan membeli cakupan, tak pernah generalitas. Setiap aturan adalah liabilitas yang harus dipelihara seseorang.

**2001.** SmarterChild menaruh resep itu di depan 30 juta pengguna instant-messenger dan menambahkan lookup backend — cuaca, saham, jadwal film — disambung ke template. Lihat dengan menyipit dan itu adalah tool calling berkostum 2001: parse intent, panggil layanan, render hasilnya ke balasan.

Lima puluh tahun, satu mekanisme, jumlah aturan menanjak. Paradigma itu berakhir bukan karena ada yang membantahnya tetapi karena biaya pemeliharaan mesin state tulisan tangan tumbuh linear dengan cakupan sementara ekspektasi pengguna tumbuh mengikuti apa pun yang mereka lihat minggu lalu.

```figure
chatbot-lineage
```

**Berbasis aturan (ELIZA, AIML, DialogFlow).** Pola tulisan tangan mencocokkan input pengguna dan menghasilkan respons. Pengklasifikasi intent merutekan ke alur terdefinisi. Mesin state pengisi-slot mengumpulkan info wajib. Berhasil brilian di dalam cakupan sempit yang dirancang untuknya. Gagal seketika di luarnya. Masih dikirim di domain kritis-keamanan (autentikasi perbankan, pemesanan maskapai) di mana halusinasi tidak ditoleransi.

**Berbasis retrieval.** Sistem ala FAQ. Encode setiap pasangan (ujaran, respons). Saat runtime, encode pesan pengguna dan ambil respons tersimpan terdekat. Bayangkan fitur klasik "artikel serupa" Zendesk. Menangani parafrasa lebih baik daripada aturan. Tanpa generasi, sehingga tanpa halusinasi.

**Neural (seq2seq).** Encoder-decoder yang dilatih pada log percakapan. Menghasilkan respons dari nol. Fasih tetapi rentan output generik ("I don't know") dan drift faktual. Tak pernah andal sejalan topik. Alasan Google, Facebook, dan Microsoft semuanya punya chatbot mengecewakan di 2016-2019.

**Agen LLM.** Model bahasa yang dibungkus loop yang merencanakan, memanggil tool, dan memverifikasi hasil. Bukan chatbot dengan prompt panjang. Loop agen: rencanakan → panggil tool → amati hasil → putuskan langkah berikutnya. Grounding retrieval-first (RAG) menjaganya dari halusinasi. Pemanggilan tool memungkinkannya benar-benar melakukan sesuatu. Ini arsitektur 2026.

Keempat paradigma bukan pengganti berurutan. Chatbot produksi 2026 merutekan melalui keempatnya: berbasis aturan untuk autentikasi dan aksi destruktif, retrieval untuk FAQ, generasi neural untuk frasa natural, agen LLM untuk query terbuka ambigu.

## Bangun Sendiri

### Langkah 1: pencocokan pola berbasis aturan

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

ELIZA dalam 20 baris. Trik refleksi ("I feel sad" → "Why do you feel sad") adalah demo psikoterapis kanonis dari Weizenbaum 1966. Masih instruktif.

### Langkah 2: berbasis retrieval (FAQ)

Cuplikan ilustratif ini membutuhkan `pip install sentence-transformers` (yang menarik torch). `code/main.py` runnable untuk pelajaran ini memakai kemiripan Jaccard stdlib saja, sehingga pelajaran berjalan tanpa dependensi eksternal.

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

Refusal berbasis ambang adalah pilihan desain kunci. Jika kecocokan terbaik tidak cukup dekat, kembalikan `None` dan biarkan sistem mengeskalasi.

### Langkah 3: generasi neural (baseline)

Pakai encoder-decoder kecil instruction-tuned (FLAN-T5) atau model percakapan fine-tune. Tak bisa dipakai produksi sendirian di 2026 (kontradiksi, drift off-topic, omong kosong faktual), tetapi dikirim di dalam sistem hibrida untuk frasa natural. Model decoder-only ala DialoGPT butuh pemisah turn eksplisit dan penanganan EOS untuk menghasilkan balasan koheren; pipeline text2text FLAN-T5 bekerja langsung untuk contoh pengajaran.

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### Langkah 4: loop agen LLM

Bentuk produksi 2026:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

Tiga hal yang perlu disebut. Tool adalah fungsi callable yang bisa dipanggil LLM. Loop berhenti ketika LLM mengembalikan jawaban final alih-alih pemanggilan tool. Budget langkah mencegah loop infinit pada tugas ambigu.

Produksi nyata menambahkan: grounding retrieval-first (suntikkan docs relevan sebelum tiap panggilan LLM), guardrail (tolak aksi destruktif tanpa konfirmasi), observabilitas (log setiap langkah), dan evaluasi (cek otomatis bahwa perilaku agen tetap on-spec).

### Langkah 5: routing hibrida

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

Polanya: aturan deterministik untuk apa pun yang destruktif, retrieval untuk FAQ kalengan, agen LLM untuk selebihnya. Inilah yang dikirim dalam sistem dukungan pelanggan 2026.

## Gunakan

Stack 2026:

| Use case | Architecture |
|---------|---------------|
| Booking, payment, authentication | Rule-based state machines + slot filling |
| Customer support FAQs | Retrieval over curated answers |
| Open-ended help chat | LLM agent with RAG + tool calls |
| Internal tools / IDE assistants | LLM agent with tool calls (search, read, write) |
| Companion / character chatbots | Tuned LLM with persona system prompt, retrieval on knowledge |

Selalu pakai routing hibrida di produksi. Tak ada satu arsitektur yang menangani setiap request dengan baik. Layer routing sendiri biasanya pengklasifikasi intent kecil.

## Mode kegagalan yang masih lolos

- **Fabrikasi percaya diri.** Agen LLM mengklaim menyelesaikan aksi yang tidak ia lakukan. Mitigasi: verifikasi hasil, log pemanggilan tool, jangan pernah biarkan LLM mengklaim telah melakukan sesuatu tanpa return tool yang sukses.
- **Injeksi prompt.** Pengguna menyisipkan teks yang menimpa system prompt. Peringkat LLM01 dalam OWASP Top 10 for LLM Applications 2025. Dua rasa: injeksi langsung (ditempel ke chat) dan injeksi tak langsung (tersembunyi dalam dokumen, email, atau output tool yang dibaca agen).

  Tingkat serangan bervariasi per skenario. Tingkat keberhasilan terukur berkisar ~0.5-8.5% lintas model frontier dalam benchmark tool-use dan coding umum. Setup berisiko tinggi spesifik (serangan adaptif terhadap agen coding AI, orkestrasi rentan) telah mencapai ~84%. CVE produksi mencakup EchoLeak (CVE-2025-32711, CVSS 9.3) — cacat eksfiltrasi data zero-click di Microsoft 365 Copilot yang dipicu email kendali-penyerang.

  Mitigasi: perlakukan input pengguna sebagai tak tepercaya di seluruh loop; sanitasi sebelum pemanggilan tool; isolasi output tool dari prompt utama; pakai pola Plan-Verify-Execute (PVE) di mana agen merencanakan dulu, lalu memverifikasi tiap aksi terhadap rencana itu sebelum mengeksekusi (ini menghentikan hasil tool menyuntikkan aksi baru tak direncanakan); wajibkan konfirmasi pengguna untuk aksi destruktif; terapkan least-privilege pada scope tool.

  Tak ada jumlah prompt engineering yang sepenuhnya menghilangkan risiko ini. Lapisan pertahanan runtime eksternal (LLM Guard, validasi allowlist, deteksi anomali semantik) dibutuhkan.
- **Scope creep.** Agen keluar tugas karena pemanggilan tool mengembalikan info yang terkait tangensial. Mitigasi: kontrak tool sempit; jaga system prompt fokus; tambah evaluasi untuk off-task rate.
- **Loop infinit.** Agen terus memanggil tool yang sama. Mitigasi: budget langkah, deduplikasi pemanggilan tool, LLM judge pada "apakah kita membuat progres."
- **Kelelahan context window.** Percakapan panjang mendorong turn paling awal keluar konteks. Mitigasi: ringkas turn lama, ambil turn lampau relevan via kemiripan, atau pakai model konteks-panjang.

## Rilis

Simpan sebagai `outputs/skill-chatbot-architect.md`:

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## Latihan

1. **Mudah.** Implementasikan respond berbasis aturan di atas dengan 10 pola untuk bot pemesanan kedai kopi. Uji edge case: pesanan ganda, modifikasi, pembatalan, intent tak jelas.
2. **Sedang.** Bangun FAQ hibrida + fallback LLM. 50 entri FAQ kalengan untuk produk SaaS, fallback LLM dengan retrieval atas situs docs. Ukur tingkat refusal dan akurasi pada 100 pertanyaan dukungan nyata.
3. **Sulit.** Implementasikan loop agen di atas dengan tiga tool (search, read-user-data, send-email). Jalankan evaluasi dengan 50 skenario uji termasuk upaya injeksi prompt. Laporkan off-task rate, tingkat tugas gagal, dan keberhasilan injeksi apa pun.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Intent | What the user wants | Categorical label (book_flight, reset_password). Routed to a handler. |
| Slot | A piece of info | Parameter the bot needs (date, destination). Slot filling is the sequence of asks. |
| RAG | Retrieval plus generation | Retrieve relevant docs, then ground the LLM's response. |
| Tool call | Function invocation | LLM emits a structured call with name + args. Runtime executes, returns result. |
| Agent loop | Plan, act, verify | Controller that runs LLM calls interleaved with tool calls until task complete. |
| Prompt injection | User attacks prompt | Malicious input that tries to override the system prompt. |

## Bacaan Lanjutan

- [Turing (1950). Computing Machinery and Intelligence](https://academic.oup.com/mind/article/LIX/236/433/986238) — the paper that made conversation the field's benchmark.
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — the original rule-based chatbot paper.
- [Colby, Weber, Hilf (1971). Artificial Paranoia](https://doi.org/10.1016/0004-3702(71)90002-6) — PARRY's affect-variable architecture, the first stateful chatbot.
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — Google's late neural-chatbot paper, just before LLM agents took over.
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — the paper that named the agent loop pattern.
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 production guidance that still holds in 2026.
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — the prompt-injection paper.
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — the ranking that made prompt injection the top security concern.
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — practical orchestration-layer defenses including Plan-Verify-Execute and user-confirmation flows.
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — the canonical zero-click data-exfiltration CVE from indirect prompt injection. Reference case for why write-access agents need runtime defenses.
