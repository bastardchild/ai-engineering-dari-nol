# Warisan FIPA-ACL dan Speech Acts

> Sebelum MCP, sebelum A2A, ada FIPA-ACL. Pada tahun 2000, IEEE Foundation for Intelligent Physical Agents meratifikasi bahasa komunikasi agent dengan dua puluh performative, dua bahasa konten, dan serangkaian interaction protocol — contract net, subscribe/notify, request-when. Ia memudar dari industri karena beban ontology terlalu berat untuk web, tetapi kebangkitan LLM dari sistem multi-agent diam-diam mengimplementasikan ulang ide yang sama tanpa semantik formal: kontrak JSON berperan sebagai performative, bahasa alami berperan sebagai ontology. Pelajaran ini membaca FIPA-ACL secara serius agar kamu bisa melihat keputusan protokol 2026 mana yang merupakan penemuan ulang, mana yang merupakan kebaruan, dan ke mana gelombang saat ini akan kembali menemukan masalah yang sudah diselesaikan pada tahun 2000-an.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## Masalahnya

Lanskap protokol agent 2026 ramai: MCP untuk tool, A2A untuk agent, ACP untuk audit enterprise, ANP untuk kepercayaan terdesentralisasi, NLIP untuk konten bahasa alami, ditambah CA-MCP dan dua lusin proposal riset. Setiap spesifikasi mengumumkan dirinya sebagai fondasional.

Pembacaan jujurnya adalah bahwa sebagian besar dari mereka sedang menemukan kembali pohon keputusan yang sangat spesifik berumur dua puluh tahun. Teori speech act dari Austin (1962) dan Searle (1969) memberi kita "ucapan adalah tindakan." KQML (1993) mengubahnya menjadi protokol wire. FIPA-ACL (diratifikasi tahun 2000) menghasilkan standardisasi rujukan: dua puluh performative, bahasa konten SL0/SL1, interaction protocol untuk contract-net dan subscribe-notify. JADE dan JACK adalah platform rujukan Java. Upaya itu memudar sekitar tahun 2010 karena beban ontology terlalu berat dan web sedang menang.

Saat kamu melihat `tools/call` MCP, lifecycle tugas A2A, atau shared context store CA-MCP, kamu sedang melihat penulisan ulang yang lebih lembut dan native-JSON dari keputusan FIPA. Mengetahui warisannya memberi tahu kamu dua hal: inovasi "baru" mana yang sebenarnya merupakan penemuan ulang, dan mode kegagalan lama mana yang akan ditemukan kembali oleh spesifikasi baru.

## Konsepnya

### Speech act, dalam satu paragraf

Austin menyadari bahwa beberapa kalimat tidak menggambarkan dunia — mereka mengubahnya. "Aku berjanji." "Aku meminta." "Aku menyatakan." Ia menyebutnya performative utterances. Searle memformalkan lima kategori: assertive, directive, commissive, expressive, declarative. KQML (Finin et al., 1993) membuatnya operasional untuk agent perangkat lunak: sebuah pesan adalah performative (tindakannya) ditambah konten (tentang apa tindakan itu). FIPA-ACL membersihkan celah KQML dan menstandardisasi sekitar dua puluh performative.

### Dua puluh performative FIPA (daftar sebagian)

| Performative | Intent |
|---|---|
| `inform` | "I tell you P is true" |
| `request` | "I ask you to do X" |
| `query-if` | "Is P true?" |
| `query-ref` | "What is the value of X?" |
| `propose` | "I propose we do X" |
| `accept-proposal` | "I accept the proposal" |
| `reject-proposal` | "I reject the proposal" |
| `agree` | "I agree to do X" |
| `refuse` | "I refuse to do X" |
| `confirm` | "I confirm P is true" |
| `disconfirm` | "I deny P" |
| `not-understood` | "Your message did not parse" |
| `cfp` | "Call for proposals on X" |
| `subscribe` | "Notify me when X changes" |
| `cancel` | "Cancel the ongoing X" |
| `failure` | "I tried X and failed" |

Daftar lengkapnya ada di `fipa00037.pdf` (FIPA ACL Message Structure). Intinya bukan menghafalnya — intinya adalah bahwa setiap satu dari ini sesuai dengan sebuah primitif yang pada akhirnya ditambahkan kembali oleh protokol LLM.

### Pesan FIPA-ACL kanonis

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

Tujuh field membawa amplop protokol; satu field (`content`) membawa payload. Sisanya persis yang kamu temukan ulang setiap kali kamu menempelkan retries, threading, dan ontology ke protokol JSON.

### Dua platform legasi

**JADE** (Java Agent DEvelopment framework, 1999–2020s) adalah runtime FIPA-compliant yang paling banyak digunakan. Agent memperluas base class, bertukar pesan ACL, berjalan di dalam container, dan berkoordinasi menggunakan "behaviors." Library interaction-protocol-nya dikirim dengan contract-net, subscribe-notify, request-when, dan propose-accept.

**JACK** (Agent Oriented Software, komersial) menekankan penalaran BDI (Belief-Desire-Intention) di atas pesan FIPA. Lebih formal, kurang diadopsi.

Keduanya menurun begitu stack web memakan use case multi-agent. MCP dan A2A adalah "container" runtime tahun 2026.

### Mengapa FIPA memudar

- **Beban ontology.** FIPA mensyaratkan ontology bersama untuk menguraikan `content`. Kesepakatan tentang ontology adalah proses standar selama bertahun-tahun. Web hanya menggunakan HTTP + JSON.
- **Semantik formal yang tidak dipakai siapa pun.** SL (Semantic Language) memberikan syarat kebenaran yang ketat, tetapi sebagian besar sistem produksi menggunakan konten bebas bentuk dan mengabaikan formalisme itu.
- **Keterkuncian tooling.** JADE hanya untuk Java; JACK komersial. Tim polyglot menyiasati keduanya.
- **Internet memenangkan stack-nya.** REST, lalu JSON-RPC, lalu gRPC menggantikan transport ACL.

### Kebangkitan LLM adalah FIPA-lite

Bandingkan sebuah `request` FIPA dengan `tools/call` MCP:

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

Amplop yang sama, sintaks berbeda. Keduanya membawa: siapa, kepada siapa, intent, payload, correlation id. Tidak ada yang merupakan revolusi atas yang lain — keduanya adalah trade-off berbeda pada desain yang sama.

Survei 2025 oleh Liu et al. ("A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP", arXiv:2505.02279) membuat garis keturunan ini eksplisit: MCP sesuai dengan speech act penggunaan-tool, A2A dengan speech act peer-agent, ACP dengan speech act jejak-audit, ANP dengan ekstensi identitas-terdesentralisasi. Spesifikasi baru adalah keturunan ACL dengan sintaks JSON dan semantik yang lebih longgar.

### Trade-off-nya, dinyatakan apa adanya

**Apa yang FIPA berikan kepadamu dan spesifikasi modern buang:**

- Semantik formal — kamu bisa membuktikan `inform` menyiratkan pengirim percaya kontennya.
- Katalog kanonis performative — kamu tidak perlu berdebat lagi "haruskah kita punya `cancel`?"
- Puluhan pola interaction-protocol — contract-net, subscribe-notify, propose-accept — dengan properti kebenaran yang diketahui.

**Apa yang spesifikasi modern berikan kepadamu dan FIPA tidak:**

- Payload native-JSON yang kompatibel dengan setiap tool modern.
- Konten bahasa alami yang bisa diinterpretasikan LLM tanpa ontology yang ditulis tangan.
- Transport stack-web (HTTP, SSE, WebSocket).
- Capability discovery melalui MCP `server/discover` dan A2A Agent Cards.

Semantik intent yang lebih longgar untuk implementasi yang lebih mudah. Itulah trade persisnya.

### Interaction protocol yang layak dipindahkan

FIPA mengirimkan ~15 interaction protocol. Tiga layak dibawa ke sistem multi-agent LLM:

1. **Contract Net Protocol (CNP).** Manager mengeluarkan `cfp` (call for proposals); penawar merespons dengan `propose`; manager menerima/menolak. Ini adalah pola pasar-tugas kanonis (Phase 16 · 16 Negotiation).
2. **Subscribe/Notify.** Subscriber mengirim `subscribe`; publisher mengirim `inform` setiap kali topik berubah. Ini adalah setiap event-bus di tahun 2026.
3. **Request-When.** "Lakukan X ketika kondisi Y terpenuhi." Tunda-tindakan dengan pra-kondisi. Analog 2026 adalah tugas tunda di engine workflow tahan lama (Phase 16 · 22 Production Scaling).

Masing-masing dipetakan bersih ke message queue modern, HTTP + polling, atau streaming SSE.

### Apa yang rusak saat kamu membuang ontology

Tanpa ontology bersama, agent menyimpulkan makna dari konten bahasa alami. Mode kegagalan 2026 yang terdokumentasi adalah **semantic drift**: dua agent menggunakan kata yang sama (`"customer"`) untuk konsep yang sedikit berbeda, agent penerima bertindak pada interpretasi yang salah, tidak ada schema validator yang menangkapnya. Persyaratan ontology FIPA akan menolak pesan itu pada waktu parsing.

Mitigasi tanpa masuk ke ontology penuh:

- JSON Schema pada `content` — menolak kesalahan struktural di wire.
- Artefak bertipe (A2A) — menolak modalitas yang salah.
- Performative eksplisit di amplop — membuat intent tidak ambigu bahkan ketika kontennya bahasa alami.

### Spesifikasi 2026, dipetakan ke warisan speech-act

| Modern spec | FIPA analog | What it keeps | What it drops |
|---|---|---|---|
| MCP `tools/call` | `request` | explicit intent, correlation id | formal semantics, ontology |
| MCP `resources/read` | `query-ref` | explicit intent, correlation id | formal semantics |
| A2A Task lifecycle | contract-net + request-when | async lifecycle, state transitions | formal completeness guarantees |
| A2A streaming events | subscribe/notify | async push | typed-predicate subscription |
| CA-MCP shared context | blackboard (Hayes-Roth 1985) | multi-writer shared memory | logical consistency model |
| NLIP | natural-language content | LLM-native | schema |

Membaca tabel dari atas ke bawah, polanya adalah: pertahankan primitif struktural, buang formalismenya, biarkan LLM menambal ambiguitasnya.

```figure
sw-contract-net
```

## Bangun

`code/main.py` mengimplementasikan translator FIPA-ACL pure-stdlib. Ia mengenkode dan mendekodekan amplop ACL kanonis dan menunjukkan bagaimana setiap bentuk pesan MCP / A2A direduksi ke tujuh field yang sama. Demo-nya:

- Mengenkode lima pesan bergaya MCP dan A2A sebagai FIPA-ACL.
- Mendekodekan FIPA-ACL kembali ke padanan modernnya.
- Menjalankan negosiasi Contract Net main-main antara satu manager dan tiga penawar menggunakan `cfp`, `propose`, `accept-proposal`, `reject-proposal`.

Jalankan:

```
python3 code/main.py
```

Keluarannya adalah jejak berdampingan yang menunjukkan setiap pesan modern dalam bentuk JSON 2026-nya dan bentuk FIPA-ACL-nya, lalu round-trip dari sebuah penawaran contract-net. Primitif protokol yang sama bertahan melalui round-trip; hanya sintaksnya yang berbeda.

## Gunakan

`outputs/skill-fipa-mapper.md` adalah skill yang membaca spesifikasi protokol-agent apa pun dan menghasilkan pemetaan FIPA-ACL. Gunakannya sebelum mengadopsi protokol baru untuk menjawab: "Apakah ini benar-benar baru, atau hanya `inform` dengan sintaks JSON?"

## Rilis

Jangan mengembalikan FIPA-ACL. Kembalikan checklist-nya:

- Apa primitif intent (performative) dari setiap pesan?
- Apakah ada correlation id untuk request-response dan pembatalan?
- Apakah ada bahasa konten eksplisit (JSON-RPC, teks biasa, artefak bertipe terstruktur)?
- Apakah interaction protocol bersifat first-class, atau kamu mengimplementasikan ulang contract-net dari nol?
- Apa yang terjadi ketika dua agent tidak setuju tentang makna konten (semantic drift)?

Dokumentasikan lima pertanyaan ini untuk protokol baru apa pun sebelum kamu merilisnya ke produksi.

## Latihan

1. Jalankan `code/main.py`. Amati encoding round-trip. Identifikasi performative FIPA mana yang sesuai dengan `tools/call`, `resources/read`, dan pembuatan tugas A2A.
2. Perluas demo contract-net dengan sebuah performative `cancel` yang memungkinkan manager menarik tugas di tengah penawaran. Mode kegagalan apa yang dipecahkan `cancel` yang tidak dipecahkan retries saja?
3. Baca FIPA ACL Message Structure (http://www.fipa.org/specs/fipa00037/) bagian 4.1–4.3. Pilih satu performative yang tidak dibahas di pelajaran ini dan jelaskan padanan JSON-RPC modernnya.
4. Baca Liu et al., arXiv:2505.02279. Untuk masing-masing MCP, A2A, ACP, ANP, daftarkan keluarga performative FIPA yang mereka pertahankan dan buang.
5. Rancang JSON-Schema minimal untuk field `content` dari sebuah performative `request` di sistem milikmu sendiri. Apa yang schema itu berikan kepadamu yang tidak diberikan oleh bahasa alami murni, dan apa biayanya?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Speech act | "An utterance that does something" | Austin/Searle: utterances as actions. The theoretical parent of ACL. |
| FIPA | "That old XML thing" | IEEE Foundation for Intelligent Physical Agents. Standardized ACL in 2000. |
| ACL | "Agent Communication Language" | FIPA's envelope format: performative + content + metadata. |
| Performative | "The verb" | The intent class of a message: `inform`, `request`, `propose`, `cfp`, etc. |
| KQML | "FIPA's predecessor" | Knowledge Query and Manipulation Language (1993). Simpler, narrower. |
| Ontology | "Shared vocabulary" | A formal definition of the concepts the content language talks about. |
| SL0 / SL1 | "FIPA content languages" | Semantic Language levels 0 and 1 — the formal content language family. |
| Contract Net | "Task market" | Manager issues cfp; bidders propose; manager accepts. The canonical interaction protocol. |
| Interaction protocol | "Pattern of messages" | A sequence of performatives with known correctness: request-when, subscribe-notify, etc. |

## Bacaan Lanjutan

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — the canonical 2025 survey connecting modern specs to FIPA heritage
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — the ratified 2000 envelope format
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — the full performative catalog
- [MCP specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) — the current stateless tool-use equivalent of `request`/`query-ref`
- [A2A specification](https://a2a-protocol.org/latest/specification/) — the modern agent-peer equivalent of contract-net and subscribe-notify
