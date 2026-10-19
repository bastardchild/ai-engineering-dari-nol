# A2A — Agent-to-Agent Protocol

> MCP bersifat agent-to-tool. A2A (Agent2Agent) bersifat agent-to-agent — protokol terbuka yang memungkinkan agent-opaque yang dibangun di framework berbeda berkolaborasi. Dirilis oleh Google pada April 2025, didonasikan ke Linux Foundation pada Juni 2025, mencapai v1.0 pada April 2026 dengan 150+ pendukung termasuk AWS, Cisco, Microsoft, Salesforce, SAP, dan ServiceNow. Ia menyerap ACP milik IBM dan menambahkan ekstensi pembayaran AP2. Pelajaran ini menelusuri Agent Card, lifecycle Task, dan dua binding transport.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Membedakan use case agent-to-tool (MCP) dari agent-to-agent (A2A).
- Menerbitkan Agent Card di `/.well-known/agent.json` dengan metadata skill dan endpoint.
- Menelusuri lifecycle Task (submitted → working → input-required → completed / failed / canceled / rejected).
- Memakai Messages dengan Parts (teks, file, data) dan Artifacts sebagai output.

## Masalahnya

Sebuah agent layanan pelanggan perlu mendelegasikan penulisan laporan ke sebuah agent penulis khusus. Opsi sebelum A2A:

- REST API kustom. Berfungsi tetapi setiap pasangan bersifat sekali-pakai.
- Codebase bersama. Menuntut kedua agent menjalankan framework yang sama.
- MCP. Tidak cocok: MCP untuk memanggil tool, bukan untuk dua agent berkolaborasi sambil mempertahankan penalaran internal masing-masing yang opaque.

A2A mengisi celah itu. Ia memodelkan interaksi sebagai satu agent mengirim Task ke agent lain, dengan lifecycle, pesan, dan artefak. State internal agent yang dipanggil tetap opaque — penelepon hanya melihat transisi state tugas dan output akhirnya.

A2A adalah protokol "biarkan agent lintas framework berbicara satu sama lain". Ia tidak menggantikan MCP; keduanya bersifat komplementer.

## Konsepnya

### Agent Card

Setiap agent yang patuh A2A menerbitkan sebuah card di `/.well-known/agent.json`:

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

Discovery berbasis URL: ambil card-nya, pelajari URL endpoint A2A, enumerasi skill.

### Agent Card Bertanda Tangan (AP2)

Ekstensi AP2 (September 2025) menambahkan tanda tangan kriptografis ke Agent Card. Sebuah penerbit menandatangani card-nya sendiri dengan JWT; konsumen memverifikasi. Mencegah impersonasi.

### Lifecycle Task

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

Klien memulai dengan `tasks/send`. Agent yang dipanggil berpindah melalui state-state; klien subscribe ke pembaruan state via SSE atau poll.

### Messages dan Parts

Sebuah message membawa satu atau lebih Parts:

- `text` — konten polos.
- `file` — blob base64 dengan mimeType.
- `data` — payload JSON bertipe (input terstruktur untuk agent yang dipanggil).

Contoh:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

Output berupa Artifacts, bukan string mentah. Sebuah Artifact adalah output yang bernama dan bertipe:

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts dapat di-stream sebagai chunk. Penelepon mengakumulasikannya.

### Dua binding transport

1. **JSON-RPC over HTTP.** Endpoint `/a2a`, POST untuk permintaan, SSE opsional untuk streaming. Binding default.
2. **gRPC.** Untuk lingkungan enterprise tempat gRPC bersifat native.

Kedua binding membawa bentuk pesan logis yang sama.

### Pelestarian opasitas

Sebuah prinsip desain kunci: state internal agent yang dipanggil bersifat opaque. Penelepon melihat state tugas dan artefak. Chain-of-thought milik agent yang dipanggil, panggilan tool-nya, delegasi sub-agent-nya — semuanya tak terlihat. Ini berbeda dari MCP, di mana panggilan tool bersifat transparan.

Alasannya: A2A memungkinkan pesaing berkolaborasi tanpa mengungkap internal. A2A bisa berupa "panggil agent layanan pelanggan ini" tanpa penelepon mempelajari bagaimana agent itu mengimplementasikan layanannya.

### Linimasa

- **2025-04-09.** Google mengumumkan A2A.
- **2025-06-23.** Didonasikan ke Linux Foundation.
- **2025-08.** Menyerap ACP milik IBM.
- **2025-09.** Ekstensi AP2 (Agent Payments) dirilis.
- **2026-04.** v1.0 dirilis dengan 150+ organisasi pendukung.

### Hubungan dengan MCP

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | JWT-signed Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

Pakai MCP ketika kamu ingin memanggil tool tertentu. Pakai A2A ketika kamu ingin mendelegasikan seluruh tugas ke agent lain. Banyak sistem production memakai keduanya: sebuah agent memakai MCP untuk lapisan tool-nya dan A2A untuk lapisan kolaborasinya.

```figure
a2a-task-lifecycle
```

## Gunakan

`code/main.py` mengimplementasikan harness A2A minimal: sebuah agent riset menerbitkan card-nya, sebuah agent penulis menerima `tasks/send` dengan parts yang memuat PDF dan instruksi teks, berpindah melalui working → input_required → working → completed, dan mengembalikan sebuah artefak teks. Semuanya stdlib; memakai transport in-memory untuk fokus pada bentuk pesan.

Yang perlu diperhatikan:

- Bentuk JSON Agent Card.
- Penugasan task id dan transisi state.
- Messages dengan parts bertipe campuran.
- Cabang input-required di tengah tugas.
- Pengembalian artefak saat selesai.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-a2a-agent-spec.md`. Diberi agent baru yang harus bisa dipanggil oleh agent lain, skill ini menghasilkan JSON Agent Card, skema skill, dan blueprint endpoint.

## Latihan

1. Jalankan `code/main.py`. Telusuri lifecycle Task lengkap, termasuk jeda input-required di mana agent yang dipanggil meminta klarifikasi.

2. Tambahkan sebuah Agent Card bertanda tangan. Tandatangani dengan HMAC di atas JSON kanonik card. Tulis sebuah verifier dan konfirmasikan bahwa ia gagal pada card yang bermutasi.

3. Implementasikan task streaming: agent penulis memancarkan tiga chunk artefak inkremental melalui SSE dan penelepon mengakumulasikannya.

4. Rancang sebuah agent A2A yang membungkus sebuah server MCP. Petakan setiap tool MCP ke sebuah skill A2A. Catat trade-off-nya — opasitas mana yang hilang?

5. Baca pengumuman A2A v1.0 dan identifikasi satu fitur yang belum diimplementasikan framework mana pun per April 2026. (Petunjuk: berkaitan dengan delegasi tugas multi-hop.)

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent-to-Agent protocol" | Open protocol for opaque agent collaboration |
| Agent Card | "`.well-known/agent.json`" | Published metadata describing an agent's skills and endpoint |
| Skill | "A callable unit" | A named operation the agent supports (analog to MCP tool) |
| Task | "Unit of delegation" | A work item with a lifecycle and final artifact |
| Message | "Task input" | Carries Parts (text, file, data) |
| Part | "Typed chunk" | `text` / `file` / `data` element of a message |
| Artifact | "Task output" | Named, typed output returned on completion |
| AP2 | "Agent Payments Protocol" | Signed Agent Cards extension for trust and payments |
| Opacity | "Black-box collaboration" | Called agent's internals are hidden from caller |
| Input-required | "Task pause" | Lifecycle state when the agent needs more info |

## Bacaan Lanjutan

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — spesifikasi A2A kanonik
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — implementasi referensi dan SDK
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — transfer governance Juni 2025
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — roadmap dan momentum mitra
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — catatan rilis v1.0 dan panduan backward-compat
