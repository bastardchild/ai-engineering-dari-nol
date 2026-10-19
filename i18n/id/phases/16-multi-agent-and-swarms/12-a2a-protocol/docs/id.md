# A2A — Protocol Agent-to-Agent

> Google mengumumkan A2A pada April 2025; per April 2026 spesifikasinya ada di https://a2a-protocol.org/latest/specification/ dan didukung 150+ organisasi. A2A adalah pasangan horizontal dari MCP (Pelajaran 13): kalau MCP bersifat vertikal (agent ↔ tools), maka A2A bersifat peer-to-peer (agent ↔ agent). A2A mendefinisikan Agent Card (discovery), tasks dengan artifacts (teks, data terstruktur, video), lifecycle task yang opak, dan auth. Sistem produksi semakin banyak memasangkan MCP dengan A2A. Google Cloud membawa dukungan A2A ke Vertex AI Agent Builder selama 2025-2026.

**Type:** Learn + Build
**Languages:** Python (stdlib, `http.server`, `json`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## Masalahnya

Agent-mu perlu memanggil agent lain di sistem lain. Bagaimana? Kamu bisa mengekspos sebuah endpoint HTTP, mendefinisikan skema JSON khusus, dan berharap pihak lain memahaminya. Setiap pasangan agent menjadi integrasi kustom.

A2A adalah wire protocol universal untuk panggilan itu. Discovery standar, model task standar, transport standar, artifacts standar. Mirip HTTP+REST tetapi untuk agent sebagai warga kelas satu.

## Konsepnya

### Empat elemennya

**Agent Card.** Sebuah dokumen JSON di `/.well-known/agent.json` yang menjelaskan agent: nama, skills, endpoints, modalitas yang didukung, dan persyaratan auth. Discovery terjadi dengan membaca kartu tersebut.

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task.** Satuan kerja. Sebuah objek async dan stateful dengan lifecycle: `submitted → working → completed / failed / canceled`. Klien mengirim sebuah task, lalu polling atau berlangganan untuk pembaruan.

**Artifact.** Tipe hasil yang diproduksi oleh sebuah task. Teks, JSON terstruktur, gambar, video, audio. Artifacts bertipe sehingga berbagai modalitas menjadi kelas satu.

**Lifecycle opak.** A2A tidak menetapkan *bagaimana* remote agent memecahkan task tersebut. Klien hanya melihat transisi state dan artifacts; implementasinya bebas memakai framework apa pun.

### Pemisahan MCP/A2A

- **MCP** (Pelajaran 13): agent ↔ tool. Agent membaca/menulis via JSON-RPC ke sebuah tool server. Stateless secara bawaan.
- **A2A**: agent ↔ agent. Protocol peer; kedua sisi adalah agent dengan penalaran masing-masing.

Sistem multi-agent produksi memakai keduanya. Sebuah peer A2A memanggil tools MCP di sisinya. Pemisahan itu menjaga kedua kekhawatiran tetap bersih.

### Alur discovery

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

Atau dengan streaming: berlangganan SSE ke `/tasks/{id}/events` untuk update push.

### Autentikasi

A2A mendukung tiga pola umum:

- **Bearer token** — OAuth2 atau opak.
- **mTLS** — mutual TLS; organisasi saling membuktikan identitas.
- **Signed requests** — HMAC atas payload-nya.

Auth dideklarasikan di Agent Card; klien menemukannya dan menaatinya.

### 150+ organisasi per April 2026

Adopsi enterprise yang mendorong skala A2A. Headline-nya: A2A menjadi cara sistem agent enterprise menembus batas kepercayaan. Google Cloud mengirimkan dukungan A2A di Vertex AI Agent Builder; Microsoft Agent Framework mendukungnya; sebagian besar framework utama (LangGraph, CrewAI, AutoGen) mengirimkan adapter A2A.

### Di mana A2A menang

- **Panggilan lintas organisasi.** Agent di perusahaan A memanggil agent di perusahaan B. Tanpa A2A, setiap pasangan adalah kontrak khusus.
- **Framework heterogen.** Agent LangGraph memanggil agent CrewAI memanggil agent Python kustom. A2A menormalkannya.
- **Artifacts bertipe.** Hasil video, JSON terstruktur, audio — semuanya kelas satu.
- **Task jangka panjang.** Lifecycle opak + polling membuat task berjam-jam menjadi mudah.

### Di mana A2A kesulitan

- **Micro-call yang sensitif latensi.** Lifecycle A2A bersifat async. Agent-to-agent sub-milidetik tidak cocok; gunakan RPC langsung.
- **Agent in-process yang ketat-sambungannya.** Jika kedua agent berjalan di proses Python yang sama, round-trip HTTP A2A berlebihan.
- **Tim kecil.** Overhead spesifikasi itu nyata; agent internal saja mungkin tidak memerlukan kerumitan formal itu.

### A2A vs ACP, ANP, NLIP

Beberapa spesifikasi terkait muncul pada 2024-2026:

- **ACP** (IBM/Linux Foundation) — pendahulu A2A, cakupan lebih sempit.
- **ANP** (Agent Network Protocol) — berat pada peer-discovery, decentralized-first.
- **NLIP** (Ecma Natural Language Interaction Protocol, distandardisasi Desember 2025) — tipe konten bahasa alami.

A2A adalah protocol peer yang paling banyak diadopsi per April 2026. Lihat arXiv:2505.02279 (Liu et al., "A Survey of Agent Interoperability Protocols") untuk perbandingannya.

```figure
sw-agent-card-discovery
```

## Bangun Sendiri

`code/main.py` mengimplementasikan sebuah server dan klien A2A-minimal memakai `http.server` dan JSON. Server-nya:

- mengekspos `/.well-known/agent.json`,
- menerima `POST /tasks`,
- mengelola state task,
- mengembalikan artifacts pada `GET /tasks/{id}`.

Kliennya:

- mengambil Agent Card,
- mengirim sebuah task,
- polling sampai selesai,
- membaca artifact-nya.

Jalankan:

```
python3 code/main.py
```

Skripnya menjalankan server di thread latar belakang, lalu menjalankan klien terhadapnya. Kamu melihat alur lengkapnya: discovery, submit, poll, artifact.

## Gunakan

`outputs/skill-a2a-integrator.md` merancang sebuah integrasi A2A: isi Agent Card, skema task, pilihan auth, streaming vs polling.

## Rilis

Checklist:

- **Pin versi spesifikasinya.** A2A masih berevolusi; Agent Card harus mendeklarasikan versi protocol.
- **Pembuatan task idempoten.** Pengiriman ganda (retry jaringan) harus menghasilkan satu task saja.
- **Skema artifacts.** Deklarasikan bentuk apa yang dikembalikan agent; konsumen sebaiknya memvalidasinya.
- **Rate limit + auth.** A2A menghadap publik; terapkan keamanan web standar.
- **Dead-letter untuk task yang gagal.** Periksa pola dari waktu ke waktu untuk tipe kegagalan yang berulang.

## Latihan

1. Jalankan `code/main.py`. Konfirmasi bahwa klien menemukan server dan menerima artifact yang benar.
2. Tambahkan skill kedua ke server (misalnya, "summarize"). Perbarui Agent Card. Tulis sebuah klien yang memilih skill berdasarkan tipe task.
3. Implementasikan sebuah endpoint streaming SSE: `/tasks/{id}/events` yang mengeluarkan perubahan state. Apa yang harus berbeda pada klien?
4. Baca spesifikasi A2A (https://a2a-protocol.org/latest/specification/). Identifikasi tiga hal yang diwajibkan spesifikasi tetapi tidak diimplementasikan demo ini.
5. Bandingkan A2A (discovery Agent Card) dengan MCP (listing kemampuan sisi server via `listTools`). Apa tradeoff antara agent yang mendeskripsikan dirinya sendiri dengan probing kemampuan?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent-to-agent" | Peer protocol for agents to call other agents across systems. Google 2025. |
| Agent Card | "The agent's business card" | JSON at `/.well-known/agent.json` describing skills, endpoints, auth. |
| Task | "The unit of work" | Async stateful object with a lifecycle; artifacts produced on completion. |
| Artifact | "The result" | Typed output: text, structured JSON, image, video, audio. First-class media. |
| Opaque lifecycle | "How it's solved is the agent's business" | Client sees state transitions; server is free to choose framework/tools. |
| Discovery | "Finding the agent" | `GET /.well-known/agent.json` returns the card. |
| MCP vs A2A | "Tools vs peers" | MCP: vertical agent ↔ tool. A2A: horizontal agent ↔ agent. |
| ACP / ANP / NLIP | "Sibling protocols" | Adjacent specs; A2A is the most-adopted 2026. |

## Bacaan Lanjutan

- [A2A specification](https://a2a-protocol.org/latest/specification/) — the canonical spec
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — April 2025 launch post
- [A2A GitHub repo](https://github.com/a2aproject/A2A) — reference implementations and SDKs
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — MCP, ACP, A2A, ANP comparison
