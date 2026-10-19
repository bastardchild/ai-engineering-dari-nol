# Handoff dan Routines — Orchestration Stateless

> Swarm dari OpenAI (Oktober 2024) memadatkan multi-agent orchestration menjadi dua primitif: **routines** (instruksi + tools sebagai system prompt) dan **handoffs** (sebuah tool yang mengembalikan Agent lain). Tanpa state machine, tanpa DSL percabangan — LLM melakukan routing dengan memanggil tool handoff yang tepat. OpenAI Agents SDK (Maret 2025) adalah penerus produksinya. Swarm sendiri tetap menjadi referensi konseptual yang paling bersih — seluruh sumber kodenya muat dalam beberapa ratus baris. Polanya viral karena permukaan API-nya kira-kira "agent = prompt + tools; handoff = fungsi yang mengembalikan agent." Keterbatasan: stateless, jadi memori menjadi masalah pemanggil.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Masalahnya

Setiap framework multi-agent ingin kamu mempelajari DSL-nya: node dan edge LangGraph, crews dan tasks CrewAI, GroupChat dan manager AutoGen. DSL-nya adalah abstraksi nyata, tapi membuat segalanya terasa lebih berat dari yang seharusnya.

Swarm mendorong ke arah sebaliknya: gunakan kemampuan tool-calling yang sudah dimiliki model. Handoff menjadi tool call. Orchestrator adalah agent yang sedang memegang percakapan. State machine tersirat di dalam system prompt para agent.

## Konsepnya

### Dua primitif

**Routine.** System prompt yang mendefinisikan peran agent dan tools yang tersedia. Bayangkan seperti sekumpulan instruksi berlingkup: "kamu adalah agent triage; jika pengguna menanyakan refund, serahkan ke agent refund."

**Handoff.** Tool yang bisa dipanggil agent dan mengembalikan objek Agent baru. Runtime Swarm mendeteksi nilai kembalian Agent tersebut dan mengganti agent aktif untuk giliran berikutnya.

Itulah seluruh abstraksinya.

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

System prompt agent triage membuatnya memilih handoff yang tepat berdasarkan pesan pengguna. Tool-calling milik LLM-lah yang melakukan routing.

### Mengapa pola ini viral

- **API kecil.** Dua konsep untuk dipelajari.
- **Memakai apa yang sudah dilakukan model.** Tool calling sudah berkelas produksi di semua provider.
- **Beban state-machine nol.** Kamu tidak menggambarkan graf; prompt para agent yang menjelaskan ke mana mereka menyerahkan.

### Pertukaran stateless-nya

Swarm secara eksplisit stateless antar-run. Framework menyimpan riwayat pesan selama satu run, tetapi tidak menyimpan apa pun secara persisten. Memori, kontinuitas, tugas jangka panjang — semuanya menjadi masalah pemanggil.

Di produksi (OpenAI Agents SDK, Maret 2025) ini adalah salah satu hal utama yang berubah: SDK menambahkan manajemen sesi bawaan, guardrails, dan tracing sambil mempertahankan primitif handoff.

### Kapan Swarm/handoff cocok

- **Pola triage.** Agent garis depan mengarahkan pengguna ke spesialis.
- **Handoff berbasis skill.** "Jika tugasnya butuh kode, panggil coder; jika butuh riset, panggil researcher."
- **Percakapan pendek dan berbatas.** Dukungan pelanggan, FAQ-ke-tiket, alur kerja sederhana.

### Kapan Swarm kesulitan

- **Sesi panjang dengan memori bersama.** Handoff menyetel ulang status percakapan ke prompt agent baru ditambah riwayat. Tidak ada status yang persisten antar agent tanpa memori yang dikelola pemanggil.
- **Eksekusi paralel.** Handoff bersifat satu-satu — agent aktif berganti. Paralelisme menuntut pemanggil mengorkestrasi beberapa run Swarm.
- **Audit dan replay.** Run stateless sulit direplay secara persis; pilihan handoff LLM bukan deterministik.

### OpenAI Agents SDK (Maret 2025)

Penerus produksinya menambahkan:

- **Status sesi.** Thread persisten antar-run.
- **Guardrails.** Hook validasi input/output.
- **Tracing.** Setiap tool call dan handoff dicatat.
- **Handoff filter.** Mengontrol konteks apa yang berpindah saat handoff.

Primitif handoff-nya bertahan; ergonomi produksi ditambahkan di sekitarnya.

### Swarm vs GroupChat

Keduanya memakai routing berbasis LLM, tetapi berbeda pada **siapa yang memilih berikutnya**:

- GroupChat: sebuah selector (fungsi atau LLM) memilih pembicara berikutnya dari luar.
- Swarm: agent saat ini memilih penerusnya dengan memanggil tool handoff.

Swarm adalah "agent memutuskan apa berikutnya"; GroupChat adalah "manager memutuskan apa berikutnya." Keputusan Swarm hidup di tool call agent aktif; keputusan GroupChat hidup di `GroupChatManager`.

```figure
sw-handoff-routing
```

## Bangun Sendiri

`code/main.py` mengimplementasikan Swarm dari nol: sebuah dataclass Agent, sebuah mekanisme handoff (tool mengembalikan Agent), dan sebuah run loop yang mendeteksi pergantian agent.

Demo: sebuah agent triage mengarahkan ke spesialis refund, sales, atau support. Setiap spesialis punya tools sendiri. Run loop mencetak setiap handoff.

Jalankan:

```
python3 code/main.py
```

## Gunakan

`outputs/skill-handoff-designer.md` merancang sebuah topologi handoff untuk tugas tertentu: agent mana saja yang ada, handoff mana saja yang bisa mereka panggil, dan konteks apa yang berpindah.

## Rilis

Checklist:

- **Logging handoff.** Setiap handoff menulis sebuah event trace berisi from-agent, to-agent, dan snapshot konteks.
- **Aturan transfer konteks.** Tentukan apa yang berpindah saat handoff: riwayat penuh (mahal), N pesan terakhir, atau ringkasan.
- **Guardrail pada handoff.** Sebuah handoff ke spesialis dengan izin tools yang berbeda harus diautentikasi — jika tidak, prompt injection bisa memaksa handoff yang tidak diinginkan.
- **Deteksi loop.** Dua agent yang saling berbalik adalah kegagalan yang umum; deteksinya dengan pemeriksaan ring sederhana last-K.
- **Agent fallback.** Jika target handoff tidak ada, jatuhkan ke default yang aman.

## Latihan

1. Jalankan `code/main.py`, triage ke agent refund. Konfirmasi bahwa agent aktif pada giliran kedua adalah refund.
2. Tambahkan sebuah aturan deteksi loop: jika dua agent yang sama sudah handoff 3 kali berturut-turut, paksa keluar. Rancang fallback-nya.
3. Baca dokumentasi OpenAI Agents SDK tentang handoff filter. Implementasikan versi "ringkas-saat-handoff": agent yang keluar memadatkan konteks menjadi ringkasan poin sebelum agent yang masuk mengambil alih.
4. Bandingkan handoff Swarm dengan selector GroupChatManager. Pola mana yang membuat prompt injection lebih buruk, dan mengapa?
5. Baca cookbook Swarm (https://developers.openai.com/cookbook/examples/orchestrating_agents). Identifikasi satu keputusan desain eksplisit yang dibuat Swarm dan yang kemudian diubah atau dipertahankan oleh OpenAI Agents SDK.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Routine | "The agent prompt" | System prompt + tool list. Defines role and available handoffs. |
| Handoff | "Transfer to another agent" | A tool the active agent can call that returns a new Agent. The runtime switches active agent. |
| Stateless | "No memory between runs" | Swarm does not persist anything; memory is the caller's responsibility. |
| Active agent | "Who's speaking now" | The agent currently holding the conversation. Handoff changes this. |
| Context transfer | "What moves on handoff" | Policy for what history the incoming agent sees: full, last N, or summarized. |
| Handoff loop | "Agents ping-pong" | Failure mode where two agents keep handing back to each other. |
| OpenAI Agents SDK | "Production Swarm" | March 2025 successor; adds sessions, guardrails, tracing on top of the handoff primitive. |
| Handoff filter | "Gate on transfer" | SDK feature to inspect and modify context at the handoff boundary. |

## Bacaan Lanjutan

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — the reference articulation
- [OpenAI Swarm repo](https://github.com/openai/swarm) — original implementation, kept as conceptual reference
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — production successor with sessions and tracing
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — how Claude Code subagents use a handoff-like pattern via `Task`
