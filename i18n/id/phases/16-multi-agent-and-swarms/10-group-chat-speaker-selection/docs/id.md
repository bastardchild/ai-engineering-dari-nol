# Group Chat dan Pemilihan Pembicara

> Orkestrasi percakaman-bersama menempatkan N agent dalam satu percakapan; sebuah fungsi selector (LLM, round-robin, atau kustom) memilih siapa yang berbicara berikutnya. Ini adalah arketip percakapan multi-agent yang muncul — agent tidak mengetahui peran mereka dalam graph statis, mereka hanya bereaksi terhadap pool bersama. AutoGen GroupChat dan AG2 GroupChat adalah implementasi rujukan: semantik GroupChat AutoGen v0.2 dipertahankan di fork AG2; AutoGen v0.4 menulis ulangnya sebagai model actor berbasis event. Microsoft menempatkan AutoGen dalam mode maintenance pada Februari 2026 dan menggabungkannya dengan Semantic Kernel menjadi Microsoft Agent Framework (RC Februari 2026). Primitif GroupChat bertahan di AG2 dan Microsoft Agent Framework — pelajari sekali, gunakan di mana pun.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Masalahnya

Graph statis (LangGraph) hebat ketika alur kerjanya diketahui. Percakaman nyata tidak statis: kadang coder bertanya kepada reviewer, kadang researcher, kadang writer. Hardcode setiap handoff yang mungkin menghasilkan ledakan edge. Kamu ingin *agent yang bereaksi terhadap pool bersama*, dengan beberapa fungsi yang memutuskan siapa yang berbicara berikutnya.

Itulah persis yang dilakukan AutoGen GroupChat.

## Konsepnya

### Bentuknya

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

Setiap agent melihat setiap pesan. Sebuah fungsi selector dipanggil di setiap giliran untuk memilih siapa yang berbicara berikutnya.

### Tiga varian selector

**Round-robin.** Siklus tetap. Deterministik. Menskalakan linear dalam N tetapi mengabaikan konteks — seorang coder mendapat giliran bahkan ketika topiknya tinjauan legal.

**Dipilih LLM.** Sebuah panggilan ke LLM yang membaca pool terbaru dan mengembalikan pembicara berikutnya terbaik. Sadar-konteks tetapi lambat: setiap giliran menambahkan satu panggilan LLM. Default AutoGen.

**Kustom.** Sebuah fungsi Python dengan logika apa pun yang kamu inginkan. Umumnya: dipilih LLM dengan aturan fallback (misalnya, "selalu berikan verifier giliran setelah coder").

### API ConversableAgent

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` memegang selector-nya. Ketika sebuah agent menyelesaikan gilirannya, manager memanggil selector, yang mengembalikan agent berikutnya. Loop berlanjut sampai kondisi terminasi.

### Terminasi

Tiga pola umum:

- **Putaran maksimum.** Batas keras pada total giliran.
- **Token "TERMINATE".** Agent bisa mengirimkan pesan sentinel; manager berhenti ketika sebuah muncul.
- **Pemeriksaan tujuan-tercapai.** Sebuah verifier ringan berjalan setiap giliran dan menghentikan chat ketika selesai.

### Garis keturunan: fork dan merger

Pada awal 2025, Microsoft memulai penulisan ulang besar AutoGen (v0.4) di sekitar model actor berbasis event. Komunitas mem-fork semantik GroupChat AutoGen v0.2 sebagai AG2, mempertahankan API yang telah diintegrasikan oleh early adopter.

Pada Februari 2026, Microsoft mengumumkan AutoGen akan masuk mode maintenance, dengan model actor berbasis event bergabung ke dalam **Microsoft Agent Framework** (RC Februari 2026, sekarang digabung dengan Semantic Kernel). Konsep GroupChat bertahan di kedua jalur; detail implementasinya berbeda. AG2 adalah upstream yang disukai untuk kode yang kompatibel v0.2.

### Kapan GroupChat cocok

- **Percakaman yang muncul.** Kamu tidak ingin menyambungkan sebelumnya setiap next-speaker yang mungkin.
- **Tugas pencampuran peran.** Coder bertanya kepada researcher, researcher bertanya kepada arsiparis, arsiparis bertanya kembali kepada coder. Alurnya bukan DAG.
- **Pemecahan masalah eksploratif.** Bayangkan "rapat brainstorming," bukan "lini perakitan."

### Kapan pola ini gagal

- **Determinisme ketat.** Selector LLM bisa tidak konsisten. Prompt yang sama, run berbeda, pembicara berikutnya berbeda.
- **Kaskade sycophancy.** Agent tunduk pada siapa pun yang berbicara dengan paling percaya diri. Counter-prompt secara eksplisit.
- **Pembengkakan context.** Setiap agent membaca setiap pesan; setelah 10 giliran context-nya besar. Gunakan projection (Lesson 15) untuk membatasi tampilan.
- **Pembicara panas.** Satu agent mendominasi percakapan karena selector menyukai keahliannya. Perkenalkan keseimbangan pembicara sebagai fitur selector.

### Group chat vs supervisor

Primitif yang sama, default berbeda:

- Supervisor: satu agent merencanakan dan yang lain mengeksekusi. Selector-nya adalah "tanyakan kepada planner apa yang harus dilakukan."
- Group chat: semua agent adalah peer; selector adalah fungsi di atas pool bersama.

Keduanya menggunakan keempat primitif dari Lesson 04. Group chat default ke orkestrasi yang dipilih LLM dan shared state full-pool.

```figure
swarm-speaker
```

## Bangun

`code/main.py` mengimplementasikan GroupChat dari nol di stdlib. Tiga agent (coder, reviewer, manager), varian round-robin dan dipilih-LLM, dan terminasi pada token `TERMINATE`.

Demo mencetak transkrip percakaman ditambah jejak keputusan selector untuk kedua varian.

Jalankan:

```
python3 code/main.py
```

## Gunakan

`outputs/skill-groupchat-selector.md` mengonfigurasi sebuah GroupChat selector untuk tugas tertentu — round-robin vs dipilih-LLM vs kustom, dan input selector apa (pesan terbaru, keahlian agent, jumlah giliran) yang digunakan.

## Rilis

Checklist:

- **Cap putaran maksimum.** Selalu. 10-20 untuk tugas tipikal.
- **Metrik keseimbangan pembicara.** Lacak giliran per agent; peringatkan ketika ketidakseimbangan melebihi ambang.
- **Token terminasi.** `TERMINATE` atau agent verifier khusus.
- **Projection atau memori berlingkup.** Setelah ~10 pesan, pertimbangkan memberi setiap agent hanya tampilan berlingkup untuk mencegah pembengkakan context.
- **Logging selector.** Untuk varian dipilih-LLM, log baik input selector maupun pilihannya. Selain itu debugging mustahil.

## Latihan

1. Jalankan `code/main.py`. Bandingkan percakaman di bawah round-robin vs dipilih-LLM. Agent mana yang mendominasi di masing-masing?
2. Tambahkan aturan "max-speaks-per-agent" di selector. Bagaimana pengaruhnya terhadap transkrip?
3. Implementasikan terminasi tujuan-tercapai: berhenti ketika reviewer mengembalikan "approved." Seberapa sering ia terpicu sebelum cap putaran?
4. Baca dokumentasi stabil AutoGen tentang GroupChat (https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html). Identifikasi default selector yang digunakan `GroupChatManager`.
5. Baca repo AG2 (https://github.com/ag2ai/ag2) dan bandingkan GroupChat v0.2-nya dengan versi v0.4 berbasis event. Properti konkret apa (throughput, fault-tolerance, composability) yang ditambahkan v0.4?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| GroupChat | "Agents in one chat room" | Shared message pool + selector function. AutoGen / AG2 primitive. |
| Speaker selection | "Who talks next" | The function that picks the next agent. Round-robin, LLM-selected, or custom. |
| GroupChatManager | "The meeting host" | AutoGen component that owns the selector and loops over turns. |
| ConversableAgent | "The base agent" | AutoGen base class; an agent that can send and receive messages. |
| Termination token | "The 'stop' word" | Sentinel string (usually `TERMINATE`) that ends the chat. |
| Hot speaker | "One agent dominates" | Failure mode where the selector keeps picking the same agent. |
| Context bloat | "Pool grows unbounded" | Each agent reads every prior message; context grows with turns. |
| Projection | "Scoped view" | Role-specific view into the shared pool to prevent context bloat. |

## Bacaan Lanjutan

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — the reference implementation
- [AG2 repo](https://github.com/ag2ai/ag2) — community AutoGen v0.2 continuation
- [Microsoft Agent Framework docs](https://learn.microsoft.com/en-us/agent-framework/) — the merged successor, RC February 2026
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) — event-driven actor model rewrite details
