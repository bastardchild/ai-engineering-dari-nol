# Mengapa Multi-Agent?

> Satu agent menabrak tembok. Langkah cerdas bukan agent yang lebih besar — melainkan lebih banyak agent.

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Identifikasi batas single-agent (meluapnya context, keahlian yang tercampur, bottleneck sekuensial) dan jelaskan kapan membagi tugas menjadi banyak agent adalah langkah yang tepat
- Bandingkan pola orkestrasi (pipeline, fan-out paralel, supervisor, hierarchical) dan pilih yang tepat untuk struktur tugas tertentu
- Rancang sistem multi-agent dengan batas peran yang jelas, shared state, dan kontrak komunikasi
- Analisis tradeoff kompleksitas multi-agent (latensi, biaya, kesulitan debugging) terhadap kesederhanaan single-agent

## Masalahnya

Kamu membangun satu agent di Phase 14. Agent itu bekerja. Ia bisa membaca file, menjalankan perintah, memanggil API, dan menalar tentang hasilnya. Lalu kamu mengarahkannya ke codebase nyata: 200 file, tiga bahasa, test yang bergantung pada infrastruktur, dan kebutuhan untuk meneliti API eksternal sebelum menulis kode.

Agent itu tersendat. Bukan karena LLM-nya bodoh, tetapi karena tugasnya melebihi kapasitas satu agent loop. Context window terisi penuh dengan isi file. Agent lupa apa yang ia baca 40 tool call sebelumnya. Ia mencoba menjadi sekaligus researcher, coder, dan reviewer, dan ketiganya dikerjakan dengan buruk.

Inilah batas single-agent. Kamu menabraknya setiap kali sebuah tugas membutuhkan:

- **Lebih banyak konteks daripada muat dalam satu window** — membaca 50 file meledak melewati 200k token
- **Keahlian berbeda di tahap berbeda** — riset membutuhkan prompting yang berbeda dari generasi kode
- **Pekerjaan yang bisa berjalan paralel** — mengapa membaca tiga file secara sekuensial ketika kamu bisa membacanya secara bersamaan?

## Konsepnya

### Batas Single-Agent

Satu agent berarti satu loop, satu context window, satu system prompt. Bayangkan begini:

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

Tiga hal yang rusak:

1. **Saturasi context** — hasil tool menumpuk. Menjelang giliran ke-30, agent telah mengonsumsi 150k token isi file, keluaran perintah, dan penalaran sebelumnya. Detail kritis dari giliran ke-5 hilang.

2. **Kebingungan peran** — system prompt yang berkata "kamu adalah researcher, coder, reviewer, dan tester" menghasilkan agent yang setengah meneliti, setengah coding, dan tidak pernah menyelesaikan review.

3. **Bottleneck sekuensial** — agent membaca file A, lalu file B, lalu file C. Tiga panggilan LLM serial. Tiga eksekusi tool serial. Tanpa paralelisme.

### Solusi Multi-Agent

Bagi pekerjaannya. Beri setiap agent satu pekerjaan, satu context window, dan satu system prompt yang disetel untuk pekerjaan itu:

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

Setiap agent memiliki:
- System prompt yang fokus ("Kamu adalah code reviewer. Satu-satumu pekerjaan adalah menemukan bug.")
- Context window-nya sendiri (tidak tercemar oleh pekerjaan agent lain)
- Kontrak input/output yang jelas (menerima catatan riset, mengeluarkan kode)

### Sistem Nyata yang Melakukannya

**Claude Code subagents** — saat Claude Code melahirkan subagent dengan `Task`, ia membuat child agent dengan tugas yang berlingkup. Parent menjaga context-nya tetap bersih. Child melakukan pekerjaan fokus dan mengembalikan ringkasan.

**Devin** — menjalankan sebuah agent planner, sebuah agent coder, dan sebuah agent browser. Planner memecah pekerjaan menjadi langkah-langkah. Coder menulis kode. Browser meneliti dokumentasi. Masing-masing punya context terpisah.

**Tim coding multi-agent (SWE-bench)** — sistem berperforma terbaik di SWE-bench menggunakan researcher yang membaca codebase, planner yang merancang perbaikan, dan coder yang mengimplementasikannya. Sistem single-agent mendapat skor lebih rendah.

**ChatGPT Deep Research** — melahirkan banyak agent pencarian secara paralel, masing-masing mengeksplorasi sudut yang berbeda, lalu mensintesis hasilnya.

### Spektrumnya

Multi-agent bukan biner. Ia adalah spektrum:

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Single agent** — satu loop, satu prompt. Cocok untuk tugas sederhana.

**Subagents** — parent melahirkan children untuk subtugas yang fokus. Parent memelihara rencana. Children melapor kembali. Inilah yang dilakukan Claude Code.

**Pipeline** — agent berjalan berurutan. Keluaran Agent A menjadi input Agent B. Cocok untuk alur kerja bertahap: riset -> kode -> review -> test.

**Team** — agent berjalan paralel dengan shared message bus. Masing-masing punya peran. Sebuah orchestrator mengoordinasi. Cocok ketika keterampilan berbeda dibutuhkan secara bersamaan.

**Swarm** — banyak agent identik atau hampir identik dengan shared state. Tanpa orchestrator tetap. Agent mengambil pekerjaan dari antrean. Cocok untuk tugas paralel berthroughput tinggi.

### Empat Pola Multi-Agent

#### Pola 1: Pipeline

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

Setiap agent mentransformasi data dan meneruskannya. Mudah untuk direasionalisasi. Kegagalan di satu tahap memblokir sisanya.

#### Pola 2: Fan-out / Fan-in

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

Bagi pekerjaan ke beberapa agent paralel, lalu gabungkan hasilnya. Cocok untuk tugas yang terurai menjadi subtugas independen.

#### Pola 3: Orchestrator-Worker

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

Sebuah orchestrator yang cerdas memutuskan apa yang harus dilakukan, mendelegasikan ke worker, dan mensintesis hasilnya. Orchestrator itu sendiri adalah agent dengan tool untuk melahirkan worker.

#### Pola 4: Peer Swarm

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

Tanpa orchestrator sentral. Agent berkomunikasi peer-to-peer. Keputusan muncul dari interaksi. Lebih sulit untuk di-debug, tetapi menskalakan ke banyak agent.

### Kapan TIDAK Menggunakan Multi-Agent

Multi-agent menambah kompleksitas. Setiap pesan antar agent adalah titik kegagalan potensial. Debugging berubah dari "baca satu percakapan" ke "telusuri pesan di lima agent."

**Tetap single-agent ketika:**
- Tugasnya muat dalam satu context window (di bawah ~100k token data kerja)
- Kamu tidak butuh system prompt berbeda untuk tahap berbeda
- Eksekusi sekuensial sudah cukup cepat
- Tugasnya cukup sederhana sehingga memecahnya menambah lebih banyak overhead daripada nilai

**Biaya kompleksitas:**
- Setiap batas agent adalah langkah kompresi lossy: context lengkap agent A diringkas menjadi pesan untuk agent B
- Logika koordinasi (siapa mengerjakan apa, kapan, urutan apa) adalah sumber bug tersendiri
- Latensi meningkat: N agent berarti minimal N panggilan LLM serial, lebih banyak jika mereka perlu bolak-balik berbicara
- Biaya berlipat: setiap agent membakar token secara independen

Aturan praktis: jika sebuah tugas membutuhkan kurang dari 20 tool call dan muat dalam 100k token, pertahankan sebagai single-agent.

```figure
swarm-messages
```

## Bangun

### Langkah 1: Single Agent yang Kewalahan

Berikut sebuah single agent yang mencoba melakukan segalanya. Ia punya satu system prompt raksasa dan satu context window yang menampung riset, kode, dan review:

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Masalah dengan pendekatan ini:
- Context window tumbuh di setiap tahap. Menjelang langkah review, context berisi catatan riset DAN kode DAN penalaran sebelumnya.
- System prompt bersifat generik. Tidak bisa disetel untuk setiap tahap.
- Tidak ada yang berjalan paralel.

### Langkah 2: Agent Spesialis

Sekarang pecah. Setiap agent mendapat satu pekerjaan:

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

Setiap spesialis punya prompt yang fokus. Masing-masing mendapat context window bersih dengan hanya input yang ia butuhkan.

### Langkah 3: Koordinasi Melalui Pesan

Sambungkan spesialis dengan passing pesan eksplisit:

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Setiap agent hanya menerima pesan yang ditujukan kepadanya. Tanpa pencemaran context. 50k token membaca dokumentasi researcher tidak pernah masuk ke context reviewer.

### Langkah 4: Bandingkan

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

Versi multi-agent menggunakan lebih banyak total token (tiga agent, tiga panggilan LLM terpisah) tetapi context setiap agent tetap bersih. Kualitas setiap tahap membaik karena system prompt-nya terspesialisasi.

## Gunakan

Pelajaran ini menghasilkan prompt yang dapat digunakan kembali untuk memutuskan kapan harus beralih ke multi-agent. Lihat `outputs/prompt-multi-agent-decision.md`.

## Latihan

1. Tambahkan spesialis keempat: agent "tester" yang menerima kode dari coder dan umpan balik review dari reviewer, lalu menulis test
2. Modifikasi pipeline agar reviewer bisa mengirim umpan balik kembali ke coder untuk loop revisi (maksimal 2 putaran)
3. Konversi pipeline sekuensial menjadi fan-out: jalankan researcher dan agent "requirements analyzer" secara paralel, lalu gabungkan keluaran mereka sebelum diteruskan ke coder

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Swarm | "A hive mind of AI agents" | A set of peer agents with shared state and no fixed leader. Behavior emerges from local interactions. |
| Orchestrator | "The boss agent" | An agent whose tools include spawning and managing other agents. It plans and delegates but may not do the actual work. |
| Coordinator | "The traffic cop" | A non-agent component (often just code, not an LLM) that routes messages between agents based on rules. |
| Consensus | "The agents agree" | A protocol where multiple agents must reach agreement before proceeding. Used when conflicting outputs need resolution. |
| Emergent behavior | "The agents figured it out themselves" | System-level patterns that arise from agent interactions but were not explicitly programmed. Can be useful or harmful. |
| Fan-out / fan-in | "Map-reduce for agents" | Splitting a task across parallel agents (fan-out), then combining their results (fan-in). |
| Message passing | "Agents talk to each other" | The communication mechanism between agents: structured data sent from one agent to another, replacing shared context windows. |

## Bacaan Lanjutan

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - survey of multi-agent patterns
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - Microsoft's multi-agent conversation framework
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - how Claude Code delegates with Task
- [CrewAI documentation](https://docs.crewai.com/) - role-based multi-agent framework
