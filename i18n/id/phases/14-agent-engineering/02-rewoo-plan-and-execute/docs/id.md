# ReWOO dan Rencana-dan-Execute: Decoupled Perencanaan

> ReAct interleaves thought dan aksi di satu stream. ReWOO separates mereka: satu besar rencana up front, maka execute. 5x fewer token, +4% accuracy pada HotpotQA, dan kamu bisa distill planner menjadi sebuah 7B model. Rencana-dan-Execute generalized itu; Rencana-dan-Bertindak scaled itu ke web navigation.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan mengapa ReWOO's Planner / Worker / Solver split saves token dan meningkatkan robustness atas ReAct's interleaved loop.
- Implementasikan sebuah rencana DAG, sebuah dependency-ordered executor, dan sebuah solver itu composes worker outputs — semua stdlib.
- Putuskan ketika sebuah tugas sebaiknya jalankan sebagai rencana-maka-execute vs interleaved ReAct, memakai 2026 "lima workflow polas" framing (Anthropic).
- Recognize ketika Rencana-dan-Bertindak's synthetic rencana data adalah needed untuk long-horizon web atau mobile tugas.

## Masalahnya

ReAct's interleaved thought-aksi-observasi loop adalah sederhana dan flexible, but setiap pemanggilan tool memiliki ke carry penuh prior context — termasuk setiap sebelumnya thought. Token usage yang terus bertambah quadratically dengan depth. Worse: ketika sebuah tool gagal mid-loop, model memiliki ke re-derive seluruh rencana dari error observasi.

ReWOO (Xu et al., arXiv:2305.18323, Dapat 2023) noticed ini dan made sebuah bet: rencana seluruh hal up front, fetch bukti di paralel, compose jawaban pada berakhir. Satu LLM panggil ke rencana, N pemanggilan tool untuk bukti (bisa menjadi paralel), satu LLM panggil ke solve. trade adalah less flexibility ( rencana adalah static) untuk much better token efficiency dan clearer mode kegagalan.

## Konsepnya

###  tiga roles

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner produces sebuah DAG. Setiap node names sebuah tool, miliknya arguments, dan yang earlier node itu depends pada (references like `#E1`, `#E2`). Workers execute node di topological order. Solver stitches everything together.

### Mengapa 5x fewer token

ReAct yang terus bertambah prompt length linearly dengan langkah count. Pada langkah 10, prompt berisi thought 1 plus aksi 1 plus observasi 1 plus thought 2 plus aksi 2 plus observasi 2, dan so pada. Setiap intermediate langkah juga redundantly mencakup original prompt.

ReWOO pays satu planner prompt (besar), N kecil worker prompts (setiap just pemanggilan tool, tidak chain), dan satu solver prompt. Pada HotpotQA paper measures ~5x fewer token sementara scoring +4 absolute accuracy.

### Mengapa itu adalah more robust

Jika worker 3 gagal di ReAct, loop memiliki ke reason out dari error mid-stream. Di ReWOO, worker 3 mengembalikan sebuah error string; solver sees itu di context dengan original rencana dan bisa degrade gracefully. Kegagalan localization adalah per-node, tidak per-langkah.

### Planner distillation

 paper's kedua hasil: karena planner melakukan tidak see observasi, kamu bisa fine-tune sebuah 7B model pada planner outputs dari sebuah 175B teacher. kecil model menangani perencanaan; besar model adalah tidak needed pada inference. Ini adalah now standard — banyak 2026 produksi agen pakai sebuah kecil planner dan sebuah besar executor atau vice-versa.

### Rencana-dan-Execute (2023)

 LangChain tim's August 2023 post generalized ReWOO menjadi sebuah pola name: Rencana-dan-Execute. Up-front planner meng-emit sebuah langkah list, executor berjalan setiap langkah, sebuah optional replanner bisa revise setelah observing hasil. Ini adalah closer ke ReAct than ReWOO ( replanner brings observasi back menjadi perencanaan) but preserves token savings.

### Rencana-dan-Bertindak (Erdogan et al., arXiv:2503.09572, ICML 2025)

Rencana-dan-Bertindak scales pola ke long-horizon web dan mobile agen. kunci contribution adalah synthetic rencana data: sebuah labeled trajectory generator produces training data di mana rencana adalah eksplisit. Used ke fine-tune planner model itu keep working past 30–50 langkah pada WebArena-like tugas di mana sebuah tunggal ReAct trajectory loses coherence.

### Ketika ke pick yang

| Pattern | When |
|---------|------|
| ReAct | Short tasks, unknown environment, need reactive exception handling |
| ReWOO | Structured tasks with known tools, token-sensitive, parallelizable evidence |
| Plan-and-Execute | Like ReWOO but with replanning after partial execution |
| Plan-and-Act | Long-horizon (>30 steps), web/mobile/computer-use |
| Tree of Thoughts | Search is worth paying for (Lesson 04) |

Anthropic's Dec 2024 guidance: mulai dengan simplest. Jika tugas adalah satu pemanggilan tool plus sebuah summary, lakukan tidak build ReWOO. Jika tugas adalah sebuah 40-langkah riset assignment, lakukan tidak lakukan ReAct alone.

```figure
rewoo-plan
```

## Bangun Sendiri

`code/main.py` implements sebuah toy ReWOO:

- `Planner` — sebuah scripted kebijakan itu meng-emit sebuah rencana DAG dari sebuah prompt.
- `Worker` — dispatches setiap node's pemanggilan tool via registry.
- `Solver` — scripted composition itu membaca bukti dan produces sebuah final jawaban.
- Dependency resolution — references like `#E1` adalah substituted dengan earlier worker outputs.

 demo menjawab "Apa adalah population dari capital dari France, rounded ke millions?" memakai sebuah dua-langkah rencana: (1) look up capital, (2) look up population, maka solve.

Jalankan:

```
python3 code/main.py
```

 trace menunjukkan penuh rencana pertama, maka worker hasil, maka solver composition. Bandingkan token count (kita print sebuah rough character count) ke sebuah ReAct-style interleaved jalankan — ReWOO wins pada ini kind dari structured tugas.

## Gunakan

LangGraph ships Rencana-dan-Execute sebagai sebuah resep (`create_react_agent` untuk ReAct, custom graphs untuk rencana-execute). CrewAI's Flow encode pola directly: kamu definisikan tugas up front dan Flow DAG executes mereka. Rencana-dan-Bertindak's synthetic data pendekatan adalah tetap mostly riset; runtime pola (eksplisit rencana DAG) ships di produksi melalui LangGraph dan CrewAI Flow.

## Rilis

`outputs/skill-rewoo-planner.md` menghasilkan sebuah ReWOO rencana DAG dari sebuah user request, given sebuah tool catalog. Itu validates rencana (acyclic, setiap reference resolved, setiap tool exists) sebelum handing off ke sebuah executor.

## Latihan

1. Parallelize worker execution untuk independent rencana node. Apa melakukan itu buy kamu pada sebuah 6-node DAG dengan 2 paralel groups?
2. Tambahkan sebuah replanner node itu fires jika apa pun worker mengembalikan sebuah error. Apa adalah terkecil change ke ReWOO itu membuat itu Rencana-dan-Execute?
3. Ganti `Planner` dengan sebuah kecil model (7B class) dan keep `Solver` pada sebuah frontier model. Bandingkan ujung-ke-ujung quality — di mana melakukan split gagal?
4. Baca Bagian 4 dari ReWOO paper pada planner distillation. Reproduce 175B -> 7B hasil conceptually: apa training data lakukan kamu butuhkan, dan bagaimana lakukan kamu score rencana quality?
5. Port toy ke Rencana-dan-Bertindak's trajectory bentuk: rencana adalah sebuah sequence, tidak sebuah DAG. Apa tradeoffs change?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| ReWOO | "Reasoning without observations" | Plan, then fetch evidence in parallel, then solve — no observations in the planning prompt |
| Plan-and-Execute | "LangChain's plan-execute pattern" | ReWOO with an optional replanner node after execution |
| Plan-and-Act | "Scaled plan-execute" | Explicit planner/executor split with synthetic plan training data for long-horizon tasks |
| Evidence reference | "#E1, #E2, ..." | Plan-node placeholder substituted with prior worker output at dispatch time |
| Planner distillation | "Small planner, big executor" | Fine-tune a small model on planner traces from a large teacher |
| Token efficiency | "Fewer round trips" | 5x fewer tokens on HotpotQA vs ReAct in the paper |
| DAG executor | "Topological dispatcher" | Runs plan nodes in dependency order; parallel at each level |

## Bacaan Lanjutan

- [Xu et al., ReWOO: Decoupling Penalaran dari Observasi (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — paper kanonikal
- [Erdogan et al., Rencana-dan-Bertindak (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — scaled planner-executor dengan synthetic rencana
- [LangGraph Rencana-dan-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) — framework resep
- [Anthropic, Building Efektif Agen](https://www.anthropic.com/research/building-effective-agents) — pick simplest pola itu bekerja
