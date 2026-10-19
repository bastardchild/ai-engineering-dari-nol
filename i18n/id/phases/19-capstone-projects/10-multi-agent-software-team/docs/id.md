# Capstone 10 — Tim Rekayasa Perangkat Lunak Multi-Agen

> Bentuk tim engineering multi-agen pada 2026 telah konvergen: seorang architect menyusun rencana, N coder bekerja paralel di worktree, seorang reviewer melakukan gating, seorang tester memverifikasi. Arsitektur "factory" milik SWE-AF, role-based prompting MetaGPT, typed actor graph AutoGen 0.4, Devin dari Cognition, dan Droids dari Factory semuanya tiba di bentuk yang sama secara independen. Worktree paralel mengubah wall-clock menjadi throughput. State bersama dan protokol handoff menjadi permukaan kegagalannya. Capstone-nya adalah membangun tim itu, mengevaluasinya di SWE-bench Pro, dan melaporkan handoff mana yang rusak dan seberapa sering.

**Type:** Capstone
**Languages:** Python / TypeScript (agents), Shell (worktree scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 17 (infrastructure)
**Phases exercised:** P11 · P13 · P14 · P15 · P16 · P17
**Time:** 40 hours

## Masalahnya

Coding harness single-agent mencapai batas pada tugas besar. Bukan karena setiap agen individual lemah, melainkan karena context 200k token tidak bisa menampung rencana arsitektur plus empat irisan codebase paralel plus komentar reviewer plus output tes. Pabrik multi-agent memecah masalah itu: seorang architect memiliki rencana, coder memiliki implementasi di worktree paralel, seorang reviewer melakukan gating, seorang tester memverifikasi. Arsitektur "factory" milik SWE-AF, peran MetaGPT, typed actor graph AutoGen — ketiganya menggambarkan bentuk yang sama.

Permukaan kegagalannya adalah handoff. Architect merencanakan sesuatu yang tidak bisa diimplementasikan para coder. Coder menghasilkan diff yang saling bertentangan. Reviewer menyetujui perbaikan yang berhalusinasi. Tester balapan dengan coder yang masih menulis. Kamu akan membangun salah satu tim itu, menjalankannya di 50 issue SWE-bench Pro, melacak setiap handoff, dan mempublikasikan post-mortem-nya.

## Konsepnya

Peran adalah agen yang bertipe. **Architect** (Claude Opus 4.7) membaca issue, menulis rencana, dan memecahnya menjadi subtask dengan antarmuka yang eksplisit. **Coders** (Claude Sonnet 4.7, N instance paralel, masing-masing di `git worktree` + sandbox Daytona) mengimplementasikan subtask secara independen. **Reviewer** (GPT-5.4) membaca diff hasil merge lalu menyetujui atau meminta perubahan spesifik. **Tester** (Gemini 2.5 Pro) menjalankan suite tes secara terisolasi dan melaporkan pass/fail beserta artefaknya.

Komunikasi terjadi lewat papan tugas bersama (berbasis file atau Redis). Setiap peran mengonsumsi tugas yang diizinkan untuk ditanganinya. Handoff adalah pesan bertipe protokol A2A. Concern koordinasi: resolusi merge conflict (peran koordinator atau three-way merge otomatis), sinkronisasi state bersama (rencana dibekukan begitu coder mulai; replan adalah event terpisah), dan penjagaan gerbang oleh reviewer (reviewer tidak boleh menyetujui perubuhannya sendiri atau perubahan yang ia usulkan).

Amplifikasi token adalah biaya tersembunyi. Setiap batas peran menambah prompt ringkasan dan konteks handoff. Satu run single-agent 40 putaran menjadi 160 putaran total di empat peran. Rubriknya secara khusus menimbang efisiensi token terhadap baseline single-agent karena pertanyaannya bukan "apakah multi-agent bekerja" tetapi "apakah ia menang per dolar".

## Arsitektur

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## Stack

- Orkestrasi: LangGraph dengan state bersama + sub-graph per agen
- Messaging: protokol A2A (Google 2025) untuk pesan antar-agen yang bertipe
- Model: Opus 4.7 (architect), Sonnet 4.7 (coder), GPT-5.4 (reviewer), Gemini 2.5 Pro (tester)
- Isolasi worktree: `git worktree add` per coder + sandbox Daytona
- Merge coordinator: three-way merge kustom + resolusi konflik yang dimediasi LLM
- Eval: SWE-bench Pro (50 issue), skenario SWE-AF, HumanEval++ untuk unit test
- Observability: Langfuse dengan span bertag peran, akuntansi token per agen
- Deployment: K8s dengan setiap peran sebagai Deployment terpisah + HPA berdasarkan backlog

```figure
ce-team-handoff
```

## Bangun Sendiri

1. **Papan tugas.** JSONL berbasis file dengan pesan bertipe: `plan_request`, `subtask`, `diff_ready`, `review_needed`, `test_needed`, `approved`, `rejected`, `replan_needed`. Agen berlangganan ke tag-tag itu.

2. **Architect.** Membaca GitHub issue, menjalankan Opus 4.7 dengan template rencana yang mensyaratkan antarmuka subtask eksplisit (berkas yang disentuh, fungsi publik, dampak pada tes). Mengeluarkan satu `plan_request` berisi DAG subtask.

3. **Coders.** N worker paralel, masing-masing mengklaim satu subtask dari papan. Setiap worker memunculkan branch `git worktree add` yang baru plus sandbox Daytona. Mengimplementasikan subtask itu. Mengeluarkan `diff_ready` berisi patch + delta tes.

4. **Merge coordinator.** Begitu semua coder selesai, melakukan three-way merge atas N branch ke sebuah branch staging. Resolusi konflik yang dimediasi LLM hanya ketika ada tumpang tindih di tingkat berkas.

5. **Reviewer.** GPT-5.4 membaca diff hasil merge. Tidak boleh menyetujui diff yang ia tulis sendiri. Mengeluarkan `approved` (no-op) atau `review_feedback` berisi permintaan perubahan spesifik yang diarahkan kembali ke coder terkait.

6. **Tester.** Gemini 2.5 Pro menjalankan suite tes di sandbox yang bersih. Menangkap artefak. Mengeluarkan `test_passed` atau `test_failed` beserta stacktrace. Tes yang gagal berputar kembali ke coder yang memiliki subtask gagal.

7. **Pembukuan handoff.** Setiap pesan yang melintasi batas peran mendapat span di Langfuse berisi ukuran payload dan model yang dipakai. Hitung amplifikasi token per subtask (coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens).

8. **Evaluasi.** Jalankan pada 50 issue SWE-bench Pro. Bandingkan pass@1 dan $-per-issue-terpecahkan terhadap baseline single-agent (satu Sonnet 4.7 di satu worktree).

9. **Post-mortem.** Untuk setiap issue yang gagal, identifikasi handoff yang rusak (rencana terlalu samar, merge conflict, reviewer false-approve, tester flake). Hasilkan histogram kegagalan handoff.

## Gunakan

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## Rilis

`outputs/skill-multi-agent-team.md` adalah deliverable-nya. Diberi URL issue dan tingkat paralelisme, tim menghasilkan PR yang siap-merge beserta akuntansi token per peran.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | Matched 50-issue subset, pass@1 |
| 20 | Parallel speedup | Wall-clock vs single-agent baseline |
| 20 | Review quality | False-approval rate on injected-bug probe |
| 20 | Token efficiency | Total tokens per solved issue vs single-agent |
| 15 | Coordination engineering | Merge-conflict resolution, handoff-failure histogram |
| **100** | | |

## Latihan

1. Suntikkan bug yang jelas ke dalam diff di tengah run (tambahan `return None` sebelum badan utama). Ukur laju false-approve reviewer. Tuning prompt reviewer sampai false-approval di bawah 5%.

2. Kurangi menjadi dua coder (architect + coder + reviewer + tester, coder menjalankan dua subtask secara berurutan). Bandingkan wall-clock dan pass rate-nya.

3. Ganti merge coordinator dengan batasan single-writer (subtask menyentuh kumpulan berkas yang saling lepas). Ukur beban perencanaan pada architect.

4. Tukar reviewer dari GPT-5.4 ke Claude Opus 4.7. Ukur laju false-approval dan delta biaya token.

5. Tambahkan peran kelima: documenter (Haiku 4.5). Setelah review, ia menghasilkan entri changelog. Ukur apakah kualitas dokumentasi sepadan dengan belanja token tambahan.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Parallel worktree | "Isolated branch" | `git worktree add` producing a fresh working tree per coder |
| Task board | "Shared message bus" | File or Redis store of typed messages agents subscribe to |
| Handoff | "Role boundary" | Any message crossing from one role's context to another's |
| Token amplification | "Multi-agent overhead" | Total tokens across roles / single-agent tokens for the same task |
| A2A protocol | "Agent-to-agent" | Google's 2025 spec for typed inter-agent messages |
| Merge coordinator | "Integrator" | Component that runs three-way merge and mediates conflicts |
| False approval | "Reviewer hallucination" | Reviewer approves a diff with known bugs |

## Bacaan Lanjutan

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — the reference 2026 multi-agent factory
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — role-based multi-agent framework
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft's typed actor framework
- [Cognition AI (Devin)](https://cognition.ai) — reference product
- [Factory Droids](https://www.factory.ai) — alternate reference product
- [Google A2A protocol](https://a2a-protocol.org/latest/) — inter-agent messaging spec
- [git worktree documentation](https://git-scm.com/docs/git-worktree) — the isolation substrate
- [SWE-bench Pro](https://www.swebench.com) — the evaluation target
