# Capstone 01 — Agen Koding Terminal-Native

> Pada 2026 bentuk sebuah coding agent sudah mapan. Sebuah harness TUI, sebuah plan ber-state, permukaan tool yang di-sandbox, dan sebuah loop yang merencanakan, bertindak, mengamati, lalu memulihkan. Claude Code, Cursor 3, dan OpenCode semuanya terlihat sama dari jarak 50 kaki. Capstone ini meminta kamu membangun satu dari ujung ke ujung — masuk lewat CLI, keluar sebagai pull request — lalu mengukurnya terhadap mini-swe-agent dan Live-SWE-agent di SWE-bench Pro. Kamu akan belajar mengapa bagian sulitnya bukan panggilan model, melainkan tool loop, sandbox, dan plafon biaya pada run 50 giliran.

**Type:** Capstone
**Languages:** TypeScript / Bun (harness), Python (eval scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and protocols), Phase 14 (agents), Phase 15 (autonomous systems), Phase 17 (infrastructure)
**Phases exercised:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Time:** 35 hours

## Masalahnya

Coding agent menjadi kategori aplikasi AI yang dominan pada 2026. Claude Code (Anthropic), Cursor 3 dengan Composer 2 dan Agent Tabs (Cursor), Amp (Sourcegraph), OpenCode (112k stars), Factory Droids, dan Google Jules semuanya merilis variasi arsitektur yang sama: sebuah harness terminal, permukaan tool ber-permission, sebuah sandbox, dan loop plan-act-observe yang dibangun di sekitar model frontier. Garis depannya sempit — Live-SWE-agent mencapai 79.2% di SWE-bench Verified dengan Opus 4.5 — tapi kerajinan engineering-nya luas. Kebanyakan mode kegagalan bukan kesalahan model. Itu adalah ketidakstabilan tool loop, context poisoning, biaya token yang meledak, dan operasi filesystem yang merusak.

Kamu tidak bisa menalar agen-agen ini dari luar. Kamu harus membangun satu, melihat loop crash pada giliran ke-47 saat ripgrep mengembalikan 8MB hasil, lalu membangun ulang lapisan truncation-nya. Itulah poin capstone ini.

## Konsepnya

Harness punya empat permukaan. **Plan** memelihara sebuah objek state bergaya TodoWrite yang ditulis ulang model setiap giliran. **Act** mengirimkan panggilan tool (read, edit, run, search, git). **Observe** menangkap stdout / stderr / exit codes, memotongnya, dan memberi ringkasan kembali. **Recover** menangani error tool tanpa meledakkan context window atau berputar selamanya. Bentuk 2026 menambah satu hal lagi: **hooks**. `PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `Notification`, `Stop`, dan `PreCompact` — titik ekstensi yang bisa dikonfigurasi tempat operator menyuntikkan policy, telemetry, dan guardrails.

Sandbox-nya E2B atau Daytona. Setiap task berjalan di devcontainer baru dengan git worktree yang di-mount read-write. Harness tidak pernah menyentuh filesystem host. Worktree dibongkar pada keberhasilan maupun kegagalan. Kontrol biaya dijalankan pada tiga lapis: plafon token per giliran, anggaran dolar per sesi, dan batas giliran keras (biasanya 50). Lapisan observability berupa span OpenTelemetry dengan semantic convention GenAI, dikirim ke Langfuse yang self-hosted.

## Arsitektur

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## Stack

- Runtime harness: Bun 1.2 + Ink 5 (React-in-terminal)
- Akses model: OpenRouter unified API dengan Claude Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro, Opus 4.5 (untuk task tersulit)
- Transport tool: Model Context Protocol StreamableHTTP (revisi MCP 2026)
- Sandbox: E2B sandboxes (JS SDK) atau Daytona devcontainers
- Pencarian kode: subprocess ripgrep, parser tree-sitter untuk 17 bahasa (pre-compiled)
- Isolasi: `git worktree add` per task, pembersihan pada sukses / gagal
- Harness eval: SWE-bench Pro (subset verified) + Terminal-Bench 2.0 + holdout 30-task milikmu sendiri
- Observability: OpenTelemetry SDK dengan semconv `gen_ai.*` → Langfuse self-hosted
- Penerbitan PR: GitHub App dengan token fine-grained, scope dibatasi ke repo target

```figure
ce-agent-loop
```

## Bangun Sendiri

1. **TUI dan command loop.** Scaffold proyek Bun dengan Ink. Terima `agent run <repo> "<task>"`. Cetak tampilan terbagi: pane plan (atas), stream tool-call (tengah), anggaran token (bawah). Tambahkan pembatalan pada Ctrl-C yang memicu hook `SessionEnd` sebelum keluar.

2. **State plan.** Definisikan schema TodoWrite yang bertipe (item pending / in_progress / done beserta catatan). Model menulis ulang seluruh state setiap giliran sebagai sebuah tool call — jangan biarkan ia memutasi secara bertahap. Simpan plan ke `.agent/state.json` supaya crash bisa dilanjutkan.

3. **Permukaan tool.** Definisikan enam tool: `read_file`, `edit_file` (dengan pratinjau diff), `ripgrep`, `tree_sitter_symbols`, `run_shell` (dengan timeout), `git` (status / diff / commit / push). Pamerkan lewat MCP StreamableHTTP supaya harness bebas transport. Setiap tool mengembalikan output terpotong (cap 4k token per panggilan).

4. **Pembungkusan sandbox.** Setiap task mem-spawn sandbox E2B. `git worktree add -b agent/$TASK_ID` sebuah branch baru. Semua panggilan tool dieksekusi di dalam sandbox. Filesystem host tidak terjangkau.

5. **Hooks.** Implementasikan semua delapan tipe hook 2026. Sambungkan setidaknya empat hook buatan pengguna: (a) guard perintah destruktif `PreToolUse` yang memblokir `rm -rf` di luar worktree, (b) akuntansi token `PostToolUse`, (c) inisialisasi anggaran `SessionStart`, (d) `Stop` yang menulis bundle trace final.

6. **Loop eval.** Clone subset 30-issue dari SWE-bench Pro Python. Jalankan harness-mu pada tiap issue. Bandingkan dengan mini-swe-agent (baseline minimal) pada pass@1, turns-per-task, dan $-per-task. Tulis hasilnya ke `eval/results.jsonl`.

7. **Kontrol biaya.** Cutoff keras: 50 giliran, 200k context, $5 per task. Hook `PreCompact` merangkum turn-turn lama menjadi sebuah blok prior-state di penanda 150k, membebaskan ruang untuk observasi baru tanpa kehilangan plan.

8. **Penerbitan PR.** Saat sukses, langkah terakhir adalah `git push` + panggilan GitHub API yang membuka PR berisi plan dan ringkasan diff pada body-nya.

## Gunakan

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## Rilis

Skill deliverable ada di `outputs/skill-terminal-coding-agent.md`. Diberi path repo dan deskripsi task, skill ini menjalankan loop plan-act-observe penuh di dalam sandbox lalu mengembalikan URL PR plus bundle trace. Rubrik untuk capstone ini:

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs baseline | Your harness vs mini-swe-agent on 30 matched Python tasks |
| 20 | Architecture clarity | Plan/act/observe separation, hook surface, tool schema — reviewed against Live-SWE-agent layout |
| 20 | Safety | Sandbox escape tests, permission prompts, destructive-command guard passes red-team |
| 20 | Observability | Trace completeness (100% of tool calls spanned), token accounting per turn |
| 15 | Developer UX | Cold-start < 2s, crash recovery resumes plan, Ctrl-C cancels mid-tool cleanly |
| **100** | | |

## Latihan

1. Tukar model backing dari Claude Sonnet 4.7 ke Qwen3-Coder-30B yang disajikan di vLLM. Bandingkan pass@1 dan $-per-task. Laporkan di mana model open tersebut berkinerja lebih buruk.

2. Tambahkan sub-agent `reviewer` yang membaca diff sebelum PR diterbitkan dan bisa meminta loop revisi. Ukur apakah review false-positive menurunkan tingkat kelulusan SWE-bench di bawah baseline single-agent (petunjuk: biasanya ya).

3. Stress-test sandbox: tulis sebuah task yang mencoba `curl` ke URL eksternal dan sebuah task yang menulis di luar worktree. Konfirmasi keduanya diblokir oleh hook PreToolUse. Catat percobaannya.

4. Implementasikan ringkasan `PreCompact` dengan model yang lebih kecil (Haiku 4.5). Ukur berapa banyak kesetiaan plan yang hilang pada kompaksi 3x.

5. Tukar transport MCP StreamableHTTP dengan stdio. Benchmark cold-start dan latensi per-panggilan. Pilih pemenangnya untuk penggunaan lokal saja.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Harness | "The agent loop" | The code surrounding the model that dispatches tools, maintains plan state, and enforces budgets |
| Hook | "Agent event listener" | A user-authored script run on one of eight lifecycle events by the harness |
| Worktree | "Git sandbox" | A linked git checkout at a separate path; disposable without touching the main clone |
| TodoWrite | "Plan state" | A typed list of pending/in-progress/done items the model rewrites each turn |
| StreamableHTTP | "MCP transport" | 2026 MCP revision: long-lived HTTP connection with bidirectional streaming; replaces SSE |
| Token ceiling | "Context budget" | Per-turn or per-session cap on input+output tokens; triggers compaction or termination |
| pass@1 | "Single-attempt pass rate" | Fraction of SWE-bench tasks solved on the first run without retry or test-set peeking |

## Bacaan Lanjutan

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — harness referensi dari Anthropic
- [Cursor 3 changelog](https://cursor.com/changelog) — catatan produk Agent Tabs dan Composer 2
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — baseline minimal untuk perbandingan harness SWE-bench
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 79.2% SWE-bench Verified dengan Opus 4.5
- [OpenCode](https://opencode.ai) — harness terbuka, 112k stars
- [SWE-bench Pro leaderboard](https://www.swebench.com) — evaluasi yang dituju capstone ini
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, metadata capability
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — skema span untuk panggilan tool dan penggunaan token
