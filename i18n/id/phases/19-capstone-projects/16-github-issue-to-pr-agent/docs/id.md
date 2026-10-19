# Capstone 16 — Agen Otonom GitHub Issue-to-PR

> Beri label pada sebuah issue, dapatkan sebuah PR — bentuk produk 2026 untuk coding agent otonom: jalankan agen di cloud sandbox, verifikasi tes lulus, dan kirimkan PR yang siap-review beserta rasionalisasi. AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud, dan Google Jules semuanya mengirimkannya. Bagian sulitnya adalah mereproduksi environment build repo secara otomatis, mencegah kebocoran kredensial, menegakkan anggaran per repo, dan memastikan agen tidak bisa force-push. Capstone ini membangun versi self-hosted-nya dan membandingkannya dari sisi biaya dan pass rate dengan alternatif yang di-hosting.

**Type:** Capstone
**Languages:** Python (agent), TypeScript (GitHub App), YAML (Actions)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P11 · P13 · P14 · P15 · P17
**Time:** 30 hours

## Masalahnya

Coding agent cloud async adalah kategori produk terpisah dari coding agent interaktif (capstone 01). UX-nya adalah sebuah label GitHub. Kamu melabeli sebuah issue `@agent fix this`, sebuah worker menyala di cloud sandbox, meng-clone repo, menjalankan tes, mengedit berkas, memverifikasi, dan membuka PR berisi rasionalisasi agen di badannya. Tidak ada loop interaktif, tidak ada terminal. AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud, Google Jules, dan Factory Droids semuanya konvergen ke bentuk ini.

Tantangan rekayasanya konkret: reproduksi environment (agen harus membangun repo dari nol tanpa image dev yang di-cache), test flake (harus dijalankan ulang atau diisolasi), scoping kredensial (sebuah GitHub App dengan izin fine-grained minimal), penegakan anggaran per repo per hari, dan kebijakan no-force-push. Capstone ini mengukur pass rate, biaya, dan keamanan dibanding alternatif yang di-hosting.

## Konsepnya

Pemicunya adalah sebuah webhook GitHub (label issue atau komentar PR). Sebuah dispatcher mengantrekan kerja ke ECS Fargate atau Lambda. Worker menarik repo ke dalam sandbox Daytona atau E2B dengan Dockerfile generik yang disimpulkan dari repo (bahasa, framework). Agen menjalankan loop mini-swe-agent atau SWE-agent v2 di atas Claude Opus 4.7 atau GPT-5.4-Codex. Ia mengiterasi: baca kode, usulkan perbaikan, terapkan patch, jalankan tes.

Verifikasi adalah langkah gating-nya. CI penuh harus lulus di dalam sandbox sebelum PR dibuka. Delta coverage dihitung; jika negatif melewati sebuah ambang, PR tetap dibuka tetapi diberi label `needs-review`. Agen memposting rasionalisasi sebagai deskripsi PR ditambah sebuah thread `@agent` yang bisa di-ping reviewer untuk tindak lanjut.

Keamanan di-scope melalui dua permukaan GitHub yang berbeda: App menyediakan installation token jangka pendek dengan `workflows: read` dan scope contents/PR yang sempit; branch protection (bukan izin aplikasi) yang menegakkan "no direct writes to `main`" dan "no force-push" — aplikasi tidak pernah ditambahkan ke daftar bypass. Akses read-only ber-scope path ke `.github/workflows` bukan primitif GitHub App yang nyata, sehingga allow-list agen untuk edit berkas harus menegakkannya di tingkat worker. Batas anggaran per repo per hari diegakkan di dispatcher (mis., maksimal 5 PR per repo per hari, $20 per PR).

## Arsitektur

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## Stack

- Pemicu: GitHub App dengan token fine-grained; receiver webhook via Lambda atau Fly.io
- Worker: task ECS Fargate (atau GitHub Actions self-hosted runner)
- Sandbox: devcontainer Daytona atau sandbox E2B per tugas
- Loop agen: baseline mini-swe-agent atau SWE-agent v2 di atas Claude Opus 4.7 / GPT-5.4-Codex
- Retrieval: repo-map tree-sitter + ripgrep
- Verifikasi: CI penuh di dalam sandbox + gerbang delta coverage
- Observability: Langfuse dengan arsip trace per PR yang tertaut dari badan PR
- Anggaran: batas dolar harian per repo; maksimal PR per repo per hari

```figure
cf-issue-to-pr
```

## Bangun Sendiri

1. **GitHub App.** Installation token fine-grained: issues read+write, pull_requests write, contents read+write, workflows read. Branch protection (satu-satunya permukaan yang bisa melakukan ini) menegakkan "no direct push to `main`" dan "no force-push"; aplikasi tidak berada di daftar bypass. Worker menegakkan "no writes under `.github/workflows`" sebagai pemeriksaan allow-list pada diff yang diusulkan, karena izin GitHub App tidak ber-scope path.

2. **Receiver webhook.** Fungsi Lambda menerima webhook label issue / komentar PR. Menyaring berdasarkan label `@agent fix this`. Mengantrekan ke SQS.

3. **Dispatcher.** Mengambil tugas dari SQS. Menegakkan anggaran per repo per hari. Menyala-kan sebuah task ECS Fargate dengan URL repo, badan issue, dan sandbox Daytona yang baru.

4. **Inferensi environment.** Deteksi bahasa (Python, Node, Go, Rust) dan package manager (uv, pnpm, go mod, cargo). Hasilkan sebuah Dockerfile secara on the fly jika belum ada.

5. **Loop agen.** mini-swe-agent atau SWE-agent v2 dengan Claude Opus 4.7. Alat: ripgrep, repo-map tree-sitter, read_file, edit_file, run_tests, git. Batas keras: biaya $20, 30 menit wall-clock, 30 putaran agen.

6. **Verifikasi.** Setelah loop selesai, jalankan suite tes penuh di dalam sandbox. Hitung delta coverage via jacoco / coverage.py. Jika CI merah: hentikan, jangan buka PR. Jika coverage turun lebih dari 2%: buka PR dengan label `needs-review`.

7. **Pemasangan PR.** Push branch agen. Buka PR lewat GitHub API dengan: judul, rasionalisasi, ringkasan diff, URL trace, biaya, putaran.

8. **Higienitas kredensial.** Worker berjalan dengan installation token GitHub App jangka pendek. Log di-scrub dari secret sebelum diarsipkan.

9. **Eval.** 30 issue internal ber-seed dengan tingkat kesulitan bervariasi. Ukur pass rate, kualitas PR (ukuran diff, style, coverage), biaya, latensi. Bandingkan dengan Cursor Background Agents dan AWS Remote SWE Agents pada issue yang sama.

## Gunakan

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## Rilis

`outputs/skill-issue-to-pr.md` adalah deliverable-nya. Sebuah GitHub App + cloud worker async yang mengubah ber-label issue menjadi PR yang siap-review dengan biaya terbatas dan kredensial yang di-scope.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Pass rate on 30 issues | End-to-end success (CI green + coverage OK) |
| 20 | PR quality | Diff size, coverage delta, style conformance |
| 20 | Cost and latency per resolved issue | $ and wall-clock per PR |
| 20 | Safety | Scoped token, per-repo budget, no force-push, credential hygiene |
| 15 | Operator UX | Rationale comments, retry affordance, @-mention follow-up |
| **100** | | |

## Latihan

1. Tambahkan mode "fix flaky test": label `@agent stabilize-flake TestX` menjalankan tes itu 50 kali di dalam sandbox dan mengusulkan perubahan minimal yang menstabilkannya.

2. Bandingkan biayanya dengan Cursor Background Agents pada tiga issue bersama. Laporkan alat mana yang menang di mana.

3. Implementasikan dashboard anggaran: biaya per repo per hari, biaya per pengguna. Beri alert pada anomali.

4. Bangun mode "dry-run" yang membuka sebuah PR draft tanpa menjalankan CI, sehingga reviewer dapat memeriksa rencananya dengan murah.

5. Tambahkan kebijakan retensi: branch PR yang berusia lebih dari 7 hari tanpa merge dihapus secara otomatis.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GitHub App | "Scoped bot identity" | App with fine-grained permissions + short-lived installation token |
| Async cloud agent | "Background agent" | Non-interactive worker that runs in a cloud sandbox, not a terminal |
| Environment inference | "Dockerfile synthesis" | Detect language + package manager, generate a Dockerfile if absent |
| Verification | "CI-in-sandbox" | Run the full test suite inside the worker before opening a PR |
| Coverage delta | "Coverage preservation" | Change in test coverage % from base to agent branch |
| Per-repo budget | "Daily ceiling" | Dollar and PR-count cap enforced at the dispatcher |
| Rationale | "PR body explanation" | Agent's summary of what changed and why; required in the PR body |

## Bacaan Lanjutan

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — the canonical async cloud agent reference
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI reference
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — commercial alternative
- [OpenAI Codex (cloud)](https://openai.com/codex) — hosted competitor
- [Google Jules](https://jules.google) — Google's hosted version
- [Factory Droids](https://www.factory.ai) — alternate commercial reference
- [GitHub App documentation](https://docs.github.com/en/apps) — scoped bot identity
- [Daytona cloud sandboxes](https://daytona.io) — reference sandbox
