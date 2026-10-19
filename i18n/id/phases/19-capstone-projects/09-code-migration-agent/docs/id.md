# Capstone 09 — Agen Migrasi Kode (Upgrade Bahasa / Runtime Tingkat Repo)

> MigrationBench dari Amazon (Java 8 ke 17) dan migrator Py2-to-Py3 App Engine dari Google menetapkan standar 2026. OpenRewrite milik Moderne melakukan rewrite AST deterministik dalam skala besar. Grit menyasar masalah yang sama dengan DSL bergaya codemod. Pola produksi menggabungkan keduanya: substrat deterministik untuk rewrite yang aman plus lapisan agen untuk kasus ambigu, sandbox untuk build per-branch, dan test harness yang berubah hijau sebelum PR dibuka. Capstone-nya adalah memigrasi 50 repo nyata dan mempublikasikan pass rate beserta taksonomi kegagalan.

**Type:** Capstone
**Languages:** Python (agent), Java / Python (targets), TypeScript (dashboard)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**Time:** 30 hours

## Masalahnya

Migrasi kode berskala besar adalah salah satu aplikasi produksi paling bersih dari coding agent 2026. Ground truth-nya jelas (apakah suite tes lulus setelah migrasi?), hadiahnya nyata (migrasi armada Java-8 adalah proyek berskala headcount), dan benchmark-nya publik (subset 50 repo MigrationBench). OpenRewrite milik Moderne menangani sisi deterministik. Lapisan agen menangani semua yang tidak bisa ditangani recipe OpenRewrite: rewrite ambigu, drift sistem build, sintaks ekor panjang, kerusakan dependency transitif.

Kamu akan membangun agen yang mengambil repo Java 8 (atau repo Python 2) dan menghasilkan branch hasil migrasi dengan CI hijau. Kamu akan mengukur pass rate, pelestarian test coverage, biaya per repo, dan membangun taksonomi kegagalan. Perbandingan berdampingan dengan baseline deterministik-murni memberi tahu di mana nilai agen sebenarnya berada.

## Konsepnya

Pipelinenya punya dua lapisan. **Deterministic substrate** (OpenRewrite untuk Java, libcst untuk Python) menjalankan sebagian besar rewrite mekanis dengan aman: import, signature metode, edit null-safety, try-with-resources, penggantian API yang sudah deprecated. Ia cepat dan menghasilkan diff yang bisa diaudit. **Agent layer** (OpenAI Agents SDK atau LangGraph di atas Claude Opus 4.7 dan GPT-5.4-Codex) menangani kasus yang tidak bisa ditangani recipe: upgrade berkas build (Maven/Gradle/pyproject), konflik dependency transitif, test flake, anotasi kustom.

Setiap repo mendapat sandbox Daytona dengan runtime target yang sudah terpasang. Agen mengiterasi: jalankan build, klasifikasikan kegagalan, terapkan perbaikan, jalankan ulang. Batas keras: 30 menit per repo, $8 per repo, 20 putaran agen. Jika semua tes lolos dan delta coverage tidak negatif, branch membuka PR. Jika tidak, repo dicatat dalam kelas kegagalan beserta buktinya.

Taksonomi kegagalan adalah deliverable-nya. Dari 50 repo, apa yang rusak? Dependency transitif? Anotasi kustom? Versi build tool? Test flake yang tidak ada hubungannya dengan migrasi? Setiap kelas mendapat hitungan dan satu diff contoh. Penulis recipe masa depan bisa menyasar tiga teratas.

## Arsitektur

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## Stack

- Substrat deterministik: OpenRewrite (Java) atau libcst (Python)
- Agen: OpenAI Agents SDK atau LangGraph di atas Claude Opus 4.7 + GPT-5.4-Codex
- Sandbox: Daytona devcontainer per branch, runtime target yang sudah terpasang (Java 17 / Python 3.12)
- Sistem build: Maven, Gradle, uv (Python)
- Benchmark: subset 50 repo Amazon MigrationBench (Java 8 ke 17), repo Py2-to-Py3 Google App Engine
- Test harness: runner paralel, coverage via Jacoco (Java) atau coverage.py (Python)
- Observability: Langfuse + trace bundle per repo dengan setiap potongan diff
- Dashboard: dashboard taksonomi kegagalan dengan hitungan per kelas dan diff contoh

```figure
ce-migration-funnel
```

## Bangun Sendiri

1. **Recipe pass.** Jalankan recipe OpenRewrite (Java) atau libcst (Python) terlebih dahulu. Tangkap 70-80% migrasi yang bersifat mekanis. Lakukan commit dengan pesan "recipe".

2. **Percobaan build.** Sandbox Daytona: pasang runtime target, jalankan build. Jika hijau, lanjut ke tes. Jika merah, serahkan ke agen.

3. **Loop agen.** LangGraph dengan alat: `run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`. Agen mengklasifikasi kegagalan (dep, sintaks, tes, build-tool) dan menerapkan perbaikan yang terarah. Jalankan ulang.

4. **Batas anggaran.** 30 menit wall-clock per repo, biaya $8, 20 putaran agen. Pelanggaran apa pun menghentikan proses dan mencatat di "budget_exhausted" beserta diff saat ini.

5. **Gerbang test + coverage.** Setelah build hijau, jalankan suite tes. Bandingkan coverage dengan repo dasar. Jika coverage turun lebih dari 2%, catat di "coverage_regression".

6. **Buka PR.** Jika berhasil, push branch-nya, buka PR berisi diff dan ringkasan recipe mana yang diterapkan serta commit mana yang ditulis oleh agen.

7. **Taksonomi kegagalan.** Untuk setiap repo yang gagal, beri tag kelas: `dep_upgrade_required`, `build_tool_drift`, `custom_annotation`, `test_flake`, `syntax_edge_case`, `budget_exhausted`. Bangun sebuah dashboard.

8. **Run 50 repo.** Jalankan di seluruh subset MigrationBench. Laporkan pass rate per kelas, biaya per repo, pelestarian coverage, dan perbandingan terhadap baseline deterministik-murni.

## Gunakan

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## Rilis

`outputs/skill-migration-agent.md` adalah deliverable-nya. Diberi sebuah repo, ia mengeksekusi recipe deterministik lalu loop agen untuk menghasilkan branch hasil migrasi yang hijau, atau mencatat repo itu dalam kelas taksonomi.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | MigrationBench pass rate | 50-repo subset pass@1 |
| 20 | Test-coverage preservation | Mean coverage delta vs base |
| 20 | Cost per migrated repo | $/repo on passing runs |
| 20 | Agent / deterministic-tool integration | Fraction of fixes that OpenRewrite handled vs agent authored |
| 15 | Failure analysis write-up | Taxonomy completeness with exemplars |
| **100** | | |

## Latihan

1. Jalankan pipeline migrasi hanya dengan OpenRewrite (tanpa agen). Bandingkan pass rate-nya dengan pipeline penuh. Identifikasi kasus di mana agen sendirilah yang menjadi pembeda.

2. Implementasikan pemeriksaan "lint-clean": setelah migrasi, jalankan style linter (spotless untuk Java, ruff untuk Python). Gagalkan PR jika muncul error lint baru. Ukur laju coverage-tetap-tapi-style-regres.

3. Tambahkan optimizer "minimal-diff": setelah branch agen lulus tes, pangkas perubahan yang tidak perlu dengan sebuah pass kedua. Laporkan pengurangan ukuran diff.

4. Perluas ke migrasi ketiga: Node 18 ke Node 22. Gunakan kembali pembungkus sandbox-nya; ganti lapisan recipe dengan codemod kustom.

5. Ukur time-to-first-green-build (TTFGB) sebagai metrik UX. Target: p50 di bawah 10 menit.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Deterministic substrate | "Recipe engine" | OpenRewrite / libcst: declarative AST rewrites with safety guarantees |
| Codemod | "Code-modifying program" | A rewrite rule that changes source code mechanically |
| Build drift | "Tool version skew" | Subtle Maven / Gradle / uv behavior changes between major versions |
| Failure class | "Taxonomy bucket" | A labeled reason a repo did not migrate: dep, syntax, test, build-tool, budget |
| Coverage delta | "Coverage preservation" | Change in test coverage % from base to migrated branch |
| Agent turn | "Tool-call round" | One plan -> act -> observe cycle in the agent loop |
| Budget exhaustion | "Hit the ceiling" | The repo consumed its 30-min / $8 / 20-turn limit without passing |

## Bacaan Lanjutan

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — the canonical 2026 benchmark
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — the deterministic substrate reference
- [OpenRewrite documentation](https://docs.openrewrite.org) — recipe authoring
- [Grit.io](https://www.grit.io) — alternate codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — the Agents SDK reference
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — alternate migration benchmark
- [libcst](https://github.com/Instagram/LibCST) — Python deterministic substrate
- [Daytona sandboxes](https://daytona.io) — reference per-branch sandbox
