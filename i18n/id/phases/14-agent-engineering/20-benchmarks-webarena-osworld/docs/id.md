# Benchmark: WebArena dan OSWorld

> WebArena tes web-agen kapabilitas lintas empat mandiri-hosted apps. OSWorld tes desktop-agen kapabilitas lintas Ubuntu, Windows, macOS. Pada release (2023–2024) keduanya showed sebuah besar gap antara best-di-class agen dan humans. gap adalah narrowing; mode kegagalan haven't changed.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 19 (SWE-bench, GAIA)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Deskripsikan WebArena's empat mandiri-hosted apps dan mengapa execution-based evaluasi matters.
- Jelaskan mengapa OSWorld memakai nyata OS screenshots alih-alih accessibility APIs.
- Name dua primary OSWorld mode kegagalan: GUI grounding dan operasional knowledge.
- Summarize apa OSWorld-G dan OSWorld-Manusia add di atas base benchmark.

## Masalahnya

Generalist agen bisa panggil tool. Bisa mereka drive sebuah browser lintas 20 clicks ke complete sebuah shopping checkout? Bisa mereka configure sebuah Linux box memakai hanya keyboard dan mouse? Ini adalah questions WebArena dan OSWorld jawaban.

## Konsepnya

### WebArena (Zhou et al., ICLR 2024)

- 812 long-horizon tugas lintas empat mandiri-hosted web apps: sebuah shopping site, sebuah forum, sebuah GitLab-like dev tool, sebuah business CMS.
- Plus utilities: map, calculator, scratchpad.
- Evaluasi adalah execution-based via gym APIs — adalah order placed, adalah issue closed, adalah CMS page updated?
- Pada release: best GPT-4 agen hit 14.41% keberhasilan vs manusia 78.24%.

 mandiri-hosted framing matters — benchmark adalah tidak flaky karena target apps adalah pinned dan reproducible.

### Extensions

- **VisualWebArena** — visually grounded tugas di mana keberhasilan depends pada interpreting images (screenshots sebagai pertama-class observasi).
- **TheAgentCompany** (Dec 2024) — adds terminal + coding; more like sebuah nyata remote-kerja environment.

### OSWorld (Xie et al., NeurIPS 2024)

- 369 nyata komputer tugas lintas Ubuntu, Windows, macOS.
- Free-form keyboard dan mouse kontrol dari nyata applications.
- 1920×1080 screenshots sebagai observasi.
- Pada release: best model 12.24% vs manusia 72.36%.

### Primary mode kegagalan

1. **GUI grounding.** Pixel → element mapping. Model struggle ke localize UI elements reliably di 1920×1080.
2. **Operational knowledge.** Yang menu memiliki setting, yang keyboard shortcut, yang preference pane. Knowledge tail itu humans build atas years.

### Follow-ups

- **OSWorld-G** — 564-sample grounding suite + Jedi training set. Decomposes grounding dari perencanaan so kamu bisa measure mereka separately.
- **OSWorld-Human** — manually curated gold aksi trajectories. Menunjukkan top agen pakai 1.4-2.7x more langkah than necessary ( trajectory-efficiency gap).

### Mengapa ini matters

Claude komputer pakai, OpenAI CUA, Gemini 2.5 Komputer Pakai (Pelajaran 21) semua train pada workloads shaped oleh WebArena dan OSWorld. benchmark adalah target; produksi model adalah shipped jawaban.

### Di mana benchmarking goes salah

- **Screenshot-only evals.** OSWorld adalah screenshot-driven; evaluating sebuah agen itu memakai DOM atau accessibility APIs pada OSWorld misses grounding challenge.
- **Ignoring trajectory length.** Scoring hanya keberhasilan-rate misses 1.4-2.7x langkah inefficiency OSWorld-Manusia surfaces.
- **Stale self-hosted apps.** WebArena's apps pin specific versions; update tanpa re-curation breaks comparability.

```figure
ae-agent-human-gap
```

## Bangun Sendiri

`code/main.py` implements sebuah toy web-agen harness:

- SEBUAH minimal "shopping app" state machine: list_items, add_to_cart, checkout.
- Gold trajectories untuk 3 tugas.
- SEBUAH scripted agen itu attempts setiap tugas.
- Execution-based evaluator (state check) dan trajectory-efficiency metrik (langkah vs gold).

Jalankan:

```
python3 code/main.py
```

Output: per-tugas tingkat keberhasilan dan trajectory efficiency, mirroring OSWorld-Manusia's methodology.

## Gunakan

- **WebArena Verified** mandiri-hosted pada sebuah internal cluster untuk continuous evaluasi.
- **OSWorld** di sebuah VM fleet untuk desktop agen.
- **Computer-use agents** (Pelajaran 21) — Claude, OpenAI CUA, Gemini — semua trained pada workloads like ini.
- **Your own product flows** — capture gold trajectories untuk kamu top 20 tugas; jalankan agen terhadap mereka weekly.

## Rilis

`outputs/skill-web-desktop-harness.md` builds sebuah web/desktop agen harness dengan execution-based eval dan trajectory efficiency metrik.

## Latihan

1. Perluas toy harness dengan sebuah kedua app (sebuah forum). Tulis 3 tugas plus gold trajectories.
2. Add trajectory-efficiency reporting per tugas. Pada kamu toy, adalah agen itu 1x, 2x, atau 3x atas gold?
3. Implementasikan sebuah "distractor" tool — satu gold trajectory tidak pernah memakai. Melakukan scripted agen dapatkan tempted?
4. Baca OSWorld-G. Bagaimana akan kamu terpisah grounding kegagalan dari perencanaan kegagalan di kamu own eval?
5. Baca WebArena's apps README. Apa breaks ketika kamu upgrade satu dari pinned app versions?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| WebArena | "Web agent benchmark" | 812 tasks across 4 self-hosted apps; gym-style evaluation |
| VisualWebArena | "Visual WebArena" | Visually grounded WebArena; screenshots are observations |
| OSWorld | "Desktop agent benchmark" | 369 tasks on real Ubuntu/Windows/macOS |
| GUI grounding | "Pixel-to-element mapping" | Model localizing UI elements in 1920x1080 |
| Operational knowledge | "OS know-how" | Which menu, which shortcut, which preference pane |
| OSWorld-G | "Grounding suite" | 564 grounding-only samples + training set |
| OSWorld-Human | "Gold trajectories" | Manual expert action sequences to measure efficiency |
| Trajectory efficiency | "Steps over gold" | Agent step count divided by human minimum |

## Bacaan Lanjutan

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — empat-app web benchmark
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — cross-OS desktop benchmark
- [Anthropic, Introducing komputer pakai](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude's benchmark-shaped kapabilitas
- [OpenAI, Komputer-Memakai Agen](https://openai.com/index/computer-using-agent/) — OSWorld dan WebArena numbers
