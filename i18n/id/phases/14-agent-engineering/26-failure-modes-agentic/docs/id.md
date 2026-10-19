# Mode kegagalan: Mengapa Agen Break

> MASFT (Berkeley, 2025) catalogs 14 multi-agen mode kegagalan di 3 categories. Microsoft's Taxonomy documents bagaimana existing AI kegagalan amplify di agentic settings. Industry field data converges pada lima recurring modes: hallucinated aksi, scope creep, cascading error, context loss, tool misuse.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 05 (Self-Refine and CRITIC), Phase 14 · 24 (Observability)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Name MASFT's tiga kegagalan categories dan pada least empat specific modes di setiap.
- Jelaskan mengapa agentic kegagalan amplifies existing AI mode kegagalan (bias, hallucination).
- Deskripsikan lima industry-recurring modes dan mereka mitigations.
- Implementasikan sebuah stdlib detector itu tags agen traces dengan kegagalan-mode labels.

## Masalahnya

Teams ship agen itu kerja pada 90% dari traces. 10% kegagalan adalah tidak random noise — mereka fall menjadi sebuah kecil number dari recurring categories. Setelah kamu bisa name mereka, kamu bisa monitor untuk mereka dan fix mereka.

## Konsepnya

### MASFT (Berkeley, arXiv:2503.13657)

Multi-agen Sistem Kegagalan Taxonomy. 14 mode kegagalan clustered menjadi 3 categories. Inter-annotator Cohen's Kappa 0.88 — categories adalah reliably distinguishable.

Pusat claim: kegagalan adalah fundamental design flaws di multi-agen sistem, tidak LLM limitations ke menjadi fixed dengan better base model.

### Microsoft Taxonomy dari Mode kegagalan di Agentic AI Sistem

- Existing AI kegagalan (bias, hallucination, data leakage) amplify di agentic settings.
- Baru kegagalan emerge dari autonomy: unintended aksi pada scale, tool misuse, mission drift.
-  whitepaper adalah risiko register untuk agentic products.

### Characterizing Faults di Agentic AI (arXiv:2603.06847)

- Kegagalan arise dari orkestrasi, internal state evolution, dan environment interaction.
- Tidak just "buruk kode" atau "buruk model output."

### LLM Agen Hallucinations Survey (arXiv:2509.18970)

Dua primary manifestations:

1. **Instruction-following Deviation** — agen doesn't follow sistem prompt.
2. **Long-range Contextual Misuse** — agen forgets atau misapplies context dari earlier giliran.

Sub-intention error: Omission (missed langkah), Redundancy (repeated langkah), Disorder (out-dari-order langkah).

###  lima industry-recurring modes

Arize, Galileo, NimbleBrain 2024-2026 field analyses converge pada:

1. **Hallucinated actions.** Agen invokes sebuah tool itu doesn't exist atau fabricates arguments.
2. **Scope creep.** Agen expands tugas melampaui user's tanya (membuat extra PRs, mengirim extra emails).
3. **Cascading errors.** Satu salah panggil triggers downstream effects. SEBUAH phantom SKU hallucination triggers empat API memanggil — sebuah multi-sistem incident.
4. **Context loss.** Long-horizon tugas forget early-giliran constraints.
5. **Tool misuse.** Memanggil right tool dengan salah arguments, atau salah tool entirely.

Cascading adalah killer. Agen cannot distinguish "I failed" dari " tugas adalah impossible" dan sering hallucinate sebuah keberhasilan message pada 400 error ke tutup loop.

### Mitigation: gates pada setiap langkah

Automated verifikasi gates pada setiap langkah dari sebuah penalaran chain, checking factual grounding terhadap environment state. Concretely:

- Per-langkah keamanan classifier (Pelajaran 21).
- Tool-panggil argument validation (Pelajaran 06).
- Cross-check retrieved content terhadap known facts (Pelajaran 05, CRITIC).
- Detect keberhasilan hallucination oleh re-probing state (adalah file actually created?).

### Di mana kegagalan monitoring goes salah

- **Tagging only crashes.** Sebagian besar agen kegagalan produce valid-looking output. Butuhkan content-level checks.
- **No baseline.** Drift detection membutuhkan sebuah terakhir-known-baik; tanpa itu kamu cannot nyatakan "ini adalah getting worse."
- **Over-alerting.** Setiap kegagalan produces sebuah page. Cluster dan rate-limit.

```figure
failure-cascade
```

## Bangun Sendiri

`code/main.py` implements sebuah stdlib kegagalan-mode tagger:

- SEBUAH synthetic trace dataset covering lima modes.
- Detector functions per mode (signature polas pada pemanggilan tool, outputs, repeat aksi).
- SEBUAH tagger itu labels setiap trace dan reports mode distribution.

Jalankan:

```
python3 code/main.py
```

Output: per-trace labels + aggregate distribution, sebuah cheap reproduction dari apa Phoenix's trace clustering surfaces.

## Gunakan

- **Phoenix** untuk produksi drift clustering (Pelajaran 24).
- **Langfuse** untuk sesi replay + annotation.
- **Custom** untuk domain-specific signatures kamu observabilitas platform bisa't detect.

## Rilis

`outputs/skill-failure-detector.md` menghasilkan kegagalan-mode detectors tailored ke kamu domain, wired ke sebuah trace store.

## Latihan

1. Tambahkan sebuah detector untuk "keberhasilan hallucination": agen mengembalikan keberhasilan but target state adalah unchanged.
2. Tag 100 nyata traces dari sebuah product kamu've built. Yang mode dominates? Apa's cost dari fixing itu?
3. Implementasikan sebuah "cascade radius" metrik: given sebuah kegagalan pada langkah N, bagaimana banyak downstream langkah melakukan itu affect?
4. Baca MASFT's 14 mode kegagalan. Pick tiga itu apply ke kamu product. Tulis detectors.
5. Wire satu detector menjadi sebuah CI job: gagal build jika >=5% dari traces tag sebuah mode.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MASFT | "Multi-agent failure taxonomy" | Berkeley 14-mode categorization |
| Cascading error | "Ripple failure" | One early mistake propagates through N steps |
| Context loss | "Forgot the constraint" | Long-horizon turn drops early-turn facts |
| Tool misuse | "Wrong tool / wrong args" | Valid call, wrong invocation |
| Success hallucination | "Faked completion" | Agent claims success on a 400; state unchanged |
| Scope creep | "Overreach" | Agent does more than asked |
| Instruction-following deviation | "Disobedience" | Ignores system prompt or user constraint |
| Sub-intention errors | "Plan bugs" | Omission, redundancy, disorder in plan execution |

## Bacaan Lanjutan

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 mode kegagalan, 3 categories
- [Microsoft, Taxonomy dari Mode kegagalan di Agentic AI Sistem](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — risiko register
- [Arize Phoenix](https://docs.arize.com/phoenix) — drift clustering dalam praktik
- [Anthropic, Building Efektif Agen](https://www.anthropic.com/research/building-effective-agents) — ketika simpler polas avoid modes entirely
