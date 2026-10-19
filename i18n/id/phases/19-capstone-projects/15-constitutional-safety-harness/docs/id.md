# Capstone 15 — Harness Keselamatan Konstitusional + Red-Team Range

> Constitutional Classifiers milik Anthropic, Llama Guard 4 dari Meta, ShieldGemma-2 dari Google, Nemotron 3 Content Safety dari NVIDIA, dan X-Guard untuk cakupan multilingual mendefinisikan stack classifier keselamatan 2026. garak, PyRIT, NVIDIA Aegis, dan promptfoo menjadi alat evaluasi adversarial standar. NeMo Guardrails v0.12 merangkainya menjadi satu pipeline produksi. Capstone ini menyambungkan semuanya: sebuah harness keselamatan berlapis di sekitar aplikasi target, agen red-team otonom yang menjalankan 6+ keluarga serangan, dan sebuah run self-critique konstitusional yang menghasilkan delta harmlessness yang terukur.

**Type:** Capstone
**Languages:** Python (safety pipeline, red team), YAML (policy configs)
**Prerequisites:** Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 18 (ethics, safety, alignment)
**Phases exercised:** P10 · P11 · P13 · P14 · P18
**Time:** 25 hours

## Masalahnya

Batas depan keselamatan LLM pada 2026 bukanlah apakah classifier bekerja (mereka bekerja, kira-kira), melainkan bagaimana menyusunnya dengan benar di sekitar aplikasi produksi tanpa over-refuse atau meninggalkan lubang yang jelas. Llama Guard 4 menangani pelanggaran kebijakan berbahasa Inggris. X-Guard (132 bahasa) menangani jailbreak multilingual. ShieldGemma-2 menangkap prompt injection berbasis gambar. NVIDIA Nemotron 3 Content Safety menutupi kategori enterprise. Constitutional Classifiers milik Anthropic adalah pendekatan terpisah yang dipakai selama training, bukan saat serving.

Evolusi serangan juga penting. PAIR dan TAP mengotomatiskan penemuan jailbreak. GCG menjalankan serangan suffix berbasis gradien. Serangan multi-turn dan code-switch mengeksploitasi memori agen. Setiap LLM yang di-deploy membutuhkan sebuah red-team range — garak dan PyRIT adalah driver kanonisnya — plus mitigasi terdokumentasi dan temuan yang diberi skor CVSS.

Kamu akan memperkeras sebuah aplikasi target (entah sebuah model instruction-tuned 8B entah salah satu RAG chatbot dari capstone lain), menjalankan 6+ keluarga serangan terhadapnya, dan menghasilkan pengukuran harmlessness sebelum/sesudah.

## Konsepnya

Pipeline keselamatan punya lima lapisan. **Input sanitize**: buang karakter zero-width, dekode base64/rot13, normalisasi Unicode. **Policy layer**: rails NeMo Guardrails v0.12 (off-domain, toxicity, ekstraksi PII). **Classifier gate**: Llama Guard 4 pada input, X-Guard pada non-Inggris, ShieldGemma-2 pada input gambar. **Model**: LLM target. **Output filter**: Llama Guard 4 pada output, scrub PII Presidio, penegakan sitasi bila berlaku. **HITL tier**: output yang ditandai berisiko tinggi masuk ke antrean Slack.

Red-team range berjalan di atas sebuah scheduler. PAIR dan TAP menemukan jailbreak secara otonom. GCG menjalankan serangan suffix berbasis gradien. Serangan encoding ASCII / base64 / rot13. Serangan multi-turn (adopsi persona, eksploitasi memori). Serangan code-switch (mencampur Inggris dengan Swahili atau Thai). Setiap run menghasilkan berkas temuan terstruktur dengan skor CVSS dan timeline disclosure.

Run self-critique konstitusional adalah intervensi saat training. Ambil 1k prompt upaya-berbahaya, minta model menyusun sebuah respons, kritik respons itu terhadap konstitusi tertulis (aturan jangan-berbuat-jelek), dan latih ulang pada loop kritik itu. Ukur delta harmlessness sebelum/sesudah pada sebuah eval held-out.

## Arsitektur

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## Stack

- Classifier keselamatan: Llama Guard 4, ShieldGemma-2, NVIDIA Nemotron 3 Content Safety, X-Guard
- Framework guardrail: NeMo Guardrails v0.12 + OPA
- Driver red-team: garak (NVIDIA), PyRIT (Microsoft Azure), NVIDIA Aegis, promptfoo
- Agen jailbreak: PAIR (Chao et al., 2023), Tree-of-Attacks (TAP), suffix GCG
- Training konstitusional: loop self-critique ala Anthropic + SFT pada kritik
- Scrub PII: Presidio
- Target: sebuah model instruction-tuned 8B atau salah satu RAG chatbot dari capstone lainnya

```figure
cf-safety-stack
```

## Bangun Sendiri

1. **Penyiapan target.** Nyalakan sebuah model instruction-tuned 8B di vLLM (atau pakai ulang sebuah RAG chatbot dari capstone lain). Inilah aplikasi yang diuji.

2. **Pembungkus pipeline keselamatan.** Sambungkan pipeline lima lapisan di sekitar target. Verifikasi setiap lapisan dapat diamati secara individual (span per lapisan di Langfuse).

3. **Cakupan classifier.** Muat Llama Guard 4, X-Guard (multilingual), ShieldGemma-2 (gambar). Jalankan masing-masing pada set berlabel kecil untuk menetapkan baseline.

4. **Scheduler red-team.** Jadwalkan garak, PyRIT, sebuah agen PAIR, sebuah agen TAP, sebuah runner GCG, penyerang multi-turn, dan penyerang code-switch. Masing-masing berjalan pada antrean terpisah.

5. **Suite serangan.** Enam keluarga serangan: (1) jailbreak otomatis PAIR, (2) tree-of-attacks TAP, (3) suffix gradien GCG, (4) encoding ASCII / base64 / rot13, (5) persona multi-turn, (6) code-switch multilingual. Laporkan laju keberhasilan per keluarga.

6. **Self-critique konstitusional.** Kurasi 1k prompt upaya-berbahaya. Untuk setiap prompt, target menyusun sebuah respons. Sebuah LLM kritikus menilai terhadap konstitusi tertulis ("jangan berbuat bahaya", "kutip bukti", "tolak permintaan ilegal"). Prompt yang dikritik ditulis ulang; target di-fine-tune pada pasangan yang sudah diperbaiki kritiknya. Ukur harmlessness sebelum/sesudah pada eval held-out.

7. **Pengukuran over-refusal.** Lacak laju false-positive pada suite prompt jinak (mis., XSTest). Target harus tetap membantu pada pertanyaan jinak.

8. **Skor CVSS.** Untuk setiap jailbreak yang berhasil, beri skor dengan CVSS 4.0 (attack vector, complexity, impact). Hasilkan timeline disclosure dan rencana mitigasi.

9. **Otomasi range.** Semua di atas berjalan di atas cron; temuan ditulis ke sebuah antrean; alert regresi over-refusal dikirim ke Slack.

## Gunakan

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## Rilis

`outputs/skill-safety-harness.md` adalah deliverable-nya. Pipeline keselamatan berlapis setingkat produksi plus sebuah red-team range yang dapat direproduksi dengan delta harmlessness sebelum/sesudah.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Attack-surface coverage | 6+ attack families exercised, 2+ languages |
| 20 | True-positive / false-positive trade-off | Attack block rate vs XSTest benign pass rate |
| 20 | Self-critique delta | Before/after harmlessness on held-out eval |
| 20 | Documentation and disclosure | CVSS-scored findings with timeline |
| 15 | Automation and repeatability | Everything runs on cron with alerts |
| **100** | | |

## Latihan

1. Jalankan plugin milik garak untuk prompt-injection pada sebuah RAG chatbot dan bandingkan laju keberhasilan serangan dengan dan tanpa lapisan output-filter.

2. Tambahkan keluarga serangan ketujuh: prompt injection tidak langsung lewat dokumen yang di-retrieve. Ukur pertahanan tambahan yang dibutuhkan.

3. Implementasikan mode "refuse-with-help": ketika guardrail memblokir, target menawarkan jawaban terkait yang lebih aman alih-alih penolakan datar. Ukur delta XSTest.

4. Celah cakupan multilingual: temukan sebuah bahasa tempat X-Guard berkinerja buruk. Usulkan sebuah dataset fine-tune yang menyasar bahasa itu.

5. Jalankan self-critique konstitusional pada sebuah model 30B dan ukur apakah delta-nya berskala.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Layered safety | "Defense in depth" | Multiple guardrails at input, gate, output, HITL |
| Llama Guard 4 | "Meta's safety classifier" | The 2026 reference input/output content classifier |
| PAIR | "Jailbreak agent" | Paper (Chao et al.) on LLM-driven jailbreak discovery |
| TAP | "Tree-of-Attacks" | Tree-search variant of PAIR |
| GCG | "Greedy coordinate gradient" | Gradient-based adversarial suffix attack |
| Constitutional self-critique | "Anthropic-style training" | Target drafts -> critic scores -> rewrite -> retrain |
| XSTest | "Benign probe set" | Benchmark for over-refusal regression |
| CVSS 4.0 | "Severity score" | Standard vulnerability scoring for safety findings |

## Bacaan Lanjutan

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — training-time reference
- [Meta Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — the 2026 input/output classifier
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — image + multimodal safety
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — enterprise reference
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132-language multilingual safety
- [garak](https://github.com/NVIDIA/garak) — NVIDIA red-team toolkit
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft red-team framework
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — rail framework
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — jailbreak agent paper
