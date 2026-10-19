# Tooling Red-Team — Garak, Llama Guard, PyRIT

> Tiga tool produksi membingkai stack red-team 2026. Llama Guard (Meta) — klasifier Llama-3.1-8B yang di-fine-tune pada 14 kategori hazard MLCommons; Llama Guard 4 2025 adalah klasifier multimodal native 12B yang diprune dari Llama 4 Scout. Garak (NVIDIA) — pemindai kerentanan LLM open-source dengan probe statis, dinamis, dan adaptif untuk halusinasi, kebocoran data, prompt injection, toksisitas, dan jailbreak. PyRIT (Microsoft) — kampanye red-team multi-turn dengan Crescendo, TAP, dan rantai konverter kustom untuk eksploitasi mendalam. Llama Guard 3 didokumentasikan dalam "Llama 3 Herd of Models" Meta (arXiv:2407.21783); Llama Guard 3-1B-INT4 dalam arXiv:2411.17713; arsitektur probe Garak dalam github.com/NVIDIA/garak. Tool ini adalah antarmuka produksi 2026 antara riset red-team (Pelajaran 12-15) dan deployment (Pelajaran 17+).

**Type:** Build
**Languages:** Python (stdlib, tool-architecture simulator and Llama Guard-style classifier mock)
**Prerequisites:** Phase 18 · 12-15 (jailbreaks and IPI)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan posisi Llama Guard 3/4 dalam stack keamanan: klasifier input, klasifier output, atau keduanya.
- Sebutkan 14 kategori hazard MLCommons dan nyatakan satu yang tak obvious (Code Interpreter Abuse).
- Jelaskan arsitektur probe Garak: probe, detektor, harness.
- Jelaskan struktur kampanye multi-turn PyRIT dan bagaimana ia berkomposisi dengan probe Garak.

## Masalahnya

Pelajaran 12-15 menyajikan permukaan serangan. Deployment produksi butuh evaluasi yang repeatable dan scalable. Tiga tool mendominasi 2026: Llama Guard (klasifier pertahanan), Garak (pemindai), PyRIT (orkestrator kampanye). Masing-masing menargetkan lapisan berbeda dari lifecycle red-team.

## Konsepnya

### Llama Guard (Meta)

Llama Guard 3 adalah model Llama-3.1-8B yang di-fine-tune untuk klasifikasi input/output atas 14 kategori AILuminate MLCommons:
- Violent crimes, non-violent crimes, sex-related, CSAM, defamation
- Specialized advice, privacy, IP, indiscriminate weapons, hate
- Suicide/self-harm, sexual content, elections, code-interpreter abuse

Mendukung 8 bahasa. Pemakaian: tempatkan sebelum LLM (moderasi input), setelah LLM (moderasi output), atau keduanya. Dua pemakaian menghasilkan distribusi training berbeda — Llama Guard 3 tersedia sebagai satu model yang menangani keduanya.

Llama Guard 3-1B-INT4 (arXiv:2411.17713, 440MB, ~30 token/detik pada CPU mobile) adalah varian edge terkuantisasi.

Llama Guard 4 (April 2025) adalah 12B, multimodal native, diprune dari Llama 4 Scout. Ia menggantikan pendahulu teks 8B maupun visi 11B dengan satu klasifier yang menelan teks + gambar.

### Garak (NVIDIA)

Pemindai kerentanan open-source. Arsitektur:
- **Probe.** Generator serangan untuk halusinasi, kebocoran data, prompt injection, toksisitas, jailbreak. Statis (prompt tetap), dinamis (prompt tergenerasi), adaptif (merespons output target).
- **Detektor.** Menskor output terhadap mode kegagalan ekspektasi — toksik, bocor, terjailbreak.
- **Harness.** Mengelola pasangan probe-detektor, menjalankan kampanye, menghasilkan laporan.

TrustyAI mengintegrasikan Garak dengan shield Llama-Stack (klasifier input Prompt-Guard-86M, klasifier output Llama-Guard-3-8B) untuk evaluasi target-tershield end-to-end. Skor bertier (TBSA) menggantikan pass/fail biner — model bisa lolos pada tier keparahan 3 dan gagal pada tier keparahan 5 pada probe sama.

### PyRIT (Microsoft)

Python Risk Identification Toolkit. Kampanye red-team multi-turn. Dibangun di sekitar:
- **Konverter.** Mentransformasi prompt seed — parafrasa, encode, terjemahan, roleplay.
- **Orkestrator.** Menjalankan kampanye: Crescendo (eskalasi), TAP (branching), RedTeaming (loop kustom).
- **Skor.** LLM-as-judge atau classifier-as-judge.

PyRIT adalah sepupu Garak yang lebih berat. Garak menjalankan ribuan probe single-turn; PyRIT menjalankan kampanye multi-turn mendalam yang dirancang meruntuhkan mode kegagalan spesifik.

### Stack-nya

Taruh Llama Guard pada kedua sisi model. Jalankan Garak tiap malam untuk regresi. Jalankan PyRIT untuk kampanye pra-rilis. Ini konfigurasi default 2026 bagi kebanyakan deployment produksi.

### Jebakan evaluasi

- **Identitas juri.** Ketiga tool bisa memakai juri LLM; kalibrasi juri mendorong ASR yang dilaporkan (Pelajaran 12). Spesifikasikan juri berdampingan tool.
- **Kedaluwarsa probe.** Probe Garak menua seiring model di-patch melawannya. Probe adaptif (berbentuk-PAIR) menua lebih lambat daripada probe statis.
- **FPR Llama Guard pada konten benign.** Versi awal Llama Guard over-flag konten politik dan LGBTQ+; kalibrasi Llama Guard 3/4 membaik tetapi tidak terkalibrasi per-deployment.

### Di mana ini berada dalam Fase 18

Pelajaran 12-15 adalah keluarga serangan. Pelajaran 16 adalah tooling produksi. Pelajaran 17 (WMDP) adalah evaluasi kapabilitas dual-use. Pelajaran 18 adalah framework keamanan frontier yang membungkus tool ini dalam struktur kebijakan.

```figure
al-guard-stack
```

## Gunakan

`code/main.py` membangun klasifier ala-Llama Guard mainan (fitur kata-kunci + semantik atas 14 kategori), harness Garak mainan (loop probe-detektor), dan rantai konverter ala-PyRIT multi-turn. Kamu bisa menjalankan ketiga tool terhadap target mock dan mengamati signature cakupan berbeda.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-red-team-stack.md`. Dengan deskripsi deployment, ia menyebut tool mana dari ketiganya yang tepat, apa yang dikonfigurasi pada masing-masing, dan kadensi regresi apa yang dijalankan.

## Latihan

1. Jalankan `code/main.py`. Bandingkan rate deteksi klasifier ala-Llama-Guard pada serangan single-turn vs multi-turn.

2. Implementasikan probe Garak baru: request harmful terenkode-base64. Ukur deteksinya oleh klasifier ala-Llama-Guard.

3. Perluas rantai konverter ala-PyRIT dengan konverter "terjemahkan ke Prancis, lalu parafrasa". Ukur ulang attack success.

4. Baca daftar kategori-hazard Llama Guard 3. Identifikasi dua kategori di mana data training realistik akan menghasilkan false-positive rate tinggi pada konten developer legit.

5. Bandingkan prinsip desain Garak dan PyRIT. Argumenkan untuk deployment di mana masing-masing adalah tool tepat.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Llama Guard | "the classifier" | Fine-tuned Llama-3.1-8B/4-12B safety classifier with 14 hazard categories |
| Garak | "the scanner" | NVIDIA open-source vulnerability scanner; probes, detectors, harnesses |
| PyRIT | "the campaign tool" | Microsoft multi-turn red-team orchestrator; converters, orchestrators, scoring |
| Prompt-Guard | "the small classifier" | Meta's 86M prompt-injection classifier, paired with Llama Guard |
| TBSA | "tier-based scoring" | Garak's tier-based pass/fail replacing binary outcomes |
| Converter chain | "paraphrase + encode + ..." | PyRIT composition primitive for building multi-step attacks |
| MLCommons hazard categories | "the 14 taxonomies" | Industry-standard taxonomy Llama Guard targets |

## Further Reading

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — the 8B classifier
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — quantized mobile classifier
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — the scanner repo and documentation
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — the campaign toolkit
