# Capstone 07 — Pipeline Fine-Tuning Ujung-ke-Ujung (Data ke SFT ke DPO ke Serve)

> Sebuah model 8B yang dilatih pada datamu sendiri, di-align dengan DPO atas preferensimu sendiri, di-quantize, di-decode secara spekulatif, dan disajikan pada $/1M token yang terukur. Stack terbuka 2026 adalah Axolotl v0.8, TRL 0.15, Unsloth untuk iterasi, GPTQ/AWQ/GGUF untuk kuantisasi, vLLM 0.7 dengan EAGLE-3 untuk serving. Tugas capstone adalah menjalankan seluruh pipeline secara reprodukibel — masuk sebagai YAML, keluar sebagai endpoint yang disajikan — dan menerbitkan model card di bawah Model Openness Framework 2026.

**Type:** Capstone
**Languages:** Python (pipeline), YAML (configs), Bash (scripts)
**Prerequisites:** Phase 2 (ML), Phase 3 (DL), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**Time:** 35 hours

## Masalahnya

Setiap tim AI yang serius pada 2026 menyimpan pipeline fine-tuning yang siap pakai. Bukan karena mereka merilis model base frontier, melainkan karena adaptasi hilir — SFT domain, DPO terhadap preferensi berlabel, draf terdistilasi untuk speculative decoding, serving dengan EAGLE-3 — adalah tempat kemenangan yang bisa diukur berada. Axolotl v0.8 menangani config SFT multi-GPU. TRL 0.15 menangani DPO dan GRPO. Unsloth memberimu iterasi single-GPU yang cepat. vLLM 0.7 dengan EAGLE-3 mendorong throughput decode 2-3x tanpa penurunan kualitas. Perkakasnya bekerja; kerajinannya ada pada YAML-nya, kebersihan data, dan disiplin eval-nya.

Kamu akan menjalankan sebuah base 8B (Llama 3.3, Qwen3, atau Gemma 3) melalui SFT lalu DPO pada data spesifik-tugas, meng-quantize untuk serving, dan mengukur gain terhadap lm-evaluation-harness, RewardBench-2, MT-Bench-v2, dan MMLU-Pro. Kamu akan menghasilkan sebuah model card di bawah Model Openness Framework 2026. Intinya adalah reproduksibilitas — satu perintah menjalankan ulang seluruh pipeline dari ujung ke ujung.

## Konsepnya

Pipeline punya lima tahap. **Data**: dedup (MinHash / Datatrove), filter kualitas (classifier bergaya Nemotron-CC), scrub PII, pemeriksaan split-hygiene terhadap kontaminasi benchmark publik. **SFT**: YAML Axolotl, ZeRO-3 di atas 8xH100, jadwal cosine, sequence packing, 2-3 epoch. **DPO atau GRPO**: config TRL, 1 epoch, pasangan preferensi baik berlabel manusia maupun dinilai model, tuning beta. **Quantize**: GPTQ + AWQ + GGUF untuk fleksibilitas deployment. **Serve**: vLLM 0.7 dengan kepala spekulatif EAGLE-3 (atau SGLang dengan SpecForge), deployment K8s, HPA pada queue-wait.

Ablasi adalah deliverable-nya: SFT-only vs SFT+DPO vs SFT+GRPO pada tiga benchmark spesifik-tugas. Metrik serving: tokens/s pada batch 1 / 8 / 32, acceptance rate EAGLE-3, $/1M token. Safety eval: tingkat kelulusan Llama Guard 4. Model card: evaluasi bias, seed reproduksibilitas, lisensi data.

## Arsitektur

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## Stack

- Data: Datatrove untuk dedup, classifier Nemotron-CC untuk kualitas, Presidio untuk PII
- Base: Llama 3.3 8B, Qwen3 14B, atau Gemma 3 12B
- SFT: Axolotl v0.8 dengan ZeRO-3, Flash Attention 3, sequence packing
- Preference tuning: TRL 0.15 untuk DPO atau GRPO; Unsloth untuk iterasi single-GPU
- Kuantisasi: GPTQ (Marlin), AWQ, GGUF via llama.cpp
- Serving: vLLM 0.7 dengan speculative decoding EAGLE-3 (atau SGLang 0.4 + SpecForge)
- Eval: lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro
- Safety eval: Llama Guard 4, ShieldGemma-2
- Infrastruktur: Kubernetes + NVIDIA device plugin, HPA pada metrik queue-wait
- Observability: W&B untuk training, Langfuse untuk inference

```figure
ce-finetune-stages
```

## Bangun Sendiri

1. **Pipeline data.** Jalankan dedup Datatrove pada korpus mentah. Terapkan classifier kualitas bergaya Nemotron-CC. Presidio meny scrub PII. Tulis split train/val dengan seed yang eksplisit.

2. **Pemeriksaan kontaminasi.** Untuk setiap split validasi, hitung MinHash terhadap test set MMLU-Pro, MT-Bench-v2, RewardBench-2. Tolak tumpang tindih apa pun.

3. **SFT Axolotl.** YAML dengan ZeRO-3, FA3, sequence packing. 2-3 epoch di atas 8xH100. Log ke W&B.

4. **DPO / GRPO TRL.** Ambil checkpoint SFT, jalankan satu epoch DPO pada pasangan preferensi (atau GRPO dengan reward yang bisa diverifikasi pada math/code). Sweep beta.

5. **Quantize.** Hasilkan tiga quant: GPTQ-INT4-Marlin, AWQ-INT4, GGUF-Q4_K_M untuk llama.cpp. Catat ukuran dan throughput nominalnya.

6. **Serve dengan speculative decoding.** Config vLLM 0.7 dengan kepala draft EAGLE-3 yang dilatih via Red Hat Speculators. Ukur acceptance rate dan latensi ekor pada batch 1 / 8 / 32. Laporkan $/1M token versus Anthropic / OpenAI pada eval yang sama.

7. **Matriks eval.** Jalankan lm-eval-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro pada base, SFT-only, SFT+DPO, SFT+GRPO. Hasilkan sebuah tabel.

8. **Safety eval.** Tingkat kelulusan Llama Guard 4 pada dev set. Filter keluaran ShieldGemma-2.

9. **Model card.** Template MOF 2026: data, training, eval, safety, license, section reproduksibilitas berisi YAML dan commit SHA.

## Gunakan

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## Rilis

`outputs/skill-finetuning-pipeline.md` menjelaskan deliverablenya. Sebuah perintah tunggal menjalankan data melalui SFT, DPO, quant, serve, lalu eval, dan mengeluarkan sebuah model card + endpoint yang disajikan.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Eval delta vs base | Measured gain on target tasks (MMLU-Pro, MT-Bench-v2, task-specific) |
| 20 | Pipeline reproducibility | One command reruns end to end with identical seeds |
| 20 | Data hygiene | Dedup rate, PII scrub coverage, contamination check green |
| 20 | Serving efficiency | tokens/s at bs=1/8/32, EAGLE-3 acceptance rate, $/1M tokens |
| 15 | Model card + safety eval | 2026 MOF completeness + Llama Guard 4 pass rate |
| **100** | | |

## Latihan

1. Jalankan SFT-only vs SFT+DPO vs SFT+GRPO pada benchmark spesifik-tugas yang sama. Laporkan metode preferensi mana yang menang dan dengan selisih berapa.

2. Tukar Llama 3.3 8B dengan Qwen3 14B. Ukur $/1M token pada kualitas yang setara.

3. Ukur acceptance rate EAGLE-3 pada data domain versus ShareGPT generik. Laporkan selisihnya dan artinya bagi anggaran latensi.

4. Suntikkan 1% kontaminasi (bocorkan jawaban MMLU-Pro ke data training) lalu jalankan ulang eval. Amati akurasi MMLU-Pro melompat secara tidak realistis. Bangun sebuah gerbang CI contamination-check yang menangkap hal ini.

5. Tambahkan LoRA SFT sebagai alternatif full fine-tune. Ukur jarak kualitasnya pada memori 10x lebih rendah.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Axolotl | "SFT trainer" | Unified YAML-driven trainer for SFT, DPO, and distillation |
| TRL | "Preference tuner" | Hugging Face library for DPO, GRPO, PPO on LLMs |
| GRPO | "Group-relative policy optimization" | DeepSeek R1's RL recipe with verifiable rewards |
| EAGLE-3 | "Speculative decoding draft" | Draft heads that predict N tokens ahead; vLLM verifies with target model |
| MOF | "Model Openness Framework" | 2026 standard for grading model releases on data, code, license |
| Contamination check | "Split hygiene" | MinHash-based detection of test-set leakage into training |
| Acceptance rate | "EAGLE / MTP metric" | Fraction of drafted tokens the target model accepts |

## Bacaan Lanjutan

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — trainer SFT / DPO referensi
- [TRL documentation](https://huggingface.co/docs/trl) — implementasi referensi DPO dan GRPO
- [Unsloth](https://github.com/unslothai/unsloth) — referensi iterasi single-GPU
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — metodologi GRPO
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — stack serving referensi
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — trainer speculative decoding alternatif
- [Model Openness Framework 2026](https://isocpp.org/) — standar penilaian rilis terbuka
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — runner eval kanonikal
