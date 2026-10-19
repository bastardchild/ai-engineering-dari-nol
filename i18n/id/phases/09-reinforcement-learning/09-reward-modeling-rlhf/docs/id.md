# Reward Modeling & RLHF

> Manusia tidak bisa menulis fungsi reward untuk "respons asisten yang baik," tetapi mereka bisa membandingkan dua respons dan memilih yang lebih baik. Fit sebuah reward model ke perbandingan itu, lalu RL-kan bahasa model terhadapnya. Christiano 2017. InstructGPT 2022. Resep yang mengubah GPT-3 menjadi ChatGPT. Di 2026 ia sebagian besar digantikan oleh DPO — tetapi model mentalnya bertahan.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45 minutes

## Masalahnya

Kamu melatih bahasa model pada objective prediksi-token-berikutnya. Ia menulis Inggris yang gramatikal. Ia juga berbohong, bertele-tele, dan menolak untuk menolak. Kamu tidak bisa memperbaiki ini dengan lebih banyak pretraining — teks web adalah masalahnya, bukan obatnya.

Kamu ingin *scalar reward* yang mengatakan "respons A lebih baik dari respons B untuk instruksi X." Menulis fungsi reward itu dengan tangan adalah mustahil. "Kepuasaan bantuan" bukan ekspresi tertutup-atas token. Tetapi manusia bisa membandingkan dua output dan menandai preferensi. Itu murah dikumpulkan dalam skala besar.

RLHF (Christiano et al. 2017; Ouyang et al. 2022) mengubah preferensi menjadi reward model, lalu mengoptimalkan LM via PPO terhadap reward itu. Dalam tiga langkah: SFT → RM → PPO. Ini adalah resep yang meluncurkan ChatGPT, Claude, Gemini, dan setiap aligned-LLM lain di 2023–2025.

Di 2026 langkah PPO sebagian besar digantikan oleh DPO (Phase 10 · 08) karena lebih murah dan hampir sebagus untuk alignment tuning. Tetapi bagian *reward model* masih mendasari setiap Best-of-N sampler, setiap pipeline RL-from-verifiable-rewards, dan setiap reasoning model yang memakai process reward model. Pahami RLHF dan kamu memahami seluruh alignment stack.

## Konsepnya

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**Stage 1: Supervised Fine-Tuning (SFT).** Mulai dari base model yang sudah di-pretrain. Fine-tune pada demonstrasi yang ditulis manusia tentang perilaku target (respons mengikuti instruksi, balasan yang membantu, dll.). Hasil: model `π_SFT` yang *cenderung ke perilaku baik* tetapi masih memiliki action space tanpa batas.

**Stage 2: training Reward Model.**

- Kumpulkan pasangan respons `(y_+, y_-)` untuk prompts `x`, dilabeli manusia sebagai "y_+ lebih dipilih daripada y_-."
- Latih reward model `R_φ(x, y)` untuk memberi skor lebih tinggi ke `y_+`.
- Loss: **Bradley-Terry pairwise logistic**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ adalah sigmoid. Perbedaan reward menyiratkan log-odds preferensi. BT telah menjadi standar sejak 1952 (Bradley-Terry) dan adalah pilihan dominan di RLHF modern.

- `R_φ` biasanya diinisialisasi dari model SFT dengan scalar head di atasnya. Transformer backbone yang sama; satu linear layer menghasilkan reward.

**Stage 3: PPO terhadap RM dengan KL penalty.**

- Inisialisasi policy yang dapat dilatih `π_θ` dari `π_SFT`. Simpan *reference* beku `π_ref = π_SFT`.
- Reward di akhir respons `y`:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL penalty mencegah `π_θ` menyimpang sembarangan dari `π_SFT` — ia adalah *regularizer*, bukan trust region keras. `β` biasanya `0.01`-`0.05`.
- Jalankan PPO (Lesson 08) dengan reward ini. Advantages dihitung pada trajectory tingkat token, tetapi RM hanya memberi skor ke seluruh respons.

**Mengapa KL?** Tanpanya, PPO akan dengan senang hati menemukan strategi reward-hacking — RM hanya dilatih pada completion in-distribution. Respons out-of-distribution bisa mendapat skor lebih tinggi dari mana pun yang ditulis manusia. KL menjaga `π_θ` dekat manifold tempat RM dilatih. Ini adalah knob paling penting dalam RLHF.

**Status 2026:**

- **DPO** (Rafailov 2023): aljabar tertutup-atas meruntuhkan Stage 2+3 menjadi satu loss supervised atas data preferensi. Tanpa RM, tanpa PPO. Kualitas sama di benchmark alignment dengan sebagian kecil compute. Dibahas di Phase 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO dengan baseline relatif-grup alih-alih critic, reward dari *verifier* (kode berjalan / jawaban matematika cocok) alih-alih RM yang dilatih manusia. Dominan untuk reasoning models. Dibahas di Phase 9 · 12.
- **Process reward models (PRMs):** memberi skor ke solusi parsial (setiap langkah penalaran), digunakan di varian RLHF dan GRPO untuk penalaran.
- **Constitutional AI / RLAIF:** gunakan LLM yang selaras untuk menghasilkan preferensi alih-alih manusia. Menskalakan anggaran preferensi.

```figure
reward-model
```

## Bangun Sendiri

Pelajaran ini menggunakan "prompts" dan "respons" sintetis kecil yang direpresentasikan sebagai string. RM adalah scorer linear di atas representasi bag-of-tokens. Tanpa LLM sungguhan — *bentuk* pipeline-nya yang penting, bukan skalanya. Lihat `code/main.py`.

### Langkah 1: data preferensi sintetis

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

Di RLHF sungguhan ini digantikan oleh labeler manusia. Bentuknya — `(prompt, preferred_response, rejected_response)` — identik.

### Langkah 2: reward model Bradley-Terry

Skor linear: `R(x, y) = w · bag(y)`. Latih untuk meminimalkan BT pairwise log-loss:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

Setelah beberapa ratus update, `w` memberi bobot positif ke token kata-baik dan negatif ke kata-buruk.

### Langkah 3: policy ala PPO di atas RM

Policy mainan kita menghasilkan satu token dari vocabulary. Kami memberi skor token di bawah RM, menghitung `log π_θ(token | prompt)`, menambahkan penalty KL-to-reference, dan menerapkan clipped surrogate PPO.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Langkah 4: pantau KL-nya

Lacak `KL(π_θ || π_ref)` rata-rata setiap update. Jika merayap melewati `~5-10` policy telah menyimpang jauh dari `π_SFT` — `β` yang terlalu rendah (drift naik) atau reward hacking sedang dimulai. Ini adalah diagnostik utama di RLHF sungguhan.

### Langkah 5: resep produksi dengan TRL

Setelah kamu memahami pipeline mainan, berikut loop yang sama seperti yang ditulis pengguna library sungguhan. [TRL](https://huggingface.co/docs/trl) Hugging Face adalah implementasi referensi — `RewardTrainer` untuk Stage 2 dan `PPOTrainer` (dengan KL-to-reference bawaan) untuk Stage 3.

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

Tiga hal yang dilakukan library untukmu. `adap_kl_ctrl=True` mengimplementasikan jadwal β-adaptif: jika KL yang diamati melebihi `target_kl`, β digandakan; jika di bawah setengah, β di-setengah. Model reference dibekukan secara konvensi — kamu tidak boleh secara tidak sengaja berbagi parameter dengan `policy`. Dan value head hidup di backbone yang sama dengan policy (`AutoModelForCausalLMWithValueHead` menempelkan scalar MLP head), itulah sebabnya TRL melaporkan `policy/kl` dan `value/loss` secara terpisah.

## Jebakan

- **Over-optimization / reward hacking.** RM tidak sempurna; `π_θ` menemukan completion adversarial yang skornya tinggi tetapi buruk. Gejala: reward naik tanpa batas sementara skor eval manusia datar atau turun. Perbaikan: berhenti lebih awal, naikkan `β`, perluas data training RM.
- **Length hacking.** RM yang dilatih pada respons yang membantu sering secara implisit memberi penghargaan panjang. Policy belajar untuk mengulur respons. Mitigasi: reward ternormalisasi-panjang, atau RLAIF dengan RM yang peka-panjang.
- **RM terlalu kecil.** RM harus setidaknya seukuran policy. RM kecil tidak bisa memberi skor yang setia ke output policy.
- **Tuning KL.** `β` terlalu rendah → menyimpang dan reward hacking. Terlalu tinggi → policy nyaris tidak berubah. Trik standarnya adalah `β` *adaptif* yang menargetkan KL tetap per langkah.
- **Noise data preferensi.** ~30% label manusia berisik atau ambigu. Kalibrasi dengan melatih RM pada data yang difilter kesepakatan atau gunakan suhu pada BT.
- **Masalah off-policy.** Data PPO sedikit off-policy setelah epoch pertama. Pantau clip fraction seperti di Lesson 08.

## Gunakan

RLHF di 2026 berlapis:

| Layer | Target | Method |
|-------|--------|--------|
| Instruction following, helpfulness, harmlessness | Alignment | DPO (Phase 10 · 08) preferred over RLHF-PPO. |
| Reasoning correctness (math, code) | Capability | GRPO with verifier reward (Phase 9 · 12). |
| Long-horizon multi-step tasks | Agentic | PPO / GRPO with process reward models over steps. |
| Safety / refusal behavior | Safety | RLHF-PPO with separate safety RM, or Constitutional AI. |
| Best-of-N at inference | Fast alignment | Use RM at decode time; no policy training needed. |
| Reward distillation | Inference compute | Train a small "reward head" on top of a frozen LM. |

RLHF adalah *metode* di 2022–2024. Di 2026, pipeline alignment produksi berbasis DPO-dulu, PPO-hanya untuk langkah yang intensif-RM atau kritis-keselamatan.

## Rilis

Simpan sebagai `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## Latihan

1. **Mudah.** Latih reward model Bradley-Terry di `code/main.py` pada 500 pasangan preferensi sintetis. Ukur pairwise accuracy pada 100 pasangan held-out. Harus melampaui 90%.
2. **Sedang.** Jalankan loop PPO-RLHF mainan dengan `β ∈ {0.0, 0.1, 1.0}`. Untuk masing-masing, plot skor RM vs KL-to-reference selama update. Run mana yang reward-hack?
3. **Sulit.** Implementasikan DPO (closed-form preference-likelihood loss) pada data preferensi yang sama dan bandingkan dengan pipeline RLHF-PPO dalam compute yang digunakan dan skor RM akhir yang dicapai.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| RLHF | "Alignment RL" | Three-stage SFT + RM + PPO pipeline (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | "The scoring net" | Learned scalar function fit to pairwise preferences via Bradley-Terry. |
| Bradley-Terry | "Pairwise logistic loss" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; the standard RM objective. |
| KL penalty | "Stay near the reference" | `β · KL(π_θ \|\| π_ref)` in the reward; the anti-reward-hacking regularizer. |
| Reward hacking | "Goodhart's law" | Policy exploits RM flaws; symptoms: reward up, human eval flat. |
| RLAIF | "AI-labeled preferences" | RLHF where labels come from another LM instead of humans. |
| PRM | "Process Reward Model" | Scores partial reasoning steps; used in reasoning pipelines. |
| Constitutional AI | "Anthropic's method" | AI-generated preferences guided by explicit rules. |

## Bacaan Lanjutan

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — the paper that started RLHF.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — the recipe behind ChatGPT.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — earlier RLHF for summarization.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO; the post-RLHF default in 2026.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF and self-critique loop.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — the HH paper.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — production `RewardTrainer` and `PPOTrainer`. Read the trainer source for the adaptive-KL and value-head details.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — the canonical walk-through of the three-stage pipeline with diagrams.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — the library; `examples/` has end-to-end RLHF scripts for Llama, Mistral, and Qwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — the reward-hypothesis view; essential prerequisite for thinking about reward hacking.
