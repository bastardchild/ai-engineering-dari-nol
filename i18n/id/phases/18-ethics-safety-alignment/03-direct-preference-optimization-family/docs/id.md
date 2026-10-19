# Keluarga Direct Preference Optimization

> Rafailov et al. (2023) menunjukkan optimum RLHF punya bentuk closed form dalam data preferensi, sehingga kamu bisa melewati reward model eksplisit dan mengoptimasi kebijakan langsung. Insight itu melahirkan keluarga — IPO, KTO, SimPO, ORPO, BPO — masing-masing memperbaiki mode kegagalan DPO. Pada 2026, algoritma alignment langsung dipakai pada lebih banyak run pasca-training frontier daripada PPO. Tetapi kurva over-optimasi dari Pelajaran 2 tetap berlaku: DAA tidak lolos dari Goodhart, mereka hanya memindahkan tempat gigitannya.

**Type:** Learn
**Languages:** Python (stdlib, six-variant preference-loss comparator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking), Phase 10 · 08 (DPO basics)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Turunkan bentuk closed form DPO dari optimum RLHF-dengan-KL.
- Nyatakan mode kegagalan yang diperbaiki masing-masing IPO, KTO, SimPO, ORPO, BPO pada DPO.
- Bedakan "selisih reward implisit" dari "kekuatan preferensi" dan jelaskan mengapa pemetaan identitas IPO penting.
- Jelaskan mengapa Rafailov et al. (NeurIPS 2024) membuktikan DAA tetap over-optimize meski tanpa RM eksplisit.

## Masalahnya

Objektif RLHF (Pelajaran 1):

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

punya optimum yang diketahui:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

Sehingga reward secara implisit didefinisikan oleh rasio kebijakan optimal terhadap referensi:

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

Substitusikan ini ke likelihood preferensi Bradley-Terry dan fungsi partisi `Z(x)` ter-cancel karena hanya bergantung pada `x`. Yang tersisa adalah loss pada parameter kebijakan saja — tanpa reward model. Itulah DPO.

Kerutannya: derivasi mengasumsikan optimum terjangkau, data preferensi in-distribution, dan kebijakan referensi adalah jangkar mode sejati. Tak satu pun berlaku persis. Setiap anggota keluarga memperbaiki asumsi yang dilanggar yang berbeda.

## Konsepnya

### DPO (Rafailov et al., 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

Apa yang bisa salah:

- Selisih reward implisit `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` tak terbatas. Preferensi mungil bisa menghasilkan selisih arbitrer besar.
- Loss mendorong log-prob chosen dan rejected ke arah berlawanan. Ia bisa mendorong log-prob absolut chosen turun selama rejected jatuh lebih cepat. Ini fenomena Degraded Chosen Response.
- Preferensi out-of-distribution (pasangan langka vs langka) menghasilkan reward implisit arbitrer.

### IPO (Azar et al., 2024)

Identity Preference Optimization mengganti log-sigmoid dengan pemetaan identitas pada probabilitas preferensi. Loss menjadi squared-error pada target terbatas:

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

Margin dibatasi `1/(2 beta)`. Kekuatan preferensi dan selisih reward-implisit proporsional. Tanpa blow-up.

### KTO (Ethayarajh et al., 2024)

Kahneman-Tversky Optimization membuang struktur berpasangan sepenuhnya. Dengan satu output berlabel dan sinyal biner "desirable" atau "undesirable", ia memetakan ke utilitas prospect-theory:

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

dengan bobot berbeda untuk gain dan loss (loss aversion). Manfaat: kamu bisa memakai data tak berpasangan, yang jauh lebih melimpah.

### SimPO (Meng et al., 2024)

Simple Preference Optimization menyelaraskan sinyal training dengan generasi. Hapus kebijakan referensi sepenuhnya dan normalisasi log-likelihood menurut panjang:

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

dengan margin `gamma` untuk stabilisasi. Normalisasi panjang menghilangkan insentif mengeksploitasi mode kegagalan bias-panjang DPO (`y_w` lebih panjang memberi selisih log-prob lebih besar by construction).

### ORPO (Hong et al., 2024)

Odds-Ratio Preference Optimization menambah term preferensi ke negative log-likelihood SFT standar:

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

Tanpa kebijakan referensi — term SFT adalah regularizer. Latih satu tahap dari model dasar ke model aligned. Tanpa checkpoint SFT terpisah.

### BPO (submisi ICLR 2026, OpenReview id=b97EwMUWu7)

Mengidentifikasi problem Degraded Chosen Responses: DPO menjaga ranking `y_w > y_l` tetapi log-prob absolut `y_w` bisa turun. BPO menambah koreksi satu-baris yang memenalti gerakan turun pada respons chosen. Dilaporkan akurasi +10,1% pada Llama-3.1-8B-Instruct pada penalaran matematika atas DPO.

### Hasil universal: DAA tetap over-optimize

Rafailov et al. "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms" (NeurIPS 2024) melatih kebijakan dengan DPO, IPO, SLiC pada banyak dataset lintas budget KL. Kurva gold-reward-vs-KL punya bentuk puncak-dan-runtuh Gao et al. yang sama. Reward implisit meng-query sampel out-of-distribution selama training; regularisasi KL tidak menstabilkan ini.

DAA tidak lolos dari Goodhart. Mereka mengubah permukaan tempat ia menggigit dari "reward model over-optimized" menjadi "rasio kebijakan referensi over-optimized." Perbaikan universal — data lebih baik, ensemble, early stopping — berlaku untuk keduanya.

### Memilih di antaranya (2026)

- Kalau kamu punya data preferensi berpasangan besar: DPO dengan beta konservatif, SimPO kalau bias panjang tampak.
- Kalau kamu punya feedback biner tak berpasangan: KTO.
- Kalau kamu menginginkan pipeline satu-tahap dari model dasar: ORPO.
- Kalau kamu melihat log-prob chosen terdegradasi dalam log DPO: BPO.
- Kalau kekuatan preferensi bervariasi lebar dan DPO saturasi: IPO.

Setiap lab menjalankan kelimanya pada baterai dan memilih pemenang per task. Tidak ada alasan optimum sama untuk penalaran matematika dan keamanan.

```figure
dpo-margin
```

## Gunakan

`code/main.py` membandingkan enam loss (DPO, IPO, KTO, SimPO, ORPO, BPO) pada dataset preferensi mainan di mana kekuatan preferensi sejati bervariasi per pasangan. Tiap loss dioptimasi terhadap 500-pasangan sampel yang sama dengan kebijakan softmax kecil. Mem-plot win rate final, drift log-prob-chosen, dan sebaran reward-implisit per metode.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-preference-loss-selector.md`. Dengan statistik dataset (berpasangan vs tak berpasangan, kekuatan preferensi variabel vs seragam, distribusi panjang) dan target (satu-tahap atau SFT-lalu-preferensi), ia merekomendasikan loss preferensi dan melaporkan mode kegagalan yang dilindunginya.

## Latihan

1. Jalankan `code/main.py`. Laporkan penurunan log-prob-chosen final untuk DPO dan BPO. BPO seharusnya mempertahankan probabilitas absolut chosen lebih tinggi — verifikasi ini.

2. Ubah data preferensi agar semua pasangan berkekuatan sama. Metode mana dari keenamnya yang paling robust? Mana yang terdegradasi? Jelaskan keunggulan IPO di sini.

3. Buat respons rejected rata-rata 2x lebih panjang daripada chosen. Tanpa mengubah apa pun, tunjukkan eksploitasi panjang DPO secara numerik dan perbaikan SimPO.

4. Rafailov et al. (NeurIPS 2024) mengklaim DAA over-optimize. Reproduksi versi satu-titik: plot divergensi KL chosen-minus-rejected dan amati over-optimasi pada DPO pada beta besar.

5. Baca abstrak paper BPO (OpenReview b97EwMUWu7). Tuliskan koreksi satu-baris yang ditambahkan BPO ke DPO. Konfirmasi terhadap implementasi di `code/main.py`.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| DPO | "RLHF without a reward model" | Loss derived from the closed-form RLHF optimum; policy parameters only |
| Implicit reward | "the log-ratio" | `beta * log(pi(y\|x) / pi_ref(y\|x))` — the DPO-implied reward |
| IPO | "bounded DPO" | Replaces log-sigmoid with identity; implicit reward gap capped by `1/(2 beta)` |
| KTO | "unpaired DPO" | Prospect-theory utility over single labels with loss aversion |
| SimPO | "reference-free DPO" | Length-normalized log-likelihood + margin; no reference policy |
| ORPO | "one-stage DPO" | NLL + odds-ratio preference term; trains from base model in one pass |
| BPO | "chosen-preserving DPO" | DPO plus a penalty for decreasing the chosen response's absolute log-prob |
| Degraded Chosen | "chosen goes down" | DPO decreases chosen log-prob so long as rejected falls faster |
| DAA | "direct alignment algorithm" | Any preference-loss method that skips an explicit RM |

## Further Reading

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
