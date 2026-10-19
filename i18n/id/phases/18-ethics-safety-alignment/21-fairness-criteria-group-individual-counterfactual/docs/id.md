# Kriteria Keadilan — Kelompok, Individual, Kontrafaktual

> Tiga keluarga membingkai literatur keadilan. Keadilan kelompok: paritas demografis, equalized odds, conditional use accuracy equality — laju yang setara di rata-rata lintas kelompok terlindungi. Keadilan individual (Dwork et al. 2012): individu serupa menerima keputusan serupa; kondisi Lipschitz pada pemetaan keputusan. Keadilan kontrafaktual (Kusner et al. 2017): sebuah keputusan adil bagi seorang individu jika keputusan itu tidak berubah ketika atribut sensitif diubah secara kontrafaktual. Hasil teoretis 2024 (NeurIPS 2024): ada trade-off CF-vs-accuracy yang melekat; sebuah metode model-agnostic mengubah prediktor optimal-tapi-tidak-adil menjadi satu CF dengan kerugian akurasi yang terbatas. Backtracking counterfactuals (arXiv:2401.13935, Januari 2024): paradigma baru yang menghindari kebutuhan intervensi pada atribut yang dilindungi hukum. Rekonsiliasi filosofis (ICLR Blogposts 2024): dengan graf kausal, memenuhi ukuran keadilan kelompok tertentu mengandaikan keadilan kontrafaktual.

**Type:** Learn
**Languages:** Python (stdlib, three-criteria comparison)
**Prerequisites:** Phase 18 · 20 (bias), Phase 02 (classical ML)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Nyatakan tiga kriteria keadilan kelompok (paritas demografis, equalized odds, conditional use accuracy equality) dan satu hasil ketidakmungkinan.
- Jelaskan keadilan individual melalui formulasi Lipschitz Dwork et al. 2012.
- Jelaskan keadilan kontrafaktual dan ketergantungannya pada graf kausal.
- Jelaskan backtracking counterfactuals dan mengapa mereka menghindari masalah intervensi pada atribut yang terlindungi.

## Masalahnya

Pelajaran 20 tentang mengukur bias. Pelajaran 21 tentang mendefinisikan standar keadilan yang harus dilayani oleh pengukuran tersebut. Ketiga keluarga memberikan standar yang berbeda secara struktural — sebuah model bisa kelompok-adil tapi individual-tidak-adil, kontrafaktual-adil tapi kelompok-tidak-adil. Memilih standar adalah keputusan kebijakan; tidak ada standar yang optimal secara universal.

## Konsepnya

### Keadilan kelompok

- **Demographic parity.** P(Y=1 | A=a) = P(Y=1 | A=a') untuk semua kelompok. Laju penerimaan yang setara.
- **Equalized odds.** P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a'). TPR dan FPR yang setara di antara kelompok.
- **Conditional use accuracy equality.** P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a'). Nilai prediktif yang setara di antara kelompok.

Ketidakmungkinan (Chouldechova, Kleinberg-Mullainathan-Raghavan 2017): ketiganya tidak dapat dipenuhi secara bersamaan di bawah base rate yang tidak setara.

### Keadilan individual

Dwork et al. 2012. Sebuah pemetaan keputusan f secara individual adil terhadap metrik kemiripan spesifik-tugas d jika |f(x) - f(x')| <= L * d(x, x') untuk suatu konstanta Lipschitz L. Individu serupa mendapat keputusan serupa.

Memerlukan definisi d. Pertanyaan kebijakan, bukan statistik.

### Keadilan kontrafaktual

Kusner et al. 2017. Sebuah keputusan secara kontrafaktual adil bagi individu i jika, di bawah model kausal populasi, keputusan itu tidak berubah ketika atribut sensitif i diubah secara kontrafaktual.

Memerlukan sebuah DAG kausal. DAG tersebut adalah pilihan pemodelan. Keadilan kontrafaktual hanya sejustifikasi dengan DAG-nya.

### Trade-off CF-vs-akurasi

NeurIPS 2024 teoretis: ada trade-off yang melekat antara keadilan kontrafaktual dan akurasi prediktif. Sebuah metode model-agnostic dapat mengubah prediktor optimal-tapi-tidak-adil menjadi satu CF, dengan biaya akurasi yang terbatas. Biaya akurasi bergantung pada magnitudo koefisien atribut sensitif dalam prediktor tidak-adil yang optimal.

### Backtracking counterfactuals

arXiv:2401.13935 (Januari 2024). Counterfactual tradisional memerlukan intervensi pada atribut sensitif — "apakah keputusan berubah jika orang ini berbeda gender." Secara hukum, ini bermasalah: atribut terlindungi tidak boleh diintervensi dalam hukum klasifikasi.

Backtracking counterfactuals membalik arahnya: alih-alih mengintervensi atributnya, tanyakan kombinasi fitur aktual individu apa yang akan menghasilkan hasil kontrafaktual tersebut. Ini menghindari keberatan hukum tersebut.

### Rekonsiliasi filosofis

ICLR Blogposts 2024. Dengan graf kausal di tangan, memenuhi ukuran keadilan-kelompok tertentu mengandaikan keadilan kontrafaktual. Ketiga keluarga bukan ortogonal; itu adalah aspek berbeda dari struktur kausal yang sama di baliknya.

Ini tidak menyelesaikan teorema ketidakmungkinan (base rate yang tidak setara tetap mencegah keadilan kelompok secara bersamaan). Tapi ini menunjukkan bahwa oposisi yang tampak antara "kelompok" dan "individual / kontrafaktual" sebagian adalah artefak dari tidak eksplisitnya model kausal.

### Di mana ini cocok di Fase 18

Pelajaran 20 adalah pengukuran bias. Pelajaran 21 adalah definisi keadilan. Pelajaran 22 adalah privasi (differential privacy). Pelajaran 23 adalah watermarking. Ini adalah pelajaran yang berdekatan dengan alokasi, melengkapi Pelajaran 7-11 yang berdekatan dengan penipuan.

```figure
an-fairness-trilemma
```

## Gunakan

`code/main.py` membangun sebuah dataset klasifikasi biner mainan dengan atribut sensitif dan base rate yang tidak setara. Hitung demographic parity, equalized odds, dan conditional use accuracy equality pada classifier sederhana. Amati ketiga metrik yang tidak sepakat. Terapkan re-weighting untuk paritas demografis dan amati biayanya pada dua lainnya.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-fairness-criterion.md`. Diberi sebuah klaim keadilan atau kebijakan, ia mengidentifikasi kriteria mana yang diklaim, apakah model dapat memenuhi kriteria yang tersisa di bawah base rate tidak-setara yang diklaim, dan DAG kausal apa yang menjadi dasar klaim tersebut.

## Latihan

1. Jalankan `code/main.py`. Laporkan tiga metrik kelompok pada data default. Terapkan re-weighting ber-target paritas demografis dan laporkan ulang.

2. Implementasikan metrik keadilan individual Dwork et al. 2012 menggunakan L2 pada fitur non-sensitif. Laporkan berapa banyak pasangan yang melanggar Lipschitz dengan konstanta L=1.

3. Baca Kusner et al. 2017. Bangun sebuah DAG kausal sederhana dua-fitur untuk penilaian resume dan identifikasi kondisi keadilan kontrafaktual yang diandaikannya.

4. Paper backtracking-counterfactuals 2024 menghindari intervensi pada atribut terlindungi. Jelaskan sebuah skenario di mana hal ini penting untuk kepatuhan hukum.

5. Rekonsiliasi ICLR 2024 berargumen bahwa keadilan kelompok dan kontrafaktual adalah aspek dari struktur yang sama. Pilih dua dari tiga kriteria di `code/main.py` dan nyatakan asumsi kausal yang akan membuatnya setara.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Demographic parity | "equal rates" | P(Y=1 | A=a) equal across groups |
| Equalized odds | "equal TPR/FPR" | Equal true-positive and false-positive rates across groups |
| Conditional use accuracy | "equal PPV/NPV" | Equal predictive values across groups |
| Individual fairness | "Lipschitz condition" | Similar individuals get similar decisions |
| Counterfactual fairness | "causal alteration invariance" | Decision unchanged under counterfactual attribute alteration |
| Backtracking counterfactual | "explain via actuals" | Counterfactual reasoned backward from outcome, not forward from attribute |
| Impossibility theorem | "the three conflict" | Chouldechova / KMR 2017: group criteria mutually exclusive under unequal base rates |

## Bacaan Lanjutan

- [Dwork et al. — Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913) — individual fairness
- [Kusner, Loftus, Russell, Silva — Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856) — counterfactual fairness
- [Chouldechova — Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056) — impossibility
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935) — new paradigm for protected-attribute interventions
