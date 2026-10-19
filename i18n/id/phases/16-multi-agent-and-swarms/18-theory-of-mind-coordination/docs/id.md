# Theory of Mind dan Koordinasi Emergen

> Li et al. (arXiv:2310.10701) menunjukkan bahwa agent LLM dalam sebuah permainan kooperatif memperlihatkan **emergent high-order Theory of Mind** (ToM) — penalaran tentang apa yang diyakini agent lain mengenai keyakinan agen ketiga — tetapi gagal pada perencanaan horizon panjang karena manajemen konteks dan halusinasi. Riedl (arXiv:2510.05174) mengukur sinergis orde-tinggi di seluruh sebuah populasi dan menemukan bahwa **hanya** kondisi prompt ToM yang menghasilkan diferensiasi terikat-identitas dan komplemen berarah-tujuan; LLM berkapasitas lebih rendah hanya menunjukkan emergence semu. Artinya, emergence koordinasi bersifat kondisional-terhadap-prompt dan bergantung-model, bukan gratis. Pelajaran ini mengimplementasikan sebuah agent ToM-aware minimal, menjalankan sebuah tugas kooperatif dengan dan tanpa prompt ToM, dan mengukur delta koordinasinya terhadap protokol Riedl 2025.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 17 (Generative Agents)
**Time:** ~75 minutes

## Masalahnya

Koordinasi multi-agent sering terlihat ajaib: agent membagi tugas, saling mengantisipasi, menghindari duplikasi. Biasanya "emergence" ini adalah artefak dari prompt engineering — seseorang menyuruh para agent untuk "berkoordinasi." Hapus prompt-nya, hilang koordinasinya.

Temuan Riedl 2025 lebih ketat: di bawah kondisi terkendali, koordinasi hanya muncul ketika agent di-prompt untuk menalar tentang **pikiran agent lain** (ToM). Tanpa prompt ToM, bahkan model yang kuat menampilkan pola koordinasi yang tidak lolos dari kontrol statistik. Ini penting untuk produksi: tim mengirimkan fitur "koordinasi multi-agent" yang bergantung-prompt dan rapuh.

Pelajaran ini memperlakukan ToM sebagai sebuah kapasitas spesifik (menalar tentang keyakinan atas keyakinan), membangun sebuah agent ToM-aware minimal, dan mengukur seperti apa koordinasi yang nyata vs seperti apa dress-up prompt.

## Konsepnya

### Apa arti ToM

Psikologi perkembangan: anak 3 tahun berpikir dunia batin siapa pun sama dengannya. Anak 5 tahun mengerti bahwa orang lain punya keyakinan berbeda. Anak 7 tahun menalar tentang keyakinan atas keyakinan ("dia berpikir bahwa saya berpikir bola itu di bawah cangkir"). Inilah ToM orde-nol, orde-pertama, dan orde-kedua.

Bagi agent LLM, ordo ToM dipetakan ke:

- **Orde-nol:** tanpa model tentang orang lain. Agent hanya bertindak berdasarkan observasinya sendiri.
- **Orde-pertama:** agent punya sebuah model atas keyakinan setiap agent lain. "Alice percaya X."
- **Orde-kedua:** agent memodelkan keyakinan rekursif. "Alice percaya bahwa Bob percaya X."

Li et al. 2023 menemukan bahwa ToM orde-pertama dan orde-kedua muncul pada agent LLM dalam permainan kooperatif tetapi merosot dengan horizon panjang dan komunikasi yang tidak andal.

### Tes Sally-Anne, secara singkat

Sebuah tes false-belief tahun 1985: Sally meletakkan sebuah marmer di keranjang A, lalu pergi. Anne memindahkannya ke keranjang B. Di mana Sally akan mencarinya ketika kembali? Seorang anak dengan ToM orde-pertama menjawab keranjang A (keyakinan Sally berbeda dari kenyataan). Anak tanpa ToM menjawab keranjang B.

LLM era GPT-4 lulus tes bergaya Sally-Anne ketika diajukan secara polos. Mereka gagal ketika narasinya panjang, adegannya berubah beberapa kali, atau pertanyaannya dirumuskan secara tidak langsung. Itulah keadaan praktis ToM pada LLM produksi di 2026.

### Pengukuran koordinasi milik Riedl

Riedl (arXiv:2510.05174) membangun sebuah tes berskala populasi: N agent, sebuah tujuan kooperatif, kondisi prompt yang bervariasi. Ukurlah:

1. **Diferensiasi terikat-identitas.** Apakah agent mengembangkan pembedaan peran yang stabil dari waktu ke waktu?
2. **Komplemen berarah-tujuan.** Apakah aksi-aksi agent saling melengkapi (subtask berbeda) alih-alih menduplikasi?
3. **Sinergis orde-tinggi.** Sebuah ukuran statistik apakah grup mencapai sesuatu yang tidak bisa dicapai subset mana pun.

Hasilnya: hanya di bawah kondisi prompt ToM ketiga metrik itu menghasilkan sinyal di atas baseline. Tanpa prompt ToM, metrik berputar di sekitar peluang (chance) untuk model berkapasitas sedang. Model besar menunjukkan sebagian koordinasi tanpa prompt ToM yang eksplisit tetapi efeknya lebih kecil daripada dengan prompting eksplisit.

### Ilusi koordinasi

Tanpa kontrol statistik, "emergence koordinasi" dalam demo sering mencerminkan:

- Prompt engineering yang membakar koordinasi ke dalamnya (system prompt yang berkata "bekerjalah bersama").
- Bias pengamat (kita melihat pola yang kita harapkan).
- Seleksi pasca-hoc atas run yang berhasil.

Sistem produksi yang memasarkan "emergence koordinasi" tanpa sinyal yang terukur sebaiknya diperlakukan sebagai pemasaran. Ukur dulu sebelum mengklaim.

### Sebuah agent ToM-aware minimal

Strukturnya:

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions
```

Atribut `other_models` adalah state ToM-nya. ToM orde-pertama hanya menyimpan satu tingkat. Orde-kedua menambahkan `other_models[i][other_models_of_j]` — apa yang menurut saya agent i percaya bahwa agent j percaya.

### Mengapa horizon panjang merugikan

Li et al. mendokumentasikan: batas konteks menyebabkan agent lupa keyakinan mana milik siapa. Halusinasi menambahkan keyakinan palsu ke model agent lain. Keduanya menghasilkan galat "saya kira dia kira X" yang menumpuk seiring waktu.

Mitigasi yang didokumentasikan dalam makalah itu dan dalam lanjutan 2024-2026:

- **State ToM eksplisit di prompt.** Format terstruktur: `{agent_id: belief_list}`. Memaksa retrieval mempertahankan pengikatan identitas-keyakinan.
- **Rantai penalaran lebih pendek.** Lebih sedikit pembaruan ToM per giliran mengurangi halusinasi yang menumpuk.
- **Store ToM eksternal.** Pertahankan model di luar konteks LLM; suntikkan hanya bagian yang relevan per giliran.

### Di mana ToM gagal di produksi

- **Setting adversarial.** Agent dengan ToM yang bagus lebih mudah dimanipulasi (kamu bisa memodelkan apa yang mereka modelkan tentangmu, lalu mengeksploitasi).
- **Tim heterogen.** Ketika modelnya berbeda, model ToM yang bekerja untuk satu lawan tidak tergeneralisasi.
- **Tugas yang bergantung ground-truth.** ToM membahas keyakinan; jika kebenarannya bergantung pada fakta, ToM bisa menjadi pengalih perhatian.

### Koordinasi yang benar-benar bisa kamu ukur

Tiga sinyal praktis bahwa koordinasi sebuah tim itu nyata, bukan sekadar dress-up prompt:

1. **Komplemen seiring waktu.** Pada sebuah tugas multi-giliran, apakah aksi-aksi agent menutupi sub-tugas yang saling lepas?
2. **Antisipasi.** Apakah aksi agent A pada giliran T+1 bergantung pada sebuah prediksi atas aksi B pada T+2 yang ternyata benar?
3. **Koreksi.** Ketika A salah-baca keyakinan B pada giliran T, apakah A mengoreksi pada giliran T+2?

Ketiganya bisa diukur dalam sebuah sistem multi-agent yang tercatat. Inilah versi substansial dari narasi "koordinasi".

```figure
sw-theory-of-mind
```

## Bangun Sendiri

`code/main.py` mengimplementasikan:

- `ToMAgent` — melacak keyakinan sendiri dan model keyakinan per agent lain.
- Sebuah tugas kooperatif: tiga agent harus mengumpulkan tiga token dari tiga kotak; setiap kotak bisa menampung satu token. Agent tidak bisa berkomunikasi; mereka menyimpulkan niat dari aksi satu sama lain.
- Dua konfigurasi: `zeroth_order` (tanpa ToM) dan `first_order` (ToM dengan model keyakinan satu-tingkat).
- Pengukuran atas 200 percobaan acak: tingkat penyelesaian, tingkat duplikasi (dua agent menyasar kotak yang sama), rata-rata giliran sampai selesai.

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan: agent orde-nol menduplikasi usaha pada tingkat ~35% dan menyelesaikan ~60% percobaan dalam 10 giliran. Agent ToM orde-pertama menduplikasi ~5% dan menyelesaikan ~95%. Delta itulah efek koordinasi yang terukur.

## Gunakan

`outputs/skill-tom-auditor.md` adalah sebuah skill yang mengaudit klaim "emergence koordinasi" dari sebuah sistem multi-agent. Memeriksa dress-up prompt, signifikansi statistik terhadap kontrol, dan komplemen yang terukur.

## Rilis

Checklist klaim koordinasi:

- **Kondisi kontrol.** Sebuah versi sistem-mu tanpa prompt koordinasi. Ukur keduanya.
- **Uji statistik.** Apakah perbedaan antara sistem dan kontrol signifikan pada `p < 0.05` dengan metrik-mu?
- **Ukuran komplemen.** Keterlepasan-aksi seiring waktu, bukan sekadar keberhasilan akhir.
- **Log kasus kegagalan.** Ketika agent berkoordinasi dengan buruk, seperti apa state ToM-nya?
- **Pengungkapan kapasitas model.** Jika efeknya lenyap pada model yang lebih kecil, katakan begitu.

## Latihan

1. Jalankan `code/main.py`. Konfirmasi ToM orde-pertama mengurangi tingkat duplikasi ~7x. Apakah selisihnya bertahan ketika kamu menskala ke 5 agent dan 5 kotak?
2. Implementasikan ToM orde-kedua (agent A memodelkan apa yang B pikirkan tentang C). Apakah ini membaik dibanding orde-pertama? Pada tugas apa?
3. Suntikkan sebuah **hallucination** ke state ToM: balikkan secara acak satu keyakinan per giliran. Seberapa besar ini merusak performa orde-pertama?
4. Baca Li et al. (arXiv:2310.10701). Reproduksi temuan "long-horizon degradation": ketika giliran tumbuh dari 10 ke 30, bagaimana performa ToM orde-pertamamu berubah?
5. Baca Riedl 2025 (arXiv:2510.05174). Implementasikan statistik sinergis orde-tinggi pada log simulasimu. Apakah efeknya ada tanpa kondisi prompt ToM?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Theory of Mind | "Understanding others' minds" | The capacity to model another agent's beliefs. Graded by order (0, 1, 2+). |
| Sally-Anne test | "The false-belief test" | 1985 developmental psychology; LLMs pass plain versions, fail complex ones. |
| First-order ToM | "A believes X" | Modeling one other's beliefs about facts. |
| Second-order ToM | "A believes B believes X" | Recursive modeling one level deeper. |
| Identity-linked differentiation | "Stable roles over time" | Riedl's metric: roles persist, not random. |
| Goal-directed complementarity | "Disjoint actions" | Agents target different subtasks, not the same one. |
| Higher-order synergy | "Group exceeds any subset" | Riedl's statistical measure for real coordination. |
| Coordination illusion | "It looks coordinated" | Prompt-dressed appearance of coordination without measurable signal. |

## Bacaan Lanjutan

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — emergent ToM in cooperative games; long-horizon failure modes
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — population-scale measurement; ToM prompting is the load-bearing condition
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — the 1978 origin of the ToM concept
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://doi.org/10.1016/0010-0277(85)90022-8) — the Sally-Anne paper (1985)
