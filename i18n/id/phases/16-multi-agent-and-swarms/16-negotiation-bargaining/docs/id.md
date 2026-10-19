# Negosiasi dan Tawar-Menawar

> Agent bernegosiasi atas sumber daya, harga, alokasi tugas, dan istilah. Rangkaian benchmark 2026 jelas: NegotiationArena (arXiv:2402.05863) menunjukkan LLM bisa meningkatkan payoff ~20% lewat manipulasi persona ("keputusasaan"); "Measuring Bargaining Abilities" (arXiv:2402.15813) menunjukkan buyer lebih sulit daripada seller dan skala tidak menolong — **OG-Narrator** mereka (generator penawaran deterministik + narator LLM) mendorong tingkat deal dari 26.67% menjadi 88.88%; Large-Scale Autonomous Negotiation Competition (arXiv:2503.06416) menjalankan ~180k negosiasi dan menemukan bahwa agent **chain-of-thought-concealing** menang dengan menyembunyikan penalaran dari lawan; Bhattacharya et al. 2025 dengan metrik Harvard Negotiation Project menempatkan Llama-3 sebagai yang paling efektif, Claude-3 paling agresif, GPT-4 paling adil. Pelajaran ini mengimplementasikan Contract Net Protocol (leluhur FIPA, Pelajaran 02), merangkai sebuah buyer/seller bergaya LLM, menjalankan sebuah dekomposisi bergaya OG-Narrator, dan mengukur bagaimana tingkat deal berubah dengan setiap pilihan struktural.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 02 (FIPA-ACL Heritage), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## Masalahnya

Dua agent perlu sepakat atas sebuah harga. Dibiarkan sendiri dengan prompt bahasa murni, LLM 2024-2026 menutup deal dengan tingkat yang mengejutkan rendahnya (~27% pada tawar-menawar yang diparameterkan ketat di arXiv:2402.15813). Skala tidak memperbaikinya: GPT-4 tidak lebih baik secara struktural dalam tawar-menawar daripada GPT-3.5; ia lebih baik pada *bahasa* tawar-menawar.

Masalah akarnya adalah LLM mencampur dua pekerjaan — memutuskan penawaran dan menarikan penawaran. OG-Narrator memisahkannya: sebuah generator penawaran deterministik menghitung langkah numerik; LLM hanya menarikan. Tingkat deal melompat ke ~89%.

Ini mencerminkan temuan multi-agent klasik: memisahkan mekanisme dari lapisan komunikasi itu menang. Contract Net Protocol (FIPA, 1996; Smith, 1980) adalah mekanisme task-market rujukannya. Colokkan sebuah LLM ke slot narasi dan kamu mendapatkan sebuah task market bertenaga LLM modern.

## Konsepnya

### Contract Net, dalam satu paragraf

Contract Net Protocol Smith tahun 1980: seorang **manager** menyiarkan sebuah **call for proposals (cfp)**; para **bidders** merespons dengan pesan **propose** yang memuat penawaran mereka; manager memilih seorang pemenang dan mengirim **accept-proposal** ke pemenang serta **reject-proposal** ke yang kalah. Pemenanglah yang mengerjakan tugasnya. Pesan opsional: **refuse** (penawar menolak mengajukan penawaran). FIPA meresmikannya sebagai interaction protocol `fipa-contract-net`.

### Mengapa OG-Narrator menang

"Measuring Bargaining Abilities of Language Models" (arXiv:2402.15813) mengamati bahwa:

- LLM sering melanggar aturan tawar-menawar (menawar pada harga yang tidak masuk akal, mengabaikan ZOPA pihak lain).
- Mereka menjangkar dengan buruk (menerima penawaran pertama yang jelek; meng-counter dengan nominal simbolis alih-alih strategis).
- Skala saja tidak memperbaiki hal-hal ini. Model yang lebih besar menghasilkan bahasa yang lebih masuk akal dengan kesalahan strategis yang serupa.

Dekomposisi OG-Narrator:

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

Generator penawarannya adalah sebuah strategi negosiasi klasik: model tawar-menawar Rubinstein, strategi Zeuthen, atau sekadar tit-for-tat sederhana atas harga. LLM-lah yang menarikan. Pesannya memuat harga deterministik dan framing bahasa alaminya.

Tingkat deal melompat karena:
- Harga tetap di dalam bargaining zone.
- Jangkaran bersifat strategis, bukan emosional.
- LLM melakukan hal yang ia kuasai: menulis.

### Temuan NegotiationArena

arXiv:2402.05863 menyediakan benchmark kanonisnya. Temuan utamanya:

- LLM bisa meningkatkan payoff ~20% dengan mengadopsi persona ("Saya putus asa menjual ini sebelum Jumat") — manipulasi persona adalah taktik nyata.
- Agent yang adil/kooperatif dieksploitasi oleh agent adversarial; pertahanan menuntut postur-balik yang eksplisit.
- Pasangan simetris konvergen ke hasil yang tidak adil pada sekitar 40% skenario benchmark.

Ini bukan "LLM itu negosiator yang buruk." Ini "LLM bernegosiasi terlalu mirip manusia, termasuk bagian-bagian yang bisa dieksploitasi."

### Penyembunyian chain-of-thought

Large-Scale Autonomous Negotiation Competition (arXiv:2503.06416) menjalankan ~180k negosiasi di berbagai strategi LLM. Para pemenang menyembunyikan penalaran mereka dari lawan:

- Jika sebuah agent mencetak "Saya hanya mau turun ke $75; harga reservasi saya $70" ke dalam scratchpad yang terlihat publik, lawan membacanya.
- Para pemenang menghitung strategi secara privat; kanal keluar hanya memuat penawaran dan narasi minimum yang diperlukan.

Ini adalah gema 2026 dari teori permainan klasik (Aumann 1976 tentang rasionalitas dan informasi): mengungkap valuasi pribadimu mengorbankan payoff. LLM tidak menyadarinya dengan naluri dan dengan senang hati mengetikkan harga reservasinya di jejak penalaran yang kemudian terlihat oleh lawan.

Intinya untuk engineering: pisahkan konteks scratchpad-pribadi dari konteks pesan-publik. Ini bukan opsional.

### Bhattacharya et al. 2025 — peringkat model

Dengan metrik Harvard Negotiation Project (negosiasi berprinsip, penghormatan BATNA, reciprositas minat):

- **Llama-3** paling efektif dalam mencapai kesepakatan (tingkat deal + payoff).
- **Claude-3** negosiator paling agresif (jangkaran tinggi, konsesi terlambat).
- **GPT-4** yang paling adil (variasi payoff paling kecil di semua pasangan).

Ini adalah potret 2025. Intinya bukan model mana yang menang pada April 2026 — melainkan bahwa base model yang berbeda punya gaya negosiasi yang konsisten. Ensemble heterogen (Pelajaran 15) memasukkan ini sebagai sumber keragaman.

### Alokasi tugas via Contract Net + LLM

Pemakaian-ulang Contract Net modern untuk multi-agent LLM:

1. Agent manager mendekomposisikan sebuah tugas menjadi unit-unit.
2. Menyiarkan `cfp` dengan deskripsi tugas ke agent pekerja.
3. Setiap pekerja mengembalikan sebuah penawaran: `(price, eta, confidence)` di mana harga bisa berupa token, unit komputasi, atau dolar.
4. Manager memilih pemenang (tunggal atau ganda, tergantung tugas) dan memberi penghargaan.
5. Pekerja yang ditolak bebas mengajukan tawaran pada tugas lain.

Ini menskalakan dengan baik melewati 100 pekerja karena koordinasinya berupa siarkan-dan-respond, bukan obrolan sinkron. Dipakai di produksi: pola orkestrasi Microsoft Agent Framework, beberapa implementasi LangGraph.

### LLM-Stakeholders Interactive Negotiation

NeurIPS 2024 (https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) memperkenalkan permainan multipihak yang bisa diberi skor dengan **skor rahasia** dan **ambang penerimaan-minimum**. Setiap stakeholder punya utilitas privat; LLM harus menyimpulkannya dari pesan. Ini adalah generalisasi tawar-menawar dua-pihak ke pembentukan koalisi N-pihak. Relevan untuk task market produksi dengan kemampuan pekerja yang heterogen.

### Aturan narasi-vs-mekanisme

Di semua benchmark negosiasi 2024-2026, aturan engineering yang konsisten adalah:

> Biarkan LLM menarikan. Jangan biarkan LLM menghitung penawaran.

Jika penawarannya harus berupa angka (harga, ETA, kuantitas), hasilkan secara deterministik dari status negosiasi dan biarkan LLM menghasilkan framing-nya. Jika penawarannya harus berupa struktur proposisi (dekomposisi tugas, penetapan peran), biarkan LLM menyusunnya, tetapi validasi terhadap sebuah schema dan pemeriksaan-kendala sebelum mengirim.

```figure
a5-og-narrator
```

## Bangun Sendiri

`code/main.py` mengimplementasikan:

- `ContractNetManager`, `ContractNetTask`, `Bid` — manager + para penawar, menyiarkan cfp, mengumpulkan proposisi, memberi penghargaan.
- `og_narrator_bargain(state, rng)` — pembeli OG-Narrator: konsesi deterministik bergaya Zeuthen menuju titik tengah.
- `seller_response(state, rng)` — kebijakan counter-offer penjual deterministik (ground truth struktural untuk kedua gaya).
- `naive_llm_bargain(state, rng)` — mensimulasikan seorang penawar semua-LLM: memilih harga dengan variasi tinggi, sering di luar ZOPA.
- Pengukuran: tingkat deal atas 1000 percobaan dengan harga reservasi segar yang di-sampling per percobaan.

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan: tingkat deal naive-LLM ~65-75%; tingkat deal OG-Narrator ~85-95%; jarak 15-25 poin adalah keuntungan struktural dari mendekomposisikan pembuatan-penawaran dari narasi. Ditambah sebuah contoh alokasi task-market Contract Net dengan tiga penawar dan satu tugas.

## Gunakan

`outputs/skill-bargainer-designer.md` merancang sebuah protokol tawar-menawar: siapa yang menghasilkan penawaran (deterministik atau LLM), siapa yang menarikan, bagaimana scratchpad privat dipisahkan dari pesan publik, dan bagaimana tingkat deal dipantau.

## Rilis

Checklist tawar-menawar produksi:

- **Scratchpad terpisah.** Status privat tidak pernah sampai ke konteks lawan. Ini tidak bisa ditawar.
- **Pembuatan penawaran deterministik.** Harga, kuantitas, ETA: hitung, jangan di-prompt.
- **Validasi semua penawaran masuk** terhadap sebuah schema. Tolak penawaran di luar ZOPA di batas protokol.
- **Batasi ronde.** Maksimal 3-5 ronde; eskalasi ke mediator saat buntu.
- **Ukur tingkat deal dan variasi payoff** terus-menerus. Tingkat deal yang turun adalah sebuah gejala — sering karena prompt drift atau serangan di sisi lawan.
- **Catat semua proposisi yang ditolak** beserta alasannya secara deterministik. Bagi manager Contract Net, penawar yang kalah perlu memahami alasannya.

## Latihan

1. Jalankan `code/main.py`. Konfirmasi OG-Narrator mengalahkan naive-LLM dalam tingkat deal. Seberapa besar selisihnya?
2. Implementasikan **peningkatan payoff berbasis persona** (arXiv:2402.05863) — pembeli mengadopsi persona "putus asa membeli minggu ini" hanya di narasi, generator penawaran tidak berubah. Apakah tingkat deal atau payoff berubah?
3. Implementasikan **penyembunyian** chain-of-thought: pertahankan sebuah string scratchpad privat yang tidak diteruskan ke lawan. Apa yang terjadi jika kamu tidak sengaja membocorkannya (simulasikan dengan menukar kanal-nya)?
4. Perluas Contract Net menjadi lelang N-penawar dengan harga reserve. Ketika semua tawaran melampaui reserve, bagaimana manager memutuskan antara harga-termurah dan kualitas-tertinggi? Aturan pemberian penghargaan mana yang kamu pilih dan mengapa?
5. Baca Bhattacharya et al. 2025 tentang metrik Harvard Negotiation Project. Implementasikan dua penawar dengan gaya berbeda (agresif vs adil). Ukur variasi payoff di bawah pasangan simetris dan asimetris.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Contract Net | "Task market" | Smith 1980, FIPA 1996. cfp + propose + accept/reject. The canonical task-market. |
| ZOPA | "Zone of possible agreement" | Overlap between buyer's max and seller's min. Offers outside it cannot close. |
| BATNA | "Best alternative to a negotiated agreement" | Your fallback if this deal fails. Sets your reservation price. |
| OG-Narrator | "Offer generator + narrator" | Decomposition: deterministic offer, LLM narration. |
| Zeuthen strategy | "Risk-minimizing concession" | Classical offer-generator that concedes based on risk limits. |
| Rubinstein bargaining | "Alternating-offer equilibrium" | Game-theoretic model for infinite-horizon bargaining with discounting. |
| CoT concealment | "Hide your reasoning" | Winners in arXiv:2503.06416 kept private scratchpads; public channel shows offer only. |
| Persona manipulation | "Emotional posturing" | arXiv:2402.05863: ~20% payoff gain from desperation/urgency personas. |

## Bacaan Lanjutan

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — the benchmark; persona manipulation and exploitation findings
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) — OG-Narrator and the buyer-harder-than-seller result
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) — ~180k negotiations; chain-of-thought concealment wins
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — multi-party scorable games with secret utilities
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — the classical mechanism, IEEE Transactions on Computers
