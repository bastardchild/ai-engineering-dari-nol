# Pelacakan State Dialog

> "I want a cheap restaurant in the north... actually make it moderate... and add Italian." Tiga turn, tiga update state. DST menjaga dict slot-value tetap sinkron supaya booking berhasil.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time:** ~75 minutes

## Masalahnya

Dalam sistem dialog berorientasi-tugas, tujuan pengguna di-encode sebagai himpunan pasangan slot-value: `{cuisine: italian, area: north, price: moderate}`. Setiap turn pengguna bisa menambah, mengubah, atau menghapus slot. Sistem harus membaca seluruh percakapan dan mengeluarkan state kini dengan benar.

Satu slot salah dan sistem memesan restoran yang salah, menjadwalkan penerbangan yang salah, atau menagih kartu yang salah. DST adalah engsel antara apa yang dikatakan pengguna dan apa yang dieksekusi backend.

Mengapa masih penting di 2026 meski ada LLM:

- Domain sensitif-compliance (perbankan, kesehatan, pemesanan maskapai) menuntut nilai slot deterministik, bukan generasi bebas-bentuk.
- Agen tool-use tetap butuh resolusi slot sebelum memanggil API.
- Koreksi multi-turn lebih sulit dari kelihatannya: "actually no, make it Thursday."

Pipeline modern: konsep DST klasik + ekstraktor LLM + guardrail output-terstruktur.

## Konsepnya

![DST: dialog history → slot-value state](../assets/dst.svg)

**Struktur tugas.** Skema mendefinisikan domain (restaurant, hotel, taxi) dan slot-nya (cuisine, area, price, people). Setiap slot bisa kosong, terisi nilai dari himpunan tertutup (price: {cheap, moderate, expensive}), atau nilai bebas-bentuk (name: "The Copper Kettle").

**Dua formulasi DST.**

- **Klasifikasi.** Untuk tiap pasangan (slot, candidate_value), prediksi ya/tidak. Berhasil untuk slot vocab-tertutup. Standar pra-2020.
- **Generasi.** Diberi dialog, hasilkan nilai slot sebagai teks bebas. Berhasil untuk slot vocab-terbuka. Default modern.

**Metrik.** Joint Goal Accuracy (JGA) — fraksi turn di mana *setiap* slot benar. Semua-atau-tidak-sama-sekali. Leaderboard MultiWOZ 2.4 memuncaki sekitar 83% di 2026.

**Arsitektur.**

1. **Berbasis aturan (regex slot + keyword).** Baseline kuat untuk domain sempit. Bisa di-debug.
2. **TripPy / BERT-DST.** Generasi berbasis-salinan dengan encoding BERT. Standar pra-LLM.
3. **LDST (LLaMA + LoRA).** LLM instruction-tuned dengan prompting domain-slot. Mencapai kualitas level-ChatGPT pada MultiWOZ 2.4.
4. **Bebas-ontologi (2024–26).** Lewati skema; hasilkan nama dan nilai slot langsung. Menangani domain terbuka.
5. **Prompt + output terstruktur (2024–26).** LLM dengan skema Pydantic + decoding terkendali. 5 baris kode, siap produksi.

### Mode kegagalan klasik

- **Ko-referensi lintas turn.** "Let's stay with the first option." Perlu me-resolve opsi mana.
- **Timpa vs tambah.** Pengguna bilang "add Italian." Apakah mengganti cuisine atau menambah?
- **Konfirmasi implisit.** "OK cool" — apakah itu menerima booking yang ditawarkan?
- **Koreksi.** "Actually make it 7 pm." Harus meng-update waktu tanpa membersihkan slot lain.
- **Koreferensi ke ujaran sistem sebelumnya.** "Yes, that one." "That" yang mana?

```figure
n5-slot-tracker
```

## Bangun Sendiri

### Langkah 1: ekstraktor slot berbasis aturan

Lihat `code/main.py`. Regex + kamus sinonim mencakup 70% ujaran kanonis dalam domain sempit:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

Rapuh di luar vocabulary kanonis. Berhasil untuk konfirmasi slot deterministik.

### Langkah 2: loop update state

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

Tiga invarian:

- Jangan pernah me-reset slot yang tidak disentuh pengguna.
- Negasi eksplisit ("never mind the cuisine") harus membersihkan.
- Koreksi pengguna ("actually...") harus menimpa, bukan menambah.

### Langkah 3: DST berbasis-LLM dengan output terstruktur

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic menjamin objek state valid. Tanpa regex, tanpa mismatch skema, tanpa slot terhalusinasi.

### Langkah 4: evaluasi JGA

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

Kalibrasi: berapa fraksi turn di mana sistem benar pada SEMUA slot? Untuk MultiWOZ 2.4, sistem teratas 2026: 80-83%. Sistem in-domain-mu harus melampaui itu pada vocabulary sempitmu atau baseline LLM mengalahkanmu.

### Langkah 5: menangani koreksi

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

Pada koreksi terdeteksi, timpa slot yang terakhir di-update alih-alih menambah. Sulit benar tanpa bantuan LLM. Pola modern: selalu biarkan LLM meregenerasi seluruh state dari histori alih-alih meng-update inkremental — ini menangani koreksi secara natural.

## Jebakan

- **Biaya regenerasi histori-penuh.** Membiarkan LLM meregenerasi state tiap turn menghabiskan total token O(n²). Batasi histori atau ringkas turn lama.
- **Drift skema.** Menambah slot baru belakangan merusak data training lama. Versikan skemamu.
- **Sensitivitas huruf.** "Italian" vs "italian" vs "ITALIAN" — normalisasi di mana-mana.
- **Warisan implisit.** Jika pengguna sebelumnya menyebut "for 4 people," permintaan baru untuk waktu berbeda tidak boleh membersihkan people. Selalu teruskan histori penuh.
- **Bebas-bentuk vs himpunan-tertutup.** Nama, waktu, dan alamat butuh slot bebas-bentuk; cuisine dan area tertutup. Campur keduanya dalam skema.

## Gunakan

Stack 2026:

| Situation | Approach |
|-----------|----------|
| Narrow domain (one or two intents) | Rule-based + regex |
| Broad domain, labeled data available | LDST (LLaMA + LoRA on MultiWOZ-style data) |
| Broad domain, no labels, prod-ready | LLM + Instructor + Pydantic schema |
| Spoken / voice | ASR + normalizer + LLM-DST |
| Multi-domain booking flow | Schema-guided LLM with per-domain Pydantic models |
| Compliance-sensitive | Rule-based primary, LLM fallback with confirmation flow |

## Rilis

Simpan sebagai `outputs/skill-dst-designer.md`:

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## Latihan

1. **Mudah.** Bangun pelacak state berbasis aturan di `code/main.py` untuk 3 slot (cuisine, area, price). Uji pada 10 dialog buatan tangan. Ukur JGA.
2. **Sedang.** Dataset sama dengan Instructor + Pydantic + LLM kecil. Bandingkan JGA. Periksa turn tersulit.
3. **Sulit.** Implementasikan keduanya dan rute: primer berbasis aturan, fallback LLM ketika primer berbasis aturan mengeluarkan <2 slot dengan keyakinan. Ukur JGA gabungan dan biaya inference per turn.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| DST | Dialogue state tracking | Maintain the slot-value dict across dialogue turns. |
| Slot | Unit of user intent | Named parameter the backend needs (cuisine, date). |
| Domain | The task area | Restaurant, hotel, taxi — sets of slots. |
| JGA | Joint Goal Accuracy | Fraction of turns where every slot is correct. All-or-nothing. |
| MultiWOZ | The benchmark | Multi-domain WOZ dataset; standard DST evaluation. |
| Ontology-free DST | No schema | Generate slot names and values directly, no fixed list. |
| Correction | "Actually..." | Turn that overwrites a previously-filled slot. |

## Bacaan Lanjutan

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — the canonical benchmark.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA instruction tuning for DST.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — the copy-based DST workhorse.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — EM-based unsupervised TOD.
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — canonical DST results.
