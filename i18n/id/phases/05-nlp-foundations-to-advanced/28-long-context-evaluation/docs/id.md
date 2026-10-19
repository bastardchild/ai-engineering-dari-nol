# Evaluasi Konteks-Panjang — NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro mengiklankan 10M token konteks. Pada 1M token, MRCR 8-needle turun ke 26.3%. Diiklankan ≠ bisa dipakai. Evaluasi konteks-panjang memberi tahu kapasitas aktual model yang kamu kirim.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60 minutes

## Masalahnya

Kamu punya kontrak 200-halaman. Model mengklaim konteks 1M-token. Kamu menempel kontrak dan bertanya: "What is the termination clause?" Model menjawab — tetapi menjawab dari halaman sampul karena klausul terminasi duduk 120k token di dalam, melewati tempat model benar-benar attend.

Inilah gap kapasitas-konteks 2026. Spec sheet bilang 1M atau 10M. Realitas bilang 60-70% dari itu bisa dipakai, dan "bisa dipakai" tergantung tugasnya.

- **Retrieval (satu needle dalam haystack):** nyaris-sempurna sampai maks yang diiklankan pada model frontier.
- **Multi-hop / agregasi:** terdegradasi tajam melewati ~128k pada kebanyakan model.
- **Penalaran atas fakta tersebar:** tugas pertama yang gagal.

Evaluasi konteks-panjang mengukur sumbu-sumbu ini. Pelajaran ini menyebutkan benchmark, apa yang sebenarnya diukur masing-masing, dan cara membangun uji needle kustom untuk domainmu.

## Konsepnya

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023).** Tanam fakta ("the magic word is pineapple") pada kedalaman terkontrol dalam konteks panjang. Minta model mengambilnya. Sapu kedalaman × panjang. Benchmark konteks-panjang orisinal. Model frontier kini menjenuhkannya; ia baseline yang perlu tetapi tidak cukup.

**RULER (Nvidia, 2024).** 13 tipe tugas lintas 4 kategori: retrieval (tunggal / multi-key / multi-value), tracing multi-hop (variable tracking), agregasi (frekuensi kata umum), QA. Panjang konteks terkonfigurasi (4k sampai 128k+). Mengungkap model yang menjenuhkan NIAH tetapi gagal pada multi-hop. Dalam rilis 2024, hanya separuh dari 17 model yang mengklaim konteks 32k+ mempertahankan kualitas pada 32k.

**LongBench v2 (2024).** 503 pertanyaan pilihan ganda, konteks 8k-2M kata, enam kategori tugas: QA dok-tunggal, QA multi-dok, in-context learning panjang, dialog panjang, repo kode, data terstruktur panjang. Benchmark produksi untuk perilaku konteks-panjang dunia-nyata.

**MRCR (Multi-Round Coreference Resolution).** Koreferensi multi-turn dalam skala. Varian 8-needle, 24-needle, 100-needle. Mengekspos berapa fakta yang bisa dijaga model sebelum attention terdegradasi.

**NoLiMa.** "Needle non-leksikal." Needle dan query tak berbagi overlap harfiah; retrieval menuntut satu langkah penalaran semantik. Lebih sulit daripada NIAH.

**HELMET.** Menggabungkan banyak dokumen, menanyakan pertanyaan dari salah satunya. Menguji attention selektif.

**BABILong.** Menanam rantai penalaran bAbI di dalam haystack tak relevan. Menguji reasoning-in-a-haystack, bukan sekadar retrieval.

### Apa yang benar-benar dilaporkan

- **Context window yang diiklankan.** Angka spec-sheet.
- **Panjang retrieval efektif.** Lolos NIAH pada ambang tertentu (mis. 90%).
- **Panjang penalaran efektif.** Lolos multi-hop atau agregasi pada ambang itu.
- **Kurva degradasi.** Akurasi vs panjang konteks, diplot per tipe tugas.

Dua angka untuk spec sheet-mu: efektif-retrieval dan efektif-penalaran. Biasanya yang efektif-penalaran adalah 25-50% dari window yang diiklankan.

```figure
gx-niah-decay
```

## Bangun Sendiri

### Langkah 1: NIAH kustom untuk domainmu

Lihat `code/main.py`. Kerangkanya:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

Sapu `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}. Plot heatmap-nya. Itulah kartu NIAH untuk model targetmu.

### Langkah 2: varian multi-needle

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

Pertanyaan seperti "What are the three magic words?" menuntut pengambilan ketiganya. Keberhasilan single-needle tidak memprediksi keberhasilan multi-needle.

### Langkah 3: tracing variabel multi-hop (ala RULER)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

Jawabannya menuntut merantai tiga penugasan. Model frontier pada 128k sering turun ke akurasi 50-70% di sini.

### Langkah 4: LongBench v2 pada stack-mu

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

Laporkan akurasi per kategori. Skor agregat menyembunyikan perbedaan besar level-tugas.

## Jebakan

- **Evaluasi NIAH-saja.** Lolos NIAH pada 1M token tak berkata apa-apa tentang multi-hop. Selalu jalankan RULER atau uji multi-hop kustom.
- **Sampling kedalaman seragam.** Banyak implementasi hanya menguji depth=0.5. Uji depth=0, 0.25, 0.5, 0.75, 1.0 — efek "lost in the middle" itu nyata.
- **Overlap leksikal dengan filler.** Jika needle berbagi keyword dengan filler, retrieval menjadi trivial. Pakai needle non-overlap ala NoLiMa.
- **Mengabaikan latensi.** Prompt 1M-token butuh 30-120 detik untuk prefill. Ukur time-to-first-token berdampingan akurasi.
- **Angka lapor-sendiri vendor.** OpenAI, Google, Anthropic semuanya menerbitkan skornya sendiri. Selalu run ulang independen pada use case-mu.

## Gunakan

Stack 2026:

| Situation | Benchmark |
|-----------|-----------|
| Quick sanity check | Custom NIAH at 3 depths × 3 lengths |
| Model selection for production | RULER (13 tasks) at your target length |
| Real-world QA quality | LongBench v2 single-doc-QA subset |
| Multi-hop reasoning | BABILong or custom variable-tracing |
| Conversational / dialogue | MRCR 8-needle at your target length |
| Model upgrade regression | Fixed in-house NIAH + RULER harness, run on every new model |

Aturan praktis untuk produksi: jangan pernah percaya context window sampai kamu punya NIAH + 1 tugas penalaran pada panjang yang dituju.

## Rilis

Simpan sebagai `outputs/skill-long-context-eval.md`:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## Latihan

1. **Mudah.** Bangun NIAH dengan 3 kedalaman (0.25, 0.5, 0.75) × 3 panjang (1k, 4k, 16k). Jalankan pada model apa pun. Plot tingkat lolos sebagai heatmap 3×3.
2. **Sedang.** Tambahkan varian 3-needle. Ukur pengambilan ketiganya pada tiap panjang. Bandingkan dengan tingkat lolos single-needle pada panjang sama.
3. **Sulit.** Konstruksikan tugas variable-tracing (X1 → X2 → X3, dengan 3 hop) yang ditanam dalam 64k filler. Ukur akurasi lintas 3 model frontier. Laporkan panjang penalaran efektif per model.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| NIAH | Needle in haystack | Plant a fact in filler, ask the model to retrieve it. |
| RULER | NIAH on steroids | 13 task types across retrieval / multi-hop / aggregation / QA. |
| Effective context | The real capacity | Length at which accuracy still holds above threshold. |
| Lost in the middle | Depth bias | Models under-attend to content in the middle of long inputs. |
| Multi-needle | Many facts at once | Multiple plants; tests attention juggling, not retrieval alone. |
| MRCR | Multi-round coref | 8, 24, or 100-needle coreference; exposes attention saturation. |
| NoLiMa | Non-lexical needle | Needle and query share no literal tokens; requires reasoning. |

## Bacaan Lanjutan

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — the original NIAH repo.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — the multi-task benchmark.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — real-world long-context eval.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — harder needles.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — reasoning-in-haystack.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — the depth-bias paper.
