# Caching Cepat dan Caching Konteks

> Perintah sistem kamu adalah 4.000 token. kamuRAGkonteksnya adalah 20.000 token. kamu mengirim keduanya dengan setiap permintaan. kamu juga membayar keduanya — setiap saat. Caching yang cepat memungkinkan penyedia menjaga awalan tersebut tetap hangat dan menagih kamu 10% dari tarif normal untuk penggunaan kembali. Jika digunakan dengan benar, ini akan mengurangi biaya inferensi sebesar 50–90% dan latensi token pertama sebesar 40–85%.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 01 (Prompt Engineering), Phase 11 · 05 (Context Engineering), Phase 11 · 11 (Caching and Cost)
**Time:** ~60 minutes

## Masalahnya

Agen pengkodean mengirimkan perintah sistem 15.000 token yang sama kepada Claude di setiap percakapan. Dua puluh putaran$3/M input tokens is $Biaya masukan sebesar 0,90 saja — sebelum pesan aktual apa pun dari pengguna. Kalikan dengan 10.000 percakapan harian dan tagihannya mencapai $9,000/dayuntuk teks yang tidak pernah berubah.

kamu tidak dapat mengecilkan prompt tanpa merusak kualitas. kamu tidak dapat menghindari pengirimannya — model membutuhkannya di setiap kesempatan. Satu-satunya langkah adalah berhenti membayar harga penuh untuk awalan yang telah dilihat oleh penyedia.

Langkah itu adalah cache yang cepat. Anthropic mengirimkannya pada Agustus 2024 (dengan perpanjangan 1 jam-TTLvarian pada tahun 2025), OpenAI mengotomatiskannya pada akhir tahun itu, Google mengirimkan cache konteks eksplisit bersama Gemini 1.5, dan ketiganya kini menawarkannya sebagai fitur kelas satu pada model perbatasan mereka.

## Konsepnya

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**Mekaniknya.** Jika awalan permintaan cocok dengan awalan dari permintaan terbaru, penyedia akan menyajikan cache KV dari proses sebelumnya alih-alih mengkodekan ulang token. kamu membayar sedikit premi tulis untuk pertama kalinya dan diskon baca yang besar setiap kali berikutnya.

**Tiga pilihan penyedia pada tahun 2026.**

| Provider | API style | Hit discount | Write premium | Default TTL | Min cacheable |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | Explicit `cache_control` markers on content blocks | 90% off input | 25% surcharge | 5 min (extendable to 1 hour) | 1,024 tokens (Sonnet/Opus), 2,048 (Haiku) |
| OpenAI | Automatic prefix detection | 50% off input | none | Up to 1 hour (best-effort) | 1,024 tokens |
| Google (Gemini) | Explicit `CachedContent` API | Storage-billed; read at ~25% of normal | Storage fee per token·hour | User-set (default 1 hour) | 4,096 tokens (Flash), 32,768 (Pro) |

**Invarian.** Ketiga awalan cache saja. Jika ada token yang berbeda di antara permintaan, semua yang terjadi setelah token yang berbeda pertama adalah sebuah kesalahan. Letakkan bagian *stabil* di atas, bagian *variabel* di bawah.

### Tata letak yang ramah cache

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

Melanggar perintah — letakkan pesan pengguna di atas perintah sistem, sisipkan pengambilan dinamis di antara beberapa pengambilan — dan cache tidak akan pernah muncul.

### Perhitungan titik impas

Premi tulis 25% Anthropic berarti blok yang di-cache harus dibaca setidaknya dua kali untuk menghemat uang. 1 tulis + 1 baca rata-rata 0,675x biaya per permintaan (menghemat 32%); 1 tulis + 10 baca rata-rata 0,205x (menghemat 80%). Aturan praktisnya: cache apa pun yang kamu harapkan untuk digunakan kembali setidaknya 3 kali dalamTTL.

```figure
prompt-cache-hit
```

## Bangun Sendiri

### Langkah 1: Caching cepat antropik dengan penanda eksplisit

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

Itu`cache_control`penanda memberitahu Anthropic untuk menyimpan blok selama 5 menit. Penggunaan kembali dalam jendela itu berhasil; digunakan kembali setelah habis masa berlakunya dan menulis lagi.

**Bidang penggunaan respons:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

Periksa kedua bidang di CI — jika`cache_read_input_tokens`tetap nol di seluruh permintaan, kunci cache kamu hilang.

### Langkah 2: diperpanjang satu jamTTL

Untuk pekerjaan batch yang berjalan lama, masa berlaku default 5 menit akan berakhir di antara pekerjaan. Tetapkan`ttl`:

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 jamTTLbiayanya 2x lipat dari premi tulis (50% dibandingkan harga dasar, bukan 25%) tetapi membayar kembali dengan cepat pada batch mana pun yang menggunakan kembali awalan lebih dari 5 kali.

### Langkah 3: Caching otomatis OpenAI

OpenAI tidak memberi kamu apa pun untuk dikonfigurasi. Awalan apa pun di atas 1.024 token yang cocok dengan permintaan terbaru akan mendapat diskon 50% secara otomatis.

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

Aturan tata letak ramah cache yang sama juga berlaku. Dua hal yang mematikan cache OpenAI namun tidak mematikan cache Anthropic: mengubah`user`bidang (digunakan sebagai komponen kunci cache) dan alat penataan ulang.

### Langkah 4: Caching konteks eksplisit Gemini

Gemini memperlakukan cache sebagai objek kelas satu yang kamu buat dan beri nama:

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3.8-flash",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini membebankan biaya penyimpanan per token·jam selama cache masih ada, dan membaca pada ~25% dari kecepatan input normal. Ini adalah bentuk yang tepat ketika kamu menggunakan kembali perintah raksasa yang sama di banyak sesi selama berhari-hari.

### Langkah 5: mengukur tingkat keberhasilan dalam produksi

Lihat`code/main.py`untuk simulasi akuntan tiga penyedia yang melacakwrite/read/missmenghitung dan menghitung biaya campuran per 1 ribu permintaan. Gerbang diterapkan pada tingkat pencapaian target — sebagian besar penyiapan Antropik produksi akan melihat >80% fraksi baca setelah pemanasan.

## Perangkap yang masih dikirimkan pada tahun 2026

- **Stempel waktu dinamis di bagian atas.**`"Current time: 2026-04-22 15:30:02"`di bagian atas prompt sistem. Setiap permintaan meleset. Pindahkan stempel waktu di bawah titik henti cache.
- **Penyusunan ulang alat.** Membuat serial alat dalam urutan yang stabil — perombakan dict antar penerapan akan menghentikan setiap hit.
- **Teks ​​bebas hampir duplikat.** "kamu sangat membantu." vs "Kamu adalah asisten yang sangat membantu." — perbedaan satu byte = kesalahan total.
- **Blok terlalu kecil.** Anthropic menerapkan lantai 1.024 token (2.048 untuk Haiku). Blok yang lebih kecil secara diam-diam tidak melakukan cache.
- **Dasbor biaya buta.** Pisahkan "token input" menjadi cache dan tidak cache. Kalau tidak, penurunan lalu lintas tampak seperti kemenangan cache.

## Gunakan

Tumpukan cache tahun 2026:

| Situation | Pick |
|-----------|------|
| Agent with stable 10k+ system prompt, many turns | Anthropic `cache_control` with 5-min TTL |
| Batch job reusing a prefix for 30+ minutes | Anthropic with `ttl: "1h"` |
| Serverless endpoints on GPT-5, no custom infra | OpenAI automatic (just make your prefix stable and long) |
| Multi-day reuse of a giant code/doc corpus | Gemini explicit `CachedContent` |
| Cross-provider fallback | Keep the cacheable prefix layout identical across providers so any hit works |

Gabungkan dengan caching semantik (Fase 11 · 11) untuk lapisan pesan pengguna: caching cepat menangani penggunaan kembali *identik-token*, caching semantik menangani penggunaan kembali *identik dengan makna.

## Rilis

Simpan`outputs/skill-prompt-caching-planner.md`:

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## Latihan

1. **Mudah.** Lakukan percakapan 10 putaran dengan perintah sistem 5.000 token melawan Claude. Jalankan tanpa`cache_control`dan kemudian dengan. Laporkan tagihan token masukan untuk masing-masing.
2. **Sedang.** Tulis test harness yang, dengan templat cepat dan log permintaan, menghitung tingkat keberhasilan yang diharapkan dan penghematan dolar per penyedia (Anthropic 5m, Anthropic 1h, OpenAI otomatis, Gemini eksplisit).
3. **Sulit.** Membuat pengoptimal tata letak: diberikan perintah dan daftar kolom yang ditandai`stable=True/False`, tulis ulang perintah untuk menempatkan satu titik henti cache pada posisi ramah cache maksimum tanpa kehilangan informasi. Verifikasi pada titik akhir Antropik yang sebenarnya.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Prompt caching | "Makes long prompts cheap" | Reusing a provider-side KV-cache for matching prefixes; 50-90% discount on repeated input tokens. |
| `cache_control` | "The Anthropic marker" | Content-block attribute that declares "everything up to here is cacheable"; `{"type": "ephemeral"}`. |
| Cache write | "Paying the premium" | The first request that populates the cache; billed at ~1.25x input rate on Anthropic, free on OpenAI. |
| Cache read | "The discount" | Subsequent requests matching the prefix; billed at 10% (Anthropic), 50% (OpenAI), ~25% (Gemini). |
| TTL | "How long it lives" | Seconds the cache stays warm; Anthropic 5m default (extendable 1h), OpenAI best-effort up to 1h, Gemini user-set. |
| Extended TTL | "1-hour Anthropic cache" | `{"type": "ephemeral", "ttl": "1h"}`; 2x write premium but worth it for batch reuse. |
| Prefix match | "Why my cache missed" | Caches only hit when every token from the start up to the breakpoint is byte-identical. |
| Context caching (Gemini) | "The explicit one" | Google's named, storage-billed cache object; best for multi-day reuse of large corpora. |

## Bacaan Lebih Lanjut

- [Antropik — Caching cepat](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —`cache_control`, 1 jamTTL, tabel titik impas.
- [OpenAI — Penyimpanan cache cepat](https://platform.openai.com/docs/guides/prompt-caching) — pencocokan awalan otomatis.
- [Google — Cache konteks](https://ai.google.dev/gemini-api/docs/caching) —`CachedContent` APIdan harga penyimpanan.
- [Rekayasa antropis — Penyimpanan cache yang cepat untuk beban kerja konteks panjang](https://www.anthropic.com/news/prompt-caching) — postingan peluncuran asli dengan nomor latensi.
- Fase 11 · 05 (Rekayasa Konteks) — di mana memotong prompt agar cache dapat mendarat.
- Fase 11 · 11 (Caching dan Biaya) — memasangkan cache cepat dengan cache semantik pada pesan pengguna.
- [Pope et al., "Inferensi Transformator Penskalaan yang Efisien" (2022)](https://arxiv.org/abs/2211.05102) — model memori KV-cache yang menampilkan caching kepada pengguna; menjelaskan mengapa awalan yang di-cache ~10× lebih murah untuk dibaca ulang daripada menghitung ulang.
- [Agrawal dkk., "SARATHI: EfisienLLMInferensi dengan Membonceng Dekode dengan Prefill yang Dipotong" (2023)](https://arxiv.org/abs/2308.16369) — pra-pengisian adalah pintasan cache fase cepat; makalah ini menjelaskan alasannyaTTFTturun drastis pada cache hit sementaraTPOTtidak terpengaruh.
- [Leviathan dkk., "Inferensi Cepat dari Transformers melalui Penguraian Kode Spekulatif" (2023)](https://arxiv.org/abs/2211.17192) — cache cepat berada di samping decoding spekulatif, Flash Attention, danMQA/GQAsebagai pengungkit yang membengkokkan kurva biaya inferensi; baca ini untuk tiga lainnya.
