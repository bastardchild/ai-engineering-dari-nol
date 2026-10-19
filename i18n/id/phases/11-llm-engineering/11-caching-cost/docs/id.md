# Caching, Pembatasan Nilai & Optimasi Biaya

> Kebanyakan startup AI tidak mati karena model yang buruk. Mereka mati karena unit ekonomi yang buruk. SatuGPT-4o biaya panggilan hanya sepersekian sen. Sepuluh ribu pengguna yang melakukan sepuluh panggilan per hari dikenakan biaya token masukan saja sebesar $250 -- sebelum kamu mengenakan biaya satu dolar. Perusahaan yang bertahan adalah perusahaan yang memperlakukan semua orangAPIpanggilan sebagai transaksi keuangan, bukan panggilan fungsi.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 Lesson 09 (Function Calling)
**Time:** ~45 minutes
**Terkait:** Fase 11 · 15 (Caching Cepat) — pelajaran ini mencakup caching lapisan aplikasi (cache semantik, cache hash yang tepat, perutean model). Pelajaran 15 mencakup caching prompt lapisan penyedia (Anthropic cache_control, OpenAI otomatis, Gemini CachedContent). Gabungkan keduanya untuk pengurangan biaya 50-95%.

## Tujuan Pembelajaran

- Menerapkan cache semantik yang menyajikan kueri berulang atau serupa dari cache alih-alih membuat yang baruAPIpanggilan
- Hitung biaya per permintaan di seluruh penyedia dan terapkan pembatasan tarif dan peringatan anggaran yang sadar token
- Bangun lapisan pengoptimalan biaya dengan kompresi cepat, perutean model (mahal vs murah), dan cache respons
- Rancang strategi caching berjenjang menggunakan pencocokan tepat, kesamaan semantik, dan caching awalan untuk jenis kueri yang berbeda

## Masalahnya

kamu membangunRAGbot obrolan. Ini bekerja dengan indah. Pengguna menyukainya.

Kemudian faktur tiba.

GPT-5 biaya$5 per million input tokens and $15 per juta output. Claude Opus 4.7 berharga$15 input / $75 keluaran. Biaya Gemini 3 Pro$1.25 input / $5 keluaran.GPT-5-mini adalah$0.25/$2. Harga di bawah ini hanya ilustrasi; selalu periksa halaman harga penyedia saat ini.

Inilah matematika yang mematikan startup:

- 10.000 pengguna aktif setiap hari
- 10 kueri per pengguna per hari
- 1.000 token masukan per kueri (perintah sistem + konteks + pesan pengguna)
- 500 token keluaran per respons

**Biaya masukan harian:** 10.000 x 10 x 1.000 / 1.000.000 x$2.50 = **$250/day**
**Biaya keluaran harian:** 10.000 x 10 x 500 / 1.000.000 x$10.00 = **$500/day**
**Total bulanan:** **$22,500/month**

Itu hanyaLLM. Tambahkan embeddings, hosting basis data vektor, infrastruktur. kamu melihat $30,000/monthuntuk chatbot.

Bagian yang brutal: 40-60% dari kueri tersebut hampir duplikat. Pengguna menanyakan pertanyaan yang sama dengan kata-kata yang sedikit berbeda. Perintah sistem kamu -- sama di setiap permintaan -- akan ditagih setiap saat. Dokumen konteks diambil olehRAGulangi pada pengguna yang bertanya tentang topik yang sama.

kamu membayar harga penuh untuk komputasi yang berlebihan.

## Konsepnya

### Anatomi Biaya suatuLLMTelepon

SetiapAPIpanggilan memiliki lima komponen biaya.

```mermaid
graph LR
    A[User Query] --> B[System Prompt<br/>500-2000 tokens]
    A --> C[Retrieved Context<br/>500-4000 tokens]
    A --> D[User Message<br/>50-500 tokens]
    B --> E[Input Cost<br/>$2.50/1M tokens]
    C --> E
    D --> E
    E --> F[Model Processing]
    F --> G[Output Cost<br/>$10.00/1M tokens]
```

Perintah sistem adalah pembunuh diam-diam. Permintaan sistem 1.500 token yang dikirim dengan setiap permintaan dikenakan biaya$3.75 per million requests just for that prefix. At 100K requests per day, that is $375/day-- $11,250/month-- untuk teks yang tidak pernah berubah.

### Caching Penyedia: Diskon Bawaan

Ketiga penyedia utama menawarkan cache cepat sisi penyedia pada tahun 2026, tetapi mekanismenya berbeda. Lihat Fase 11 · 15 untuk penyelaman lebih dalam.

| Provider | Mechanism | Discount | Minimum | Cache Duration |
|----------|-----------|----------|---------|----------------|
| Anthropic | Explicit cache_control markers | 90% on cache hits (pay 25% extra on write) | 1,024 tokens (Sonnet/Opus), 2,048 (Haiku) | 5 min default; 1h extended (2x write premium) |
| OpenAI | Automatic prefix matching | 50% on cache hits | 1,024 tokens | Best-effort up to 1 hour |
| Google Gemini | Explicit CachedContent API | ~75% reduction (plus storage) | 4,096 (Flash) / 32,768 (Pro) | User-configurable TTL |

**Pendekatan Antropik** bersifat eksplisit. kamu menandai bagian dari prompt kamu dengan`cache_control: {"type": "ephemeral"}`. Permintaan pertama membayar premi tulis sebesar 25%. Permintaan selanjutnya dengan awalan yang sama mendapatkan diskon 90%. Permintaan sistem 2.000 token membutuhkan biaya$0.005 normally costs $0,000625 pada cache yang ditemukan. Lebih dari 100 ribu permintaan, itu menghemat $437.50/day.

**Pendekatan OpenAI** bersifat otomatis. Awalan cepat apa pun yang cocok dengan permintaan sebelumnya mendapat diskon 50%. Tidak diperlukan penanda. Dampaknya: diskon lebih sedikit, kontrol lebih sedikit, namun tidak ada upaya implementasi.

### Caching Semantik: Lapisan Kustom kamu

Caching penyedia hanya berfungsi untuk awalan yang identik. Caching semantik menangani kasus yang lebih sulit: kueri berbeda dengan arti yang sama.

“Apa kebijakan pengembaliannya?” dan "Bagaimana cara mengembalikan barang?" adalah string yang berbeda tetapi maksudnya sama. Cache semantik menyematkan kedua kueri, menghitung kesamaan kosinus, dan mengembalikan respons yang disimpan dalam cache jika kesamaan melebihi ambang batas (biasanya 0,92-0,95).

```mermaid
flowchart TD
    A[User Query] --> B[Embed Query]
    B --> C{Similar query<br/>in cache?}
    C -->|sim > 0.95| D[Return Cached Response]
    C -->|sim < 0.95| E[Call LLM API]
    E --> F[Cache Response<br/>with Embedding]
    F --> G[Return Response]
    D --> G
```

Biaya penyematan dapat diabaikan. Text-embedding-3-small OpenAI berharga $0,02 per juta token. Memeriksa cache hampir tidak memerlukan biaya apa pun dibandingkan dengan penuhLLMpanggilan.

### Caching Tepat: Hash dan Pencocokan

Untuk panggilan deterministik (suhu=0, model yang sama, perintah yang sama), caching yang tepat lebih sederhana dan lebih cepat. Hash prompt lengkap, periksa cache, kembalikan jika ditemukan.

Ini berfungsi sempurna untuk:
- Perintah sistem + konteks tetap + pertanyaan pengguna yang identik
- Pemanggilan fungsi dengan definisi alat yang identik
- Pemrosesan batch dimana dokumen yang sama diproses beberapa kali

### Pembatasan Tarif: Melindungi Anggaran kamu

Pembatasan tarif bukan hanya soal keadilan. Ini tentang kelangsungan hidup.

**Algoritme keranjang token:** setiap pengguna mendapat satu keranjang berisi N token yang diisi ulang dengan kecepatan R per detik. Permintaan menggunakan token dari bucket. Jika bucket kosong, permintaan ditolak. Hal ini memungkinkan terjadinya burst (gunakan keranjang penuh sekaligus) sambil menerapkan tarif rata-rata.

**Kuota per pengguna:** ditetapkandaily/monthlybatas token per tingkat pengguna.

| Tier | Daily Token Limit | Max Requests/min | Model Access |
|------|------------------|------------------|-------------|
| Free | 50,000 | 10 | GPT-4o-mini only |
| Pro | 500,000 | 60 | GPT-4o, Claude Sonnet |
| Enterprise | 5,000,000 | 300 | All models |

### Perutean Model: Model yang Tepat untuk Pekerjaan yang Tepat

Tidak semua permintaan membutuhkanGPT-4o.

"Jam berapa tokonya tutup?" tidak memerlukan a$10/M-output model. GPT-4o-mini at $0.60/Moutput menanganinya dengan sempurna. Claude Haiku seharga $1.25/Moutput menanganinya. Pengklasifikasi sederhana mengarahkan kueri murah ke model murah dan kueri kompleks ke model mahal.

```mermaid
flowchart TD
    A[User Query] --> B[Complexity Classifier]
    B -->|Simple: lookup, FAQ| C[GPT-4o-mini<br/>$0.15/$0.60 per 1M]
    B -->|Medium: analysis, summary| D[Claude Sonnet<br/>$3.00/$15.00 per 1M]
    B -->|Complex: reasoning, code| E[GPT-4o / Claude Opus<br/>$2.50/$10.00+]
```

Router yang disetel dengan baik menghemat 40-70% biaya model saja.

### Pelacakan Biaya: Ketahui Kemana Uang Pergi

kamu tidak dapat mengoptimalkan apa yang tidak kamu ukur. Catat setiapAPIpanggilan dengan:

- Stempel waktu
- Nama model
- Token masukan
- Token keluaran
- Latensi (ms)
- Biaya yang dihitung ($)
- ID Pengguna
- Tembolokhit/miss
- Kategori permintaan

Data ini mengungkapkan fitur mana yang mahal, pengguna mana yang merupakan konsumen berat, dan fitur caching mana yang memiliki dampak paling besar.

### Batching: Diskon Massal

Batch OpenAIAPImemproses permintaan secara asinkron dengan diskon 50%. kamu mengirimkan batch hingga 50.000 permintaan, dan hasilnya akan muncul dalam waktu 24 jam.

Gunakan pengelompokan untuk:
- Pemrosesan dokumen setiap malam
- Klasifikasi massal
- Evaluasi berjalan
- Jalur pengayaan data

Bukan untuk: kueri yang dihadapi pengguna secara real-time (masalah latensi).

### Peringatan Anggaran dan Pemutus Arus

Pemutus sirkuit berhenti mengeluarkan uang ketika kamu mencapai batas. Tanpanya, bug atau penyalahgunaan dapat menghabiskan anggaran bulanan kamu dalam hitungan jam.

Tetapkan tiga ambang batas:
1. **Peringatan** (70% anggaran): kirim peringatan
2. **Throttle** (85% anggaran): beralih ke model yang lebih murah saja
3. **Hentikan** (95% anggaran): tolak permintaan baru, kembalikan respons yang disimpan dalam cache saja

### Tumpukan Optimasi

Terapkan teknik ini secara berurutan. Setiap lapisan menyatu dengan lapisan sebelumnya.

| Layer | Technique | Typical Savings | Implementation Effort |
|-------|-----------|----------------|----------------------|
| 1 | Provider prompt caching | 30-50% | Low (add cache markers) |
| 2 | Exact caching | 10-20% | Low (hash + dict) |
| 3 | Semantic caching | 15-30% | Medium (embeddings + similarity) |
| 4 | Model routing | 40-70% | Medium (classifier) |
| 5 | Rate limiting | Budget protection | Low (token bucket) |
| 6 | Prompt compression | 10-30% | Medium (rewrite prompts) |
| 7 | Batching | 50% on eligible | Low (batch API) |

SEBUAHRAGaplikasi yang menerapkan lapisan 1-5 biasanya mengurangi biaya$22,500/month to $4.000-6,000/month.Itulah bedanya membakar landasan dan membangun bisnis.

### Penghematan Nyata: Sebelum dan Sesudah

Berikut adalah rincian nyata untuk aRAGchatbot melayani 10.000DAU.

| Metric | Before Optimization | After Optimization | Savings |
|--------|--------------------|--------------------|---------|
| Monthly LLM cost | $22,500 | $5,200 | 77% |
| Avg cost per query | $0.0075 | $0.0017 | 77% |
| Cache hit rate | 0% | 52% | -- |
| Queries routed to mini | 0% | 65% | -- |
| P95 latency | 2,800ms | 900ms (cache hits: 50ms) | 68% |
| Monthly embedding cost | $0 | $180 | (new cost) |
| Total monthly cost | $22,500 | $5,380 | 76% |

Biaya penyematan untuk cache semantik ($180/month) membayar sendiri dalam satu jam pertama setelah cache ditemukan.

```figure
semantic-cache
```

## Bangun Sendiri

### Langkah 1: Kalkulator Biaya

Buat kalkulator biaya token yang mengetahui harga saat ini untuk model-model utama.

```python
import hashlib
import time
import json
import math
from dataclasses import dataclass, field


MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    "o3": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "o3-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.55},
    "o4-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.275},
    "claude-opus-4": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-haiku-3.5": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cached_input": 0.3125},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60, "cached_input": 0.0375},
}


def calculate_cost(model, input_tokens, output_tokens, cached_input_tokens=0):
    if model not in MODEL_PRICING:
        return {"error": f"Unknown model: {model}"}
    pricing = MODEL_PRICING[model]
    non_cached = input_tokens - cached_input_tokens
    input_cost = (non_cached / 1_000_000) * pricing["input"]
    cached_cost = (cached_input_tokens / 1_000_000) * pricing["cached_input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total = input_cost + cached_cost + output_cost
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "input_cost": round(input_cost, 6),
        "cached_input_cost": round(cached_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total, 6),
    }
```

### Langkah 2: Tembolok yang Tepat

Hash prompt lengkap dan kembalikan respons cache untuk permintaan serupa.

```python
class ExactCache:
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _hash(self, model, messages, temperature):
        key_data = json.dumps({"model": model, "messages": messages, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, model, messages, temperature=0.0):
        if temperature > 0:
            self.misses += 1
            return None
        key = self._hash(model, messages, temperature)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self.hits += 1
                entry["access_count"] += 1
                return entry["response"]
            del self.cache[key]
        self.misses += 1
        return None

    def put(self, model, messages, temperature, response):
        if temperature > 0:
            return
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        key = self._hash(model, messages, temperature)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        }

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.cache),
        }
```

### Langkah 3: Cache Semantik

Sematkan kueri dan kembalikan respons yang disimpan dalam cache ketika kesamaan melebihi ambang batas.

```python
def simple_embed(text):
    words = text.lower().split()
    vocab = {}
    for w in words:
        vocab[w] = vocab.get(w, 0) + 1
    norm = math.sqrt(sum(v * v for v in vocab.values()))
    if norm == 0:
        return {}
    return {k: v / norm for k, v in vocab.items()}


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    all_keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in all_keys)
    return dot


class SemanticCache:
    def __init__(self, similarity_threshold=0.85, max_size=500, ttl_seconds=3600):
        self.entries = []
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, query):
        query_embedding = simple_embed(query)
        now = time.time()
        best_match = None
        best_sim = 0.0
        for entry in self.entries:
            if now - entry["timestamp"] > self.ttl:
                continue
            sim = cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match = entry
        if best_match and best_sim >= self.threshold:
            self.hits += 1
            best_match["access_count"] += 1
            return {"response": best_match["response"], "similarity": round(best_sim, 4), "original_query": best_match["query"]}
        self.misses += 1
        return None

    def put(self, query, response):
        if len(self.entries) >= self.max_size:
            self.entries.sort(key=lambda e: e["timestamp"])
            self.entries.pop(0)
        self.entries.append({
            "query": query,
            "embedding": simple_embed(query),
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        })

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.entries),
        }
```

### Langkah 4: Pembatas Nilai

Pembatas tingkat keranjang token dengan kuota per pengguna.

```python
class TokenBucketRateLimiter:
    def __init__(self):
        self.buckets = {}
        self.tiers = {
            "free": {"capacity": 50_000, "refill_rate": 500, "max_requests_per_min": 10},
            "pro": {"capacity": 500_000, "refill_rate": 5_000, "max_requests_per_min": 60},
            "enterprise": {"capacity": 5_000_000, "refill_rate": 50_000, "max_requests_per_min": 300},
        }

    def _get_bucket(self, user_id, tier="free"):
        if user_id not in self.buckets:
            tier_config = self.tiers.get(tier, self.tiers["free"])
            self.buckets[user_id] = {
                "tokens": tier_config["capacity"],
                "capacity": tier_config["capacity"],
                "refill_rate": tier_config["refill_rate"],
                "last_refill": time.time(),
                "request_timestamps": [],
                "max_rpm": tier_config["max_requests_per_min"],
                "tier": tier,
                "total_tokens_used": 0,
            }
        return self.buckets[user_id]

    def _refill(self, bucket):
        now = time.time()
        elapsed = now - bucket["last_refill"]
        refill = int(elapsed * bucket["refill_rate"])
        if refill > 0:
            bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + refill)
            bucket["last_refill"] = now

    def check(self, user_id, tokens_needed, tier="free"):
        bucket = self._get_bucket(user_id, tier)
        self._refill(bucket)
        now = time.time()
        bucket["request_timestamps"] = [t for t in bucket["request_timestamps"] if now - t < 60]
        if len(bucket["request_timestamps"]) >= bucket["max_rpm"]:
            return {"allowed": False, "reason": "rate_limit", "retry_after_seconds": 60 - (now - bucket["request_timestamps"][0])}
        if bucket["tokens"] < tokens_needed:
            deficit = tokens_needed - bucket["tokens"]
            wait = deficit / bucket["refill_rate"]
            return {"allowed": False, "reason": "token_limit", "tokens_available": bucket["tokens"], "retry_after_seconds": round(wait, 1)}
        return {"allowed": True, "tokens_available": bucket["tokens"]}

    def consume(self, user_id, tokens_used, tier="free"):
        bucket = self._get_bucket(user_id, tier)
        bucket["tokens"] -= tokens_used
        bucket["request_timestamps"].append(time.time())
        bucket["total_tokens_used"] += tokens_used

    def get_usage(self, user_id):
        if user_id not in self.buckets:
            return {"error": "User not found"}
        b = self.buckets[user_id]
        return {
            "user_id": user_id,
            "tier": b["tier"],
            "tokens_remaining": b["tokens"],
            "capacity": b["capacity"],
            "total_tokens_used": b["total_tokens_used"],
            "utilization": round(b["total_tokens_used"] / b["capacity"], 4) if b["capacity"] else 0,
        }
```

### Langkah 5: Pelacak Biaya

Catat setiap panggilan dan hitung total yang berjalan.

```python
class CostTracker:
    def __init__(self, monthly_budget=1000.0):
        self.logs = []
        self.monthly_budget = monthly_budget
        self.alerts = []

    def log_call(self, model, input_tokens, output_tokens, cached_input_tokens=0, latency_ms=0, user_id="anonymous", cache_status="miss"):
        cost = calculate_cost(model, input_tokens, output_tokens, cached_input_tokens)
        entry = {
            "timestamp": time.time(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "latency_ms": latency_ms,
            "cost": cost["total_cost"],
            "user_id": user_id,
            "cache_status": cache_status,
        }
        self.logs.append(entry)
        self._check_budget()
        return entry

    def _check_budget(self):
        total = self.total_cost()
        pct = total / self.monthly_budget if self.monthly_budget > 0 else 0
        if pct >= 0.95 and not any(a["level"] == "stop" for a in self.alerts):
            self.alerts.append({"level": "stop", "message": f"Budget 95% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})
        elif pct >= 0.85 and not any(a["level"] == "throttle" for a in self.alerts):
            self.alerts.append({"level": "throttle", "message": f"Budget 85% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})
        elif pct >= 0.70 and not any(a["level"] == "warning" for a in self.alerts):
            self.alerts.append({"level": "warning", "message": f"Budget 70% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})

    def total_cost(self):
        return round(sum(e["cost"] for e in self.logs), 6)

    def cost_by_model(self):
        by_model = {}
        for e in self.logs:
            m = e["model"]
            if m not in by_model:
                by_model[m] = {"calls": 0, "cost": 0, "input_tokens": 0, "output_tokens": 0}
            by_model[m]["calls"] += 1
            by_model[m]["cost"] = round(by_model[m]["cost"] + e["cost"], 6)
            by_model[m]["input_tokens"] += e["input_tokens"]
            by_model[m]["output_tokens"] += e["output_tokens"]
        return by_model

    def cache_savings(self):
        cache_hits = [e for e in self.logs if e["cache_status"] == "hit"]
        if not cache_hits:
            return {"saved": 0, "cache_hits": 0}
        saved = 0
        for e in cache_hits:
            full_cost = calculate_cost(e["model"], e["input_tokens"], e["output_tokens"])
            saved += full_cost["total_cost"]
        return {"saved": round(saved, 4), "cache_hits": len(cache_hits)}

    def summary(self):
        if not self.logs:
            return {"total_calls": 0, "total_cost": 0}
        total_latency = sum(e["latency_ms"] for e in self.logs)
        cache_hits = sum(1 for e in self.logs if e["cache_status"] == "hit")
        return {
            "total_calls": len(self.logs),
            "total_cost": self.total_cost(),
            "avg_cost_per_call": round(self.total_cost() / len(self.logs), 6),
            "avg_latency_ms": round(total_latency / len(self.logs), 1),
            "cache_hit_rate": round(cache_hits / len(self.logs), 4),
            "cost_by_model": self.cost_by_model(),
            "cache_savings": self.cache_savings(),
            "budget_remaining": round(self.monthly_budget - self.total_cost(), 2),
            "budget_utilization": round(self.total_cost() / self.monthly_budget, 4) if self.monthly_budget > 0 else 0,
            "alerts": self.alerts,
        }
```

### Langkah 6: Perute Model

Rutekan kueri ke model termurah yang dapat menanganinya.

```python
SIMPLE_KEYWORDS = ["what time", "hours", "address", "phone", "price", "return policy", "hello", "hi", "thanks", "yes", "no"]
COMPLEX_KEYWORDS = ["analyze", "compare", "explain why", "write code", "debug", "architect", "design", "trade-off", "evaluate"]


def classify_complexity(query):
    q = query.lower()
    if len(q.split()) <= 5 or any(kw in q for kw in SIMPLE_KEYWORDS):
        return "simple"
    if any(kw in q for kw in COMPLEX_KEYWORDS):
        return "complex"
    return "medium"


def route_model(query, tier="pro"):
    complexity = classify_complexity(query)
    routing_table = {
        "simple": {"free": "gpt-4.1-nano", "pro": "gpt-4o-mini", "enterprise": "gpt-4o-mini"},
        "medium": {"free": "gpt-4o-mini", "pro": "claude-sonnet-4", "enterprise": "claude-sonnet-4"},
        "complex": {"free": "gpt-4o-mini", "pro": "gpt-4o", "enterprise": "claude-opus-4"},
    }
    model = routing_table[complexity].get(tier, "gpt-4o-mini")
    return {"query": query, "complexity": complexity, "model": model, "tier": tier}
```

### Langkah 7: Jalankan Demonya

```python
def simulate_llm_call(model, query):
    input_tokens = len(query.split()) * 4 + 500
    output_tokens = 150 + (len(query.split()) * 2)
    latency = 200 + (output_tokens * 2)
    return {
        "model": model,
        "response": f"[Simulated {model} response to: {query[:50]}...]",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency,
    }


def run_demo():
    print("=" * 60)
    print("  Caching, Rate Limiting & Cost Optimization Demo")
    print("=" * 60)

    print("\n--- Model Pricing ---")
    for model, pricing in list(MODEL_PRICING.items())[:6]:
        cost_1k = calculate_cost(model, 1000, 500)
        print(f"  {model}: ${cost_1k['total_cost']:.6f} per 1K in + 500 out")

    print("\n--- Cost Comparison: 100K Requests ---")
    for model in ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4", "claude-haiku-3.5"]:
        cost = calculate_cost(model, 1000 * 100_000, 500 * 100_000)
        print(f"  {model}: ${cost['total_cost']:.2f}")

    print("\n--- Anthropic Cache Savings ---")
    no_cache = calculate_cost("claude-sonnet-4", 2000, 500, 0)
    with_cache = calculate_cost("claude-sonnet-4", 2000, 500, 1500)
    saving = no_cache["total_cost"] - with_cache["total_cost"]
    print(f"  Without cache: ${no_cache['total_cost']:.6f}")
    print(f"  With 1500 cached tokens: ${with_cache['total_cost']:.6f}")
    print(f"  Savings per call: ${saving:.6f} ({saving/no_cache['total_cost']*100:.1f}%)")

    exact_cache = ExactCache(max_size=100, ttl_seconds=300)
    semantic_cache = SemanticCache(similarity_threshold=0.75, max_size=100)
    rate_limiter = TokenBucketRateLimiter()
    tracker = CostTracker(monthly_budget=100.0)

    print("\n--- Exact Cache ---")
    messages_1 = [{"role": "user", "content": "What is the return policy?"}]
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.0)
    print(f"  First lookup: {'HIT' if result else 'MISS'}")
    exact_cache.put("gpt-4o-mini", messages_1, 0.0, "You can return items within 30 days.")
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.0)
    print(f"  Second lookup: {'HIT' if result else 'MISS'} -> {result}")
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.7)
    print(f"  With temp=0.7: {'HIT' if result else 'MISS (non-deterministic, skip cache)'}")
    print(f"  Stats: {exact_cache.stats()}")

    print("\n--- Semantic Cache ---")
    test_queries = [
        ("What is the return policy?", "Items can be returned within 30 days with receipt."),
        ("How do I return an item?", None),
        ("What are your store hours?", "We are open 9am-9pm Monday through Saturday."),
        ("When does the store open?", None),
        ("Tell me about quantum computing", "Quantum computers use qubits..."),
        ("Explain quantum mechanics", None),
    ]
    for query, response in test_queries:
        cached = semantic_cache.get(query)
        if cached:
            print(f"  '{query[:40]}' -> CACHE HIT (sim={cached['similarity']}, original='{cached['original_query'][:40]}')")
        elif response:
            semantic_cache.put(query, response)
            print(f"  '{query[:40]}' -> MISS (stored)")
        else:
            print(f"  '{query[:40]}' -> MISS (no match)")
    print(f"  Stats: {semantic_cache.stats()}")

    print("\n--- Rate Limiting ---")
    for i in range(12):
        check = rate_limiter.check("user_1", 1000, "free")
        if check["allowed"]:
            rate_limiter.consume("user_1", 1000, "free")
        status = "OK" if check["allowed"] else f"BLOCKED ({check['reason']})"
        if i < 5 or not check["allowed"]:
            print(f"  Request {i+1}: {status}")
    print(f"  Usage: {rate_limiter.get_usage('user_1')}")

    print("\n--- Model Routing ---")
    routing_queries = [
        "What time do you close?",
        "Summarize this quarterly earnings report",
        "Analyze the trade-offs between microservices and monoliths",
        "Hello",
        "Write code for a binary search tree with deletion",
    ]
    for q in routing_queries:
        route = route_model(q, "pro")
        print(f"  '{q[:50]}' -> {route['model']} ({route['complexity']})")

    print("\n--- Full Pipeline: Before vs After Optimization ---")
    queries = [
        "What is the return policy?",
        "How do I return something?",
        "What are your hours?",
        "When do you open?",
        "Explain the difference between TCP and UDP",
        "Compare TCP vs UDP protocols",
        "Hello",
        "What is your phone number?",
        "Write a Python function to sort a list",
        "Analyze the pros and cons of serverless architecture",
    ]

    print("\n  [Before: no caching, single model (gpt-4o)]")
    tracker_before = CostTracker(monthly_budget=1000.0)
    for q in queries:
        result = simulate_llm_call("gpt-4o", q)
        tracker_before.log_call("gpt-4o", result["input_tokens"], result["output_tokens"], latency_ms=result["latency_ms"], cache_status="miss")
    before = tracker_before.summary()
    print(f"  Total cost: ${before['total_cost']:.6f}")
    print(f"  Avg cost/call: ${before['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {before['avg_latency_ms']}ms")

    print("\n  [After: caching + routing + rate limiting]")
    exact_c = ExactCache()
    semantic_c = SemanticCache(similarity_threshold=0.75)
    tracker_after = CostTracker(monthly_budget=1000.0)

    for q in queries:
        messages = [{"role": "user", "content": q}]
        cached = exact_c.get("gpt-4o", messages, 0.0)
        if cached:
            tracker_after.log_call("gpt-4o-mini", 0, 0, latency_ms=5, cache_status="hit")
            continue
        sem_cached = semantic_c.get(q)
        if sem_cached:
            tracker_after.log_call("gpt-4o-mini", 0, 0, latency_ms=15, cache_status="hit")
            continue
        route = route_model(q)
        result = simulate_llm_call(route["model"], q)
        tracker_after.log_call(route["model"], result["input_tokens"], result["output_tokens"], latency_ms=result["latency_ms"], cache_status="miss")
        exact_c.put(route["model"], messages, 0.0, result["response"])
        semantic_c.put(q, result["response"])

    after = tracker_after.summary()
    print(f"  Total cost: ${after['total_cost']:.6f}")
    print(f"  Avg cost/call: ${after['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {after['avg_latency_ms']}ms")
    print(f"  Cache hit rate: {after['cache_hit_rate']:.0%}")

    if before["total_cost"] > 0:
        savings_pct = (1 - after["total_cost"] / before["total_cost"]) * 100
        print(f"\n  SAVINGS: {savings_pct:.1f}% cost reduction")
        print(f"  Latency improvement: {(1 - after['avg_latency_ms'] / before['avg_latency_ms']) * 100:.1f}% faster")

    print("\n--- Budget Alerts Demo ---")
    alert_tracker = CostTracker(monthly_budget=0.01)
    for i in range(5):
        alert_tracker.log_call("gpt-4o", 5000, 2000, latency_ms=500)
    print(f"  Total spent: ${alert_tracker.total_cost():.6f} / ${alert_tracker.monthly_budget}")
    for alert in alert_tracker.alerts:
        print(f"  ALERT [{alert['level'].upper()}]: {alert['message']}")

    print("\n--- Cost Breakdown by Model ---")
    multi_tracker = CostTracker(monthly_budget=500.0)
    for _ in range(50):
        multi_tracker.log_call("gpt-4o-mini", 800, 200, latency_ms=150)
    for _ in range(30):
        multi_tracker.log_call("claude-sonnet-4", 1500, 500, latency_ms=400)
    for _ in range(10):
        multi_tracker.log_call("gpt-4o", 2000, 800, latency_ms=600)
    for _ in range(10):
        multi_tracker.log_call("claude-opus-4", 3000, 1000, latency_ms=1200)
    breakdown = multi_tracker.cost_by_model()
    for model, data in sorted(breakdown.items(), key=lambda x: x[1]["cost"], reverse=True):
        print(f"  {model}: {data['calls']} calls, ${data['cost']:.6f}, {data['input_tokens']:,} in / {data['output_tokens']:,} out")
    print(f"  Total: ${multi_tracker.total_cost():.6f}")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
```

## Gunakan

### Caching Prompt Antropik

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-5",
#     max_tokens=1024,
#     system=[
#         {
#             "type": "text",
#             "text": "You are a helpful customer support agent for Acme Corp...",
#             "cache_control": {"type": "ephemeral"},
#         }
#     ],
#     messages=[{"role": "user", "content": "What is the return policy?"}],
# )
#
# print(f"Input tokens: {response.usage.input_tokens}")
# print(f"Cache creation tokens: {response.usage.cache_creation_input_tokens}")
# print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")
```

Panggilan pertama menulis ke cache (25% premium). Setiap panggilan berikutnya dengan awalan prompt sistem yang sama dibaca dari cache (diskon 90%). Cache bertahan selama 5 menit dan menyetel ulang pengatur waktu pada setiap pukulan.

### Caching Otomatis OpenAI

```python
# from openai import OpenAI
#
# client = OpenAI()
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "You are a helpful customer support agent..."},
#         {"role": "user", "content": "What is the return policy?"},
#     ],
# )
#
# print(f"Prompt tokens: {response.usage.prompt_tokens}")
# print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
# print(f"Completion tokens: {response.usage.completion_tokens}")
```

OpenAI melakukan cache secara otomatis. Awalan cepat apa pun yang berjumlah 1.024+ token yang cocok dengan permintaan terbaru akan mendapat diskon 50%. Tidak diperlukan perubahan kode -- cukup periksa`prompt_tokens_details.cached_tokens`sebagai tanggapan untuk memverifikasi bahwa itu berfungsi.

### Kelompok OpenAIAPI

```python
# import json
# from openai import OpenAI
#
# client = OpenAI()
#
# requests = []
# for i, query in enumerate(queries):
#     requests.append({
#         "custom_id": f"request-{i}",
#         "method": "POST",
#         "url": "/v1/chat/completions",
#         "body": {
#             "model": "gpt-4o-mini",
#             "messages": [{"role": "user", "content": query}],
#         },
#     })
#
# with open("batch_input.jsonl", "w") as f:
#     for r in requests:
#         f.write(json.dumps(r) + "\n")
#
# batch_file = client.files.create(file=open("batch_input.jsonl", "rb"), purpose="batch")
# batch = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")
# print(f"Batch ID: {batch.id}, Status: {batch.status}")
```

kumpulanAPImemberikan diskon tetap 50% untuk semua token. Hasil tiba dalam waktu 24 jam. Sempurna untuk beban kerja non-waktu nyata: evaluasi, pelabelan data, ringkasan massal.

### Cache Semantik Produksi dengan Redis

```python
# import redis
# import numpy as np
# from openai import OpenAI
#
# r = redis.Redis()
# client = OpenAI()
#
# def get_embedding(text):
#     response = client.embeddings.create(model="text-embedding-3-small", input=text)
#     return response.data[0].embedding
#
# def semantic_cache_lookup(query, threshold=0.95):
#     query_emb = np.array(get_embedding(query))
#     keys = r.keys("cache:emb:*")
#     best_sim, best_key = 0, None
#     for key in keys:
#         stored_emb = np.frombuffer(r.get(key), dtype=np.float32)
#         sim = np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb))
#         if sim > best_sim:
#             best_sim, best_key = sim, key
#     if best_sim >= threshold and best_key:
#         response_key = best_key.decode().replace("cache:emb:", "cache:resp:")
#         return r.get(response_key).decode()
#     return None
```

Dalam produksi, ganti pemindaian linier dengan indeks vektor (Redis Vector Search, Pinecone, atau pgvector). Pemindaian linier berfungsi untuk <1.000 entri. Selain itu, gunakanANN(perkiraan tetangga terdekat) untuk pencarian O(log n).

## Rilis

Pelajaran ini menghasilkan`outputs/prompt-cost-optimizer.md`-- prompt yang dapat digunakan kembali untuk menganalisis file kamuLLMaplikasi dan merekomendasikan optimalisasi biaya spesifik dengan proyeksi penghematan.

Itu juga menghasilkan`outputs/skill-cost-patterns.md`-- kerangka keputusan untuk memilih strategi caching yang tepat, konfigurasi pembatasan laju, dan aturan perutean model untuk kasus penggunaan kamu.

## Latihan

1. **TerapkanLRUeviction untuk cache semantik.** Ganti eviction terlama-pertama dengan yang paling jarang digunakan. Lacak waktu akses terakhir untuk setiap entri dan keluarkan entri dengan waktu akses terlama ketika cache penuh. Bandingkan tingkat keberhasilan antara kedua strategi tersebut pada 100 kueri.

2. **Buatlah alat proyeksi biaya.** Diberikan logAPIpanggilan (log CostTracker), memproyeksikan biaya bulanan berdasarkan rata-rata 7 hari terakhir. Akun untukweekday/weekendpola. Memicu peringatan jika proyeksi biaya bulanan melebihi anggaran lebih dari 20%.

3. **Terapkan cache semantik berjenjang.** Gunakan dua ambang batas kesamaan: 0,98 untuk klik berkeyakinan tinggi (segera kembali) dan 0,90 untuk klik dengan keyakinan sedang (kembali dengan penafian: "Berdasarkan pertanyaan serupa sebelumnya..."). Lacak dari tingkat mana setiap klik berasal dan ukur perbedaan kepuasan pengguna.

4. **Buat pengklasifikasi perutean model.** Ganti pengklasifikasi berbasis kata kunci dengan pengklasifikasi berbasis penyematan. Sematkan 50 kueri berlabel (simple/medium/complex), lalu klasifikasikan kueri baru dengan mencari contoh berlabel terdekat. Ukur akurasi klasifikasi terhadap set pengujian yang terdiri dari 20 kueri.

5. **Menerapkan pemutus sirkuit dengan tingkat degradasi.** Dengan anggaran 70%, catat peringatan. Pada 85%, secara otomatis mengalihkan semua perutean ke model termurah (gpt-4o-mini). Pada tingkat 95%, hanya melayani respons yang di-cache dan menolak kueri baru. Uji dengan menyimulasikan 1.000 permintaan dengan anggaran $1,00 dan verifikasi setiap pemicu ambang batas dengan benar.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Prompt caching | "Cache the system prompt" | Provider-level caching where repeated prompt prefixes get a discount (90% Anthropic, 50% OpenAI) -- no code changes for OpenAI, explicit markers for Anthropic |
| Semantic caching | "Smart caching" | Embedding the query, computing similarity to past queries, and returning the cached response if similarity exceeds a threshold -- catches paraphrases that exact matching misses |
| Exact caching | "Hash caching" | Hashing the full prompt (model + messages + temperature) and returning the cached response for identical inputs -- only works for temperature=0 deterministic calls |
| Token bucket | "Rate limiter" | An algorithm where each user has a bucket of N tokens that refills at rate R per second -- allows bursts up to N while enforcing an average rate of R |
| Model routing | "Cheapskate routing" | Using a classifier to send simple queries to cheap models (GPT-4o-mini, Haiku) and complex queries to expensive models (GPT-4o, Opus) -- saves 40-70% on model costs |
| Cost tracking | "Metering" | Logging every API call with model, tokens, latency, cost, and user ID so you know exactly where money goes and which features are expensive |
| Circuit breaker | "Kill switch" | Automatically degrading service (cheaper models, cached-only) or stopping requests entirely when spending approaches the budget limit |
| Batch API | "Bulk discount" | OpenAI's asynchronous processing at 50% discount -- submit up to 50,000 requests, get results within 24 hours |
| Prompt compression | "Token diet" | Rewriting system prompts and context to use fewer tokens while preserving meaning -- shorter prompts cost less and often perform better |
| Cache hit rate | "Cache efficiency" | The percentage of requests served from cache instead of calling the LLM -- 40-60% is typical for production chatbots, saves proportionally on cost |

## Bacaan Lebih Lanjut

- [Panduan Caching Prompt Antropik](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) -- dokumen resmi untuk penanda cache_control eksplisit Anthropic, harga, dan perilaku seumur hidup cache
- [Caching Perintah OpenAI](https://platform.openai.com/docs/guides/prompt-caching) -- Caching otomatis OpenAI, cara memverifikasi cache yang ditemukan melalui bidang penggunaan, dan panjang awalan minimum
- [Kelompok OpenAIAPI](https://platform.openai.com/docs/guides/batch) -- Diskon 50% untuk pemrosesan asinkron,JSONLformat, jendela penyelesaian 24 jam, dan batas permintaan 50 ribu
- [GPTCache](https://github.com/zilliztech/GPTCache) -- pustaka cache semantik sumber terbuka yang mendukung beberapa backend penyematan, penyimpanan vektor, dan kebijakan penggusuran
- [Router Model Mars](https://docs.withmartian.com) -- perutean model produksi yang secara otomatis memilih model termurah yang mampu menangani setiap kueri
- [Bukan Berlian](https://www.notdiamond.ai) -- Model router berbasis ML yang mempelajari pola lalu lintas kamu untuk dioptimalkancost/qualitypengorbanan antar penyedia
- [Helikon](https://www.helicone.ai) --LLMplatform observasi dengan pelacakan biaya, cache, pembatasan tarif, dan peringatan anggaran sebagai lapisan proksi
- [Dean & Barroso, "Ekor dalam Skala" (CACM2013)](https://research.google/pubs/the-tail-at-scale/) -- latensi, throughput,TTFT/TPOTpersentil, dan permintaan lindung nilai; model biaya di belakang "pilih model termurah yang masih memenuhiP95."
- [Kwon dkk., "Manajemen Memori yang Efisien untuk Penyajian Model Bahasa Besar dengan PagedAttention" (SOSP2023)](https://arxiv.org/abs/2309.06180) -- kertas vLLM; mengapa halaman KV-cache + pengelompokan berkelanjutan mengalahkan server naif 24× dalam hal throughput, lapisan infra di bawah "caching dan biaya".
- [Dao dkk., "FlashAttention-2: Perhatian Lebih Cepat dengan Paralelisme dan Partisi Kerja yang Lebih Baik" (ICLR2024)](https://arxiv.org/abs/2307.08691) -- pengurangan biaya tingkat kernel ortogonal untuk meminta caching; membaca bersama decoding spekulatif danGQAuntuk gambaran kurva biaya penuh.
