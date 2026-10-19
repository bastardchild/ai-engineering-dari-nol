# Capstone 06 — Agen Troubleshooting DevOps untuk Kubernetes

> DevOps Agent milik AWS sudah GA, Resolve AI menerbitkan playbook K8s-nya, NeuBird mendemokan monitoring semantik, dan Metoro menautkan AI SRE ke SLO per-service. Bentuk produksinya sudah mapan: sebuah webhook alert memicu, sebuah agen membaca telemetry, menelusuri graph objek K8s, memeringkat hipotesis akar masalah, lalu memposting brief Slack dengan tombol persetujuan. Read-only secara default. Setiap remediation dijaga manusia. Capstone ini adalah agen itu, dievaluasi pada 20 insiden sintetis dan dibandingkan dengan Agen milik AWS pada tiga kasus bersama.

**Type:** Capstone
**Languages:** Python (agent), TypeScript (Slack integration)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and MCP), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P11 · P13 · P14 · P15 · P17 · P18
**Time:** 30 hours

## Masalahnya

Narasi SRE 2025-2026 menjadi: "Agen AI men-triage insiden, manusia menyetujui remediation." AWS DevOps Agent, Resolve AI, NeuBird, Metoro, PagerDuty AIOps semuanya merilis bentuk ini di produksi. Agen membaca metrik Prometheus, log Loki, trace Tempo, kube-state-metrics, dan sebuah knowledge graph berisi objek K8s. Ia menghasilkan hipotesis akar masalah yang berperingkat dengan sitasi telemetry dalam waktu di bawah lima menit. Ia tidak pernah mengeksekusi perintah destruktif tanpa persetujuan manusia eksplisit lewat Slack.

Sebagian besar kerja beratnya ada pada scoping dan keselamatan, bukan penalaran. Agen memerlukan permukaan RBAC read-only-by-default, server tool MCP yang dikeraskan, dan log audit untuk setiap perintah yang dipertimbangkan versus dieksekusi. Ia perlu tahu kapan ia berada di luar batas kemampuannya dan harus eskalasi. Dan ia harus berjalan cukup murah sehingga kaskade OOM-kill tidak menghasilkan tagihan agen $5k.

## Konsepnya

Agen beroperasi di atas sebuah knowledge graph. Node adalah objek K8s (Pod, Deployment, Service, Node, HPA, PVC) plus sumber telemetry (deret Prometheus, stream Loki, trace Tempo). Edge mengkodekan kepemilikan (Pod -> ReplicaSet -> Deployment), penjadwalan (Pod -> Node), dan pengamatan (Pod -> deret Prometheus). Graph dijaga segar oleh sebuah sink kube-state-metrics dan di-sample ulang pada setiap alert.

Saat alert menyala, agen menentukan akar masalah dari objek yang terdampak. Ia menelusuri edge, menarik potongan telemetry yang relevan (15 menit terakhir), lalu menyusun sebuah hipotesis. Hipotesis diperingkat berdasarkan bukti: berapa banyak sitasi telemetry yang mendukungnya, seberapa baru, seberapa spesifik. Hipotesis top-3 dikirim ke Slack dengan visualisasi graph-path dan tombol persetujuan untuk aksi remediation.

Remediation dijaga. Aksi default yang diizinkan bersifat read-only. Aksi destruktif (scaling down, rollback, menghapus Pod) memerlukan persetujuan Slack; hook rollback ArgoCD memerlukan auth token yang tidak pernah dipegang agen. Log audit mencatat setiap perintah yang *dipertimbangkan* agen — bukan hanya yang dieksekusi — supaya proses review menangkap near-miss.

## Arsitektur

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## Stack

- Sumber observability: Prometheus, Loki, Tempo, kube-state-metrics
- Knowledge graph: Neo4j (managed) atau kuzu (embedded) berisi objek K8s + edge telemetry
- Agen: LangGraph dengan allow-list per-tool, read-only secara default
- Transport tool: FastMCP di atas StreamableHTTP; server terpisah untuk tool destruktif di balik gerbang persetujuan
- Model: Claude Sonnet 4.7 untuk penalaran akar masalah, Gemini 2.5 Flash untuk peringkasan log
- Remediation: webhook rollback ArgoCD, eskalasi PagerDuty, kartu persetujuan Slack
- Audit: log terstruktur append-only (dipertimbangkan, dieksekuti, disetujui, hasil)
- Deployment: deployment K8s dengan role RBAC sempit miliknya sendiri; namespace terpisah

```figure
ce-rootcause-walk
```

## Bangun Sendiri

1. **Ingestion graph.** Sink kube-state-metrics ke Neo4j/kuzu setiap 30 detik. Node: Pod, Deployment, Node, Service, PVC, HPA. Edge: OWNED_BY, SCHEDULED_ON, EXPOSES, MOUNTS, SCALES. Edge overlay telemetry: OBSERVED_BY (sebuah Pod diamati oleh sebuah deret Prometheus).

2. **Penerima alert.** Endpoint FastAPI yang menerima webhook PagerDuty atau Alertmanager. Ekstrak objek yang terdampak dan pelanggaran SLO-nya.

3. **Permukaan tool read-only.** Bungkus kubectl, query Prometheus, Loki logql, Tempo traceql lewat FastMCP. Setiap tool punya verb RBAC sempit ("get", "list", "describe"). Tidak ada "delete", "exec", "scale" di server default.

4. **Agen akar masalah.** LangGraph dengan tiga node: `sample` menarik potongan telemetry 15-menit terakhir, `walk` men-query graph untuk objek tetangga, `hypothesize` menyusun kandidat akar masalah yang berperingkat dengan sitasi telemetry.

5. **Skoring bukti.** Setiap hipotesis punya skor = recency * specificity * invers panjang graph-path * jumlah sitasi. Kembalikan top-3.

6. **Brief Slack.** Posting sebuah lampiran berisi hipotesis, visualisasi graph-path (gambar subgraph yang di-render di sisi server), dan tombol persetujuan untuk setidaknya satu aksi remediation.

7. **Gerbang remediation.** Tool destruktif (scale down, rollback, delete) berada di server MCP kedua di balik token persetujuan. Agen hanya bisa memanggilnya setelah kartu Slack disetujui manusia.

8. **Log audit.** JSONL append-only: untuk setiap kandidat perintah, catat apakah ia dipertimbangkan, apakah ia dieksekusi, dan siapa yang menyetujuinya. Kirim ke S3 setiap hari.

9. **Suite insiden sintetis.** Bangun 20 skenario: kaskade OOMKill, DNS flap, HPA thrash, PVC penuh, noisy neighbor, sidecar cacat, rollout ConfigMap yang buruk, rotasi sertifikat, image-pull backoff, dll. Nilai agen pada akurasi akar masalah dan time-to-hypothesis.

## Gunakan

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## Rilis

`outputs/skill-devops-agent.md` adalah deliverable-nya. Diberi sebuah kluster K8s dan sumber alert, agen menghasilkan hipotesis akar masalah yang berperingkat serta alur remediation yang dijaga Slack.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | RCA accuracy on scenario suite | ≥80% correct root cause across 20 synthetic incidents |
| 20 | Safety | Destructive-action guard never fires without Slack approval in the audit log |
| 20 | Time-to-hypothesis | p50 under 5 minutes from alert to Slack brief |
| 20 | Explainability | Every hypothesis has graph paths and telemetry citations |
| 15 | Integration completeness | PagerDuty, Slack, ArgoCD, Prometheus end-to-end working |
| **100** | | |

## Latihan

1. Jalankan agenmu pada tiga insiden yang sama dengan yang didemokan DevOps Agent milik AWS. Terbitkan perbandingan berdampingan. Laporkan di mana agen berbeda.

2. Tambahkan audit "near-miss" yang menandai perintah apa pun yang *dipertimbangkan* agen dan akan bersifat destruktif tanpa persetujuan. Ukur tingkat near-miss selama satu minggu.

3. Tukar model hipotesis dari Claude Sonnet 4.7 ke Llama 3.3 70B self-hosted. Ukur selisih akurasi RCA dan dolar per insiden.

4. Bangun sebuah filter kausal: bedakan puncak telemetry yang berkorelasi dengan akar masalah yang sebenarnya. Latih classifier kecil pada label 20-skenario.

5. Tambahkan dry-run rollback: rollback ArgoCD terhadap kluster staging dengan manifest yang sama. Verifikasi rencana rollback di kluster live sebelum tombol persetujuan Slack.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| K8s knowledge graph | "Cluster graph" | Nodes = K8s objects + telemetry series; edges = ownership, scheduling, observation |
| Read-only-by-default | "Scoped RBAC" | Agent's service account has only get/list/describe verbs; destructive verbs live in a separate server behind approval |
| Audit log | "Considered vs executed" | Append-only record of every candidate command, whether it ran, who approved |
| Hypothesis ranking | "Evidence score" | Recency × specificity × graph-path length inverse × citation count |
| Slack approval card | "HITL gate" | Interactive Slack message with remediation buttons; agent cannot proceed until a human clicks |
| Telemetry citation | "Evidence pointer" | A Prometheus query, Loki selector, or Tempo trace URL that supports a claim |
| MTTR | "Time to resolution" | Wall-clock from alert fire to SLO recovery |

## Bacaan Lanjutan

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — referensi kanonikal 2026
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — referensi kompetitor
- [NeuBird semantic monitoring](https://www.neubird.ai) — pendekatan graph semantik
- [Metoro AI SRE](https://metoro.io) — framing produksi SLO-first
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — sumber state kluster
- [LangGraph](https://langchain-ai.github.io/langgraph/) — orkestrator agen referensi
- [FastMCP](https://github.com/jlowin/fastmcp) — framework server MCP berbasis Python
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — target remediation yang dijaga gerbang
