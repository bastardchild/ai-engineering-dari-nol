# Compliance — SOC 2, HIPAA, GDPR, PCI-DSS, EU AI Act, ISO 42001

> Cakupan multi-framework adalah table stakes untuk deal enterprise 2026. **EU AI Act**: berlaku sejak 1 Agustus 2024. Kebanyakan syarat high-risk ditegakkan 2 Agustus 2026. Denda hingga €15M atau 3% omzet tahunan global untuk kewajiban sistem-high-risk (Art. 99(4)); hingga €35M atau 7% untuk praktik AI terlarang (Art. 99(3)). Berlaku global bila melayani pengguna UE. **Colorado AI Act**: efektif 30 Juni 2026 (ditunda dari Februari 2026 oleh SB25B-004) — penilaian dampak untuk sistem high-risk, hak banding atas keputusan AI. Virginia serupa untuk kredit/pekerjaan/perumahan/pendidikan. **SOC 2 Type II**: syarat B2B AI de facto (Type II, bukan Type I, untuk fintech). **GDPR**: denda spesifik-AI terdokumentasi terbesar adalah €30,5M terhadap Clearview AI (DPA Belanda, Sep 2024); Garante Italia menerbitkan €15M terhadap OpenAI pada Des 2024 (kemudian dibatalkan pada banding Maret 2026). Redaksi PII real-time pada inferensi adalah standar yang defensible; cleanup pasca-pemrosesan tidak cukup. **HIPAA**: terikat-healthcare — tidak boleh mengirim PHI ke layanan AI eksternal tanpa BAA. **PCI-DSS**: cakupan lapisan-interaksi-AI butuh konfigurasi + perjanjian kontraktual, bukan otomatis. **ISO 42001**: standar tata-kelola AI yang muncul, syarat pengadaan yang tumbuh berdampingan ISO 27001. Profil referensi: OpenAI memelihara SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS untuk komponen pembayaran ChatGPT. Pemetaan lintas-framework mengurangi kelelahan audit: kontrol akses memetakan lintas ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a).

**Type:** Learn
**Languages:** (Python optional — compliance is policy + process, not code)
**Prerequisites:** Phase 17 · 25 (Security), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Enumerasikan tujuh framework 2026 yang relevan untuk produk LLM dan pasangkan masing-masing dengan segmen pelanggan.
- Kutip timeline penegakan EU AI Act (berlaku Agustus 2024; penegakan high-risk Agustus 2026) dan plafon denda dua-tier (€15M / 3% untuk kewajiban high-risk, €35M / 7% untuk praktik terlarang).
- Jelaskan mengapa cleanup PII pasca-pemrosesan tidak cukup untuk GDPR dan sebutkan redaksi real-time lapisan-inferensi sebagai standar defensible.
- Jelaskan pemetaan kontrol lintas-framework (mis. kontrol akses memetakan ke ISO 27001 A.5.15-5.18 + GDPR Art. 32 + HIPAA §164.312(a)).

## Masalahnya

Pengadaan pelanggan enterprise meminta SOC 2 Type II, GDPR, BAA HIPAA, ISO 27001, dan "pernyataan compliance EU AI Act." Timmu punya SOC 2 Type I. Kamu enam bulan dari Type II dan belum memulai catatan Article 30 GDPR.

Cakupan multi-framework bukan masalah LLM — melainkan masalah enterprise-SaaS, dengan overlay spesifik-LLM. Tim pengadaan pada 2026 menginginkan matriks dengan baris per framework dan kolom per kontrol, bukan PDF.

## Konsepnya

### Tujuh framework

| Framework | Scope | LLM-specific requirement |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS baseline | Process controls audited over 6-12 months |
| HIPAA | US healthcare | BAA required; PHI cannot leave infrastructure without signed agreement |
| GDPR | EU users | Real-time PII redaction; data subject rights; Article 30 records |
| PCI-DSS | Payment data | Configuration + contracts for AI touching payment |
| EU AI Act | Serving EU users | Risk tier classification; high-risk systems: conformity assessment, documentation, logging |
| Colorado AI Act | Serving CO residents | Impact assessments; right to appeal |
| ISO 42001 | AI governance | Emerging; pairs with ISO 27001 |

### Timeline EU AI Act

- 1 Agustus 2024: berlaku.
- 2 Februari 2025: praktik-AI terlarang ditegakkan.
- 2 Agustus 2026: sistem high-risk ditegakkan (penilaian kesesuaian, dokumentasi, logging).
- Agustus 2027: sistem high-risk dalam produk di bawah legislasi harmonisasi.

Tier risiko: Tak Dapat Diterima (dilarang), High-risk (kesesuaian + logging), Limited-risk (transparansi), Minimal-risk (tanpa constraint). Kebanyakan SaaS LLM B2B adalah limited-risk; high-risk berlaku untuk pekerjaan, kredit, pendidikan, penegakan hukum, migrasi, layanan esensial.

Denda (Article 99): hingga €15M atau 3% omzet tahunan global untuk pelanggaran kewajiban sistem-high-risk (Art. 99(4)); hingga €35M atau 7% untuk praktik AI terlarang (Art. 99(3)); mana yang lebih tinggi berlaku.

### GDPR — redaksi real-time adalah standar

Cleanup pasca-pemrosesan (redaksi PII setelah LLM melihatnya) bukan posture defensible — model sudah melihat datanya. Redaksi real-time lapisan-inferensi adalah standar 2026:

- Pengenalan entitas sebelum panggilan LLM.
- Tokenisasi konsisten (pendekatan Mesh) menjaga semantik.
- Simpan hanya prompt terredaksi + mentah opt-in yang disetujui.

Penegakan terbaru: €30,5M terhadap Clearview AI (DPA Belanda, Sep 2024) adalah denda GDPR spesifik-AI terdokumentasi terbesar hingga kini; €15M terhadap OpenAI (Garante Italia, Des 2024) adalah denda spesifik-LLM terbesar, meski dibatalkan pada banding Maret 2026 dan putusannya masih dalam review lanjutan. Klaim pasca-pemrosesan gagal dalam audit.

### HIPAA — BAA bukan opsional

Kamu tidak boleh mengirim PHI ke layanan AI eksternal tanpa Business Associate Agreement yang ditandatangani. Ketiga platform LLM hyperscaler (Bedrock, Azure OpenAI, Vertex) menawarkan BAA. API langsung OpenAI menawarkan BAA. API langsung Anthropic menawarkan BAA. Konfirmasi sebelum mengirim PHI.

### SOC 2 Type II

Type I: kontrol dirancang dan didokumentasikan.
Type II: kontrol beroperasi efektif selama 6-12 bulan.

Pengadaan B2B pada 2026 default ke Type II. Type I adalah starter; Type II adalah gerbang.

Pendorong audit umum: log akses (siapa melihat apa), manajemen perubahan (bagaimana di-deploy), penilaian risiko (kuartalan), respons insiden (teruji?). Log audit dari Fase 17 · 25 bisa dipakai ulang langsung.

### Pemetaan lintas-framework

Satu kebijakan kontrol akses memenuhi banyak kontrol framework:

| Control | Frameworks |
|---------|-----------|
| Access logging | ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a) |
| Change management | ISO 27001 A.8.32, PCI DSS Req. 6, HIPAA breach-notification scope |
| Encryption in transit | ISO 27001 A.8.24, GDPR Art. 32, HIPAA §164.312(e) |
| Secrets management | ISO 27001 A.8.19, PCI DSS Req. 8, SOC 2 CC6.1 |

Tool compliance (Drata, Vanta, Secureframe) mengotomatisasi pemetaan ini. Layak dengan biayanya pada skala.

### ISO 42001 — muncul

Dipublikasikan akhir 2023. Syarat pengadaan yang tumbuh berdampingan ISO 27001. Framework tata-kelola AI mencakup manajemen risiko, kualitas data, transparansi, oversight manusia.

### Profil referensi OpenAI

OpenAI memelihara SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS untuk komponen pembayaran ChatGPT. Itu kira-kira table stakes enterprise pada 2026.

### Angka yang perlu kamu ingat

- Denda EU AI Act: hingga €15M / 3% (kewajiban high-risk, Art. 99(4)); hingga €35M / 7% (praktik terlarang, Art. 99(3)).
- Penegakan high-risk EU AI Act: 2 Agustus 2026.
- Denda GDPR spesifik-AI terdokumentasi terbesar: €30,5M, Clearview AI (DPA Belanda, Sep 2024).
- Denda GDPR spesifik-LLM terbesar: €15M, OpenAI (Garante Italia, Des 2024; dibatalkan pada banding Maret 2026).
- Window SOC 2 Type II: 6-12 bulan kontrol yang dioperasikan.
- Tanggal efektif Colorado AI Act: 30 Juni 2026 (ditunda dari Februari 2026 oleh SB25B-004).

```figure
i4-control-matrix
```

## Gunakan

`code/main.py` adalah spreadsheet pemetaan-compliance dalam Python — dengan kontrol tertentu, mendaftar framework yang dipenuhinya.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-compliance-matrix.md`. Dengan segmen pelanggan dan geografi, ia menspesifikasikan framework dan kontrol yang dibutuhkan.

## Latihan

1. Pelanggan enterprise pertamamu mewajibkan SOC 2 Type II, BAA HIPAA, pernyataan EU AI Act. Apa posture compliance minimum-viable untuk memenangkan deal?
2. Klasifikasikan tiga produk LLM hipotetis pada tier risiko EU AI Act. Apa yang berubah pada high-risk?
3. Kamu tak sengaja mengirim PHI ke provider tanpa BAA. Jalani respons insidennya.
4. Argumenkan apakah ISO 42001 "perlu pada 2026" bagi vendor AI mid-market.
5. Petakan field log audit LLM-mu (Fase 17 · 25) ke minimal tiga kontrol framework.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| SOC 2 Type II | "audited controls" | Controls operating over 6-12 months, independently attested |
| HIPAA BAA | "healthcare contract" | Business Associate Agreement; required for PHI |
| GDPR | "EU privacy" | Real-time PII redaction is the defensible 2026 standard |
| EU AI Act | "EU AI rules" | High-risk enforcement August 2026; €15M / 3% (high-risk obligations) — €35M / 7% (prohibited practices) |
| Colorado AI Act | "US AI state law" | June 30, 2026 effective (delayed by SB25B-004); impact assessments |
| ISO 42001 | "AI governance" | Emerging framework for AI risk + transparency |
| ISO 27001 | "security ISMS" | Information Security Management System baseline |
| Conformity assessment | "EU AI doc package" | High-risk requirement: docs, testing, logging |
| Cross-framework mapping | "one control, many frames" | Single policy satisfies multiple framework controls |

## Further Reading

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — reference compliance profile.
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — primary source.
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — primary source.
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI management system standard.
