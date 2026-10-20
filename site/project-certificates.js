(function (scope) {
  'use strict';
  function validate(report, project) {
    if (!report || report.schemaVersion !== 1 || !Array.isArray(report.projects)) throw new Error('Pilih laporan JSON dari grader proyek.');
    var matches = report.projects.filter(function (p) { return p.id === project.id; });
    if (matches.length !== 1) throw new Error('Laporan ini harus berisi satu hasil untuk proyek ini.');
    var result = matches[0];
    if (result.mode !== 'learner') throw new Error('Jalankan grader pada workspace kamu sendiri tanpa --solution.');
    if (!project.manifestHash || result.manifestHash !== project.manifestHash) throw new Error('Proyek telah berubah. Jalankan grader saat ini lagi.');
    if (!result.allStagesPassed || !result.certificateEligible || !Array.isArray(result.stages) || result.stages.length !== project.stages.length) throw new Error('Luluskan setiap tahap dengan --all --strict sebelum mengimpor laporan.');
    project.stages.forEach(function (stage, i) {
      var row = result.stages[i];
      if (!row || row.id !== stage.id || row.status !== 'pass' || !Number.isInteger(row.tests) || row.tests < 1 || row.skippedTests !== 0) throw new Error('Setiap tahap memerlukan tes yang lulus tanpa yang dilewati.');
    });
    if (!report.generatedAt || !Number.isFinite(Date.parse(report.generatedAt))) throw new Error('Laporan tidak memiliki tanggal penyelesaian yang valid.');
    return result;
  }
  function escape(value) { return String(value).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function certificate(report, project, name) {
    var result = validate(report, project);
    var tests = result.stages.reduce(function (sum, stage) { return sum + stage.tests; }, 0);
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>Penyelesaian proyek</title><style>body{font-family:Georgia,serif;color:#172436;background:#f7f6ef;max-width:900px;margin:70px auto;padding:50px;border:3px solid #3553ff}h1{font-size:44px}h2{font-size:32px}p{line-height:1.7}small{display:block;margin-top:60px}code{overflow-wrap:anywhere}@media print{body{margin:0}}</style><p>AI ENGINEERING FROM SCRATCH</p><h1>Sertifikat penyelesaian proyek</h1><p>Diberikan kepada</p><h2>' + escape(name.trim().slice(0, 100)) + '</h2><p>Karena telah menyelesaikan <strong>' + escape(project.title) + '</strong>.</p><p>' + project.stages.length + ' tahap · ' + tests + ' tes lulus · ' + escape(report.generatedAt.slice(0, 10)) + '</p><p>Bahasa: ' + escape(project.languages.join(', ')) + '</p><small>Kursus komunitas independen. Berdasarkan laporan grader lokal yang disediakan peserta; bukan kredensial yang diawasi, diverifikasi secara independen, atau dari vendor.</small><p>Sidik jari kurikulum: <code>' + escape(result.manifestHash) + '</code></p></html>';
  }
  function mount(host, project) {
    host.innerHTML = '<div class="pj-eyebrow">SERTIFIKAT PROYEK</div><h3>Simpan bukti pekerjaan kamu.</h3><p>Luluskan setiap tahap di workspace kamu, lalu impor laporan grader untuk mengunduh sertifikat penyelesaian.</p><pre class="pj-cmd"><code>python3 scripts/project_test.py ' + escape(project.id) + ' --all --strict --path my-' + escape(project.id) + ' --report completion.json</code></pre><div class="pj-certificate-fields"><label>Nama kamu<input id="pjCertificateName" maxlength="100" autocomplete="name"></label><label>Laporan grader<input id="pjCertificateFile" type="file" accept="application/json,.json"></label><button type="button" class="pj-action" id="pjCertificateDownload" disabled>Unduh sertifikat</button></div><p id="pjCertificateStatus" role="status">Sertifikat menggunakan bukti lokal. Ini bukan kredensial yang diawasi atau dari vendor.</p>';
    var report = null;
    var status = host.querySelector('#pjCertificateStatus');
    var button = host.querySelector('#pjCertificateDownload');
    var name = host.querySelector('#pjCertificateName');
    function enabled() { button.disabled = !report || !name.value.trim(); }
    name.addEventListener('input', enabled);
    host.querySelector('#pjCertificateFile').addEventListener('change', async function (event) {
      report = null; enabled();
      try {
        var file = event.target.files[0];
        if (!file || file.size > 2000000) throw new Error('Pilih laporan JSON grader yang lebih kecil dari 2 MB.');
        var parsed = JSON.parse(await file.text());
        validate(parsed, project); report = parsed;
        status.textContent = 'Semua tahap lulus dalam laporan peserta yang diimpor. Masukkan nama kamu untuk mengunduh.';
      } catch (error) { status.textContent = error.message; }
      enabled();
    });
    button.addEventListener('click', function () {
      if (button.disabled) return;
      var url = URL.createObjectURL(new Blob([certificate(report, project, name.value)], { type: 'text/html' }));
      var link = document.createElement('a'); link.href = url; link.download = project.id + '-certificate.html'; link.click();
      setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    });
  }
  var api = { validate: validate, certificate: certificate, mount: mount };
  if (typeof module !== 'undefined') module.exports = api;
  else scope.AIFSProjectCertificates = api;
}(typeof window !== 'undefined' ? window : globalThis));
