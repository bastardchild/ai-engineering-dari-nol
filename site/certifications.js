(function () {
  'use strict';

  var root = document.documentElement;
  var assessmentTimer = null;
  var GITHUB_BLOB_BASE = 'https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/';

  function esc(value) {
    var div = document.createElement('div');
    div.textContent = value == null ? '' : String(value);
    return div.innerHTML;
  }

  function attr(value) {
    return esc(value).replace(/`/g, '&#96;');
  }

  function data() {
    return typeof CERTIFICATIONS !== 'undefined' && CERTIFICATIONS
      ? CERTIFICATIONS
      : { programs: [], tracks: [], lessonsByPath: {}, assessmentsById: {} };
  }

  function tracks() {
    return Array.isArray(data().tracks) ? data().tracks : [];
  }

  function programs() {
    return Array.isArray(data().programs) ? data().programs : [];
  }

  function programForTrack(track) {
    var id = track && track.programId;
    return programs().find(function (program) { return program.id === id; }) || {};
  }

  function githubBlobUrl(relativePath) {
    return relativePath ? GITHUB_BLOB_BASE + relativePath : '';
  }

  function query(name) {
    return new URLSearchParams(window.location.search).get(name) || '';
  }

  function prefersReducedMotion() {
    return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  }

  function scrollToPageTop() {
    window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
  }

  function focusWithoutScrolling(element) {
    if (!element || typeof element.focus !== 'function') return;
    try {
      element.focus({ preventScroll: true });
    } catch (_) {
      element.focus();
    }
  }

  function updateAssessmentResultUrl(id, showResult) {
    if (!window.history || typeof window.history.replaceState !== 'function') return;
    try {
      var params = new URLSearchParams(window.location.search);
      params.set('id', id);
      if (showResult) params.set('result', 'latest');
      else params.delete('result');
      window.history.replaceState(null, '', (window.location.pathname || 'assessment.html') + '?' + params.toString() + (window.location.hash || ''));
    } catch (_) {}
  }

  function findTrack(id) {
    return tracks().find(function (track) {
      return track.id === id || track.slug === id || track.examCode === id;
    }) || null;
  }

  function lessonRefPath(ref) {
    return typeof ref === 'string' ? ref : ref && ref.path ? ref.path : '';
  }

  function findCoreLesson(path) {
    if (typeof PHASES === 'undefined' || !Array.isArray(PHASES)) return null;
    for (var i = 0; i < PHASES.length; i++) {
      for (var j = 0; j < PHASES[i].lessons.length; j++) {
        var lesson = PHASES[i].lessons[j];
        var match = lesson.url && lesson.url.match(/(phases\/[^/?#]+\/[^/?#]+)/);
        if (match && match[1] === path) {
          return {
            path: path,
            name: lesson.name,
            summary: lesson.summary || '',
            type: lesson.type || '',
            languages: lesson.lang || '',
            phaseId: PHASES[i].id,
            phaseName: PHASES[i].name,
          };
        }
      }
    }
    return null;
  }

  function findKnownLesson(path) {
    var lessonsByPath = data().lessonsByPath || {};
    if (Object.prototype.hasOwnProperty.call(lessonsByPath, path) && lessonsByPath[path]) return lessonsByPath[path];
    return findCoreLesson(path);
  }

  function findLesson(path) {
    return findKnownLesson(path) || {
      path: path,
      name: String(path || '').split('/').pop().replace(/^\d+-/, '').replace(/-/g, ' '),
      summary: '',
    };
  }

  function trackContainsLesson(track, path) {
    return !!(track && Array.isArray(track.lessons) && track.lessons.some(function (ref) {
      return lessonRefPath(ref) === path;
    }));
  }

  function lessonIsComplete(path) {
    if (!path || !window.AIFSProgress || typeof window.AIFSProgress.isLessonComplete !== 'function') return false;
    try {
      return !!window.AIFSProgress.isLessonComplete(path);
    } catch (_) {
      return false;
    }
  }

  function validatedInternalLessonReference(ref) {
    var path = lessonRefPath(ref);
    if (!/^(?:certifications\/[a-z0-9][a-z0-9-]*\/lessons\/[^/?#]+|phases\/[^/?#]+\/[^/?#]+)$/.test(path)) return null;
    var lesson = findKnownLesson(path);
    if (!lesson) return null;
    return {
      path: path,
      lesson: lesson,
      label: ref && typeof ref === 'object' ? ref.label || ref.title || '' : '',
    };
  }

  function lessonReferenceHref(path, sourceTrack) {
    var href = 'lesson?path=' + encodeURIComponent(path);
    if (sourceTrack) {
      href += trackContainsLesson(sourceTrack, path)
        ? '&track=' + encodeURIComponent(sourceTrack.id)
        : '&fromTrack=' + encodeURIComponent(sourceTrack.id);
      return href;
    }
    var containingTrack = tracks().find(function (track) { return trackContainsLesson(track, path); });
    return href + (containingTrack ? '&track=' + encodeURIComponent(containingTrack.id) : '');
  }

  function renderInternalLessonReference(ref, sourceTrack) {
    var validated = validatedInternalLessonReference(ref);
    if (!validated) return '';
    var crossTrack = !!(sourceTrack && !trackContainsLesson(sourceTrack, validated.path));
    var label = validated.label || validated.lesson.name;
    if (crossTrack) label += ' · Suplemen untuk ' + (sourceTrack.examCode || sourceTrack.shortName || sourceTrack.id);
    return '<a href="' + attr(lessonReferenceHref(validated.path, sourceTrack)) + '">' + esc(label) + ' →</a>';
  }

  function examValue(track, names, fallback) {
    var exam = track && track.exam ? track.exam : {};
    for (var i = 0; i < names.length; i++) {
      if (exam[names[i]] !== undefined && exam[names[i]] !== null && exam[names[i]] !== '') return exam[names[i]];
      if (track && track[names[i]] !== undefined && track[names[i]] !== null && track[names[i]] !== '') return track[names[i]];
    }
    return fallback;
  }

  function minutes(track) {
    return examValue(track, ['durationMinutes', 'minutes', 'timeLimitMinutes'], 'Tidak tercantum');
  }

  function unpublished(track, flag) {
    return !!(track && track.exam && track.exam[flag] === false);
  }

  function questions(track) {
    if (unpublished(track, 'itemCountPublished')) return 'Tidak dipublikasikan';
    return examValue(track, ['questionCount', 'questions', 'items'], 'Tidak tercantum');
  }

  function passing(track) {
    if (unpublished(track, 'passingScorePublished')) return 'Tidak dipublikasikan';
    return examValue(track, ['passingScaledScore', 'passingScore', 'passingScaled'], 'Lihat panduan resmi');
  }

  function price(track) {
    var value = examValue(track, ['price', 'fee', 'priceUsd', 'feeUsd'], 'Lihat penyedia resmi');
    if (typeof value === 'number') return '$' + value;
    return value;
  }

  function formatDate(value) {
    if (!value) return '';
    var date = new Date(value);
    if (isNaN(date.getTime())) return String(value);
    return date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
  }

  function setTheme(preferred) {
    var stored = preferred || '';
    if (!stored) {
      try { stored = localStorage.getItem('theme') || ''; } catch (_) {}
    }
    var theme = stored || (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    root.setAttribute('data-theme', theme);
    var icon = document.getElementById('themeIcon');
    if (icon) icon.textContent = theme === 'light' ? 'N' : 'D';
  }

  function initTheme() {
    setTheme();
    var button = document.getElementById('themeToggle');
    if (!button) return;
    button.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      try { localStorage.setItem('theme', next); } catch (_) {}
      setTheme(next);
    });
  }

  function metaChip(text) {
    return text ? '<span class="cert-meta-chip">' + esc(text) + '</span>' : '';
  }

  function examFacts(track) {
    return [
      { value: questions(track), label: 'Jumlah soal', unpublished: unpublished(track, 'itemCountPublished') },
      { value: String(minutes(track)).match(/^\d+$/) ? minutes(track) + ' menit' : minutes(track), label: 'Batas waktu' },
      { value: passing(track), label: 'Skor kelulusan', unpublished: unpublished(track, 'passingScorePublished') },
      { value: price(track), label: 'Biaya ujian' },
      { value: examValue(track, ['format'], 'Buku tertutup'), label: 'Format' },
      { value: examValue(track, ['validityMonths', 'validForMonths'], 'Lihat penyedia'), label: 'Masa berlaku' },
    ];
  }

  function renderCardFacts(track, limit) {
    return examFacts(track).filter(function (fact) { return !fact.unpublished; }).slice(0, limit || 3).map(function (fact) {
      var value = fact.value;
      if (fact.label === 'Masa berlaku' && typeof value === 'number') value += ' bulan';
      return '<div class="cert-card-fact"><strong>' + esc(value) + '</strong><span class="cert-fact-label">' + esc(fact.label) + '</span></div>';
    }).join('');
  }

  function renderTrackBadge(track) {
    var badge = track && track.badge;
    if (!badge || !badge.imageUrl) return '';
    var width = Number(badge.width) || 600;
    var height = Number(badge.height) || width;
    var shapeClass = badge.shape === 'square' ? ' cert-track-badge--square' : '';
    return '<img class="cert-track-badge' + shapeClass + '" src="' + attr(badge.imageUrl) + '" width="' + width + '" height="' + height + '" loading="lazy" decoding="async" referrerpolicy="no-referrer" alt="" aria-hidden="true">';
  }

  function accessNoticeHtml(program, includeAllLinks) {
    var notice = program && program.accessNotice;
    if (!notice) return '';
    var links = Array.isArray(program.officialLinks) ? program.officialLinks.filter(function (link) {
      if (!link || !link.url) return false;
      return includeAllLinks || String(link.label || '').toLowerCase().indexOf('faq') !== -1;
    }) : [];
    return '<div><strong>' + esc(program.accessNoticeTitle || 'Akses ujian resmi') + '</strong><p>' + esc(notice) + '</p></div>' +
      (links.length ? '<div class="cert-source-links" aria-label="Sumber sertifikasi resmi">' + links.map(function (link) {
        return '<a href="' + attr(link.url) + '" target="_blank" rel="noopener">' + esc(link.label || 'Sumber resmi') + ' ↗</a>';
      }).join('') + '</div>' : '');
  }

  function renderAccessNotice(id, program, includeAllLinks) {
    var mount = document.getElementById(id);
    if (!mount) return;
    var html = accessNoticeHtml(program, includeAllLinks);
    mount.hidden = !html;
    mount.innerHTML = html;
  }

  function disclaimerParagraphsHtml(program) {
    return '<p>' + esc(program.disclaimer) + '</p>' +
      (program.scoringNotice ? '<p>' + esc(program.scoringNotice) + '</p>' : '');
  }

  function programNoticeHtml(program, heading) {
    return '<strong>' + esc(heading) + '</strong>' + disclaimerParagraphsHtml(program);
  }

  function renderProgramNotice(id, program, heading) {
    var mount = document.getElementById(id);
    if (!mount || !program || !program.disclaimer) return;
    mount.innerHTML = programNoticeHtml(program, heading);
  }

  function programLinksHtml(program) {
    var label = program.shortName || program.name || 'sertifikasi ini';
    var links = [];
    if (program.learnerGuidePath) {
      links.push('<a class="cert-action secondary" href="' + attr(githubBlobUrl(program.learnerGuidePath)) + '" target="_blank" rel="noopener" aria-label="' + attr('Pelajari ' + label + ' dengan tutor AI di GitHub, terbuka di tab baru') + '">Pelajari dengan tutor AI di GitHub ↗</a>');
    }
    if (program.tutorSkillPath) {
      links.push('<a class="cert-action secondary" href="' + attr(githubBlobUrl(program.tutorSkillPath)) + '" target="_blank" rel="noopener" aria-label="' + attr('Baca skill tutor ' + label + ' di GitHub, terbuka di tab baru') + '">Baca skill tutor ↗</a>');
    }
    return links.length ? '<div class="cert-track-hero-actions cert-program-actions">' + links.join('') + '</div>' : '';
  }

  function joinLabels(labels) {
    if (labels.length < 2) return labels.join('');
    return labels.slice(0, -1).join(', ') + ' dan ' + labels[labels.length - 1];
  }

  function latestVerified(list) {
    return list.map(function (program) {
      return program.verifiedAt || program.lastVerified || program.updatedAt || '';
    }).filter(Boolean).sort().pop() || '';
  }

  function renderTrackCard(track, index) {
    var domains = Array.isArray(track.domains) ? track.domains.length : 0;
    var lessonCount = Array.isArray(track.lessons) ? track.lessons.length : 0;
    var delay = Math.min(index * 30, 80);
    var badge = renderTrackBadge(track);
    return '<a class="cert-track-card cert-catalog-arrival" style="--cert-arrival-delay:' + delay + 'ms" href="certification?id=' + encodeURIComponent(track.id) + '">' +
      '<div class="cert-card-top"><span class="cert-card-code">' + esc(track.examCode || track.shortName || track.slug) + '</span><span class="cert-status">' + esc(track.level || 'Jalur belajar') + '</span></div>' +
      '<div class="cert-card-identity' + (badge ? ' has-badge' : '') + '"><h3>' + esc(track.credential || track.title || track.shortName || track.id) + '</h3>' + badge + '</div>' +
      '<p>' + esc(track.summary || track.audience || 'Rute praktis melalui blueprint sertifikasi ini.') + '</p>' +
      '<div class="cert-card-facts">' + renderCardFacts(track, 3) + '</div>' +
      '<div class="cert-card-footer"><span>' + lessonCount + ' pelajaran · ' + domains + ' domain</span><span>Buka jalur →</span></div>' +
    '</a>';
  }

  function renderCatalog() {
    var certs = data();
    var summary = document.getElementById('certProgramSummary');
    var meta = document.getElementById('certProgramMeta');
    var mount = document.getElementById('certProgramSections');
    if (!mount) return;

    var available = programs().filter(function (program) {
      return tracks().some(function (track) { return track.programId === program.id; });
    });
    if (!available.length) {
      mount.innerHTML = '<div class="cert-container cert-section"><div class="cert-empty">Track sertifikasi sedang disusun. Jalankan <code>node site/build.js</code> setelah menambahkan manifest program.</div></div>';
      if (summary) summary.textContent = 'Katalog sertifikasi lokal belum dibuat.';
      return;
    }

    var labels = joinLabels(available.map(function (program) { return program.shortName || program.name || program.id; }));
    if (summary) summary.textContent = 'Persiapan gratis, independen, dan open-source untuk kredensial teknik AI. Tersedia jalur ' + labels + ' sekarang, dengan lebih banyak keluarga sertifikasi segera hadir.';
    if (meta) {
      var verified = latestVerified(available);
      meta.innerHTML = metaChip(labels + ' tersedia sekarang') +
        metaChip(tracks().length + ' track berbasis peran') +
        metaChip(Object.keys(certs.lessonsByPath || {}).length + ' pelajaran sertifikasi') +
        metaChip(verified ? 'Terverifikasi ' + formatDate(verified) : 'Materi sumber dengan versi');
    }
    var notice = document.getElementById('certProgramNotice');
    var disclaimed = available.filter(function (program) { return program.disclaimer; });
    if (notice && disclaimed.length) {
      notice.innerHTML = '<strong>Persiapan independen</strong>' + disclaimed.map(disclaimerParagraphsHtml).join('');
    }

    var cardIndex = 0;
    mount.innerHTML = available.map(function (program) {
      var headingId = 'certProgram-' + program.id;
      var access = accessNoticeHtml(program, true);
      var cards = tracks().filter(function (track) { return track.programId === program.id; }).map(function (track) {
        return renderTrackCard(track, cardIndex++);
      }).join('');
      return '<section class="cert-container cert-section cert-program-section" aria-labelledby="' + attr(headingId) + '">' +
        '<div class="cert-section-heading"><div><div class="cert-eyebrow">' + esc(program.provider || 'Program sertifikasi') + '</div><h2 id="' + attr(headingId) + '">' + esc(program.name || program.id) + '</h2></div><p>' + esc(program.summary || '') + '</p></div>' +
        (access ? '<aside class="cert-access-notice" aria-label="' + attr((program.shortName || program.name || 'Sertifikasi') + ': akses ujian resmi') + '">' + access + '</aside>' : '') +
        programLinksHtml(program) +
        '<div class="cert-track-grid">' + cards + '</div>' +
      '</section>';
    }).join('');
  }

  function completedLessons(track) {
    var count = 0;
    var refs = Array.isArray(track.lessons) ? track.lessons : [];
    refs.forEach(function (ref) {
      var path = lessonRefPath(ref);
      if (lessonIsComplete(path)) count++;
    });
    return count;
  }

  function assessmentVersion(assessment) {
    return assessment && assessment.version != null ? assessment.version : 1;
  }

  function assessmentBest(id) {
    var assessment = data().assessmentsById[id] || null;
    if (!window.AIFSCertProgress || typeof window.AIFSCertProgress.getBest !== 'function') return null;
    try {
      var best = window.AIFSCertProgress.getBest(id, assessmentVersion(assessment));
      return isRecord(best) && isFiniteNumber(best.percent) && best.percent >= 0 && best.percent <= 100 ? best : null;
    } catch (_) {
      return null;
    }
  }

  function assessmentAttempts(id, version) {
    if (!window.AIFSCertProgress || typeof window.AIFSCertProgress.getAttempts !== 'function') return [];
    try {
      var attempts = window.AIFSCertProgress.getAttempts(id, version);
      if (!Array.isArray(attempts)) return [];
      return attempts.filter(function (attempt) {
        return isRecord(attempt) && isRecord(attempt.answers) &&
          isFiniteNumber(attempt.startedAt) && attempt.startedAt > 0 &&
          isFiniteNumber(attempt.durationMs) && attempt.durationMs >= 0;
      });
    } catch (_) {
      return [];
    }
  }

  function progressIsPersistent() {
    return !!(window.AIFSCertProgress && window.AIFSCertProgress.isPersistent && window.AIFSCertProgress.isPersistent());
  }

  function renderTrackSeo(track) {
    var name = track.credential || track.title || track.shortName || track.id;
    document.title = name + ' - Persiapan sertifikasi gratis';
    var description = track.summary || 'Persiapan sertifikasi gratis dan independen dengan pelajaran dan latihan orisinal.';
    var desc = document.querySelector('meta[name="description"]');
    if (desc) desc.setAttribute('content', description);
    var canonical = document.querySelector('link[rel="canonical"]');
    if (canonical) canonical.setAttribute('href', 'https://aiengineeringfromscratch.com/certification?id=' + encodeURIComponent(track.id));
  }

  function renderTrackNotFound(hero) {
    document.title = 'Track sertifikasi tidak ditemukan - AI Engineering from Scratch';
    var description = document.querySelector('meta[name="description"]');
    if (description) description.setAttribute('content', 'Pilih track persiapan sertifikasi yang tersedia.');
    var canonical = document.querySelector('link[rel="canonical"]');
    if (canonical) canonical.setAttribute('href', 'https://aiengineeringfromscratch.com/certifications.html');
    var breadcrumb = document.getElementById('trackBreadcrumb');
    if (breadcrumb) breadcrumb.textContent = 'Tidak ditemukan';
    var main = document.getElementById('main');
    if (main) {
      Array.from(main.querySelectorAll('.cert-section, .cert-access-notice, .cert-notice')).forEach(function (section) {
        section.hidden = true;
      });
    }
    hero.classList.add('is-not-found');
    hero.innerHTML = '<div class="cert-track-not-found-copy"><div class="cert-eyebrow">TRACK TIDAK DITEMUKAN</div><h1>Pilih jalur sertifikasi yang tersedia.</h1><p class="cert-track-summary">ID track ini tidak cocok dengan katalog sertifikasi saat ini.</p><div class="cert-track-hero-actions"><a class="cert-action" href="certifications.html">Semua sertifikasi</a></div></div>';
  }

  function renderTrack() {
    var id = query('id') || query('track');
    var track = findTrack(id);
    var hero = document.getElementById('trackHero');
    if (!hero) return;
    if (!track) {
      renderTrackNotFound(hero);
      return;
    }

    renderTrackSeo(track);
    var program = programForTrack(track);
    var tutorUrl = githubBlobUrl(program.learnerGuidePath);
    var breadcrumb = document.getElementById('trackBreadcrumb');
    if (breadcrumb) breadcrumb.textContent = track.examCode || track.shortName || track.id;
    var refs = Array.isArray(track.lessons) ? track.lessons : [];
    var complete = completedLessons(track);
    var firstIncomplete = refs.find(function (ref) {
      var path = lessonRefPath(ref);
      return path && !lessonIsComplete(path);
    }) || refs[0];
    var firstPath = lessonRefPath(firstIncomplete);

    hero.innerHTML = '<div class="cert-eyebrow">' + esc(track.examCode || track.shortName || 'TRACK SERTIFIKASI') + '</div>' +
      '<h1>' + esc(track.credential || track.title || track.shortName || track.id) + '</h1>' +
      '<p class="cert-track-summary">' + esc(track.summary || track.audience || '') + '</p>' +
      '<div class="cert-meta-row">' + examFacts(track).map(function (fact) {
        var value = fact.value;
        if (fact.label === 'Masa berlaku' && typeof value === 'number') value += ' bulan';
        return metaChip(fact.label + ': ' + value);
      }).join('') + '</div>' +
      '<div class="cert-track-hero-actions">' +
        (firstPath ? '<a class="cert-action" href="lesson?path=' + encodeURIComponent(firstPath) + '&track=' + encodeURIComponent(track.id) + '">' + (complete ? 'Lanjutkan jalur' : 'Mulai belajar') + '</a>' : '') +
        '<a class="cert-action secondary" href="#trackAssessments">Latihan kesiapan</a>' +
        (tutorUrl ? '<a class="cert-action secondary" href="' + attr(tutorUrl) + '" target="_blank" rel="noopener" aria-label="Pelajari dengan tutor AI di GitHub (terbuka di tab baru)">Pelajari dengan tutor AI di GitHub ↗</a>' : '') +
        (track.exam && track.exam.officialGuideUrl ? '<a class="cert-action secondary" href="' + attr(track.exam.officialGuideUrl) + '" target="_blank" rel="noopener">Panduan ujian resmi</a>' : '') +
      '</div>';

    renderAccessNotice('trackAccessNotice', program, false);
    renderProgramNotice('trackProgramNotice', program, 'Persiapan independen');
    renderTrackProgress(track);
    renderDomains(track);
    renderLessons(track);
    renderDeepDives(track);
    renderAssessments(track);
    renderStudyPlans(track);

    if (window.AIFSProgress) {
      window.AIFSProgress.onChange(function () {
        renderTrackProgress(track);
        renderLessons(track);
      });
    }
    if (window.AIFSCertProgress) {
      window.AIFSCertProgress.onChange(function () { renderAssessments(track); });
    }
  }

  function renderTrackProgress(track) {
    var mount = document.getElementById('trackProgress');
    if (!mount) return;
    var total = Array.isArray(track.lessons) ? track.lessons.length : 0;
    var complete = completedLessons(track);
    var percent = total ? Math.round((complete / total) * 100) : 0;
    var assessments = Array.isArray(track.assessments) ? track.assessments : [];
    var attempted = assessments.filter(function (meta) { return assessmentBest(meta.id); }).length;
    mount.innerHTML = '<div class="cert-progress-head"><div><div class="cert-eyebrow">PROGRES LOKAL</div><h2 id="progressTitle">Progres kamu</h2></div><div class="cert-progress-number">' + percent + '%</div></div>' +
      '<div class="cert-progress-bar" role="progressbar" aria-labelledby="progressTitle" aria-valuemin="0" aria-valuemax="100" aria-valuenow="' + percent + '"><div class="cert-progress-fill" style="--cert-progress:' + (percent / 100) + '"></div></div>' +
      '<div class="cert-progress-detail"><span>' + complete + ' dari ' + total + ' pelajaran selesai</span><span>' + attempted + ' dari ' + assessments.length + ' asesmen dicoba</span><span>' + (progressIsPersistent() ? 'Hanya tersimpan di browser ini' : 'Penyimpanan browser tidak tersedia; progres bersifat sementara') + '</span></div>';
  }

  function renderDomains(track) {
    var mount = document.getElementById('trackDomains');
    if (!mount) return;
    var domains = Array.isArray(track.domains) ? track.domains : [];
    mount.innerHTML = domains.length ? domains.map(function (domain) {
      var objectives = Array.isArray(domain.objectives) ? domain.objectives : [];
      return '<article class="cert-domain-row"><div class="cert-domain-weight">' + esc(domain.weight != null ? domain.weight + '%' : 'INTI') + '</div><div><h3>' + esc(domain.name || domain.title || domain.id) + '</h3><div class="cert-card-code">' + esc(domain.id || '') + '</div></div><ul class="cert-objectives">' + objectives.map(function (objective) { return '<li>' + esc(objective) + '</li>'; }).join('') + '</ul></article>';
    }).join('') : '<div class="cert-empty">Detail domain belum tersedia di manifest track ini.</div>';
  }

  function renderLessons(track) {
    var mount = document.getElementById('trackLessons');
    if (!mount) return;
    var refs = Array.isArray(track.lessons) ? track.lessons : [];
    mount.innerHTML = refs.length ? refs.map(function (ref, index) {
      var normalized = typeof ref === 'string' ? { path: ref } : ref;
      var path = normalized.path;
      var lesson = findLesson(path);
      var done = lessonIsComplete(path);
      var domains = Array.isArray(normalized.domains) ? normalized.domains : [];
      var origin = path.indexOf('phases/') === 0 ? 'Kurikulum inti' : (normalized.role || lesson.type || 'Pelajaran sertifikasi');
      return '<article class="cert-lesson-row' + (done ? ' is-complete' : '') + '">' +
        '<div class="cert-lesson-num">' + String(index + 1).padStart(2, '0') + '</div>' +
        '<div class="cert-lesson-copy"><h3>' + esc(lesson.name) + '</h3><p>' + esc((done ? 'Selesai · ' : '') + origin + (lesson.summary ? ' · ' + lesson.summary : '')) + '</p></div>' +
        '<div class="cert-domain-chips">' + domains.map(function (domain) { return '<span class="cert-domain-chip">' + esc(domain) + '</span>'; }).join('') + '</div>' +
        '<a class="cert-lesson-open" href="lesson?path=' + encodeURIComponent(path) + '&track=' + encodeURIComponent(track.id) + '">' + (done ? 'Tinjau' : 'Buka') + ' →</a>' +
      '</article>';
    }).join('') : '<div class="cert-empty">Pelajaran belum ditambahkan ke track ini.</div>';
  }

  function renderDeepDives(track) {
    var section = document.getElementById('trackDeepDiveSection');
    var mount = document.getElementById('trackDeepDives');
    if (!section || !mount) return;
    var refs = Array.isArray(track.deepDives) ? track.deepDives : [];
    section.hidden = refs.length === 0;
    if (!refs.length) {
      mount.innerHTML = '';
      return;
    }
    mount.innerHTML = refs.map(function (ref) {
      var normalized = typeof ref === 'string' ? { path: ref } : ref || {};
      var path = normalized.path || '';
      var lesson = findLesson(path);
      var title = normalized.label || lesson.name;
      var reason = normalized.reason || lesson.summary || 'Konteks tambahan dari kurikulum inti AI Engineering from Scratch.';
      return '<article class="cert-deep-dive-row">' +
        '<div class="cert-deep-dive-badge">OPSIONAL</div>' +
        '<div class="cert-lesson-copy"><h3>' + esc(title) + '</h3><p>' + esc(reason) + '</p></div>' +
        '<a class="cert-lesson-open" href="' + attr(lessonReferenceHref(path, track)) + '">Buka pelajaran →</a>' +
      '</article>';
    }).join('');
  }

  function renderAssessments(track) {
    var mount = document.getElementById('trackAssessments');
    if (!mount) return;
    var assessments = Array.isArray(track.assessments) ? track.assessments : [];
    mount.classList.toggle('cert-two-card-grid', assessments.length === 2 || assessments.length === 4);
    mount.innerHTML = assessments.length ? assessments.map(function (meta) {
      var assessment = data().assessmentsById[meta.id] || meta;
      var best = assessmentBest(meta.id);
      var version = assessmentVersion(assessment);
      var attempts = assessmentAttempts(meta.id, version);
      var latest = attempts[0] || null;
      var count = assessment.questionCount || meta.questionCount || (Array.isArray(assessment.questions) ? assessment.questions.length : 0);
      var limit = assessment.timeLimitMinutes || meta.timeLimitMinutes || 0;
      return '<article class="cert-assessment-card"><div class="cert-assessment-head"><span class="cert-assessment-kind">' + esc(assessment.kind || 'Latihan') + '</span><span class="cert-card-code">' + count + ' soal' + (limit ? ' · ' + limit + ' menit' : '') + '</span></div><h3>' + esc(assessment.title || meta.title || 'Asesmen latihan') + '</h3><p>' + esc(assessment.summary || assessment.description || 'Latihan skenario orisinal dengan umpan balik setelah pengiriman.') + '</p><div class="cert-assessment-best">' + (best ? 'Skor latihan terbaik: ' + best.percent + '%' : 'Belum dicoba di perangkat ini') + '</div><div class="cert-track-hero-actions"><a class="cert-action" href="assessment.html?id=' + encodeURIComponent(meta.id) + '">' + (latest ? 'Coba lagi' : 'Mulai latihan') + '</a>' + (latest ? '<a class="cert-action secondary" href="assessment.html?id=' + encodeURIComponent(meta.id) + '&result=latest">Tinjau hasil terbaru</a>' : '') + '</div></article>';
    }).join('') : '<div class="cert-empty">Asesmen latihan sedang disiapkan untuk track ini.</div>';
  }

  function normalizePlans(value) {
    if (Array.isArray(value)) return value;
    if (!value || typeof value !== 'object') return [];
    return Object.keys(value).map(function (key) {
      var plan = value[key];
      if (typeof plan === 'string') return { title: key, description: plan };
      return Object.assign({ id: key, title: plan.title || key }, plan);
    });
  }

  function renderStudyPlans(track) {
    var mount = document.getElementById('trackStudyPlans');
    if (!mount) return;
    var plans = normalizePlans(track.studyPlans);
    mount.classList.toggle('cert-two-card-grid', plans.length === 2);
    mount.innerHTML = plans.length ? plans.map(function (plan) {
      var steps = plan.steps || plan.schedule || plan.weeks || plan.milestones || [];
      if (!Array.isArray(steps) && steps && typeof steps === 'object') {
        steps = Object.keys(steps).map(function (key) { return key + ': ' + steps[key]; });
      }
      if (!Array.isArray(steps)) steps = [];
      var pace = plan.duration || plan.pace || (plan.durationDays ? plan.durationDays + ' hari' : plan.id || 'Rencana belajar');
      var workload = plan.hoursPerWeek ? plan.hoursPerWeek + ' jam per minggu' : '';
      return '<article class="cert-plan-card"><div class="cert-card-code">' + esc(pace) + '</div><h3>' + esc(plan.title || plan.label || plan.name || 'Rencana belajar') + '</h3><p>' + esc(plan.description || plan.summary || workload) + '</p>' + (steps.length ? '<ul>' + steps.map(function (step) { return '<li>' + esc(typeof step === 'string' ? step : step.title || step.name || JSON.stringify(step)) + '</li>'; }).join('') + '</ul>' : '') + '</article>';
    }).join('') : '<div class="cert-empty">Gunakan urutan pelajaran di atas sebagai rencana belajar bawaan.</div>';
  }

  function normalizeQuestion(question, index) {
    var correct = question.correct;
    if (!Array.isArray(correct)) correct = correct == null ? [] : [correct];
    return {
      id: question.id || 'q-' + (index + 1),
      domain: question.domain || 'general',
      objective: question.objective || '',
      type: question.type === 'multiple' || correct.length > 1 ? 'multiple' : 'single',
      prompt: question.prompt || question.question || '',
      options: Array.isArray(question.options) ? question.options : [],
      correct: correct.map(Number).sort(function (a, b) { return a - b; }),
      explanation: question.explanation || '',
      references: Array.isArray(question.references) ? question.references : [],
    };
  }

  function isRecord(value) {
    return !!value && typeof value === 'object' && !Array.isArray(value);
  }

  function isFiniteNumber(value) {
    return typeof value === 'number' && isFinite(value);
  }

  function validDraftForAssessment(draft, version, questions, limit) {
    if (!isRecord(draft) || !isRecord(draft.answers)) return false;
    if (String(draft.version) !== String(version)) return false;
    if (!isFiniteNumber(draft.startedAt) || draft.startedAt <= 0) return false;
    if (!isFiniteNumber(draft.deadlineAt) || draft.deadlineAt < 0) return false;
    var expectedDeadline = limit ? draft.startedAt + limit * 60000 : 0;
    if (draft.deadlineAt !== expectedDeadline) return false;

    var questionsById = Object.create(null);
    questions.forEach(function (question) { questionsById[question.id] = question; });
    return Object.keys(draft.answers).every(function (id) {
      var question = questionsById[id];
      var selected = draft.answers[id];
      if (!question || !Array.isArray(selected)) return false;
      if (question.type === 'single' && selected.length > 1) return false;
      var seen = {};
      return selected.every(function (choice) {
        if (!isFiniteNumber(choice) || Math.floor(choice) !== choice || choice < 0 || choice >= question.options.length || seen[choice]) return false;
        seen[choice] = true;
        return true;
      });
    });
  }

  function sameAnswers(a, b) {
    var left = (Array.isArray(a) ? a : []).map(Number).filter(isFiniteNumber).sort(function (x, y) { return x - y; });
    var right = (Array.isArray(b) ? b : []).map(Number).filter(isFiniteNumber).sort(function (x, y) { return x - y; });
    return left.length === right.length && left.every(function (value, index) { return value === right[index]; });
  }

  function scoreAssessmentAnswers(questions, answers) {
    var safeAnswers = isRecord(answers) ? answers : {};
    var correct = 0;
    var domains = {};
    questions.forEach(function (question) {
      var ok = sameAnswers(safeAnswers[question.id], question.correct);
      if (ok) correct++;
      if (!domains[question.domain]) domains[question.domain] = { correct: 0, total: 0 };
      domains[question.domain].total++;
      if (ok) domains[question.domain].correct++;
    });
    return {
      correct: correct,
      total: questions.length,
      percent: questions.length ? Math.round((correct / questions.length) * 100) : 0,
      domains: domains,
    };
  }

  function findAssessmentMeta(id) {
    var found = null;
    tracks().some(function (track) {
      var meta = (track.assessments || []).find(function (item) { return item.id === id; });
      if (!meta) return false;
      found = { meta: meta, track: track };
      return true;
    });
    return found;
  }

  function renderAssessment() {
    var id = query('id');
    var mount = document.getElementById('assessmentMount');
    var located = findAssessmentMeta(id);
    var assessment = data().assessmentsById[id];
    if (!mount) return;
    if (!assessment && located && located.meta.path && window.AIFSContentSource) {
      fetch(window.AIFSContentSource.repoUrl(located.meta.path)).then(function (response) {
        if (!response.ok) throw new Error('Could not load practice data');
        return response.json();
      }).then(function (loaded) {
        assessment = Object.assign({}, loaded, located.meta, { id: id });
        renderAssessmentLoaded(mount, assessment, located.track);
      }).catch(function () { renderAssessmentError(mount); });
      return;
    }
    if (!assessment) {
      renderAssessmentError(mount);
      return;
    }
    var track = located ? located.track : findTrack(assessment.track);
    renderAssessmentLoaded(mount, assessment, track);
  }

  function renderAssessmentError(mount) {
    mount.innerHTML = '<div class="cert-error"><h1>Asesmen tidak ditemukan</h1><p>Set latihan ini tidak tersedia dalam data sertifikasi yang dihasilkan.</p><a class="cert-action" href="certifications.html">Semua sertifikasi</a></div>';
  }

  function renderAssessmentLoaded(mount, assessment, track, forceForm) {
    renderProgramNotice('assessmentProgramNotice', programForTrack(track), 'Latihan independen');
    var questions = (assessment.questions || []).map(normalizeQuestion);
    if (!questions.length) {
      mount.innerHTML = '<div class="cert-error"><h1>Latihan sedang disusun</h1><p>Asesmen ini memiliki metadata tetapi belum ada soal.</p><a class="cert-action" href="' + (track ? 'certification?id=' + encodeURIComponent(track.id) : 'certifications.html') + '">Kembali ke track</a></div>';
      return;
    }
    var title = assessment.title || 'Asesmen latihan';
    var version = assessmentVersion(assessment);
    document.title = title + ' - AI Engineering from Scratch';
    var existing = assessmentAttempts(assessment.id, version);
    if (!forceForm && query('result') === 'latest' && existing && existing[0]) {
      renderAssessmentResults(mount, assessment, track, questions, existing[0]);
      return;
    }

    var limit = Number(assessment.timeLimitMinutes || 0);
    var draft = window.AIFSCertProgress && window.AIFSCertProgress.getDraft(assessment.id, version);
    if (draft && !validDraftForAssessment(draft, version, questions, limit)) {
      if (window.AIFSCertProgress) window.AIFSCertProgress.clearDraft(assessment.id, version);
      draft = null;
    }
    if (!draft) {
      var now = Date.now();
      draft = { answers: {}, startedAt: now, deadlineAt: limit ? now + limit * 60000 : 0, version: version };
      if (window.AIFSCertProgress) window.AIFSCertProgress.saveDraft(assessment.id, version, draft);
    }

    mount.innerHTML = '<div class="assessment-shell"><div class="cert-breadcrumb"><a href="certifications.html">Sertifikasi</a><span>/</span>' + (track ? '<a href="certification?id=' + encodeURIComponent(track.id) + '">' + esc(track.examCode || track.shortName || track.id) + '</a><span>/</span>' : '') + '<span>Latihan</span></div><section class="assessment-hero"><div class="cert-eyebrow">' + esc(assessment.kind || 'LATIHAN') + '</div><h1>' + esc(title) + '</h1><p>' + esc(assessment.summary || assessment.description || 'Kerjakan skenario orisinal, lalu kirim untuk melihat penjelasan dan umpan balik per domain.') + '</p><p class="assessment-scoring-note">Kirim kapan pun kamu siap. Soal tanpa jawaban dihitung salah.</p><div class="cert-meta-row">' + metaChip(questions.length + ' soal orisinal') + metaChip(limit ? limit + ' menit' : 'Tanpa batas waktu') + metaChip(progressIsPersistent() ? 'Tersimpan lokal' : 'Progres hanya bertahan di halaman ini') + '</div></section>' +
      '<div class="cert-notice"><strong>Hanya skor latihan</strong><p>Hasil kamu adalah persentase yang dibuat oleh kursus open-source ini. Ini bukan skor ujian terukur resmi, keputusan kredensial, atau jaminan kelulusan.</p></div>' +
      (limit ? '<div class="cert-timer" id="assessmentTimer" role="timer" aria-live="off" aria-label="Waktu tersisa"><span>Waktu tersisa</span><strong id="assessmentTimerValue">' + formatRemaining(draft.deadlineAt - Date.now()) + '</strong></div>' : '') +
      '<form class="assessment-form" id="assessmentForm" tabindex="-1" aria-label="Soal asesmen">' + questions.map(function (question, index) { return renderQuestion(question, index, draft.answers[question.id] || []); }).join('') + '<div class="cert-submit-row"><p>Kamu bisa mengirim percobaan sebagian. Soal tanpa jawaban dihitung salah, dan penjelasan muncul setelah pengiriman.</p><button class="cert-action" type="submit">Kirim latihan</button></div></form></div>';

    var form = document.getElementById('assessmentForm');
    form.addEventListener('change', function (event) {
      var input = event.target;
      if (!input.matches('input[data-question]')) return;
      var qid = input.getAttribute('data-question');
      var question = questions.find(function (item) { return item.id === qid; });
      if (!question) return;
      if (question.type === 'multiple') {
        draft.answers[qid] = Array.from(form.querySelectorAll('input[data-question="' + cssEscape(qid) + '"]:checked')).map(function (node) { return Number(node.value); });
      } else {
        draft.answers[qid] = [Number(input.value)];
      }
      Array.from(form.querySelectorAll('input[data-question="' + cssEscape(qid) + '"]')).forEach(function (node) {
        var option = node.closest('.cert-option');
        if (option) option.classList.toggle('is-selected', node.checked);
      });
      if (window.AIFSCertProgress) window.AIFSCertProgress.saveDraft(assessment.id, version, draft);
    });
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      submitAssessment(mount, assessment, track, questions, draft, false);
    });
    startTimer(mount, assessment, track, questions, draft);
  }

  function cssEscape(value) {
    if (window.CSS && window.CSS.escape) return window.CSS.escape(value);
    return String(value).replace(/(["\\])/g, '\\$1');
  }

  function renderQuestion(question, index, selected) {
    var inputType = question.type === 'multiple' ? 'checkbox' : 'radio';
    selected = Array.isArray(selected) ? selected : [];
    return '<fieldset class="cert-question"><legend><div class="cert-question-head"><span class="cert-question-kicker">Soal ' + (index + 1) + ' · ' + esc(question.domain) + '</span><span class="cert-question-kicker">' + (question.type === 'multiple' ? 'Pilih semua yang berlaku' : 'Pilih satu') + '</span></div><div class="cert-question-prompt">' + esc(question.prompt) + '</div></legend><div class="cert-option-list">' + question.options.map(function (option, optionIndex) {
      var isSelected = selected.map(Number).indexOf(optionIndex) !== -1;
      var checked = isSelected ? ' checked' : '';
      return '<label class="cert-option' + (isSelected ? ' is-selected' : '') + '"><input type="' + inputType + '" name="question-' + attr(question.id) + '" value="' + optionIndex + '" data-question="' + attr(question.id) + '"' + checked + '><span>' + esc(option) + '</span></label>';
    }).join('') + '</div></fieldset>';
  }

  function formatRemaining(ms) {
    var total = Math.max(0, Math.ceil(ms / 1000));
    var hours = Math.floor(total / 3600);
    var mins = Math.floor((total % 3600) / 60);
    var secs = total % 60;
    return (hours ? String(hours).padStart(2, '0') + ':' : '') + String(mins).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
  }

  function startTimer(mount, assessment, track, questions, draft) {
    if (assessmentTimer) clearInterval(assessmentTimer);
    if (!draft.deadlineAt) return;
    function tick() {
      var remaining = draft.deadlineAt - Date.now();
      var value = document.getElementById('assessmentTimerValue');
      var timer = document.getElementById('assessmentTimer');
      if (value) value.textContent = formatRemaining(remaining);
      if (timer) timer.classList.toggle('is-low', remaining <= 5 * 60000);
      if (remaining <= 0) {
        clearInterval(assessmentTimer);
        assessmentTimer = null;
        submitAssessment(mount, assessment, track, questions, draft, true);
      }
    }
    tick();
    assessmentTimer = setInterval(tick, 1000);
  }

  function submitAssessment(mount, assessment, track, questions, draft, timedOut) {
    if (assessmentTimer) {
      clearInterval(assessmentTimer);
      assessmentTimer = null;
    }
    var score = scoreAssessmentAnswers(questions, draft.answers);
    var submittedAt = Date.now();
    var attempt = {
      answers: draft.answers,
      correct: score.correct,
      total: score.total,
      percent: score.percent,
      domains: score.domains,
      startedAt: draft.startedAt,
      submittedAt: submittedAt,
      durationMs: submittedAt - draft.startedAt,
      timedOut: !!timedOut,
      version: assessmentVersion(assessment),
    };
    if (window.AIFSCertProgress) window.AIFSCertProgress.recordAttempt(assessment.id, assessmentVersion(assessment), attempt);
    renderAssessmentResults(mount, assessment, track, questions, attempt);
    updateAssessmentResultUrl(assessment.id, true);
    scrollToPageTop();
    focusWithoutScrolling(document.getElementById('assessmentResultsSummary'));
  }

  function renderRemediation(track, questions, attempt) {
    if (!attempt) return '';
    var groups = Object.create(null);
    var answers = isRecord(attempt.answers) ? attempt.answers : {};
    questions.forEach(function (question, index) {
      if (sameAnswers(answers[question.id] || [], question.correct)) return;
      var id = question.domain || 'general';
      if (!groups[id]) groups[id] = { id: id, references: [], paths: Object.create(null), reviewIds: [] };
      groups[id].reviewIds.push('review-' + (question.id || index + 1));
      question.references.forEach(function (ref) {
        var validated = validatedInternalLessonReference(ref);
        if (!validated || groups[id].paths[validated.path]) return;
        groups[id].paths[validated.path] = true;
        groups[id].references.push(ref);
      });
    });

    var recomputed = scoreAssessmentAnswers(questions, answers);
    var missed = Object.keys(groups).map(function (id) {
      var score = recomputed.domains[id] || { correct: 0, total: groups[id].reviewIds.length };
      var percent = score.total ? Math.round((score.correct / score.total) * 100) : 0;
      groups[id].score = score;
      groups[id].percent = percent;
      return groups[id];
    }).sort(function (a, b) {
      return a.percent - b.percent;
    });
    if (!missed.length) return '';

    var cards = missed.map(function (item) {
      var domain = track && (track.domains || []).find(function (entry) { return entry.id === item.id; });
      var links = item.references.map(function (ref) {
        return '<li>' + renderInternalLessonReference(ref, track) + '</li>';
      }).join('');
      if (!links) {
        links = '<li><a href="#' + attr(item.reviewIds[0]) + '">Tinjau penjelasan yang terlewat →</a></li>';
      }
      return '<article class="cert-remediation-domain"><h3>' + esc(domain ? domain.name : item.id) + '</h3><span>' + item.score.correct + '/' + item.score.total + ' · ' + item.percent + '%</span><ul class="cert-remediation-links">' + links + '</ul></article>';
    }).join('');

    return '<section class="cert-remediation" aria-labelledby="remediationTitle"><div class="cert-eyebrow">REMEDIASI TERARAH</div><h2 id="remediationTitle">Pelajari keputusan yang kamu lewatkan</h2><p>Setiap pelajaran di bawah berasal dari referensi internal pada soal yang kamu lewatkan. Tinjau kembali keputusan itu, lalu mulai percobaan baru.</p><div class="cert-remediation-grid">' + cards + '</div></section>';
  }

  function renderAssessmentResults(mount, assessment, track, questions, attempt) {
    var attemptAnswers = isRecord(attempt.answers) ? attempt.answers : {};
    var scoreSummary = scoreAssessmentAnswers(questions, attemptAnswers);
    var domainHtml = Object.keys(scoreSummary.domains).map(function (id) {
      var score = scoreSummary.domains[id];
      var percent = score.total ? Math.round((score.correct / score.total) * 100) : 0;
      var domain = track && (track.domains || []).find(function (item) { return item.id === id; });
      return '<div class="cert-domain-score"><strong>' + esc(domain ? domain.name : id) + '</strong><span>' + score.correct + '/' + score.total + ' · ' + percent + '%</span></div>';
    }).join('');
    var remediationHtml = renderRemediation(track, questions, attempt);
    var latestResultHref = 'assessment.html?id=' + encodeURIComponent(assessment.id) + '&result=latest';
    mount.innerHTML = '<div class="assessment-shell cert-results"><div class="cert-breadcrumb"><a href="certifications.html">Sertifikasi</a><span>/</span>' + (track ? '<a href="certification?id=' + encodeURIComponent(track.id) + '">' + esc(track.examCode || track.shortName || track.id) + '</a><span>/</span>' : '') + '<span>Hasil</span></div><section class="cert-results-summary" id="assessmentResultsSummary" tabindex="-1" aria-label="Hasil asesmen"><div class="cert-results-head"><div><div class="cert-eyebrow">HASIL LATIHAN</div><div class="cert-results-score">' + scoreSummary.percent + '%</div></div><div><strong>' + scoreSummary.correct + ' dari ' + scoreSummary.total + ' jawaban tepat</strong><p>' + (attempt.timedOut ? 'Dikirim saat waktu habis.' : 'Dikirim dalam ' + formatDuration(attempt.durationMs)) + '</p></div></div><div class="cert-domain-score-grid">' + domainHtml + '</div><div class="cert-notice"><strong>Bukan skor terukur resmi</strong><p>Persentase ini hanya mengukur set latihan orisinal ini. Ini tidak dapat memprediksi atau menjamin hasil sertifikasi terukur dari penyedia.</p></div><div class="cert-track-hero-actions"><button class="cert-action" type="button" id="assessmentRetake">Mulai percobaan baru</button><a class="cert-action secondary" href="' + attr(latestResultHref) + '">Lihat lagi hasil terbaru</a>' + (track ? '<a class="cert-action secondary" href="certification?id=' + encodeURIComponent(track.id) + '">Kembali ke track</a>' : '') + '</div></section>' + remediationHtml + '<section aria-labelledby="reviewTitle"><div class="cert-section-heading"><div><div class="cert-eyebrow">TINJAUAN</div><h2 id="reviewTitle">Keputusan dan penjelasan</h2></div><p>Gunakan setiap kesalahan untuk mengidentifikasi domain dan aturan keputusan yang perlu kamu tinjau kembali.</p></div>' + questions.map(function (question, index) { return renderReview(question, index, attemptAnswers[question.id] || [], track); }).join('') + '</section></div>';
    var retake = document.getElementById('assessmentRetake');
    if (retake) retake.addEventListener('click', function () {
      if (window.AIFSCertProgress) window.AIFSCertProgress.clearDraft(assessment.id, assessmentVersion(assessment));
      updateAssessmentResultUrl(assessment.id, false);
      renderAssessmentLoaded(mount, assessment, track, true);
      scrollToPageTop();
      focusWithoutScrolling(document.getElementById('assessmentForm'));
    });
  }

  function formatDuration(ms) {
    if (!isFiniteNumber(ms) || ms < 0) return 'durasi tidak diketahui';
    var mins = Math.floor(ms / 60000);
    var secs = Math.floor((ms % 60000) / 1000);
    return mins + 'mnt ' + secs + 'dtk';
  }

  function renderReview(question, index, picked, track) {
    picked = Array.isArray(picked) ? picked : [];
    var ok = sameAnswers(picked, question.correct);
    var pickedLabels = picked.map(function (choice) { return question.options[Number(choice)] || ''; }).filter(Boolean);
    var correctLabels = question.correct.map(function (choice) { return question.options[Number(choice)] || ''; }).filter(Boolean);
    var refs = question.references.map(function (ref) {
      if (typeof ref === 'string') {
        if (/^https?:\/\//i.test(ref)) {
          var sourceLabel = 'Sumber eksternal';
          try {
            var sourceUrl = new URL(ref);
            sourceLabel = (/\.pdf$/i.test(sourceUrl.pathname) ? 'Panduan ujian' : sourceUrl.hostname.replace(/^www\./, '')) + ' ↗';
          } catch (_) {}
          return '<li><a href="' + attr(ref) + '" target="_blank" rel="noopener">' + esc(sourceLabel) + '</a></li>';
        }
        if (/^(certifications|phases)\//.test(ref)) {
          var internalLink = renderInternalLessonReference(ref, track);
          return internalLink ? '<li>' + internalLink + '</li>' : '<li>' + esc(ref) + '</li>';
        }
        return '<li>' + esc(ref) + '</li>';
      }
      if (!ref) return '';
      var label = ref.label || ref.title || ref.url || ref.path || 'Referensi';
      if (ref.url) return '<li><a href="' + attr(ref.url) + '" target="_blank" rel="noopener">' + esc(label) + ' ↗</a></li>';
      if (ref.path) {
        var objectLink = renderInternalLessonReference(ref, track);
        return objectLink ? '<li>' + objectLink + '</li>' : '<li>' + esc(label) + '</li>';
      }
      return '<li>' + esc(label) + '</li>';
    }).join('');
    return '<article class="cert-review ' + (ok ? 'is-correct' : 'is-wrong') + '" id="review-' + attr(question.id || index + 1) + '"><div class="cert-question-kicker">Soal ' + (index + 1) + ' · ' + esc(question.domain) + ' · ' + (ok ? 'Benar' : 'Tinjau') + '</div><h3>' + esc(question.prompt) + '</h3><div class="cert-review-answer"><strong>Jawaban kamu</strong><p>' + esc(pickedLabels.length ? pickedLabels.join('; ') : 'Tidak ada jawaban') + '</p></div>' + (!ok ? '<div class="cert-review-answer"><strong>Jawaban benar</strong><p>' + esc(correctLabels.join('; ')) + '</p></div>' : '') + '<div class="cert-review-explanation"><strong>Mengapa</strong><p>' + esc(question.explanation || 'Belum ada penjelasan yang ditambahkan.') + '</p></div>' + (refs ? '<ul class="cert-reference-list">' + refs + '</ul>' : '') + '</article>';
  }

  function init() {
    initTheme();
    var page = document.body.getAttribute('data-cert-page');
    if (page === 'catalog') renderCatalog();
    else if (page === 'track') renderTrack();
    else if (page === 'assessment') renderAssessment();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}());
