(function () {
  'use strict';

  const publicationUrl = 'https://thatdevopsguy.substack.com';
  let dialog;
  let trigger;

  function signupSection(variant) {
    const lesson = variant === 'lesson';
    const id = variant === 'dialog' ? 'newsletter-modal' : 'newsletter';
    const section = document.createElement('section');
    section.id = id;
    section.className = 'newsletter-section newsletter-' + variant;
    if (variant === 'home') section.classList.add('container');
    section.setAttribute('aria-labelledby', id + '-title');
    section.innerHTML = `<div class="newsletter-strip">
      <div class="newsletter-copy">
        <h2 id="${id}-title">${lesson ? 'AI Engineering Newsletter' : 'Belajar AI dari nol. Tetap selangkah di depan.'}</h2>
        <p>${lesson ? 'Pelajaran praktis dan kabar terbaru seputar AI, DevOps, dan cloud native. Satu email per minggu.' : 'Pelajaran praktis, alat yang layak dicoba, dan perkembangan penting minggu ini di AI, DevOps, dan cloud native. Satu email gratis, setiap minggu.'}</p>
      </div>
      <div class="newsletter-signup">
        <form action="${publicationUrl}/subscribe" method="get" aria-describedby="${id}-note">
          <label class="newsletter-email-label" for="${id}-email">Alamat email</label>
          <div class="newsletter-input-row">
            <input id="${id}-email" name="email" type="email" placeholder="Alamat email kamu" autocomplete="email" required>
            <button type="submit">Berlangganan gratis <span aria-hidden="true">↗</span></button>
          </div>
        </form>
        <p id="${id}-note" class="newsletter-note">Gratis. Berhenti kapan saja. Selesaikan pendaftaran di Substack.</p>
      </div>
    </div>`;
    return section;
  }

  function newsletterLink(label = 'Newsletter') {
    const link = document.createElement('a');
    link.href = publicationUrl + '/subscribe';
    link.textContent = label;
    link.dataset.newsletterOpen = '';
    if (typeof HTMLDialogElement !== 'undefined' && HTMLDialogElement.prototype.showModal) {
      link.setAttribute('aria-haspopup', 'dialog');
    }
    return link;
  }

  function addNavigation() {
    const nav = document.querySelector('.header-nav');
    if (nav && !nav.querySelector('[data-newsletter-open]')) {
      nav.insertBefore(newsletterLink(), nav.querySelector('.header-mobile-tools'));
    }
    const footer = document.querySelector('.footer-links');
    if (footer && !footer.querySelector('[data-newsletter-open]')) footer.prepend(newsletterLink());
  }

  function createDialog() {
    const element = document.createElement('dialog');
    element.className = 'newsletter-modal';
    element.setAttribute('aria-labelledby', 'newsletter-modal-title');
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'newsletter-modal-close';
    close.textContent = 'Tutup ×';
    close.setAttribute('aria-label', 'Tutup pendaftaran newsletter');
    close.autofocus = true;
    close.addEventListener('click', function () { element.close(); });
    element.append(close, signupSection('dialog'));
    element.addEventListener('click', function (event) {
      if (event.target === element) element.close();
    });
    element.addEventListener('close', function () {
      const focusTarget = trigger?.getClientRects().length ? trigger : document.querySelector('.header-menu-toggle');
      focusTarget?.focus({ preventScroll: true });
    });
    document.body.append(element);
    return element;
  }

  function enableDialog() {
    document.addEventListener('click', function (event) {
      const link = event.target.closest('a[data-newsletter-open]');
      if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      if (typeof HTMLDialogElement === 'undefined' || !HTMLDialogElement.prototype.showModal) return;
      event.preventDefault();
      trigger = link;
      if (!dialog) dialog = createDialog();
      dialog.showModal();
    });
  }

  function revealAnchor() {
    if (location.hash !== '#newsletter' && location.hash !== '#newsletter-cta') return;
    document.fonts.ready.then(function () {
      requestAnimationFrame(function () {
        document.getElementById(location.hash.slice(1))?.scrollIntoView({ block: 'start', behavior: 'instant' });
      });
    });
  }

  function addPagePlacement() {
    let page = location.pathname.split('/').pop().replace(/\.html$/, '');
    const aliases = { path: 'prereqs', roadmap: 'prereqs', docs: 'developer' };
    page = aliases[page] || page;
    if (page === 'about') {
      document.querySelector('.about')?.append(signupSection('about'));
      return;
    }
    const placements = {
      'learning-paths': ['#overview', 'after', 'Terus kembangkan skill AI kamu, setiap minggu.', 'learning-paths-container'],
      catalog: ['.catalog-header', 'append', 'Dapatkan pelajaran praktis dan kabar AI minggu ini.'],
      certifications: ['.cert-hero', 'after', 'Terus belajar seiring alat dan praktik AI berkembang.', 'cert-container'],
      certification: ['#trackProgramNotice', 'before', 'Terus berkembang melampaui rencana belajar kamu.', 'cert-container'],
      glossary: ['.glossary-explorer', 'after', 'Ubah konsep AI menjadi keterampilan praktis.'],
      prereqs: ['.roadmap-workspace', 'after', 'Jaga pembelajaran kamu tetap terhubung dengan hal berikutnya.', 'roadmap-container'],
      developer: ['.trust-page', 'append', 'Ikuti alat dan ide yang membentuk rekayasa AI.']
    };
    const placement = placements[page];
    if (!placement) return;
    const [selector, position, message, container = ''] = placement;
    const target = document.querySelector(selector);
    if (!target) return;
    const callout = document.createElement('aside');
    callout.id = 'newsletter-cta';
    callout.className = 'newsletter-callout ' + container;
    callout.setAttribute('aria-label', 'AI Engineering Newsletter');
    const row = document.createElement('div');
    row.className = 'newsletter-callout-row';
    const copy = document.createElement('p');
    copy.textContent = message;
    row.append(copy, newsletterLink('Dapatkan newsletter gratis →'));
    callout.append(row);
    target[position](callout);
  }

  function addLessonPlacement() {
    const content = document.getElementById('lessonContent');
    if (!content) return;
    function insert() {
      const next = content.querySelector('.lesson-nav-bottom');
      if (!next || content.querySelector('#newsletter')) return false;
      next.before(signupSection('lesson'));
      revealAnchor();
      return true;
    }
    if (insert()) return;
    const observer = new MutationObserver(function () {
      if (insert()) observer.disconnect();
    });
    observer.observe(content, { childList: true, subtree: true });
  }

  function initialize() {
    if (!document.querySelector('.site-header') || document.querySelector('[data-newsletter-open]')) return;
    addNavigation();
    enableDialog();
    const readers = document.querySelector('.learners-strip');
    if (readers) {
      readers.before(signupSection('home'));
      const colophon = document.querySelector('.colophon p');
      if (colophon) colophon.textContent = colophon.textContent.replace('No paywall, no signup.', 'Tanpa paywall, tanpa perlu akun.');
    } else {
      addPagePlacement();
      addLessonPlacement();
    }
    revealAnchor();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize);
  else initialize();
})();
