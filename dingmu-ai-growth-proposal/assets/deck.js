(() => {
  const slides = [...document.querySelectorAll('.slide')];
  const counter = document.getElementById('counter');
  const prev = document.getElementById('prev');
  const next = document.getElementById('next');
  const audio = document.getElementById('audio');
  const narrate = document.getElementById('narrate');
  const listDialog = document.getElementById('listDialog');
  const transcriptDialog = document.getElementById('transcriptDialog');
  const chapterList = document.getElementById('chapterList');
  const transcriptTitle = document.getElementById('transcriptTitle');
  const transcriptText = document.getElementById('transcriptText');
  const toast = document.getElementById('toast');
  let index = 0;
  let touchX = null;

  const say = (message) => {
    toast.textContent = message;
    toast.classList.add('show');
    window.setTimeout(() => toast.classList.remove('show'), 1600);
  };

  const noteFor = (slide) => (slide.querySelector('.notes')?.textContent || '').trim();

  slides.forEach((slide, i) => {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.innerHTML = `<span>${String(i + 1).padStart(2, '0')}</span><span>${slide.dataset.title}</span>`;
    button.addEventListener('click', () => { show(i); listDialog.close(); });
    li.append(button);
    chapterList.append(li);
  });

  const chapterButtons = [...chapterList.querySelectorAll('button')];

  function stopAudio() {
    audio.pause();
    audio.currentTime = 0;
    narrate.innerHTML = '▶ <span>語音播放</span>';
    narrate.setAttribute('aria-label', '播放女性講解');
  }

  function show(nextIndex) {
    stopAudio();
    index = Math.max(0, Math.min(slides.length - 1, nextIndex));
    slides.forEach((slide, i) => slide.classList.toggle('is-active', i === index));
    chapterButtons.forEach((button, i) => button.classList.toggle('active', i === index));
    counter.value = `${index + 1} / ${slides.length}`;
    counter.textContent = `${index + 1} / ${slides.length}`;
    prev.disabled = index === 0;
    next.disabled = index === slides.length - 1;
    transcriptTitle.textContent = slides[index].dataset.title;
    transcriptText.textContent = noteFor(slides[index]);
    document.title = `${slides[index].dataset.title}｜鼎沐 AI 成長提案`;
    history.replaceState(null, '', `#${index + 1}`);
    slides[index].scrollTop = 0;
  }

  function toggleNarration() {
    if (!audio.paused) {
      audio.pause();
      narrate.innerHTML = '▶ <span>語音播放</span>';
      return;
    }
    const src = slides[index].dataset.audio;
    if (!src) return say('本頁沒有旁白音檔');
    if (!audio.src.endsWith(src)) audio.src = src;
    audio.play().then(() => {
      narrate.innerHTML = 'Ⅱ <span>語音暫停</span>';
      narrate.setAttribute('aria-label', '暫停女性講解');
    }).catch(() => say('瀏覽器阻擋播放，請再按一次講解'));
  }

  prev.addEventListener('click', () => show(index - 1));
  next.addEventListener('click', () => show(index + 1));
  narrate.addEventListener('click', toggleNarration);
  audio.addEventListener('ended', () => { narrate.innerHTML = '↻ <span>重新播放</span>'; });
  document.getElementById('listBtn').addEventListener('click', () => listDialog.showModal());
  document.getElementById('transcriptBtn').addEventListener('click', () => transcriptDialog.showModal());
  document.querySelectorAll('.close').forEach(button => button.addEventListener('click', () => button.closest('dialog').close()));
  document.querySelectorAll('dialog').forEach(dialog => dialog.addEventListener('click', event => {
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
  }));
  document.addEventListener('keydown', event => {
    if (event.key === 'ArrowRight' || event.key === 'PageDown') show(index + 1);
    if (event.key === 'ArrowLeft' || event.key === 'PageUp') show(index - 1);
    if (event.key === 'Escape') { if (listDialog.open) listDialog.close(); if (transcriptDialog.open) transcriptDialog.close(); }
  });
  document.addEventListener('touchstart', event => { touchX = event.changedTouches[0].clientX; }, {passive:true});
  document.addEventListener('touchend', event => {
    if (touchX === null) return;
    const delta = event.changedTouches[0].clientX - touchX;
    if (Math.abs(delta) > 70) show(index + (delta < 0 ? 1 : -1));
    touchX = null;
  }, {passive:true});
  const fromHash = Number(location.hash.slice(1));
  show(Number.isFinite(fromHash) && fromHash > 0 ? fromHash - 1 : 0);
})();
