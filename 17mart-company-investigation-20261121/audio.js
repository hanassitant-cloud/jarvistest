(()=>{
  const audio=document.getElementById('narration');
  const play=document.getElementById('play');
  const status=document.getElementById('audio-status');
  const auto=document.getElementById('autoplay');
  const dialog=document.getElementById('transcript');
  const slides=[...document.querySelectorAll('.slide')];
  let index=-1;

  function activeIndex(){return slides.findIndex(s=>s.classList.contains('is-active'));}
  function sourceFor(i){return `audio/slide-${String(i+1).padStart(2,'0')}.m4a`;}
  function pageLabel(i){return `第 ${i+1} 頁 · 台灣繁中自然女聲`;}
  function sync(force=false){
    const n=activeIndex();
    if(n<0||(!force&&n===index))return;
    audio.pause();
    index=n;
    audio.src=sourceFor(n);
    audio.load();
    play.innerHTML='▶<span class="oc-ppt-remote-small">講解</span>';
    status.textContent=pageLabel(n);
    dialog.querySelector('p').textContent=slides[n].querySelector('.notes')?.textContent?.trim()||'';
  }
  async function start(){
    try{await audio.play();}
    catch(e){status.textContent='請再點一次「講解」以啟動聲音';}
  }
  function advance(){
    if(auto.checked&&index<slides.length-1){
      document.querySelector('.ppt-next')?.click();
      setTimeout(()=>{sync(true);start();},180);
    }else status.textContent='本頁講解結束';
  }

  play.addEventListener('click',()=>audio.paused?start():audio.pause());
  audio.addEventListener('play',()=>play.innerHTML='Ⅱ<span class="oc-ppt-remote-small">暫停</span>');
  audio.addEventListener('pause',()=>play.innerHTML='▶<span class="oc-ppt-remote-small">講解</span>');
  audio.addEventListener('error',()=>status.textContent='音檔載入失敗，請重試或閱讀講稿');
  audio.addEventListener('ended',advance);
  document.getElementById('script').addEventListener('click',()=>{sync(true);dialog.showModal();});
  document.getElementById('close-script').addEventListener('click',()=>dialog.close());
  new MutationObserver(()=>sync()).observe(document.querySelector('.deck'),{subtree:true,attributes:true,attributeFilter:['class']});
  sync(true);
})();
