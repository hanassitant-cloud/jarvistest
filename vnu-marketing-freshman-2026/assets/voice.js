(() => {
 const slides=[...document.querySelectorAll('.deck > .slide')];
 const audio=new Audio();audio.id='narration-audio';audio.preload='metadata';audio.setAttribute('playsinline','');document.body.append(audio);
 const panel=document.createElement('div');panel.className='voice-panel';panel.setAttribute('aria-label','自然女聲旁白');
 panel.innerHTML='<div class="voice-label">台灣中文・自然女聲 <span>合成旁白</span></div><div class="voice-controls"><button type="button" class="voice-play">▶ 播放旁白</button><button type="button" class="voice-replay">↺ 重播</button><label><input type="checkbox" class="voice-continuous"> 連續導覽</label><span class="voice-time">0:00</span></div><div class="voice-status" aria-live="polite">按播放開始本頁講解</div>';
 const btn=panel.querySelector('.voice-play'),status=panel.querySelector('.voice-status'),timer=panel.querySelector('.voice-time');
 let page=-1,want=false,serial=0;
 const time=n=>Number.isFinite(n)?Math.floor(n/60)+':'+String(Math.floor(n%60)).padStart(2,'0'):'0:00';
 function update(){btn.textContent=want?'Ⅱ 暫停旁白':'▶ 播放旁白';btn.setAttribute('aria-pressed',String(want));}
 async function play(){const id=serial;want=true;update();status.textContent='正在載入第 '+(page+1)+' 頁旁白…';try{await audio.play();if(id===serial)status.textContent='正在講解第 '+(page+1)+'／20 頁';}catch(e){if(id===serial){want=false;update();status.textContent='請再按播放；若無法播放，可重新整理頁面。';}}}
 function change(){const p=slides.findIndex(s=>s.classList.contains('is-active'));if(p<0||p===page)return;page=p;serial++;audio.pause();audio.src='audio-neural/slide-'+String(page+1).padStart(2,'0')+'.mp3';audio.load();slides[page].querySelector('.main').prepend(panel);timer.textContent='0:00';status.textContent='第 '+(page+1)+'／20 頁・按播放開始講解';if(want)play();else update();}
 btn.onclick=()=>{if(want){want=false;audio.pause();update();status.textContent='已暫停，再按播放可繼續';}else play();};
 panel.querySelector('.voice-replay').onclick=()=>{audio.currentTime=0;play();};
 audio.addEventListener('timeupdate',()=>{timer.textContent=time(audio.currentTime)+' / '+time(audio.duration);});
 audio.addEventListener('ended',()=>{if(panel.querySelector('input').checked&&page<slides.length-1){want=true;document.querySelector('.ppt-next').click();}else{want=false;update();status.textContent=page===slides.length-1?'全篇講解完畢':'本頁講解完畢';}});
 audio.addEventListener('error',()=>{want=false;update();status.textContent='音檔暫時讀取失敗，請重新整理後再試。';});
 new MutationObserver(change).observe(document.querySelector('.deck'),{subtree:true,attributes:true,attributeFilter:['class']});change();
})();
