// Extracted reference runtime. index.html is self-contained; this file is included for deployment handoff.
const slides=[...document.querySelectorAll('.slide')];let index=0;
function goto(i){index=Math.max(0,Math.min(slides.length-1,i));slides[index].scrollIntoView({behavior:'smooth',block:'start'});history.replaceState(null,'','#s'+(index+1));update();}
function update(){document.getElementById('pageno').textContent=(index+1)+' / '+slides.length;}
function nearest(){let best=0,dist=1e9;slides.forEach((s,i)=>{const d=Math.abs(s.getBoundingClientRect().top-80);if(d<dist){dist=d;best=i}});index=best;update();}
document.getElementById('prev').onclick=()=>goto(index-1);document.getElementById('next').onclick=()=>goto(index+1);
document.getElementById('notes').onclick=()=>{document.querySelectorAll('details').forEach(d=>d.open=!d.open)};
addEventListener('scroll',()=>requestAnimationFrame(nearest));addEventListener('keydown',e=>{if(['ArrowRight',' '].includes(e.key)){e.preventDefault();goto(index+1)}if(e.key==='ArrowLeft'){e.preventDefault();goto(index-1)}if(e.key.toLowerCase()==='s'){document.getElementById('notes').click()}if(e.key==='Home')goto(0);if(e.key==='End')goto(slides.length-1)});
const m=location.hash.match(/s(\d+)/);if(m) index=Number(m[1])-1;update();
