const fs = require('fs');
const path = require('path');

const root = __dirname;
const jsonPath = path.join(root, 'catalog.json');
const entries = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));

const next = {
  slug: 'fsmlab-ai-growth-proposal',
  name: 'FSMLAB 飛創美｜AI 成長規劃 HTML PPT',
  desc: '依黃鳳美 Nina / FSMLAB 飛創美近兩年公開資料，整理品牌定位、產業方向、AI 肌膚檢測、LINE AI 顧問、內容行銷、訓練招商與 90 天落地路線圖。',
  source: 'Jason 2026-06-14 指示 + FSMLAB 官網、公開新聞、公開體驗文、公司登記與產業資料整理',
  date: '2026-06-14',
  category: '品牌分析',
  url: 'https://hanassitant-cloud.github.io/jarvistest/fsmlab-ai-growth-proposal/',
  zip_url: 'https://hanassitant-cloud.github.io/jarvistest/fsmlab-ai-growth-proposal.zip',
  notes: '公開資料顧問提案版；量子、非侵入、改善效果等字眼均作為品牌宣傳與市場定位分析，不作醫療或臨床功效承諾。已保留逐頁真人語音稿與 audio/slide-xx.m4a 播放結構；目前待 Jason 核准真人音源後再切入正式音檔，不使用機器感 TTS 充數。'
};

const idx = entries.findIndex(e => e.slug === next.slug);
if (idx >= 0) entries[idx] = { ...entries[idx], ...next };
else entries.unshift(next);

entries.sort((a, b) => String(b.date || '').localeCompare(String(a.date || '')));
fs.writeFileSync(jsonPath, JSON.stringify(entries, null, 2) + '\n');

function esc(s) {
  return String(s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

const cats = [...new Set(entries.map(e => e.category).filter(Boolean))];
const cards = entries.map(e => {
  const data = `${e.name} ${e.desc} ${e.source} ${e.notes || ''} ${e.category}`;
  const zip = e.zip_url ? `<a class="zip" href="${esc(e.zip_url)}" target="_blank" rel="noopener">ZIP 下載</a>` : '';
  const notes = e.notes ? `<p class="source"><b>備註：</b>${esc(e.notes)}</p>` : '';
  return `<article class="card" data-name="${esc(data)}" data-cat="${esc(e.category)}">
  <div class="meta"><span>${esc(e.category)}</span><time>${esc(e.date)}</time></div>
  <h2>${esc(e.name)}</h2>
  <p class="desc">${esc(e.desc)}</p>
  <p class="source"><b>資料來源：</b>${esc(e.source)}</p>
  ${notes}
  <div class="actions"><a class="open" href="${esc(e.url)}" target="_blank" rel="noopener">打開 HTML PPT</a>${zip}<code>/${esc(e.slug)}/</code></div>
</article>`;
}).join('\n');

const chips = ['全部', ...cats].map((cat, i) =>
  `<button class="chip${i === 0 ? ' active' : ''}" data-cat="${esc(cat)}">${esc(cat)}</button>`
).join('');

const html = `<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>文文 HTML PPT 作品總表</title>
  <meta name="description" content="文文已發布 HTML PPT 作品總表，依日期由新到舊排序。">
  <style>
    :root{--bg:#f7fbff;--ink:#102033;--muted:#64748b;--blue:#0ea5e9;--teal:#14b8a6;--line:rgba(15,23,42,.14);--shadow:0 24px 70px rgba(15,23,42,.12)}
    *{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#fff,#eefaff 68%,#f7fbff);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Noto Sans TC","PingFang TC",Arial,sans-serif;letter-spacing:0}.wrap{max-width:1180px;margin:0 auto;padding:42px 20px 80px}.home{display:inline-flex;align-items:center;gap:8px;text-decoration:none;color:#075985;background:#e0f2fe;border:1px solid #bae6fd;border-radius:999px;padding:9px 13px;font-weight:800}.hero{padding:42px 0 26px}.eyebrow{color:#0f766e;font-weight:900;letter-spacing:0;font-size:14px;text-transform:uppercase}h1{font-size:clamp(34px,6vw,68px);line-height:1.03;margin:10px 0 16px;letter-spacing:0}.lead{font-size:19px;line-height:1.65;color:#40536b;max-width:850px;margin:0}.stats{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}.stat{background:rgba(255,255,255,.8);border:1px solid var(--line);border-radius:8px;padding:14px 16px;box-shadow:var(--shadow)}.stat b{font-size:28px}.stat span{display:block;color:var(--muted);font-size:13px}.tools{display:grid;grid-template-columns:1fr auto;gap:14px;margin:24px 0}.search{width:100%;border:1px solid var(--line);border-radius:8px;padding:14px 16px;font-size:16px;background:#fff}.chips{display:flex;gap:8px;flex-wrap:wrap}.chip{border:1px solid var(--line);background:#fff;border-radius:999px;padding:10px 13px;font-weight:800;color:#334155;cursor:pointer}.chip.active{background:#0f172a;color:#fff}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{background:rgba(255,255,255,.9);border:1px solid var(--line);border-radius:8px;padding:22px;box-shadow:0 18px 48px rgba(15,23,42,.1)}.meta{display:flex;justify-content:space-between;gap:12px;color:#64748b;font-size:13px;font-weight:900}.meta span{color:#0369a1}.card h2{font-size:24px;margin:12px 0 10px;line-height:1.25}.desc,.source{line-height:1.62;color:#475569}.source{font-size:14px;color:#64748b}.actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:16px}.actions a{text-decoration:none;border-radius:999px;padding:10px 14px;font-weight:900}.open{background:#0f172a;color:#fff}.zip{background:#e0f2fe;color:#075985}.actions code{color:#64748b;font-size:12px}@media(max-width:820px){.tools{grid-template-columns:1fr}.grid{grid-template-columns:1fr}.wrap{padding:28px 14px 60px}.card h2{font-size:21px}}
  </style>
</head>
<body>
  <main class="wrap">
    <a class="home" href="../">返回入口</a>
    <section class="hero">
      <div class="eyebrow">HTML PPT Catalog</div>
      <h1>文文 HTML PPT 作品總表</h1>
      <p class="lead">這裡整理已發布的 HTML PPT、圖文簡報與提案頁面，依日期由新到舊排序。每個項目包含展示連結、ZIP 下載、資料來源與分類，方便 Jason 快速查找與交付。</p>
      <div class="stats"><div class="stat"><b>${entries.length}</b><span>已登錄作品</span></div><div class="stat"><b>${cats.length}</b><span>分類</span></div><div class="stat"><b>GitHub Pages</b><span>公開展示層</span></div></div>
    </section>
    <section class="tools"><input id="q" class="search" placeholder="搜尋名稱、描述或資料來源"><div class="chips" id="chips">${chips}</div></section>
    <section class="grid" id="grid">${cards}</section>
  </main>
  <script>
    const q=document.getElementById('q'), cards=[...document.querySelectorAll('.card')], chips=[...document.querySelectorAll('.chip')]; let cat='全部';
    function apply(){const term=(q.value||'').trim().toLowerCase(); cards.forEach(c=>{const okCat=cat==='全部'||c.dataset.cat===cat; const okTerm=!term||c.dataset.name.toLowerCase().includes(term); c.style.display=okCat&&okTerm?'block':'none';})}
    q.addEventListener('input',apply); chips.forEach(b=>b.addEventListener('click',()=>{chips.forEach(x=>x.classList.remove('active')); b.classList.add('active'); cat=b.dataset.cat; apply();}));
  </script>
</body>
</html>
`;

fs.writeFileSync(path.join(root, 'index.html'), html);
