#!/usr/bin/env python3
"""Генерирует автономный HTML: вкладка «Поиск» по истории + вкладка «Статистика»
с графиками. Открывается локально в браузере (данные не покидают машину).

  uv run dashboard.py            # собрать и открыть на вкладке статистики
  uv run dashboard.py --search   # открыть на вкладке поиска
"""
import html
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "dashboard.html")


def rows():
    db = sqlite3.connect(os.path.join(BASE, "history.sqlite3"))
    db.row_factory = sqlite3.Row
    try:
        r = db.execute("SELECT * FROM transcriptions ORDER BY ts DESC").fetchall()
    except sqlite3.OperationalError:
        r = []
    db.close()
    return r


def dict_sizes():
    def count(fn):
        try:
            with open(os.path.join(BASE, fn)) as f:
                return sum(1 for l in f if l.strip() and not l.startswith("#"))
        except FileNotFoundError:
            return 0
    return count("terms.txt"), count("auto_terms.txt")


def compute(rs):
    total = len(rs)
    words = sum(len(re.findall(r"\w+", r["text"] or "")) for r in rs)
    secs = sum(r["duration"] or 0 for r in rs)
    corrected = sum(1 for r in rs if (r["text"] or "") != (r["raw_text"] or ""))
    by_app = Counter(r["app"] or "?" for r in rs)
    by_day = defaultdict(lambda: [0, 0])  # date -> [диктовок, исправлено]
    by_hour = Counter()
    for r in rs:
        d = datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d")
        by_day[d][0] += 1
        if (r["text"] or "") != (r["raw_text"] or ""):
            by_day[d][1] += 1
        by_hour[datetime.fromtimestamp(r["ts"]).hour] += 1
    manual, auto = dict_sizes()

    def col(r, name):
        try:
            return r[name]
        except (IndexError, KeyError):
            return None
    tps = [col(r, "gen_tps") for r in rs if col(r, "gen_tps")]
    asr_ms = [col(r, "asr_ms") for r in rs if col(r, "asr_ms")]
    llm_ms = [col(r, "llm_ms") for r in rs if col(r, "llm_ms")]
    avg = lambda xs: round(sum(xs) / len(xs)) if xs else 0
    return {
        "total": total, "words": words, "secs": secs, "corrected": corrected,
        "corr_rate": round(100 * corrected / total) if total else 0,
        "avg_words": round(words / total, 1) if total else 0,
        "wpm_saved": round(words / (secs / 60)) if secs else 0,
        "by_app": by_app.most_common(8),
        "by_day": sorted(by_day.items()),
        "by_hour": [by_hour.get(h, 0) for h in range(24)],
        "dict_manual": manual, "dict_auto": auto,
        "gen_tps_avg": avg(tps), "asr_ms_avg": avg(asr_ms), "llm_ms_avg": avg(llm_ms),
        "tps_series": list(reversed(tps))[-40:],  # хронологически, последние 40
    }


def search_data(rs):
    return [{
        "t": datetime.fromtimestamp(r["ts"]).strftime("%d.%m %H:%M"),
        "app": r["app"] or "?",
        "text": r["text"] or "",
        "raw": (r["raw_text"] or "") if (r["raw_text"] or "") != (r["text"] or "") else "",
    } for r in rs]


# ————— SVG-примитивы (палитра из dataviz-гайда) —————
PAL = {"series": "#2a78d6", "series2": "#8ab4e8", "accent": "#eb6834",
       "good": "#008300"}


def bars_h(data, unit="", accent_key=None):
    """Горизонтальные бары: [(label, value, sublabel?)]. Длина = величина."""
    if not data:
        return '<p class="empty">нет данных</p>'
    mx = max(v for _, v, *_ in data) or 1
    rowh, gap, labelw, valw = 26, 8, 150, 56
    w = 620
    barw = w - labelw - valw
    svg = [f'<svg viewBox="0 0 {w} {len(data)*(rowh+gap)}" class="chart">']
    for i, (label, val, *rest) in enumerate(data):
        y = i * (rowh + gap)
        bw = max(3, barw * val / mx)
        svg.append(f'<text x="{labelw-10}" y="{y+rowh/2+4}" text-anchor="end" '
                   f'class="lbl">{html.escape(str(label)[:20])}</text>')
        svg.append(f'<rect x="{labelw}" y="{y+2}" width="{bw}" height="{rowh-4}" '
                   f'rx="4" fill="var(--series)"/>')
        svg.append(f'<text x="{labelw+bw+8}" y="{y+rowh/2+4}" class="val">'
                   f'{val}{unit}</text>')
    svg.append("</svg>")
    return "".join(svg)


def timeline(by_day):
    """Столбики по дням: всего (светлый) + исправлено (акцент поверх)."""
    if not by_day:
        return '<p class="empty">нет данных</p>'
    w, h, pad = 620, 180, 28
    n = len(by_day)
    mx = max(tot for _, (tot, _) in by_day) or 1
    slot = (w - pad) / max(n, 1)
    bw = min(28, slot * 0.7)
    svg = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    # сетка
    for gy in (0, 0.5, 1):
        yy = pad + (h - 2*pad) * (1 - gy)
        svg.append(f'<line x1="{pad}" y1="{yy}" x2="{w}" y2="{yy}" class="grid"/>')
        svg.append(f'<text x="2" y="{yy+4}" class="tick">{round(mx*gy)}</text>')
    for i, (day, (tot, corr)) in enumerate(by_day):
        x = pad + slot * i + (slot - bw) / 2
        ht = (h - 2*pad) * tot / mx
        hc = (h - 2*pad) * corr / mx
        yb = h - pad - ht
        svg.append(f'<rect x="{x}" y="{yb}" width="{bw}" height="{ht}" rx="4" '
                   f'fill="var(--series2)"><title>{day}: {tot} диктовок</title></rect>')
        if corr:
            svg.append(f'<rect x="{x}" y="{h-pad-hc}" width="{bw}" height="{hc}" '
                       f'rx="4" fill="var(--accent)"><title>{corr} исправлено</title></rect>')
        if n <= 20 or i % 2 == 0:
            svg.append(f'<text x="{x+bw/2}" y="{h-pad+14}" text-anchor="middle" '
                       f'class="tick">{day[8:]}.{day[5:7]}</text>')
    svg.append("</svg>")
    return "".join(svg)


def hours(by_hour):
    w, h, pad = 620, 120, 22
    mx = max(by_hour) or 1
    slot = (w - pad) / 24
    bw = slot * 0.7
    svg = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for hh, v in enumerate(by_hour):
        x = pad + slot * hh + (slot - bw) / 2
        ht = (h - 2*pad) * v / mx
        svg.append(f'<rect x="{x}" y="{h-pad-ht}" width="{bw}" height="{max(0,ht)}" '
                   f'rx="3" fill="var(--series)"><title>{hh}:00 — {v}</title></rect>')
        if hh % 3 == 0:
            svg.append(f'<text x="{x+bw/2}" y="{h-pad+13}" text-anchor="middle" '
                       f'class="tick">{hh}</text>')
    svg.append("</svg>")
    return "".join(svg)


def sparkline(vals, unit=""):
    if not vals:
        return '<p class="empty">пока нет замеров LLM — подиктуй фразы с чисткой</p>'
    w, h, pad = 620, 150, 26
    mx, mn = max(vals), min(vals)
    rng = (mx - mn) or 1
    n = len(vals)
    step = (w - 2*pad) / max(n - 1, 1)
    pts = [(pad + i*step, h - pad - (h - 2*pad) * (v - mn) / rng) for i, v in enumerate(vals)]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = d + f" L{pts[-1][0]:.1f},{h-pad} L{pts[0][0]:.1f},{h-pad} Z"
    avg = sum(vals) / n
    ya = h - pad - (h - 2*pad) * (avg - mn) / rng
    svg = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    svg.append(f'<path d="{area}" fill="var(--series)" opacity="0.10"/>')
    svg.append(f'<line x1="{pad}" y1="{ya:.1f}" x2="{w-pad}" y2="{ya:.1f}" class="grid"/>')
    svg.append(f'<text x="{w-pad}" y="{ya-5:.1f}" text-anchor="end" class="tick">'
               f'среднее {round(avg)}{unit}</text>')
    svg.append(f'<path d="{d}" fill="none" stroke="var(--series)" stroke-width="2" '
               f'stroke-linejoin="round"/>')
    lx, ly = pts[-1]
    svg.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="4" fill="var(--series)"/>')
    svg.append(f'<text x="{lx:.1f}" y="{ly-9:.1f}" text-anchor="end" class="val">'
               f'{round(vals[-1])}{unit}</text>')
    svg.append("</svg>")
    return "".join(svg)


def donut(rate):
    r, c = 52, 2 * 3.14159 * 52
    off = c * (1 - rate / 100)
    return f'''<svg viewBox="0 0 140 140" class="donut">
      <circle cx="70" cy="70" r="52" fill="none" stroke="var(--track)" stroke-width="16"/>
      <circle cx="70" cy="70" r="52" fill="none" stroke="var(--accent)" stroke-width="16"
        stroke-linecap="round" stroke-dasharray="{c:.1f}" stroke-dashoffset="{off:.1f}"
        transform="rotate(-90 70 70)"/>
      <text x="70" y="66" text-anchor="middle" class="donut-num">{rate}%</text>
      <text x="70" y="86" text-anchor="middle" class="donut-lbl">исправлено</text>
    </svg>'''


def tile(label, value, sub=""):
    return (f'<div class="tile"><div class="tile-l">{label}</div>'
            f'<div class="tile-v">{value}</div>'
            f'<div class="tile-s">{sub}</div></div>')


def build(active="stats"):
    rs = rows()
    s = compute(rs)
    sd = search_data(rs)
    mins = round(s["secs"] / 60)

    apps = [(a, n) for a, n in s["by_app"]]
    stats_html = f'''
      <div class="tiles">
        {tile("Всего слов надиктовано", f'{s["words"]:,}'.replace(",", " "),
              f'{s["total"]} диктовок · {mins} мин речи')}
        {tile("В среднем за диктовку", s["avg_words"], "слов")}
        {tile("Темп речи", s["wpm_saved"], "слов/мин")}
        {tile("Словарь", f'{s["dict_manual"]}+{s["dict_auto"]}',
              "ручных + авто")}
      </div>
      <div class="grid2">
        <div class="card"><h3>Диктовок по дням
          <span class="key"><i class="sw2"></i>всего <i class="swa"></i>исправлено</span></h3>
          {timeline(s["by_day"])}</div>
        <div class="card center"><h3>Доля исправлений LLM</h3>
          {donut(s["corr_rate"])}
          <p class="note">{s["corrected"]} из {s["total"]} фраз правились чисткой</p></div>
      </div>
      <div class="grid2">
        <div class="card"><h3>Куда диктуешь</h3>{bars_h(apps)}</div>
        <div class="card"><h3>В какое время суток</h3>{hours(s["by_hour"])}</div>
      </div>
      <h2 class="sec">Производительность</h2>
      <div class="tiles">
        {tile("Скорость генерации LLM", s["gen_tps_avg"] or "—", "токенов/с в среднем")}
        {tile("Распознавание", f'{s["asr_ms_avg"]} мс' if s["asr_ms_avg"] else "—", "на фразу")}
        {tile("LLM-чистка", f'{s["llm_ms_avg"]} мс' if s["llm_ms_avg"] else "—",
              "когда включается")}
        {tile("Модели", "MLX", "Whisper + Qwen на GPU")}
      </div>
      <div class="card"><h3>Скорость генерации LLM по фразам
        <span class="key">токенов/с, последние {len(s["tps_series"])}</span></h3>
        {sparkline(s["tps_series"], " т/с")}</div>'''

    return PAGE.replace("__ACTIVE__", active) \
               .replace("__STATS__", stats_html) \
               .replace("__DATA__", json.dumps(sd, ensure_ascii=False))


PAGE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dictate — статистика и история</title>
<style>
:root{
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --border:rgba(11,11,11,.10);
  --series:#2a78d6; --series2:#8ab4e8; --accent:#eb6834; --track:#e6ecf5;
}
@media (prefers-color-scheme:dark){:root{
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --border:rgba(255,255,255,.10);
  --series:#3987e5; --series2:#2f5c8f; --accent:#d95926; --track:#26313f;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:20px 28px 0;max-width:1320px;margin:0 auto}
h1{font-size:20px;margin:0 0 14px}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--border)}
.tab{padding:9px 18px;cursor:pointer;color:var(--ink2);border-bottom:2px solid transparent;
  font-weight:500;user-select:none}
.tab.on{color:var(--ink);border-color:var(--series)}
main{max-width:1320px;margin:0 auto;padding:22px 28px 60px}
.view{display:none}.view.on{display:block}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px}
.tile-l{color:var(--ink2);font-size:13px}
.tile-v{font-size:30px;font-weight:600;margin:6px 0 2px;letter-spacing:-.01em}
.tile-s{color:var(--muted);font-size:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.sec{font-size:15px;font-weight:600;margin:22px 0 12px;color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px}
.card.center{display:flex;flex-direction:column;align-items:center}
.card h3{margin:0 0 14px;font-size:14px;font-weight:600;display:flex;
  justify-content:space-between;align-items:center;gap:12px}
.key{font-weight:400;color:var(--muted);font-size:12px;display:flex;gap:10px;align-items:center}
.key i{display:inline-block;width:12px;height:8px;border-radius:2px;vertical-align:middle;margin-right:3px}
.sw2{background:var(--series2)}.swa{background:var(--accent)}
.chart{width:100%;height:auto;overflow:visible}
.lbl{fill:var(--ink2);font-size:13px}.val{fill:var(--ink);font-size:13px;font-weight:600}
.tick{fill:var(--muted);font-size:11px}
.grid{stroke:var(--grid);stroke-width:1}
.donut{width:150px;height:150px}
.donut-num{fill:var(--ink);font-size:26px;font-weight:600}
.donut-lbl{fill:var(--muted);font-size:11px}
.note{color:var(--muted);font-size:12px;margin:10px 0 0}
.empty{color:var(--muted);padding:20px;text-align:center}
/* поиск */
.searchbar{display:flex;gap:10px;margin-bottom:16px}
#q{flex:1;padding:11px 14px;border:1px solid var(--border);border-radius:10px;
  background:var(--surface);color:var(--ink);font-size:15px}
#count{color:var(--muted);align-self:center;font-size:13px;white-space:nowrap}
.row{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:12px 15px;margin-bottom:8px}
.row-h{display:flex;gap:10px;color:var(--muted);font-size:12px;margin-bottom:5px}
.row-h .app{color:var(--series);font-weight:600}
.row-t{white-space:pre-wrap}
.row-raw{color:var(--muted);font-size:12.5px;margin-top:5px}
mark{background:rgba(235,104,52,.28);color:inherit;border-radius:2px;padding:0 1px}
@media(max-width:820px){.tiles{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}
</style></head><body>
<header>
  <h1>🎤 dictate</h1>
  <div class="tabs">
    <div class="tab" data-v="stats">Статистика</div>
    <div class="tab" data-v="search">Поиск истории</div>
  </div>
</header>
<main>
  <section class="view" id="stats">__STATS__</section>
  <section class="view" id="search">
    <div class="searchbar">
      <input id="q" placeholder="Искать по надиктованному тексту…" autofocus>
      <span id="count"></span>
    </div>
    <div id="results"></div>
  </section>
</main>
<script>
const DATA=__DATA__;
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function hl(s,q){if(!q)return esc(s);
  return esc(s).replace(new RegExp('('+q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')','gi'),'<mark>$1</mark>')}
function render(q){
  q=q.trim();const ql=q.toLowerCase();
  const r=DATA.filter(d=>!ql||d.text.toLowerCase().includes(ql)||d.raw.toLowerCase().includes(ql));
  document.getElementById('count').textContent=r.length+' из '+DATA.length;
  document.getElementById('results').innerHTML=r.slice(0,400).map(d=>
    '<div class="row"><div class="row-h"><span class="app">'+esc(d.app)+
    '</span><span>'+esc(d.t)+'</span></div><div class="row-t">'+hl(d.text,q)+'</div>'+
    (d.raw?'<div class="row-raw">сырой: '+hl(d.raw,q)+'</div>':'')+'</div>').join('')||
    '<p class="empty">ничего не найдено</p>';
}
function show(v){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.dataset.v===v));
  document.querySelectorAll('.view').forEach(s=>s.classList.toggle('on',s.id===v));
  if(v==='search')document.getElementById('q').focus();
  location.hash=v;
}
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>show(t.dataset.v));
document.getElementById('q').addEventListener('input',e=>render(e.target.value));
render('');
show(location.hash==='#search'?'search':'__ACTIVE__');
</script></body></html>"""


def main():
    active = "search" if "--search" in sys.argv else "stats"
    with open(OUT, "w") as f:
        f.write(build(active))
    subprocess.run(["open", OUT])


if __name__ == "__main__":
    main()
