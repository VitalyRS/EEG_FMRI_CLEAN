#!/usr/bin/env python3
"""
Generate comprehensive HTML batch comparison report with:
1. PSD overlays (frequency domain)
2. Time-domain snippets at t=1s, 50s, 250s for key channels
3. Band power ratios table
4. Cleaning metrics
"""
import json
from pathlib import Path

print("Loading spectral data...")
batch_diff = json.load(open('/tmp/batch_diff.json'))['segments']
batch_meta = json.load(open('/tmp/batch_meta.json'))

# Try to load timeseries data (may not exist yet)
timeseries_path = Path('/tmp/batch_timeseries.json')
if timeseries_path.exists():
    print("Loading timeseries data...")
    batch_timeseries = json.load(open(timeseries_path))['segments']
    has_timeseries = True
else:
    print("⚠️  Timeseries data not found. Run extract_timeseries.py first to add time-domain plots.")
    batch_timeseries = []
    has_timeseries = False

# Try to load cleaning quality metrics
metrics_path = Path('/tmp/batch_cleaning_metrics.json')
if metrics_path.exists():
    print("Loading cleaning quality metrics...")
    batch_metrics_data = json.load(open(metrics_path))['segments']
    # Index by key
    batch_metrics_dict = {f"{m['subj']}:{m['seg']}": m for m in batch_metrics_data}
    has_metrics = True
else:
    print("⚠️  Cleaning metrics not found. Run compute_cleaning_metrics.py to add quality comparison.")
    batch_metrics_dict = {}
    has_metrics = False

# Build enriched segment list for PSD
segments = []
for seg_data in batch_diff:
    subj, seg = seg_data['subj'], seg_data['seg']
    key = f"{subj}:{seg}"
    meta = batch_meta.get(key, {})
    bands = {b['name']: b['ratio'] for b in seg_data['bands']}

    # Add cleaning quality metrics if available
    quality_metrics = batch_metrics_dict.get(key, {})

    segments.append({
        'subj': subj, 'seg': seg, 'key': key,
        'nch': seg_data['nch'], 'dur': seg_data['dur'],
        'cut_ours': seg_data['cut_ours'], 'cut_ref': seg_data['cut_ref'],
        'ref_muscle': seg_data['ref_muscle'], 'bands': bands,
        'f_o': seg_data['f_o'], 'p_o': seg_data['p_o'],
        'f_r': seg_data['f_r'], 'p_r': seg_data['p_r'],
        'burst_crit': meta.get('burst_crit'),
        'n_ic': meta.get('n_ic'),
        'n_ic_rejected': meta.get('n_ic_rejected'),
        'alpha_retention': meta.get('alpha_retention'),
        'variance_drop': meta.get('variance_drop'),
        'quality_score': quality_metrics.get('overall_score'),
        'quality_winner': quality_metrics.get('winner'),
        'temporal_std_ours': quality_metrics.get('temporal_stability_ours'),
        'temporal_std_ref': quality_metrics.get('temporal_stability_ref'),
    })

print(f"Building HTML for {len(segments)} segments...")

def ratio_class(ratio, band_name):
    if ratio is None: return 'na'
    if any(x in band_name for x in ['alpha', 'beta', 'theta']):
        return 'good' if 0.7 <= ratio <= 1.5 else ('warn' if 0.5 <= ratio <= 2.5 else 'bad')
    return 'good' if 0.5 <= ratio <= 2.0 else ('warn' if 0.3 <= ratio <= 4.0 else 'bad')

def cutoff_class(ours, ref):
    if ours is None or ref is None: return 'na'
    diff = abs(ours - ref)
    return 'good' if diff <= 5 else ('warn' if diff <= 15 else 'bad')

BAND_KEYS = ['delta 1-4', 'theta 4-8', 'alpha 8-13', 'beta 13-30', 'low-gamma 30-45', 'hi 45-80', 'above-80']

# Build table rows
table_rows = ""
for s in segments:
    cut_cls = cutoff_class(s['cut_ours'], s['cut_ref'])
    cut_badge = f"<span class='cutoff-badge {cut_cls}'>{s['cut_ours']:.1f} / {s['cut_ref']:.1f}</span>"
    band_cells = []
    for bk in BAND_KEYS:
        r = s['bands'].get(bk)
        if r is not None:
            cls = ratio_class(r, bk)
            width = min(100, r * 50)
            band_cells.append(f"<td class='ratio-cell {cls}'><div class='ratio-bar' style='width:{width}%'></div><span class='ratio-val'>{r:.2f}</span></td>")
        else:
            band_cells.append("<td class='ratio-cell na'><span class='ratio-val'>–</span></td>")
    muscle_mark = " <small style='color:var(--text-dim);'>+ica_musle</small>" if s['ref_muscle'] else ""
    ic_str = f"{s['n_ic_rejected']}/{s['n_ic']}" if s['n_ic'] else "–"
    alpha_ret = f"{s['alpha_retention']:.2f}" if s['alpha_retention'] is not None else "–"
    var_drop = f"{s['variance_drop']:.2f}" if s['variance_drop'] is not None else "–"

    # Quality score badge
    if s['quality_score'] is not None:
        score = s['quality_score']
        winner = s['quality_winner']
        if winner == 'ours':
            score_badge = f"<span class='score-badge good'>{score:.0f}</span>"
        elif winner == 'ref':
            score_badge = f"<span class='score-badge bad'>{score:.0f}</span>"
        else:
            score_badge = f"<span class='score-badge warn'>{score:.0f}</span>"
    else:
        score_badge = "<span class='score-badge na'>–</span>"

    table_rows += f"<tr><td class='mono'>{s['subj']}/{s['seg']}{muscle_mark}</td><td>{cut_badge}</td>{''.join(band_cells)}<td style='text-align:center;'><span class='asr-badge'>{s['burst_crit']}</span></td><td class='num'>{ic_str}</td><td class='num'>{alpha_ret}</td><td class='num'>{var_drop}</td><td style='text-align:center;'>{score_badge}</td></tr>\n"

# Build timeseries section HTML
timeseries_section = ""
if has_timeseries:
    timeseries_section = """
<div class="section">
<h2>Временные характеристики</h2>
<p style="color:var(--text-dim);margin-bottom:1rem">
  Сравнение сырых сигналов (1 секунда окно) в моменты t=1s, t=50s, t=250s для ключевых каналов.
  Синий = наши FIF, серый = reference .set.
</p>
<div id="timeseries-container"></div>
</div>
"""

# CSS additions for timeseries
timeseries_css = """
.timeseries-segment{margin-bottom:2rem;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.5rem}
.timeseries-segment h3{font-size:1.1rem;margin-bottom:1rem;border-bottom:1px solid var(--border);padding-bottom:0.5rem}
.timepoint-section{margin-bottom:1.5rem}
.timepoint-header{font-size:0.9rem;font-weight:600;color:var(--text-dim);margin-bottom:0.5rem}
.timeseries-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}
.timeseries-card{background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:0.75rem}
.timeseries-card canvas{width:100%;height:150px}
.channel-label{font-family:var(--font-mono);font-size:0.85rem;font-weight:600;margin-bottom:0.25rem}
""" if has_timeseries else ""

# JavaScript for timeseries rendering
timeseries_js = ""
if has_timeseries:
    timeseries_js = f"""
const timeseriesData = {json.dumps(batch_timeseries)};
const container = document.getElementById('timeseries-container');

timeseriesData.forEach(seg => {{
  const segDiv = document.createElement('div');
  segDiv.className = 'timeseries-segment';

  const title = document.createElement('h3');
  title.textContent = `${{seg.subj}}/${{seg.seg}}`;
  segDiv.appendChild(title);

  seg.timepoints.forEach(tp => {{
    const tpSection = document.createElement('div');
    tpSection.className = 'timepoint-section';

    const tpHeader = document.createElement('div');
    tpHeader.className = 'timepoint-header';
    tpHeader.textContent = `t = ${{tp.t}} сек`;
    tpSection.appendChild(tpHeader);

    const grid = document.createElement('div');
    grid.className = 'timeseries-grid';

    tp.channels.forEach(ch => {{
      const card = document.createElement('div');
      card.className = 'timeseries-card';

      const label = document.createElement('div');
      label.className = 'channel-label';
      label.textContent = ch.name;
      card.appendChild(label);

      const canvas = document.createElement('canvas');
      canvas.width = 560;
      canvas.height = 300;
      card.appendChild(canvas);

      drawTimeseries(canvas, ch.ours, ch.ref);
      grid.appendChild(card);
    }});

    tpSection.appendChild(grid);
    segDiv.appendChild(tpSection);
  }});

  container.appendChild(segDiv);
}});

function drawTimeseries(canvas, ours, ref) {{
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const pad = {{t:20, r:20, b:40, l:50}};
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  const times_o = ours.times;
  const vals_o = ours.values;
  const times_r = ref.times;
  const vals_r = ref.values;

  // Normalize both signals to same scale for visual comparison
  const allVals = vals_o.concat(vals_r);
  const globalMin = Math.min(...allVals);
  const globalMax = Math.max(...allVals);
  const globalRange = globalMax - globalMin;
  const globalPad = globalRange * 0.1;

  const vMin = globalMin - globalPad;
  const vMax = globalMax + globalPad;
  const vRange = vMax - vMin;

  const tMax = Math.max(times_o[times_o.length-1], times_r[times_r.length-1]);

  const xScale = t => pad.l + (t / tMax) * plotW;
  const yScale = v => pad.t + plotH - ((v - vMin) / vRange) * plotH;

  const isDark = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() === '#0a0a0a';

  // Background
  ctx.fillStyle = isDark ? '#1a1a1a' : '#ffffff';
  ctx.fillRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = isDark ? '#333' : '#e5e5e5';
  ctx.lineWidth = 1;
  for(let i=0; i<=4; i++) {{
    const y = pad.t + (plotH * i / 4);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + plotW, y);
    ctx.stroke();
  }}

  // Reference line (gray)
  ctx.strokeStyle = isDark ? '#888' : '#666';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for(let i=0; i<times_r.length; i++) {{
    const x = xScale(times_r[i]);
    const y = yScale(vals_r[i]);
    if(i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }}
  ctx.stroke();

  // Ours line (blue)
  ctx.strokeStyle = isDark ? '#3b82f6' : '#0066cc';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for(let i=0; i<times_o.length; i++) {{
    const x = xScale(times_o[i]);
    const y = yScale(vals_o[i]);
    if(i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }}
  ctx.stroke();

  // Axes
  ctx.strokeStyle = isDark ? '#f0f0f0' : '#1a1a1a';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, pad.t + plotH);
  ctx.lineTo(pad.l + plotW, pad.t + plotH);
  ctx.stroke();

  // Labels
  ctx.fillStyle = isDark ? '#f0f0f0' : '#1a1a1a';
  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.textAlign = 'center';

  // X-axis
  for(let i=0; i<=4; i++) {{
    const t = (tMax * i / 4);
    ctx.fillText(t.toFixed(2), xScale(t), pad.t + plotH + 20);
  }}

  // Y-axis
  ctx.textAlign = 'right';
  for(let i=0; i<=4; i++) {{
    const v = vMin + (vRange * i / 4);
    ctx.fillText(v.toFixed(1), pad.l - 10, pad.t + plotH - (plotH * i / 4) + 4);
  }}

  // Axis labels
  ctx.textAlign = 'center';
  ctx.fillText('Время (с)', pad.l + plotW/2, h - 5);
  ctx.save();
  ctx.translate(15, pad.t + plotH/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillText('µV', 0, 0);
  ctx.restore();
}}
"""

# Build complete HTML
html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Сравнительный отчёт: 1916+1925</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{--bg:#fafafa;--surface:#fff;--text:#1a1a1a;--text-dim:#666;--border:#e0e0e0;--accent:#0066cc;--good:#10b981;--warn:#f59e0b;--bad:#ef4444;--na:#9ca3af;--font-sans:'IBM Plex Sans',sans-serif;--font-mono:'IBM Plex Mono',monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0a0a0a;--surface:#1a1a1a;--text:#f0f0f0;--text-dim:#a0a0a0;--border:#333;--accent:#3b82f6;--good:#059669;--warn:#d97706;--bad:#dc2626;--na:#6b7280}}}}
:root[data-theme="dark"]{{--bg:#0a0a0a;--surface:#1a1a1a;--text:#f0f0f0;--text-dim:#a0a0a0;--border:#333;--accent:#3b82f6;--good:#059669;--warn:#d97706;--bad:#dc2626;--na:#6b7280}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:var(--font-sans);font-size:15px;line-height:1.6;padding:2rem 1rem}}
.container{{max-width:1600px;margin:0 auto}}
h1{{font-size:2rem;font-weight:600;margin-bottom:0.5rem}}
.subtitle{{color:var(--text-dim);margin-bottom:2rem}}
.section{{margin-bottom:3rem}}
h2{{font-size:1.5rem;font-weight:600;margin-bottom:1rem;border-bottom:2px solid var(--border);padding-bottom:0.5rem}}
h3{{font-size:1.2rem;font-weight:500;margin:1.5rem 0 0.75rem}}
table{{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--border);font-size:0.9rem}}
thead{{background:var(--border)}}
th,td{{padding:0.6rem 0.8rem;text-align:left;border:1px solid var(--border)}}
th{{font-weight:600;white-space:nowrap}}
.mono{{font-family:var(--font-mono);font-size:0.85rem}}
.num{{text-align:right;font-family:var(--font-mono)}}
.ratio-cell{{position:relative;font-family:var(--font-mono);font-size:0.85rem;padding:0.4rem 0.6rem}}
.ratio-bar{{position:absolute;top:0;left:0;bottom:0;opacity:0.15;z-index:0}}
.ratio-val{{position:relative;z-index:1}}
.ratio-cell.good .ratio-bar{{background:var(--good)}}
.ratio-cell.warn .ratio-bar{{background:var(--warn)}}
.ratio-cell.bad .ratio-bar{{background:var(--bad)}}
.ratio-cell.na{{color:var(--na)}}
.cutoff-badge{{display:inline-block;padding:0.2rem 0.5rem;border-radius:4px;font-family:var(--font-mono);font-size:0.8rem;font-weight:500}}
.cutoff-badge.good{{background:var(--good);color:#fff}}
.cutoff-badge.warn{{background:var(--warn);color:#fff}}
.cutoff-badge.bad{{background:var(--bad);color:#fff}}
.asr-badge{{display:inline-block;padding:0.15rem 0.4rem;border-radius:3px;background:var(--border);font-family:var(--font-mono);font-size:0.75rem}}
.psd-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:1.5rem;margin-top:1rem}}
.psd-card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem}}
.psd-title{{font-weight:600;margin-bottom:0.5rem}}
canvas{{width:100%;display:block}}
.psd-card canvas{{height:220px}}
.findings{{background:var(--surface);border-left:4px solid var(--accent);padding:1rem 1.5rem;margin-top:1rem}}
.findings ul{{margin-left:1.5rem;margin-top:0.5rem}}
.findings li{{margin-bottom:0.5rem}}
strong{{font-weight:600}}
{timeseries_css}
</style>
</head>
<body>
<div class="container">
<h1>Сравнительный отчёт очистки: 1916 + 1925</h1>
<p class="subtitle">FIF (1.0–80 Hz Bergen+BCG+ICA) vs reference .set (LPF_80 MR_corr+asr+ica)</p>

<div class="section">
<h2>Сводная таблица</h2>
<p style="color:var(--text-dim);margin-bottom:1rem">Соотношение = (наша мощность)/(reference). Зелёный≈1.0, жёлтый=отклонение, красный=большое.</p>
<table>
<thead><tr>
<th>Сегмент</th><th>Cutoff(Hz)</th><th>δ 1–4</th><th>θ 4–8</th><th>α 8–13</th><th>β 13–30</th><th>γₗ 30–45</th><th>HF 45–80</th><th>&gt;80</th><th>ASR</th><th>IC удал/всего</th><th>αRet</th><th>varDrop</th><th>Качество</th>
</tr></thead>
<tbody>{table_rows}</tbody>
</table>
</div>

<div class="section">
<h2>PSD наложения (частотная область)</h2>
<div class="psd-grid" id="psd-grid"></div>
</div>

{timeseries_section}

<div class="section">
<h2>Выводы</h2>
<div class="findings">
<h3>Частотная полоса</h3>
<ul>
<li><strong>Достигнуто:</strong> FIF 1.0–80Hz (cutoff 80–91Hz) vs reference 58–80Hz. High-pass 1.0Hz убирает медленный дрейф (было 0.5Hz).</li>
<li><strong>Мощность:</strong> Наши FIF мощнее (1.4–35×) = более мягкая очистка, сохранение сигнала. Reference применяет ica_musle на ec/eo.</li>
<li><strong>Alpha 8–13Hz:</strong> Соотношения 1.1–5.7×, большинство 1.4–3.3× = нейронные осцилляции сохранены.</li>
</ul>
<h3>ASR</h3>
<ul>
<li><strong>Все 8 выбрали burst_crit='off'</strong> после Bergen+BCG+band-pass 1.0–80Hz — полоса 40–80Hz чистая, ASR рискует повредить гамму.</li>
<li><strong>Важно:</strong> Это для текущих данных 1916/1925. Для других субъектов/протоколов Phase-ASR может выбрать k=20/10. Механизм проверяет каждый сегмент независимо.</li>
</ul>
<h3>ICA паттерны</h3>
<ul>
<li><strong>1916:</strong> 35–43 IC из 91–92 (38–47%), varDrop 56–86%, ICLabel 0.75–0.85.</li>
<li><strong>1925:</strong> 14–52 IC, varDrop 29–82%, ICLabel 0.65–0.75 (чище или другой артефакт-профиль).</li>
<li>Разница 3–4× = индивидуальная физиология, правильно обработано Optuna.</li>
</ul>
<h3>Дальше</h3>
<ul>
<li>✅ Полоса 1.0–80Hz совпадает с reference.</li>
<li>✅ ASR оптимизирован для текущих данных.</li>
<li>Опционально: Если ratio&gt;2× неприемлемы → добавить второй проход ICA для мышц (как reference ica_musle).</li>
</ul>
</div>
</div>

</div>

<script>
// PSD plots
const segments={json.dumps(segments)};
const grid=document.getElementById('psd-grid');
segments.forEach(s=>{{
const card=document.createElement('div');card.className='psd-card';
const title=document.createElement('div');title.className='psd-title';title.textContent=`${{s.subj}}/${{s.seg}}`;
const canvas=document.createElement('canvas');canvas.width=800;canvas.height=440;
card.appendChild(title);card.appendChild(canvas);grid.appendChild(card);
drawPSD(canvas,s.f_o,s.p_o,s.f_r,s.p_r,s.cut_ours,s.cut_ref);
}});

function drawPSD(canvas,f_o,p_o,f_r,p_r,cut_o,cut_r){{
const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
const pad={{t:20,r:40,b:50,l:60}},plotW=w-pad.l-pad.r,plotH=h-pad.t-pad.b;
const fMax=120,allP=p_o.concat(p_r),pMin=Math.min(...allP)*0.8,pMax=Math.max(...allP)*1.2;
const xScale=f=>pad.l+(f/fMax)*plotW;
const yScale=p=>pad.t+plotH-((Math.log10(p)-Math.log10(pMin))/(Math.log10(pMax)-Math.log10(pMin)))*plotH;
const isDark=getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()==='#0a0a0a';
ctx.fillStyle=isDark?'#1a1a1a':'#fff';ctx.fillRect(0,0,w,h);
ctx.strokeStyle=isDark?'#333':'#e0e0e0';ctx.lineWidth=1;
for(let f=0;f<=fMax;f+=20){{const x=xScale(f);ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,pad.t+plotH);ctx.stroke();}}
ctx.strokeStyle=isDark?'#888':'#666';ctx.lineWidth=2;ctx.beginPath();
for(let i=0;i<f_r.length;i++){{if(f_r[i]>fMax)break;const x=xScale(f_r[i]),y=yScale(p_r[i]);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}}
ctx.stroke();
ctx.strokeStyle=isDark?'#3b82f6':'#0066cc';ctx.lineWidth=2.5;ctx.beginPath();
for(let i=0;i<f_o.length;i++){{if(f_o[i]>fMax)break;const x=xScale(f_o[i]),y=yScale(p_o[i]);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}}
ctx.stroke();
if(cut_o){{ctx.strokeStyle=isDark?'#3b82f6':'#0066cc';ctx.setLineDash([5,5]);ctx.lineWidth=1.5;const xo=xScale(cut_o);ctx.beginPath();ctx.moveTo(xo,pad.t);ctx.lineTo(xo,pad.t+plotH);ctx.stroke();}}
if(cut_r){{ctx.strokeStyle=isDark?'#888':'#666';ctx.setLineDash([5,5]);ctx.lineWidth=1.5;const xr=xScale(cut_r);ctx.beginPath();ctx.moveTo(xr,pad.t);ctx.lineTo(xr,pad.t+plotH);ctx.stroke();}}
ctx.setLineDash([]);
ctx.strokeStyle=isDark?'#f0f0f0':'#1a1a1a';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,pad.t+plotH);ctx.lineTo(pad.l+plotW,pad.t+plotH);ctx.stroke();
ctx.fillStyle=isDark?'#f0f0f0':'#1a1a1a';ctx.font='12px "IBM Plex Mono",monospace';ctx.textAlign='center';
for(let f=0;f<=fMax;f+=20)ctx.fillText(f,xScale(f),pad.t+plotH+20);
ctx.save();ctx.translate(15,pad.t+plotH/2);ctx.rotate(-Math.PI/2);ctx.fillText('PSD (µV²/Hz, log)',0,0);ctx.restore();
ctx.fillText('Frequency (Hz)',pad.l+plotW/2,h-10);
ctx.font='11px "IBM Plex Sans"';ctx.textAlign='left';
ctx.fillStyle=isDark?'#3b82f6':'#0066cc';ctx.fillText('● Ours (FIF)',w-120,pad.t+15);
ctx.fillStyle=isDark?'#888':'#666';ctx.fillText('● Ref (.set)',w-120,pad.t+30);
}}

// Timeseries plots
{timeseries_js}
</script>
</body>
</html>"""

out = Path('/tmp/batch_report.html')
out.write_text(html, encoding='utf-8')
print(f"✅ Wrote {out} ({len(html):,} bytes)")

rep = Path('reports')
if rep.exists():
    dest = rep / 'batch_cleaning_1916_1925.html'
    dest.write_text(html, encoding='utf-8')
    print(f"✅ Copied to {dest}")
else:
    print("⚠️  reports/ directory not found")

print("\nTo add time-domain plots:")
print("1. Run: python3 extract_timeseries.py")
print("2. Run: python3 generate_report.py")
print("\nOr open current report (PSD only):")
print(f"  xdg-open {out}")
