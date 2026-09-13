#!/usr/bin/env python3
"""Render a directory of schema-1 bench results into one self-contained page.

    python3 bench/report.py                 # -> bench/report.html (thin CLI shim)
    GET /api/bench/report                   # -> the same page, live, for the UI

Moved here from ``bench/report.py`` so the product can serve the report it
already knew how to write to disk. ``bench/report.py`` is now a shim over this
module and keeps its old command line; the renderer is unchanged, so a page
served from the Bench view and a page committed to the repo are the same page.

No CDN, no build step, no JavaScript: inline CSS tokens and inline SVG charts, so
the file works opened from disk, pasted into an artifact host, served by the
marketing site, or dropped into an iframe. Dark by default with a light palette
under prefers-color-scheme.

A missing section renders as "Not measured" in words. That is the whole point of
the format: bench/SCHEMA.md forbids filling a gap with an estimate, so the page
has to be able to say "we did not measure this" without it looking like a zero.
stdlib only.
"""
import argparse
import html
import json
import pathlib
import sys

CSS = """
:root{
  --ground:#0b0d0c; --s1:#131614; --s2:#1b201d; --line:#2a302c;
  --ink:#e9ece9; --steel:#8d958f; --accent:#76b900; --accent-dim:#4d7a08;
  --cool:#5aa9c8; --warn:#d9a13b; --good:#76b900;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,"DejaVu Sans Mono",Consolas,monospace;
  --pad:clamp(18px,4vw,56px);
}
@media (prefers-color-scheme:light){
  :root{
    --ground:#f7f8f6; --s1:#ffffff; --s2:#eef1ec; --line:#d8ded6;
    --ink:#161915; --steel:#5f6862; --accent:#4d7a08; --accent-dim:#76b900;
    --cool:#22708f; --warn:#8a5d06;
  }
}
*{box-sizing:border-box;margin:0;padding:0}
html{background:var(--ground);-webkit-text-size-adjust:100%}
body{font-family:var(--sans);color:var(--ink);background:var(--ground);
  line-height:1.55;font-size:15px}
.wrap{max-width:1120px;margin:0 auto;padding:var(--pad)}
a{color:var(--accent)}
h1{font-size:clamp(28px,5vw,44px);line-height:1.1;letter-spacing:-.02em;font-weight:650}
h1 .g{color:var(--accent)}
h2{font-size:clamp(19px,2.6vw,25px);margin:56px 0 6px;letter-spacing:-.01em;font-weight:620}
h3{font-size:19px;font-weight:620;letter-spacing:-.01em}
.lede{color:var(--steel);max-width:70ch;margin-top:14px}
.lede strong{color:var(--ink);font-weight:600}
.sub{color:var(--steel);font-size:13.5px;margin-top:4px;max-width:78ch}
.top{border-bottom:1px solid var(--line);padding-bottom:26px}
.brandrow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
.tag{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);border:1px solid var(--accent-dim);border-radius:3px;padding:3px 8px}
.counts{font-family:var(--mono);font-size:12px;color:var(--steel)}

/* leaderboard */
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--s1);
  margin-top:18px}
table{border-collapse:collapse;width:100%;min-width:860px;font-size:13.5px}
th,td{text-align:left;padding:10px 13px;border-bottom:1px solid var(--line);
  white-space:nowrap;vertical-align:baseline}
th{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--steel);
  font-weight:600;background:var(--s2)}
tbody tr:last-child td{border-bottom:none}
td.num{font-family:var(--mono);text-align:right}
td.num b{color:var(--accent);font-weight:600}
td.mdl{white-space:normal;min-width:190px}
td.mdl b{font-weight:620}
td.mdl span{display:block;color:var(--steel);font-size:12px;font-family:var(--mono)}
.dim{color:var(--steel)}

/* run blocks */
.mrun{border:1px solid var(--line);border-radius:10px;background:var(--s1);
  margin-top:22px;overflow:hidden}
.mhead{padding:18px 20px;border-bottom:1px solid var(--line);background:var(--s2)}
.mtitle{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.badge{font-family:var(--mono);font-size:11px;color:var(--accent);
  border:1px solid var(--accent-dim);border-radius:3px;padding:2px 7px}
.mmeta{font-family:var(--mono);font-size:12px;color:var(--steel);margin-top:8px;
  line-height:1.8;word-break:break-word}
.mmeta b{color:var(--ink);font-weight:500}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  border-bottom:1px solid var(--line)}
.stat{padding:15px 20px;border-right:1px solid var(--line)}
.stat:last-child{border-right:none}
.stat .k{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--steel)}
.stat .v{font-family:var(--mono);font-size:23px;color:var(--accent);line-height:1.25;
  margin-top:3px;font-weight:600}
.stat .v.na{color:var(--steel);font-size:15px;font-weight:400}
.stat .u{font-family:var(--mono);font-size:11.5px;color:var(--steel)}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
.panel{padding:18px 20px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
.panel.wide{grid-column:1/-1;border-right:none}
.ptitle{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--steel);
  font-weight:600;margin-bottom:4px}
.pnote{font-size:12.5px;color:var(--steel);margin-top:8px}
.chart{width:100%;height:auto;display:block;margin-top:10px;overflow:visible}
.nm{font-family:var(--mono);font-size:13px;color:var(--steel);
  border:1px dashed var(--line);border-radius:5px;padding:9px 11px;margin-top:10px}
.tscroll{overflow-x:auto;margin-top:10px}
.minitable{width:100%;min-width:0;font-size:12.5px}
.minitable th,.minitable td{padding:5px 8px;white-space:nowrap}
.minitable td{font-family:var(--mono)}
.chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}
.chip{font-family:var(--mono);font-size:12px;color:var(--ink);background:var(--s2);
  border:1px solid var(--line);border-radius:999px;padding:4px 11px}
.chip i{font-style:normal;color:var(--steel)}
.flags{font-family:var(--mono);font-size:12px;color:var(--steel);background:var(--s2);
  border:1px solid var(--line);border-radius:6px;padding:11px 13px;margin-top:10px;
  overflow-x:auto;white-space:pre-wrap;word-break:break-word;line-height:1.65}
.flags b{color:var(--ink);font-weight:500}
ul.notes{margin:10px 0 0 17px;font-size:13.5px;color:var(--steel)}
ul.notes li{margin-bottom:5px}
ul.notes li::marker{color:var(--accent)}
.method{border:1px solid var(--line);border-radius:10px;background:var(--s1);
  padding:20px;margin-top:18px}
.method h4{font-size:13.5px;margin:16px 0 4px;font-weight:620}
.method h4:first-child{margin-top:0}
.method p{font-size:13.5px;color:var(--steel);max-width:82ch}
.method code{font-family:var(--mono);font-size:12.5px;color:var(--ink);
  background:var(--s2);border:1px solid var(--line);border-radius:3px;padding:1px 5px}
footer{margin-top:52px;padding-top:20px;border-top:1px solid var(--line);
  font-family:var(--mono);font-size:12px;color:var(--steel);
  display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap}
footer b{color:var(--accent);font-weight:500}
"""


# ---------------------------------------------------------------- formatting

def e(s):
    return html.escape(str(s), quote=True)


def num(v, nd=1):
    """Numbers a reader can scan: no trailing .0 noise, em-dash-free placeholder."""
    if v is None:
        return "-"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def ctx(n):
    if not n:
        return "-"
    if n >= 1_000_000:
        return f"{n / 1_048_576:.0f}M"
    if n >= 1024:
        return f"{n // 1024}K"
    return str(n)


def toks(n):
    if n is None:
        return "-"
    if n >= 1000:
        return f"{n / 1000:.0f}k" if n >= 10000 else f"{n / 1000:.1f}k"
    return str(n)


def ms(v):
    """TTFT reads better in seconds once it passes a second."""
    if v is None:
        return "-"
    if v >= 10000:
        return f"{v / 1000:.0f}s"
    if v >= 1000:
        return f"{v / 1000:.1f}s"
    return f"{v:.0f}ms"


def stamp_txt(s):
    s = str(s or "")
    if len(s) >= 15 and "-" in s:
        d, t = s.split("-", 1)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]} UTC"
    return s


def params_txt(m):
    p, a = m.get("params_b"), m.get("active_b")
    if not p:
        return "-"
    if m.get("arch") == "moe" and a and a != p:
        return f"{num(p)}B / {num(a)}B active"
    return f"{num(p)}B dense"


# ---------------------------------------------------------------- reading

def load_runs(d):
    runs = []
    for f in sorted(d.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except ValueError as ex:
            print(f"  skipped {f.name}: not valid JSON ({ex})", file=sys.stderr)
            continue
        if r.get("schema") != 1:
            print(f"  skipped {f.name}: schema {r.get('schema')!r}, expected 1",
                  file=sys.stderr)
            continue
        r["_file"] = f.name
        runs.append(r)
    runs.sort(key=lambda r: str(r.get("stamp", "")), reverse=True)
    return runs


def res(r, key):
    return (r.get("results") or {}).get(key)


def single_tps(r):
    s = res(r, "single_stream") or {}
    return s.get("decode_tok_s")


def best_conc(r):
    """The 16-stream row if it exists, else the widest sweep that ran. Returns
    (streams, aggregate) so the column can name the stream count it is quoting
    instead of implying every run was measured at 16."""
    rows = [c for c in (res(r, "concurrency") or []) if c.get("aggregate_tok_s")]
    if not rows:
        return None, None
    for c in rows:
        if c.get("streams") == 16:
            return 16, c["aggregate_tok_s"]
    top = max(rows, key=lambda c: c.get("streams") or 0)
    return top.get("streams"), top.get("aggregate_tok_s")


# ---------------------------------------------------------------- svg charts

def _sc(v, v0, v1, o0, o1):
    if v1 == v0:
        return (o0 + o1) / 2
    return o0 + (v - v0) * (o1 - o0) / (v1 - v0)


def svg_prefill(rows):
    """TTFT and decode against prompt length, both on a linear x so the shape of
    the curve is the real shape: prefill grows faster than the prompt does."""
    pts = [r for r in rows if r.get("prompt_tokens") and r.get("ttft_ms") is not None]
    if len(pts) < 2:
        return ""
    W, H = 620, 262
    L, R, T, B = 56, 52, 26, 38
    xmax = max(p["prompt_tokens"] for p in pts) * 1.04
    ymax = max(p["ttft_ms"] for p in pts) * 1.14 or 1
    decs = [p["decode_tok_s"] for p in pts if p.get("decode_tok_s")]
    dmax = (max(decs) * 1.25) if decs else 0
    def X(v):
        return _sc(v, 0, xmax, L, W - R)

    def Y(v):
        return _sc(v, 0, ymax, H - B, T)

    def D(v):
        return _sc(v, 0, dmax, H - B, T)

    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
         f'aria-label="time to first token against prompt tokens">']
    for i in range(5):                                   # grid + left axis
        gv = ymax * i / 4
        y = Y(gv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" '
                 f'stroke="var(--line)" stroke-width="1"/>')
        o.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="12.5" '
                 f'fill="var(--steel)" font-family="var(--mono)">{e(ms(gv))}</text>')
    if dmax:                                             # right axis for decode
        for i in (2, 4):
            dv = dmax * i / 4
            o.append(f'<text x="{W - R + 8}" y="{D(dv) + 4:.1f}" font-size="12" '
                     f'fill="var(--cool)" font-family="var(--mono)">{num(dv, 0)}</text>')
        dpath = " ".join(f"{X(p['prompt_tokens']):.1f},{D(p['decode_tok_s']):.1f}"
                         for p in pts if p.get("decode_tok_s"))
        if dpath:
            o.append(f'<polyline points="{dpath}" fill="none" stroke="var(--cool)" '
                     f'stroke-width="1.6" stroke-dasharray="5 4"/>')
            for p in pts:
                if p.get("decode_tok_s"):
                    o.append(f'<circle cx="{X(p["prompt_tokens"]):.1f}" '
                             f'cy="{D(p["decode_tok_s"]):.1f}" r="3.6" fill="var(--cool)"/>')
    path = " ".join(f"{X(p['prompt_tokens']):.1f},{Y(p['ttft_ms']):.1f}" for p in pts)
    o.append(f'<polyline points="{path}" fill="none" stroke="var(--accent)" stroke-width="2.6"/>')
    for p in pts:
        x, y = X(p["prompt_tokens"]), Y(p["ttft_ms"])
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="var(--accent)"/>')
        o.append(f'<text x="{x:.1f}" y="{y - 11:.1f}" text-anchor="middle" font-size="12.5" '
                 f'fill="var(--accent)" font-family="var(--mono)">{e(ms(p["ttft_ms"]))}</text>')
        o.append(f'<text x="{x:.1f}" y="{H - B + 15:.1f}" text-anchor="middle" '
                 f'font-size="12.5" fill="var(--steel)" font-family="var(--mono)">'
                 f'{e(toks(p["prompt_tokens"]))}</text>')
    o.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" '
             f'stroke="var(--line)" stroke-width="1"/>')
    o.append(f'<text x="{L}" y="{H - 5}" font-size="12" fill="var(--steel)" '
             f'font-family="var(--mono)">prompt tokens (server-reported)</text>')
    o.append(f'<text x="{L}" y="15" font-size="12" fill="var(--accent)" '
             f'font-family="var(--mono)">TTFT</text>')
    if dmax:
        o.append(f'<text x="{L + 52}" y="15" font-size="12" fill="var(--cool)" '
                 f'font-family="var(--mono)">decode tok/s (right)</text>')
    o.append("</svg>")
    return "".join(o)


def svg_concurrency(rows):
    """Aggregate against per-stream, same axis because both are tok/s. The gap
    between the two bars is the throughput the batch buys and the latency each
    user pays for it."""
    rows = [r for r in rows if r.get("aggregate_tok_s")]
    if not rows:
        return ""
    W, H = 620, 250
    L, R, T, B = 52, 14, 26, 44
    ymax = max(r["aggregate_tok_s"] for r in rows) * 1.16 or 1
    def Y(v):
        return _sc(v, 0, ymax, H - B, T)

    slot = (W - L - R) / len(rows)
    bw = min(38, slot * 0.34)

    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
         f'aria-label="aggregate and per-stream tokens per second by stream count">']
    for i in range(5):
        gv = ymax * i / 4
        y = Y(gv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" '
                 f'stroke="var(--line)" stroke-width="1"/>')
        o.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="12.5" '
                 f'fill="var(--steel)" font-family="var(--mono)">{num(gv, 0)}</text>')
    for i, r in enumerate(rows):
        cx = L + slot * (i + 0.5)
        agg, per = r["aggregate_tok_s"], r.get("per_stream_tok_s")
        x0 = cx - (bw + 2) if per else cx - bw / 2
        h = (H - B) - Y(agg)
        o.append(f'<rect x="{x0:.1f}" y="{Y(agg):.1f}" width="{bw:.1f}" height="{h:.1f}" '
                 f'fill="var(--accent)" rx="2"/>')
        o.append(f'<text x="{x0 + bw / 2:.1f}" y="{Y(agg) - 6:.1f}" text-anchor="middle" '
                 f'font-size="13" fill="var(--accent)" font-family="var(--mono)">'
                 f'{num(agg, 0)}</text>')
        if per:
            h2 = (H - B) - Y(per)
            o.append(f'<rect x="{cx + 2:.1f}" y="{Y(per):.1f}" width="{bw:.1f}" '
                     f'height="{h2:.1f}" fill="var(--cool)" rx="2"/>')
            o.append(f'<text x="{cx + 2 + bw / 2:.1f}" y="{Y(per) - 6:.1f}" '
                     f'text-anchor="middle" font-size="13" fill="var(--cool)" '
                     f'font-family="var(--mono)">{num(per, 0)}</text>')
        o.append(f'<text x="{cx:.1f}" y="{H - B + 16:.1f}" text-anchor="middle" '
                 f'font-size="14" fill="var(--ink)" font-family="var(--mono)">'
                 f'{r.get("streams", "?")}</text>')
        if r.get("median_ttft_ms") is not None:
            o.append(f'<text x="{cx:.1f}" y="{H - B + 30:.1f}" text-anchor="middle" '
                     f'font-size="13" fill="var(--steel)" font-family="var(--mono)">'
                     f'{e(ms(r["median_ttft_ms"]))}</text>')
    o.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" '
             f'stroke="var(--line)" stroke-width="1"/>')
    o.append(f'<text x="{L}" y="15" font-size="12" fill="var(--accent)" '
             f'font-family="var(--mono)">aggregate tok/s</text>')
    if any(r.get("per_stream_tok_s") for r in rows):
        o.append(f'<text x="{L + 124}" y="15" font-size="12" fill="var(--cool)" '
                 f'font-family="var(--mono)">per stream</text>')
    o.append(f'<text x="{W - R}" y="15" text-anchor="end" font-size="11.5" '
             f'fill="var(--steel)" font-family="var(--mono)">'
             f'streams, median TTFT under each</text>')
    o.append("</svg>")
    return "".join(o)


def svg_reasoning(d):
    """Wall clock for the same answer, thinking off against thinking on."""
    off, on = d.get("thinking_off_wall_s"), d.get("thinking_on_wall_s")
    if off is None and on is None:
        return ""
    W, H = 620, 126
    # L has to clear "thinking off" at 13 units of mono, or the bar starts on top
    # of its own label.
    L, R = 104, 96
    vmax = max(v for v in (off, on) if v is not None) * 1.1 or 1
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
         f'aria-label="wall clock with thinking off against thinking on">']
    for i, (lbl, v, col, tk) in enumerate((
            ("off", off, "var(--cool)", d.get("thinking_off_tokens")),
            ("on", on, "var(--accent)", d.get("thinking_on_tokens")))):
        y = 22 + i * 46
        o.append(f'<text x="0" y="{y + 15}" font-size="13" fill="var(--ink)" '
                 f'font-family="var(--mono)">thinking {lbl}</text>')
        if v is None:
            o.append(f'<text x="{L}" y="{y + 15}" font-size="11.5" fill="var(--steel)" '
                     f'font-family="var(--mono)">not measured</text>')
            continue
        w = _sc(v, 0, vmax, 0, W - L - R)
        o.append(f'<rect x="{L}" y="{y}" width="{max(w, 2):.1f}" height="22" '
                 f'fill="{col}" rx="3"/>')
        lab = f"{v:.1f}s" + (f"  ({tk} tok)" if tk else "")
        o.append(f'<text x="{L + max(w, 2) + 8:.1f}" y="{y + 15}" font-size="12.5" '
                 f'fill="var(--steel)" font-family="var(--mono)">{e(lab)}</text>')
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- blocks

def leaderboard(runs):
    o = ['<div class="tablewrap"><table><thead><tr>'
         '<th>Model</th><th>Params / active</th><th>Quant</th><th>Placement</th>'
         '<th style="text-align:right">Single stream</th>'
         '<th style="text-align:right">Batched</th>'
         '<th style="text-align:right">Rubric</th>'
         '<th style="text-align:right">Context</th>'
         '</tr></thead><tbody>']
    # Fastest single stream first; a run that did not measure it sinks to the
    # bottom rather than reading as slow.
    for r in sorted(runs, key=lambda x: (single_tps(x) is not None, single_tps(x) or 0),
                    reverse=True):
        m, p = r.get("model") or {}, r.get("placement") or {}
        st = single_tps(r)
        cs, ca = best_conc(r)
        rb = r.get("rubric") or {}
        rub = (f"{rb.get('pass')}/{rb.get('total')}"
               if rb.get("total") else '<span class="dim">-</span>')
        place = e(p.get("node") or "-")
        if p.get("tp"):
            place += f' <span class="dim">TP{e(p["tp"])}</span>'
        o.append(
            f'<tr><td class="mdl"><b>{e(m.get("name") or m.get("id") or "?")}</b>'
            f'<span>{e(r.get("label") or "")}</span></td>'
            f'<td>{e(params_txt(m))}</td>'
            f'<td>{e(m.get("quant") or "-")}</td>'
            f'<td>{place}</td>'
            f'<td class="num">'
            + (f'<b>{num(st)}</b> tok/s' if st else '<span class="dim">-</span>')
            + '</td><td class="num">'
            + (f'<b>{num(ca, 0)}</b> tok/s <span class="dim">@{cs}</span>'
               if ca else '<span class="dim">-</span>')
            + f'</td><td class="num">{rub}</td>'
            f'<td class="num">{e(ctx(m.get("context")))}</td></tr>')
    o.append("</tbody></table></div>")
    return "".join(o)


def stat(k, v, u="", na=False):
    cls = "v na" if na else "v"
    return (f'<div class="stat"><div class="k">{e(k)}</div>'
            f'<div class="{cls}">{v}</div>'
            + (f'<div class="u">{e(u)}</div>' if u else "") + "</div>")


def panel(title, body, note=""):
    return (f'<div class="panel"><div class="ptitle">{e(title)}</div>{body}'
            + (f'<div class="pnote">{note}</div>' if note else "") + "</div>")


NM = '<div class="nm">Not measured</div>'


def run_block(r):
    m, p = r.get("model") or {}, r.get("placement") or {}
    s = r.get("settings") or {}
    o = ['<section class="mrun">']

    # ---- head
    o.append('<div class="mhead"><div class="mtitle">'
             f'<h3>{e(m.get("name") or m.get("id") or "?")}</h3>'
             + (f'<span class="badge">{e(r.get("label"))}</span>' if r.get("label") else "")
             + f'<span class="dim" style="font-family:var(--mono);font-size:12px">'
               f'{e(stamp_txt(r.get("stamp")))}</span></div>')
    meta = [f'<b>{e(m.get("id") or "?")}</b>']
    bits = []
    if p.get("node"):
        bits.append(f'node <b>{e(p["node"])}</b>')
    if p.get("gpu"):
        bits.append(f'{e(p["gpu"])} x{e(p.get("gpus", 1))}')
    if p.get("tp"):
        bits.append(f'TP{e(p["tp"])}')
    if p.get("engine_image"):
        bits.append(f'engine <b>{e(p["engine_image"])}</b>')
    if p.get("ainode"):
        bits.append(f'ainode {e(p["ainode"])}')
    if p.get("kv_cache_dtype"):
        bits.append(f'kv {e(p["kv_cache_dtype"])}')
    if p.get("gpu_memory_utilization") is not None:
        bits.append(f'gmu {e(p["gpu_memory_utilization"])}')
    if p.get("max_model_len"):
        bits.append(f'served ctx {e(ctx(p["max_model_len"]))}')
    if s.get("thinking") is False:
        bits.append("thinking off")
    elif s.get("thinking") is True:
        bits.append("thinking on")
    if p.get("stacked_with"):
        bits.append("stacked with " + e(", ".join(p["stacked_with"])))
    meta.append(" &middot; ".join(bits))
    meta.append(f'source <b>{e(r.get("source") or "?")}</b>')
    o.append(f'<div class="mmeta">{"<br>".join(meta)}</div></div>')

    # ---- headline stats
    st = res(r, "single_stream") or {}
    su = res(r, "sustained") or {}
    cs, ca = best_conc(r)
    pf = [x for x in (res(r, "prefill") or []) if x.get("prompt_tokens")]
    rt = res(r, "reasoning_tax") or {}
    o.append('<div class="stats">')
    o.append(stat("Single stream",
                  f'{num(st.get("decode_tok_s"))}' if st.get("decode_tok_s")
                  else "not measured",
                  'tok/s' + (f' · TTFT {ms(st.get("ttft_ms"))}'
                              if st.get("ttft_ms") is not None else ""),
                  na=not st.get("decode_tok_s")))
    o.append(stat("Sustained",
                  num(su.get("decode_tok_s")) if su.get("decode_tok_s") else "not measured",
                  f'tok/s over {toks(su.get("gen_tokens"))} tok' if su.get("gen_tokens")
                  else "one long generation",
                  na=not su.get("decode_tok_s")))
    o.append(stat("Batched", num(ca, 0) if ca else "not measured",
                  f'tok/s at {cs} streams' if ca else "concurrency sweep",
                  na=not ca))
    deep = max(pf, key=lambda x: x["prompt_tokens"]) if pf else None
    o.append(stat("Deepest prefill",
                  e(ms(deep.get("ttft_ms"))) if deep and deep.get("ttft_ms") is not None
                  else (toks(deep["prompt_tokens"]) if deep else "not measured"),
                  f'TTFT at {toks(deep["prompt_tokens"])} prompt tok' if deep
                  else "prefill curve",
                  na=not deep))
    if rt:
        o.append(stat("Reasoning tax",
                      f'{num(rt.get("tax_x"), 2)}x' if rt.get("tax_x") else "measured",
                      "wall clock, thinking on vs off", na=not rt.get("tax_x")))
    o.append("</div>")

    # ---- panels
    o.append('<div class="panels">')
    rows = res(r, "prefill")
    if rows:
        chart = svg_prefill(rows)
        tbl = ['<div class="tscroll"><table class="minitable"><thead><tr><th>prompt tok</th><th>TTFT</th>'
               '<th>prefill tok/s</th><th>decode tok/s</th></tr></thead><tbody>']
        for x in rows:
            tbl.append(f'<tr><td>{e(toks(x.get("prompt_tokens")))}</td>'
                       f'<td>{e(ms(x.get("ttft_ms")))}</td>'
                       f'<td>{num(x.get("prefill_tok_s"), 0)}</td>'
                       f'<td>{num(x.get("decode_tok_s"))}</td></tr>')
        tbl.append("</tbody></table></div>")
        note = ("prefill tok/s is prompt_tokens/TTFT, so it is a floor: TTFT carries "
                "queueing as well as the forward pass.")
        if len(rows) > 1 and rows[0].get("decode_tok_s") and rows[-1].get("decode_tok_s"):
            d = (1 - rows[-1]["decode_tok_s"] / rows[0]["decode_tok_s"]) * 100
            # Negative happens on spec-decode engines, where acceptance rate moves
            # with content: over a short sweep that swing can beat the depth
            # penalty. Reporting it as a rise beats printing "falls -14%".
            verb = "falls" if d >= 0 else "rises"
            note = (f'Decode {verb} {abs(d):.0f}% between the shallowest and deepest '
                    f'prompt. ') + note
        o.append(panel("Prefill against prompt length", chart + "".join(tbl), note))
    else:
        o.append(panel("Prefill against prompt length", NM))

    crows = res(r, "concurrency")
    if crows:
        note = ""
        ok = [c for c in crows if c.get("aggregate_tok_s")]
        if len(ok) > 1:
            lo, hi = ok[0], ok[-1]
            note = (f'{hi["streams"]} streams deliver '
                    f'{hi["aggregate_tok_s"] / lo["aggregate_tok_s"]:.1f}x the aggregate '
                    f'of {lo["streams"]}. Batching is where this hardware pays off.')
        o.append(panel("Concurrency", svg_concurrency(crows), note))
    else:
        o.append(panel("Concurrency", NM))

    if rt:
        extra = []
        if rt.get("thinking_on_tokens"):
            extra.append(f'{rt["thinking_on_tokens"]} tokens with thinking on')
        if rt.get("thinking_off_tokens"):
            extra.append(f'{rt["thinking_off_tokens"]} with it off')
        o.append(panel("Reasoning tax", svg_reasoning(rt),
                       e(", ".join(extra)) if extra else ""))
    else:
        o.append(panel("Reasoning tax", NM))

    tel = res(r, "telemetry")
    if tel:
        chips = []
        if tel.get("gpu_mem_used_gb") is not None:
            chips.append(f'<span class="chip"><i>GPU mem</i> {num(tel["gpu_mem_used_gb"])}'
                         + (f' / {num(tel.get("gpu_mem_total_gb"))}'
                            if tel.get("gpu_mem_total_gb") else "") + " GB</span>")
        if tel.get("gpu_util_pct") is not None:
            chips.append(f'<span class="chip"><i>GPU util</i> '
                         f'{num(tel["gpu_util_pct"], 0)}%</span>')
        if tel.get("temp_c") is not None:
            chips.append(f'<span class="chip"><i>temp</i> {num(tel["temp_c"], 0)} C</span>')
        if tel.get("samples"):
            chips.append(f'<span class="chip"><i>samples</i> {e(tel["samples"])}</span>')
        o.append(panel("Device telemetry", f'<div class="chips">{"".join(chips)}</div>',
                       e(tel.get("reading") or "")))
    else:
        o.append(panel("Device telemetry", NM,
                       "Run with --ainode to record GPU load beside the numbers."))

    if p.get("flags"):
        src = p.get("flags_source") or ""
        o.append(panel("Engine flags",
                       '<div class="flags">' + e(" ".join(p["flags"])) + "</div>",
                       e(src)))
    notes = r.get("notes") or []
    rb = r.get("rubric") or {}
    if rb.get("notes"):
        notes = [f'Rubric {rb.get("pass")}/{rb.get("total")}: {rb["notes"]}'] + list(notes)
    if notes:
        o.append('<div class="panel wide"><div class="ptitle">Notes</div><ul class="notes">'
                 + "".join(f"<li>{e(n)}</li>" for n in notes) + "</ul></div>")
    o.append("</div></section>")
    return "".join(o)


METHOD = """
<h4>Nothing is loaded, unloaded or restarted</h4>
<p>Every run is pure inference load against whatever the node was already serving.
A benchmark that reloads the model measures the loader, and it takes the node away
from whoever was using it.</p>
<h4>Token counts come from the server</h4>
<p>Prompt sizes are the engine's own <code>usage.prompt_tokens</code>, requested via
<code>stream_options.include_usage</code>, never a chars-per-token estimate: a bad guess
would shift the whole prefill x-axis invisibly. Generated counts are
<code>usage.completion_tokens</code>, never a count of SSE chunks, because under
speculative decoding (DSpark, MTP) one chunk can carry several accepted tokens and
chunk-counting silently halves the rate.</p>
<h4>Prefix caching cannot fake a deep prompt</h4>
<p>Every prompt carries a unique nonce at the front, so <code>--enable-prefix-caching</code>
has nothing to hit. A nonce at the end would leave the whole prefix cacheable and the
depth numbers would be cache hits wearing a costume.</p>
<h4>Decode excludes prefill</h4>
<p>The decode clock starts at the first content delta, not at request send. Folding
TTFT into the rate turns a decode number into an agent-loop number and the two get
quoted interchangeably. Tokens emitted by a reasoning parser count as generated
tokens, because they cost decode time like any other.</p>
<h4>Prefill tok/s is a floor</h4>
<p>It is <code>prompt_tokens / TTFT</code>. TTFT includes queueing and scheduling, and the
OpenAI-compatible API exposes no engine-internal prefill timing, so the real forward
pass is at least this fast and possibly faster.</p>
<h4>Telemetry is the peak, not the average</h4>
<p>GPU load, memory and temperature are sampled from AINode's <code>/api/nodes</code> while
the bench runs, and the peak is reported: the peak is what the node had to survive.
A tok/s number without the load context beside it is not evidence.</p>
<h4>"Not measured" is a real answer</h4>
<p>A section that did not run renders as those words. Nothing on this page is
extrapolated, rounded up from a similar model, or carried over from another run.</p>
"""


def render(runs):
    nodes = sorted({(r.get("placement") or {}).get("node") for r in runs
                    if (r.get("placement") or {}).get("node")})
    models = sorted({(r.get("model") or {}).get("id") for r in runs
                     if (r.get("model") or {}).get("id")})
    o = ['<!doctype html><html lang="en"><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         "<title>AINode Bench</title>",
         '<meta name="description" content="What AINode-served models actually do on '
         'NVIDIA GB10 hardware: prefill against prompt length, sustained decode, '
         'concurrency, and the reasoning tax, with device telemetry beside every number.">',
         f"<style>{CSS}</style>",
         '<body><div class="wrap">',
         '<header class="top"><div class="brandrow"><span class="tag">AINode bench</span>',
         f'<span class="counts">{len(runs)} run{"s" if len(runs) != 1 else ""} &middot; '
         f'{len(models)} model{"s" if len(models) != 1 else ""}'
         + (f' &middot; {len(nodes)} node{"s" if len(nodes) != 1 else ""}' if nodes else "")
         + "</span></div>",
         "<h1>What these models <span class=\"g\">actually do</span><br>on the hardware "
         "in front of us</h1>",
         '<p class="lede">A spec sheet quotes one number from one prompt on an empty box. '
         'This measures the rest: <strong>how prefill scales with prompt length</strong>, '
         'whether throughput <strong>holds over a long generation</strong>, what happens '
         'when <strong>several people use the node at once</strong>, and what a reasoning '
         'model charges for thinking. Device telemetry is recorded beside every number, '
         'and every run names the node, engine image and vLLM flags it ran on.</p>',
         '<p class="sub">GB10 decode is memory-bandwidth bound, so single-stream tok/s is '
         'close to a property of the model and the quantisation. The column worth reading '
         'is the batched one.</p>',
         "</header>"]
    if not runs:
        o.append('<h2>No results yet</h2><p class="sub">Drop a schema-1 JSON into '
                 'bench/results/ and re-run <code>python3 bench/report.py</code>.</p>')
    else:
        o.append("<h2>Leaderboard</h2>")
        o.append('<p class="sub">Sorted by single-stream decode. "Batched" is the '
                 'aggregate across the widest concurrency sweep that ran, with the stream '
                 'count beside it. Rubric is the hand-scored quality pass, where one '
                 'was done.</p>')
        o.append(leaderboard(runs))
        o.append("<h2>Every run</h2>")
        o.append('<p class="sub">Newest first. One block per model, placement and day.</p>')
        for r in runs:
            o.append(run_block(r))
    o.append("<h2>How it is measured</h2>")
    o.append(f'<div class="method">{METHOD}</div>')
    o.append('<footer><span>AINode bench &middot; generated by bench/report.py from '
             'bench/results/*.json</span><span>Powered by <b>argentos.ai</b></span>'
             "</footer></div></body></html>")
    return "".join(o)


def render_dir(d):
    """Render every schema-1 result in a directory. A directory that does not
    exist yet renders the empty page rather than raising: a fresh node has no
    results and the Bench view still has to load."""
    d = pathlib.Path(d)
    return render(load_runs(d) if d.is_dir() else [])


def main(argv=None, results=None, out=None):
    ap = argparse.ArgumentParser(prog="bench/report.py",
                                 description="render bench/results/*.json to one page")
    ap.add_argument("--results", default=str(results) if results else None, required=not results)
    ap.add_argument("--out", default=str(out) if out else None, required=not out)
    a = ap.parse_args(argv)
    d = pathlib.Path(a.results)
    if not d.is_dir():
        sys.exit(f"no results directory at {d}")
    runs = load_runs(d)
    outp = pathlib.Path(a.out)
    outp.write_text(render(runs))
    kb = outp.stat().st_size / 1024
    print(f"  {len(runs)} run(s) -> {outp} ({kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
