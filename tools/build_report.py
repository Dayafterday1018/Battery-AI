"""Build the model-improvement report page from the metrics that evaluate.py and eval_ocr.py saved.

Every number on the page comes from reports/metrics/*.json; nothing is typed by hand.
Run:  python tools/build_report.py [--team "Team 3"] [--demo]
Out:  reports/report.html   (self-contained; publish it as an artifact and share the link)

Runs are picked by --stage: the latest "baseline" and the latest "improved" run of each model kind.
--demo puts a banner on the page; use it only for runs on FAKE_* data.
"""
import argparse
import base64
import html
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "reports" / "metrics"
REPORTS = ROOT / "reports"

e = lambda s: html.escape(str(s))
pct = lambda x: f"{x * 100:.1f}%"


def load_runs():
    runs = [json.loads(p.read_text()) for p in sorted(METRICS.glob("*.json"))]
    return [r for r in runs if r.get("kind") == "classifier"], [r for r in runs if r.get("kind") == "ocr"]


def pick(runs, stage, best=False):
    """Latest run of a stage; with best=True the most accurate one (every run still appears in the log)."""
    tagged = [r for r in runs if r.get("stage") == stage]
    if tagged:
        return max(tagged, key=lambda r: (r["accuracy"], r["run_at"])) if best else tagged[-1]
    if not runs:
        return None
    return runs[0] if stage == "baseline" else runs[-1]


def pill(ok):
    return f'<span class="pill {"pass" if ok else "fail"}">{"Meets target" if ok else "Below target"}</span>'


def delta(a, b):
    if a is None or b is None:
        return ""
    d = (b - a) * 100
    sign = "+" if d >= 0 else "−"
    return f'<span class="delta {"up" if d >= 0 else "down"}">{sign}{abs(d):.1f} pts</span>'


def scale_bar(base, final, target):
    """Accuracy on a 50–100% scale with the target marked."""
    lo = 0.5
    x = lambda v: max(0.0, min(1.0, (v - lo) / (1 - lo))) * 100
    base_mark = f'<span class="tick base" style="left:{x(base):.2f}%" title="baseline {pct(base)}"></span>' if base is not None else ""
    return (f'<div class="scale" role="img" aria-label="accuracy {pct(final)} against target {pct(target)}">'
            f'<span class="fill" style="width:{x(final):.2f}%"></span>{base_mark}'
            f'<span class="tick target" style="left:{x(target):.2f}%"></span></div>'
            f'<div class="scale-axis"><span>50%</span><span>target {pct(target)}</span><span>100%</span></div>')


def kpi(title, unit_note, base, final, target):
    if final is None:
        return (f'<article class="kpi"><h3>{e(title)}</h3><p class="empty">No run yet. '
                f'{e(unit_note)}</p></article>')
    b = base["accuracy"] if base and base is not final else None
    return f'''<article class="kpi">
  <div class="kpi-head"><h3>{e(title)}</h3>{pill(final["accuracy"] >= target)}</div>
  <p class="kpi-num">{pct(final["accuracy"])}</p>
  <p class="kpi-sub">{"baseline " + pct(b) + " · " if b is not None else ""}{delta(b, final["accuracy"])} <span class="muted">{e(unit_note)}</span></p>
  {scale_bar(b, final["accuracy"], target)}
</article>'''


def changes_list(items):
    if not items:
        return '<p class="muted">No changes recorded for this run.</p>'
    return "<ol class=\"changes\">" + "".join(f"<li>{e(c)}</li>" for c in items) + "</ol>"


def table(headers, rows, cls=""):
    head = "".join(f"<th>{e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def confusion(r):
    c = r["confusion"]
    return f'''<figure class="cm">
  <figcaption>{e(r["model"])} <span class="muted">· threshold {r["threshold"]}</span></figcaption>
  <div class="cm-grid">
    <span></span><span class="cm-h">pred. normal</span><span class="cm-h">pred. laptop</span>
    <span class="cm-h row">normal</span><span class="cell ok">{c["normal_as_normal"]}</span><span class="cell bad" title="never billed">{c["normal_as_laptop"]}</span>
    <span class="cm-h row">laptop</span><span class="cell warn" title="one extra OCR call">{c["laptop_as_normal"]}</span><span class="cell ok">{c["laptop_as_laptop"]}</span>
  </div>
</figure>'''


def img_tag(path, alt):
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure class="curve"><img src="data:image/png;base64,{data}" alt="{e(alt)}"><figcaption>{e(alt)}</figcaption></figure>'


def brand_chart(base, final, target):
    """Horizontal bars per brand: baseline (light) and improved (solid), target line."""
    brands = sorted(final["per_brand"], key=lambda b: -final["per_brand"][b]["total"])
    row_h, label_w, chart_w, pad = 34, 96, 440, 14
    h = pad * 2 + row_h * len(brands) + 22
    x = lambda v: label_w + v * chart_w
    out = [f'<svg class="brand-chart" viewBox="0 0 {label_w + chart_w + 70} {h}" role="img" aria-label="OCR accuracy per brand">']
    for v in (0, 0.5, 1.0):
        out.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{pad - 4}" y2="{h - 22}" class="grid"/>'
                   f'<text x="{x(v):.1f}" y="{h - 6}" class="axis" text-anchor="middle">{int(v * 100)}%</text>')
    out.append(f'<line x1="{x(target):.1f}" x2="{x(target):.1f}" y1="{pad - 6}" y2="{h - 22}" class="target-line"/>'
               f'<text x="{x(target) + 4:.1f}" y="{pad + 2}" class="axis target-text">target {int(target * 100)}%</text>')
    for i, b in enumerate(brands):
        y = pad + i * row_h + 6
        f = final["per_brand"][b]
        acc = f["correct"] / f["total"]
        out.append(f'<text x="{label_w - 10}" y="{y + 15}" class="label" text-anchor="end">{e(b)}</text>')
        if base and b in base["per_brand"]:
            bb = base["per_brand"][b]
            out.append(f'<rect x="{x(0)}" y="{y}" width="{bb["correct"] / bb["total"] * chart_w:.1f}" height="9" class="bar-base"/>')
        out.append(f'<rect x="{x(0)}" y="{y + 11}" width="{acc * chart_w:.1f}" height="11" class="bar-final"/>')
        out.append(f'<text x="{x(acc) + 6:.1f}" y="{y + 21}" class="value">{f["correct"]}/{f["total"]}</text>')
    out.append("</svg>")
    return "".join(out)


def cls_section(base, final):
    if not final:
        return '<p class="empty">Run <code>python src/evaluate.py --model … --stage baseline improved</code> first.</p>'
    cols = [r for r in (base, final) if r] if base is not final else [final]
    metrics = [("Test accuracy", "accuracy", pct), ("Recall · normal", "recall_normal", pct),
               ("Recall · laptop", "recall_laptop", pct), ("Precision · normal", "precision_normal", pct),
               ("Precision · laptop", "precision_laptop", pct), ("Decision threshold", "threshold", str),
               ("Input size (px)", "img", str), ("Model file (MB)", "model_mb", str),
               ("Seconds per photo (CPU)", "sec_per_photo_cpu", str), ("Test photos", "test_photos", str)]
    rows = [[e(label)] + [e(f(r[key])) if r.get(key) is not None else "—" for r in cols] for label, key, f in metrics]
    heads = ["Metric"] + [f'{r.get("stage") or "run"} · {r["model"]}' for r in cols]
    errors = final.get("errors", [])
    err_rows = [[f"<code>{e(x['file'])}</code>", e(x["actual"]), e(x["predicted"]), e(x["p_laptop"])] for x in errors[:12]]
    curve = img_tag(REPORTS / f'{final["model"]}_curves.png', f'Training curves · {final["model"]}')
    gap = ""
    if final.get("best_val_accuracy") is not None and final.get("train_accuracy_at_best") is not None:
        gap = (f'<p class="note">Best validation accuracy {pct(final["best_val_accuracy"])} with training accuracy '
               f'{pct(final["train_accuracy_at_best"])} at the same epoch.</p>')
    return f'''
<div class="split">
  <div class="col"><h3>What we changed</h3>{changes_list(final.get("changes", []))}
    <p class="note">The threshold is tuned on the validation split. The test set is read once, by <code>src/evaluate.py</code>.</p></div>
  <div class="col">{table(heads, rows, "metrics")}</div>
</div>
<h3>Confusion matrices</h3>
<p class="note">A normal battery called laptop is a fee never billed (red). A laptop called normal costs one extra OCR call (amber).</p>
<div class="cms">{"".join(confusion(r) for r in cols)}</div>
{curve}{gap}
<h3>Test photos the final model got wrong <span class="count">{len(errors)}</span></h3>
{table(["File", "Actual", "Predicted", "p(laptop)"], err_rows) if errors else '<p class="muted">None.</p>'}'''


def ocr_path(runs):
    """Every OCR run in order, with what changed since the previous one (read from the metrics files)."""
    if len(runs) < 2:
        return ""
    rows, prev = [], None
    for r in sorted(runs, key=lambda r: r["run_at"]):
        what = []
        if prev is None:
            what.append("Starting point")
        else:
            if r["version"] != prev["version"]:
                what.append(f'Pipeline {prev["version"]} → {r["version"]}: prompt with the brand list, photo preprocessing, retries')
            if r["model"] != prev["model"]:
                what.append(f'Vision model {prev["model"]} → {r["model"]}')
            what = what or ["Re-run with the same settings"]
        acc = f'{pct(r["accuracy"])} ({r["correct"]}/{r["gold_photos"]})'
        rows.append([e(f'{r["version"]} · {r["model"]}'), e("; ".join(what)),
                     e(acc) + (" " + delta(prev["accuracy"], r["accuracy"]) if prev else "")])
        prev = r
    return "<h3>How OCR accuracy moved, run by run</h3>" + table(["Run", "What changed", "Gold-set accuracy"], rows, "metrics")


def ocr_section(base, final, target, runs=()):
    if not final:
        return '<p class="empty">Run <code>python src/eval_ocr.py --version v1 --stage baseline</code> and <code>--version v2 --stage improved</code> first.</p>'
    cols = [r for r in (base, final) if r] if base is not final else [final]
    metrics = [("Gold-set accuracy", lambda r: f'{pct(r["accuracy"])} ({r["correct"]}/{r["gold_photos"]})'),
               ("Pipeline version", lambda r: r["version"]), ("Vision model", lambda r: r["model"]),
               ("Seconds per photo", lambda r: r["sec_per_photo"]), ("Model calls per photo", lambda r: r["calls_per_photo"])]
    rows = [[e(label)] + [e(f(r)) for r in cols] for label, f in metrics]
    heads = ["Metric"] + [f'{r.get("stage") or "run"} · {r["version"]}' for r in cols]
    miss_rows = [[f"<code>{e(m['file'])}</code>", e(m["expected"]), e(m["got"]), e(m["text_seen"]) or "—"]
                 for m in final.get("misses", [])[:15]]
    legend = ('<p class="legend"><span class="sw base"></span>baseline <span class="sw final"></span>final</p>'
              if base and base is not final else "")
    changes = list(final.get("changes", []))
    if base and base.get("model") != final.get("model"):
        changes.insert(0, f'Vision model {base.get("model")} → {final.get("model")}')
    return f'''
<div class="split">
  <div class="col"><h3>What we changed</h3>{changes_list(changes)}</div>
  <div class="col">{table(heads, rows, "metrics")}</div>
</div>
{ocr_path(list(runs))}
<h3>Accuracy per brand</h3>{legend}
<div class="chart-wrap">{brand_chart(base if base is not final else None, final, target)}</div>
<h3>Gold-set photos still missed <span class="count">{len(final.get("misses", []))}</span></h3>
{table(["File", "Expected", "Got", "Text the model saw"], miss_rows) if miss_rows else '<p class="muted">None.</p>'}'''


def runs_log(cls_runs, ocr_runs):
    rows = []
    for r in cls_runs:
        rows.append([e(r["run_at"]), "Classifier", e(r.get("stage") or "—"), e(r["model"]), e(pct(r["accuracy"])), e(r.get("note", ""))])
    for r in ocr_runs:
        rows.append([e(r["run_at"]), "Brand OCR", e(r.get("stage") or "—"), e(f'{r["version"]} · {r["model"]}'), e(pct(r["accuracy"])), e(r.get("note", ""))])
    rows.sort(key=lambda r: r[0])
    return table(["Run", "Model", "Stage", "Version", "Accuracy", "Note"], rows)


CSS = """
<style>
/* Layout: one reading column; verdict tiles, then a section per model; tables scroll inside their own box. */
:root{
  --bg:#f3f4f6; --surface:#ffffff; --ink:#1b1f24; --muted:#5b6470; --line:#d9dde3;
  --copper:#b2622a; --copper-soft:#e9c7ab; --pass:#2f7d4f; --fail:#b3362c; --warn:#b07a12;
  --display:"Barlow Condensed","Arial Narrow",sans-serif; --body:"Source Sans 3","Segoe UI",system-ui,sans-serif;
  --mono:"JetBrains Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#14171b; --surface:#1d2127; --ink:#e7e9ec; --muted:#9aa3ae; --line:#323842;
  --copper:#e08a4c; --copper-soft:#5b3a22; --pass:#5cbf86; --fail:#ef7b70; --warn:#e2b04a; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#14171b; --surface:#1d2127; --ink:#e7e9ec; --muted:#9aa3ae; --line:#323842;
  --copper:#e08a4c; --copper-soft:#5b3a22; --pass:#5cbf86; --fail:#ef7b70; --warn:#e2b04a; color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:16px/1.55 var(--body);
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.page{max-width:1040px;margin:0 auto;padding-block:40px 64px;display:flex;flex-direction:column;gap:40px}
h1,h2,h3{font-family:var(--display);text-wrap:balance;margin:0;line-height:1.1}
h1{font-size:clamp(2.2rem,5vw,3.4rem);font-weight:700;letter-spacing:.005em}
h2{font-size:1.9rem;font-weight:700;display:flex;align-items:baseline;gap:12px}
h3{font-size:1.25rem;font-weight:600;margin-top:8px}
.eyebrow{font:600 .78rem/1 var(--display);letter-spacing:.14em;text-transform:uppercase;color:var(--copper)}
header{display:flex;flex-direction:column;gap:10px}
.meta{color:var(--muted);margin:0}
.demo{border:2px dashed var(--fail);color:var(--fail);padding:12px 16px;border-radius:6px;font-weight:600;margin:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:20px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:20px 22px;display:flex;flex-direction:column;gap:8px}
.kpi-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
.kpi-num{font:700 3.2rem/1 var(--display);margin:4px 0 0;font-variant-numeric:tabular-nums}
.kpi-sub{margin:0;color:var(--muted);font-size:.95rem}
.pill{font:600 .75rem/1 var(--body);padding:6px 10px;border-radius:99px;letter-spacing:.02em;white-space:nowrap}
.pill.pass{color:var(--pass);border:1px solid var(--pass)}
.pill.fail{color:var(--fail);border:1px solid var(--fail)}
.delta{font-weight:600;font-variant-numeric:tabular-nums}.delta.up{color:var(--pass)}.delta.down{color:var(--fail)}
.scale{position:relative;height:10px;background:var(--line);border-radius:5px;margin-top:8px}
.scale .fill{position:absolute;inset:0 auto 0 0;background:var(--copper);border-radius:5px}
.scale .tick{position:absolute;top:-5px;width:2px;height:20px;transform:translateX(-1px)}
.scale .tick.target{background:var(--ink)}
.scale .tick.base{background:var(--muted);opacity:.8}
.scale-axis{display:flex;justify-content:space-between;font:.75rem var(--mono);color:var(--muted)}
section{display:flex;flex-direction:column;gap:16px}
.section-num{font:600 1rem var(--mono);color:var(--copper)}
.split{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.2fr);gap:28px;align-items:start}
@media (max-width:760px){.split{grid-template-columns:minmax(0,1fr)}}
.col{min-width:0;display:flex;flex-direction:column;gap:10px}
.changes{margin:0;padding-left:1.3em;display:flex;flex-direction:column;gap:6px}
.changes li::marker{color:var(--copper);font-family:var(--mono);font-size:.85em}
.note{color:var(--muted);font-size:.92rem;margin:0}
.muted{color:var(--muted)}
.empty{color:var(--muted);border:1px dashed var(--line);border-radius:6px;padding:14px;margin:0}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.93rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{font:600 .74rem var(--display);letter-spacing:.1em;text-transform:uppercase;color:var(--muted);white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
table.metrics td:not(:first-child){font-family:var(--mono);font-size:.88rem}
table.metrics td:first-child{white-space:nowrap}
code{font:.85em var(--mono);word-break:break-all}
.count{font:600 .9rem var(--mono);color:var(--muted)}
.cms{display:flex;flex-wrap:wrap;gap:28px}
.cm{margin:0;display:flex;flex-direction:column;gap:8px}
.cm figcaption{font-weight:600}
.cm-grid{display:grid;grid-template-columns:auto 92px 92px;gap:4px;align-items:center}
.cm-h{font:600 .72rem var(--display);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:center}
.cm-h.row{text-align:right;padding-right:8px}
.cell{font:600 1.3rem var(--mono);text-align:center;padding:14px 0;border-radius:4px;background:var(--surface);border:1px solid var(--line)}
.cell.ok{color:var(--pass)}.cell.bad{color:var(--fail)}.cell.warn{color:var(--warn)}
.curve{margin:0;display:flex;flex-direction:column;gap:6px}
.curve img{border:1px solid var(--line);border-radius:6px;background:#fff}
.curve figcaption{color:var(--muted);font-size:.85rem}
.chart-wrap{overflow-x:auto}
.brand-chart{width:100%;max-width:640px;min-width:420px}
.brand-chart .grid{stroke:var(--line);stroke-width:1}
.brand-chart .target-line{stroke:var(--ink);stroke-width:1.5;stroke-dasharray:4 3}
.brand-chart .axis{fill:var(--muted);font:11px var(--mono)}
.brand-chart .target-text{fill:var(--ink)}
.brand-chart .label{fill:var(--ink);font:600 13px var(--body)}
.brand-chart .value{fill:var(--muted);font:12px var(--mono)}
.brand-chart .bar-base{fill:var(--copper-soft)}
.brand-chart .bar-final{fill:var(--copper)}
.legend{display:flex;gap:14px;align-items:center;margin:0;color:var(--muted);font-size:.88rem}
.sw{display:inline-block;width:14px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
.sw.base{background:var(--copper-soft)}.sw.final{background:var(--copper)}
.steps{margin:0;padding-left:1.3em;display:flex;flex-direction:column;gap:6px}
pre{font:.85rem/1.6 var(--mono);background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px;overflow-x:auto;margin:0}
footer{color:var(--muted);font-size:.85rem;border-top:1px solid var(--line);padding-top:16px}
</style>"""


def build(team, demo):
    cls_runs, ocr_runs = load_runs()
    cb, cf = pick(cls_runs, "baseline"), pick(cls_runs, "improved")
    ob, of = pick(ocr_runs, "baseline"), pick(ocr_runs, "improved", best=True)
    cls_target = (cf or {}).get("target", 0.98)
    ocr_target = (of or {}).get("target", 0.85)
    when = datetime.now().strftime("%Y-%m-%d")
    test_n = cf["test_photos"] if cf else "—"
    gold_n = of["gold_photos"] if of else "—"
    banner = ('<p class="demo">Demo run on synthetic FAKE_* photos. These numbers show how the page works; '
              'they are not results on Client W\'s photos.</p>') if demo else ""
    return f'''<title>BatteryScan Model Improvements</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=JetBrains+Mono:wght@400;600&family=Source+Sans+3:wght@400;600&display=swap">
{CSS}
<main class="page">
<header>
  <span class="eyebrow">BatteryScan · Client W · week 2</span>
  <h1>Model improvement report</h1>
  <p class="meta">{e(team) + " · " if team else ""}{when} · classifier test set {test_n} photos · OCR gold set {gold_n} photos</p>
  {banner}
</header>

<div class="kpis">
  {kpi("Battery classification", f"target ≥ {pct(cls_target)} on the test set", cb, cf, cls_target)}
  {kpi("Brand OCR", f"target ≥ {pct(ocr_target)} on the gold set", ob, of, ocr_target)}
</div>

<section>
  <h2><span class="section-num">2.x</span>Battery classification · laptop vs normal</h2>
  {cls_section(cb, cf)}
</section>

<section>
  <h2><span class="section-num">3.1</span>Brand OCR on normal batteries</h2>
  {ocr_section(ob, of, ocr_target, ocr_runs)}
</section>

<section>
  <h2>How we tested</h2>
  <ol class="steps">
    <li>Photos split 80 / 20 into train and test, stratified by label and grouped so two shots of the same battery never land on both sides.</li>
    <li>20% of the training photos held out for validation; early stopping and the decision threshold use validation only.</li>
    <li>Each model is scored once on the test set by <code>src/evaluate.py</code>; OCR is scored on the hand-checked gold set by <code>src/eval_ocr.py</code>.</li>
    <li>This page is generated from the saved metrics files by <code>tools/build_report.py</code>.</li>
  </ol>
  <pre>python src/train_transfer.py --name mobilenet_baseline --img 128 --fine-tune-layers 0 --no-augment --no-class-weight --no-tune
python src/train_transfer.py --name mobilenet
python src/evaluate.py --model models/mobilenet_baseline.keras models/mobilenet.keras --stage baseline improved
python src/eval_ocr.py --version v1 --stage baseline
python src/eval_ocr.py --version v2 --stage improved
python src/eval_ocr.py --version v2 --model qwen2.5vl:7b --stage improved
python tools/build_report.py</pre>
  <h3>Every evaluation run</h3>
  {runs_log(cls_runs, ocr_runs)}
</section>

<footer>Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} from reports/metrics/ · BatteryScan AI solution demo</footer>
</main>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="")
    ap.add_argument("--demo", action="store_true", help="mark the page as a synthetic-data demo")
    ap.add_argument("--out", type=Path, default=REPORTS / "report.html")
    args = ap.parse_args()
    if not METRICS.exists():
        raise SystemExit("No reports/metrics/ yet: run src/evaluate.py and src/eval_ocr.py first.")
    args.out.write_text(build(args.team, args.demo), encoding="utf-8")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
