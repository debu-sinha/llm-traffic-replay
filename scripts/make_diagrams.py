#!/usr/bin/env python3
"""Generate the repo diagrams: editable .excalidraw sources + SVG renders.

One geometry spec drives both outputs, so the picture in the README and the
editable file at excalidraw.com never drift apart. Rerun after editing SPEC.
"""
import json, random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)
rnd = random.Random(42)

BLUE, GREEN, YELLOW, PINK, GRAY = "#a5d8ff", "#b2f2bb", "#ffec99", "#ffc9c9", "#e9ecef"

ARCH = {
    "boxes": [
        # id, x, y, w, h, fill, title, sub
        ("cfg",     40,  40, 210, 70, YELLOW, "profile + run configs", "JSON, provenance labels"),
        ("profile", 320, 40, 210, 70, BLUE,   "profile.py", "quantiles -> per-request\ntoken plan (lognormal fits)"),
        ("pool",    600, 40, 210, 70, BLUE,   "prefix_pool.py", "doc + prefix length,\nZipf popularity"),
        ("sched",   320, 170, 210, 70, BLUE,  "schedule.py", "bursty arrivals,\nrate_scale, shard"),
        ("textgen", 600, 170, 210, 70, GREEN, "textgen.py", "shared-prefix text,\ncalibrated chars/token"),
        ("runner",  880, 105, 210, 70, GREEN, "runner.py", "paced dispatch,\nbounded thread pool"),
        ("client",  880, 260, 210, 70, GREEN, "client.py", "streaming HTTP,\nTTFT on first content"),
        ("endpoint",600, 390, 210, 70, PINK,  "endpoint", "real PT / pay-per-token\nor bundled mock"),
        ("metrics", 1160, 260, 210, 70, GRAY, "metrics.py", "percentiles +\nbelievability block"),
        ("results", 1160, 390, 210, 70, GRAY, "results/", "requests.jsonl,\nsummary.json, report.md"),
    ],
    "arrows": [  # from-id, to-id, label, optional waypoints
        ("cfg", "profile", "", None),
        ("profile", "pool", "prefix tokens", None),
        ("pool", "textgen", "doc id + cut", None),
        ("sched", "runner", "timestamps",
         [(425, 240), (425, 252), (940, 252), (940, 178)]),
        ("textgen", "runner", "messages", None),
        ("runner", "client", "paced submit", None),
        ("client", "endpoint", "SSE stream", None),
        ("client", "metrics", "timings + usage", None),
        ("metrics", "results", "", None),
    ],
    "size": (1420, 510),
    "title": "llm-traffic-replay: architecture",
}

SEQ = {
    "boxes": [
        ("runner", 120, 40, 180, 50, GREEN, "runner", ""),
        ("worker", 520, 40, 180, 50, GREEN, "worker thread", ""),
        ("ep",     920, 40, 180, 50, PINK,  "endpoint", ""),
    ],
    "lifelines": [("runner", 210), ("worker", 610), ("ep", 1010)],
    "messages": [  # y, from-x, to-x, label
        (140, 210, 610, "submit at scheduled time (lag recorded)"),
        (200, 610, 1010, "POST chat completion, stream=true  (t_send)"),
        (260, 1010, 610, "role-only chunk  (TTFB, not TTFT)"),
        (320, 1010, 610, "first content delta  (TTFT)"),
        (380, 1010, 610, "final chunk + usage, [DONE]  (E2E)"),
        (440, 610, 210, "RequestResult: timings, tokens, cached, doc id"),
    ],
    "size": (1220, 520),
    "title": "llm-traffic-replay: request sequence",
}


def _base(el_type, x, y, w, h):
    return {
        "id": f"el{rnd.randrange(10**9)}", "type": el_type, "x": x, "y": y,
        "width": w, "height": h, "angle": 0, "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 1,
        "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": None, "seed": rnd.randrange(10**9),
        "versionNonce": rnd.randrange(10**9), "isDeleted": False,
        "boundElements": None, "updated": 1, "link": None, "locked": False,
    }


def ex_rect(x, y, w, h, fill):
    e = _base("rectangle", x, y, w, h)
    e["backgroundColor"] = fill
    e["roundness"] = {"type": 3}
    return e


def ex_text(x, y, w, h, text, size=16, align="center"):
    e = _base("text", x, y, w, h)
    e.update({"text": text, "fontSize": size, "fontFamily": 1,
              "textAlign": align, "verticalAlign": "top",
              "containerId": None, "originalText": text,
              "autoResize": True, "lineHeight": 1.25})
    return e


def ex_arrow(x1, y1, x2, y2, dashed=False, pts=None):
    if pts is None:
        pts = [(x1, y1), (x2, y2)]
    x1, y1 = pts[0]
    x2, y2 = pts[-1]
    e = _base("arrow", x1, y1, x2 - x1, y2 - y1)
    e.update({"points": [[px - x1, py - y1] for (px, py) in pts],
              "lastCommittedPoint": None, "startBinding": None,
              "endBinding": None, "startArrowhead": None,
              "endArrowhead": "arrow", "elbowed": False})
    if dashed:
        e["strokeStyle"] = "dashed"
    return e


def centers(spec):
    return {b[0]: (b[1] + b[3] / 2, b[2] + b[4] / 2, b[1], b[2], b[3], b[4])
            for b in spec["boxes"]}


def edge_points(a, b):
    ax, ay, ax0, ay0, aw, ah = a
    bx, by, bx0, by0, bw, bh = b
    dx, dy = bx - ax, by - ay
    if abs(dx) > abs(dy):
        p1 = (ax0 + (aw if dx > 0 else 0), ay)
        p2 = (bx0 + (0 if dx > 0 else bw), by)
    else:
        p1 = (ax, ay0 + (ah if dy > 0 else 0))
        p2 = (bx, by0 + (0 if dy > 0 else bh))
    return p1, p2


def build_excalidraw(spec, with_lifelines=False):
    els = []
    for (bid, x, y, w, h, fill, title, sub) in spec["boxes"]:
        els.append(ex_rect(x, y, w, h, fill))
        label = title + ("\n" + sub if sub else "")
        els.append(ex_text(x + 8, y + 8, w - 16, h - 16, label,
                           size=14 if sub else 16))
    if with_lifelines:
        for (_, lx) in spec["lifelines"]:
            els.append(ex_arrow(lx, 95, lx, spec["size"][1] - 30, dashed=True))
        for (y, x1, x2, label) in spec["messages"]:
            els.append(ex_arrow(x1, y, x2, y))
            els.append(ex_text(min(x1, x2) + 20, y - 22, abs(x2 - x1) - 40,
                               18, label, size=13, align="left"))
    else:
        c = centers(spec)
        for (f, t, label, via) in spec["arrows"]:
            if via:
                pts = via
                p1, p2 = pts[0], pts[-1]
            else:
                p1, p2 = edge_points(c[f], c[t])
                pts = [p1, p2]
            els.append(ex_arrow(*p1, *p2, pts=pts))
            if label:
                mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                els.append(ex_text(mx - 60, my - 24, 140, 16, label, size=12))
    return {"type": "excalidraw", "version": 2,
            "source": "llm-traffic-replay/scripts/make_diagrams.py",
            "elements": els,
            "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
            "files": {}}


FONT = ("font-family='-apple-system, BlinkMacSystemFont, Segoe UI, "
        "Helvetica, Arial, sans-serif'")


def svg_of(spec, with_lifelines=False):
    W, H = spec["size"]
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' "
           f"width='{W}' height='{H}'>",
           "<defs><marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' "
           "markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
           "<path d='M 0 1 L 9 5 L 0 9' fill='none' stroke='#1e1e1e' "
           "stroke-width='1.4' stroke-linecap='round'/></marker></defs>",
           f"<rect width='{W}' height='{H}' fill='white'/>",
           f"<text x='{W/2}' y='26' text-anchor='middle' {FONT} "
           f"font-size='19' fill='#1e1e1e'>{spec['title']}</text>"]

    def box(x, y, w, h, fill, title, sub):
        out.append(f"<rect x='{x}' y='{y}' rx='10' width='{w}' height='{h}' "
                   f"fill='{fill}' stroke='#1e1e1e' stroke-width='1.3'/>")
        ty = y + 22 if sub else y + h / 2 + 5
        out.append(f"<text x='{x + w/2}' y='{ty}' text-anchor='middle' {FONT} "
                   f"font-size='15' font-weight='bold' fill='#1e1e1e'>{title}</text>")
        for i, line in enumerate((sub.split("\n") if sub else [])):
            out.append(f"<text x='{x + w/2}' y='{ty + 17 + i * 15}' "
                       f"text-anchor='middle' {FONT} font-size='12' "
                       f"fill='#343a40'>{line}</text>")

    def arrow(x1, y1, x2, y2, dashed=False, pts=None):
        dash = " stroke-dasharray='7 6'" if dashed else ""
        marker = "" if dashed else " marker-end='url(#ah)'"
        if pts is None:
            pts = [(x1, y1), (x2, y2)]
        d = "M " + " L ".join(f"{px} {py}" for (px, py) in pts)
        out.append(f"<path d='{d}' stroke='#1e1e1e' "
                   f"stroke-width='1.3' fill='none'{dash}{marker}/>")

    for (bid, x, y, w, h, fill, title, sub) in spec["boxes"]:
        box(x, y, w, h, fill, title, sub)
    if with_lifelines:
        for (_, lx) in spec["lifelines"]:
            arrow(lx, 95, lx, H - 30, dashed=True)
        for (y, x1, x2, label) in spec["messages"]:
            arrow(x1, y, x2, y)
            out.append(f"<text x='{(x1 + x2)/2}' y='{y - 8}' "
                       f"text-anchor='middle' {FONT} font-size='12.5' "
                       f"fill='#1e1e1e'>{label}</text>")
    else:
        c = centers(spec)
        for (f, t, label, via) in spec["arrows"]:
            if via:
                pts = via
                p1, p2 = pts[0], pts[-1]
            else:
                p1, p2 = edge_points(c[f], c[t])
                pts = [p1, p2]
            arrow(*p1, *p2, pts=pts)
            if label:
                out.append(f"<text x='{(pts[0][0]+pts[-1][0])/2}' y='{min(py for _, py in pts) - 6}' "
                           f"text-anchor='middle' {FONT} font-size='11.5' "
                           f"fill='#495057'>{label}</text>")
    out.append("</svg>")
    return "\n".join(out)


for name, spec, seq in (("architecture", ARCH, False),
                        ("request-sequence", SEQ, True)):
    (OUT / f"{name}.excalidraw").write_text(
        json.dumps(build_excalidraw(spec, seq), indent=1))
    (OUT / f"{name}.svg").write_text(svg_of(spec, seq))
    print("wrote", name, ".excalidraw + .svg")
