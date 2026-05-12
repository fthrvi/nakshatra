#!/usr/bin/env python3
"""Render the Nakshatra SVG assets to PNG via headless Chrome."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

JOBS = [
    # (svg, png_out, viewport_w, viewport_h)
    ("logo-mark.svg",   "logo-mark-512.png",  512, 512),
    ("logo-mark.svg",   "logo-mark-1024.png", 1024, 1024),
    ("logo-lockup.svg", "logo-lockup-1600.png", 1600, 512),
    ("og-card.svg",     "og-card-1200x630.png", 1200, 630),
]

WRAPPER_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  body {{ width: {w}px; height: {h}px; overflow: hidden; }}
  svg {{ display: block; width: {w}px; height: {h}px; }}
</style></head>
<body>{svg}</body></html>"""

for svg_name, png_name, w, h in JOBS:
    svg_path = HERE / svg_name
    png_path = HERE / png_name
    tmp_html = HERE / f"_tmp_{svg_name}.html"
    tmp_html.write_text(WRAPPER_HTML.format(w=w, h=h, svg=svg_path.read_text()))

    cmd = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--default-background-color=00000000",  # transparent
        f"--window-size={w},{h}",
        f"--screenshot={png_path}",
        "--virtual-time-budget=2000",
        tmp_html.as_uri(),
    ]
    print(f"[render] {svg_name} -> {png_name} ({w}x{h})")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + "\n" + r.stderr + "\n")
        sys.exit(r.returncode)
    tmp_html.unlink(missing_ok=True)
    print(f"        wrote {png_path.stat().st_size:,} bytes")

print("[render] done")
