#!/usr/bin/env python3
"""Convert a markdown file to a PDF via headless Chrome.

Usage:  python3 md_to_pdf.py [stem]
        (default stem = "paper-draft" → paper-draft.md → paper-draft.pdf)
"""
import subprocess
import sys
from pathlib import Path
import markdown

HERE = Path(__file__).parent
STEM = sys.argv[1] if len(sys.argv) > 1 else "paper-draft"
MD = HERE / f"{STEM}.md"
HTML = HERE / f"{STEM}.html"
PDF = HERE / f"{STEM}.pdf"

CSS = """
@page { size: Letter; margin: 0.85in 0.85in 0.95in 0.85in; }
html { -webkit-print-color-adjust: exact; }
body {
  font-family: "Charter", "Iowan Old Style", "Palatino", "Georgia", serif;
  font-size: 10.8pt;
  line-height: 1.45;
  color: #111;
  max-width: 6.75in;
  margin: 0 auto;
}
/* Title: keep on its own opening band */
h1 {
  font-size: 19pt;
  font-weight: 700;
  margin: 0 0 0.35em 0;
  line-height: 1.2;
  letter-spacing: -0.005em;
  page-break-after: avoid;
}
/* The bold lead lines right under the title (Status / Repository / Author) */
h1 + p, h1 + p + p, h1 + p + p + p, h1 + p + p + p + p {
  margin: 0.15em 0;
  font-size: 10pt;
  color: #444;
}
h2 {
  font-size: 13.5pt;
  margin-top: 1.5em;
  margin-bottom: 0.4em;
  border-bottom: 1px solid #777;
  padding-bottom: 3px;
  page-break-after: avoid;
}
h3 {
  font-size: 11.6pt;
  margin-top: 1.15em;
  margin-bottom: 0.3em;
  page-break-after: avoid;
}
h4 {
  font-size: 10.8pt;
  font-style: italic;
  margin-top: 0.9em;
  margin-bottom: 0.25em;
  page-break-after: avoid;
}
p  { margin: 0.55em 0; text-align: justify; hyphens: auto; }
ul, ol { margin: 0.45em 0 0.65em 1.5em; padding: 0; }
li { margin: 0.18em 0; }
code, pre, tt {
  font-family: "SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
  font-size: 9.1pt;
}
code {
  background: #f1f1f3;
  padding: 0 3px;
  border-radius: 2px;
  font-size: 9.5pt;
}
pre {
  background: #f6f6f7;
  border: 1px solid #d8d8dc;
  border-radius: 3px;
  padding: 8px 11px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
  line-height: 1.38;
  page-break-inside: avoid;
  margin: 0.7em 0;
}
pre code { background: transparent; padding: 0; font-size: 9.1pt; }
table {
  border-collapse: collapse;
  margin: 0.65em 0;
  font-size: 9.6pt;
  width: 100%;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #b8b8be;
  padding: 4px 8px;
  text-align: left;
  vertical-align: top;
}
th { background: #ececef; font-weight: 700; }
blockquote {
  border-left: 3px solid #aaa;
  margin: 0.7em 0;
  padding: 0.1em 0 0.1em 0.85em;
  color: #444;
}
hr { border: 0; border-top: 1px solid #999; margin: 1.3em 0; }
a { color: #1a4fa0; text-decoration: none; }
a:hover { text-decoration: underline; }
strong { font-weight: 700; }
em { font-style: italic; }
/* Abstract paragraph styling */
h2:first-of-type + p {
  text-align: justify;
}
"""

def main():
    md_text = MD.read_text()
    html_body = markdown.markdown(
        md_text,
        extensions=["extra", "tables", "fenced_code", "codehilite", "toc", "sane_lists"],
        extension_configs={"codehilite": {"noclasses": True, "pygments_style": "friendly"}},
    )
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Nakshatra — Draft Technical Report</title>
<style>{CSS}</style></head>
<body>
{html_body}
</body></html>"""
    HTML.write_text(html_doc)
    print(f"[md_to_pdf] wrote {HTML} ({len(html_doc):,} chars)")

    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={PDF}",
        "--print-to-pdf-no-header",
        "--no-pdf-header-footer",
        "--virtual-time-budget=5000",
        HTML.as_uri(),
    ]
    print(f"[md_to_pdf] running headless Chrome -> {PDF}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + "\n" + r.stderr + "\n")
        sys.exit(r.returncode)
    print(f"[md_to_pdf] done: {PDF} ({PDF.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
