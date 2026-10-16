"""Inject model.json + backtest_summary.json into the app template.

Writes:
  app/index.html            - page content for the Claude artifact (the artifact adds its own <html> skeleton)
  server/public/index.html  - full standalone page for the hosted website / phone home-screen app
"""
import json, pathlib

root = pathlib.Path(__file__).resolve().parent.parent
tpl = (root / "app" / "template.html").read_text()
model = (root / "out" / "model.json").read_text()
bt = json.loads((root / "out" / "backtest_summary.json").read_text())
page = tpl.replace("/*MODEL*/null", model).replace("/*BACKTEST*/null", json.dumps(bt, separators=(",", ":")))
(root / "app" / "index.html").write_text(page)

head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<meta name="apple-mobile-web-app-capable" content="yes">'
        '<meta name="apple-mobile-web-app-title" content="Next Read">'
        '<meta name="theme-color" content="#23543a">'
        '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
        'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>')
pub = root / "server" / "public"
pub.mkdir(parents=True, exist_ok=True)
(pub / "index.html").write_text(head + page + "</body></html>")
print("ok", len(page) // 1024, "KB")
