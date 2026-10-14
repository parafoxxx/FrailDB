"""Assemble the deployable static site.

site/template.html + site/frailtydb.json  ->  site/index.html   (the page, data inlined)
                                          ->  dist/            (what you deploy)
dist/ contains:
  index.html        the whole portal (one self-contained page; data inlined, hash routing)
  frailtydb.json    the same database as a public machine-readable file
  .nojekyll, robots.txt, 404.html
"""
import os, shutil, json

here = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(here, "site", "template.html"), encoding="utf-8").read()
raw = open(os.path.join(here, "site", "frailtydb.json"), encoding="utf-8").read()
page_body = tpl.replace("/*__FRAILDB_DATA__*/", raw.replace("</", "<\\/"))
open(os.path.join(here, "site", "index.html"), "w", encoding="utf-8").write(page_body)

# A standalone web page needs its own document shell (the Artifact publisher adds one automatically).
shell = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
         '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
         '<meta name="description" content="FrailDB: Fried frailty phenotype and Rockwood frailty index results across SHARE, HRS, ELSA, MHAS and CLHLS.">\n'
         '<style>html{color-scheme:light dark}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n'
         '</head>\n<body>\n')
dist = os.path.join(here, "dist")
os.makedirs(dist, exist_ok=True)
# move <title> and the font <link>s (everything before the first <style>) into <head>
cut = page_body.index("<style>")
html = shell.replace("</head>", page_body[:cut] + "</head>") + page_body[cut:] + "\n</body>\n</html>\n"
open(os.path.join(dist, "index.html"), "w", encoding="utf-8").write(html)
open(os.path.join(dist, "404.html"), "w", encoding="utf-8").write(
    shell + '<p style="font-family:system-ui;padding:24px">Page not found. <a href="./">Go to FrailDB</a></p></body></html>')
shutil.copyfile(os.path.join(here, "site", "frailtydb.json"), os.path.join(dist, "frailtydb.json"))
open(os.path.join(dist, ".nojekyll"), "w").close()
open(os.path.join(dist, "robots.txt"), "w").write("User-agent: *\nAllow: /\n")

meta = json.loads(raw)["meta"]
print(f"site/index.html {len(page_body) // 1024} KB · dist/ ready (version {meta['version']}, built {meta['built']})")
