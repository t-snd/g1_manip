#!/bin/sh
# REPORT.md を A4 の PDF（docs/report/REPORT.pdf）にして、ページ数を出す。1 枚に収まっているかの確認用。
#   sh docs/report/make_pdf.sh
# Markdown → HTML は marked（npx）、HTML → PDF は Google Chrome のヘッドレス印刷。
set -eu
cd "$(dirname "$0")/../.."
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HTML=REPORT.tmp.html  # 画像の相対パスを REPORT.md と揃えるため、リポジトリ直下に置く
{
  cat <<'EOF'
<!doctype html><html lang="ja"><head><meta charset="utf-8"><style>
@page { size: A4; margin: 10mm 12mm; }
body { font-family: "Hiragino Sans", sans-serif; font-size: 8.4pt; line-height: 1.48; color: #111; }
h1 { font-size: 13.5pt; margin: 0 0 2px; }
h1 + p { margin: 0 0 4px; color: #555; font-size: 8pt; }
h2 { font-size: 10pt; margin: 7px 0 2px; border-bottom: 1px solid #ccc; }
p, ul, ol { margin: 2px 0; }
ul, ol { padding-left: 1.3em; }
li { margin: 1px 0; }
table { border-collapse: collapse; margin: 3px 0; font-size: 8pt; }
th, td { border: 1px solid #ccc; padding: 1px 6px; }
th { white-space: nowrap; }
img { margin: 0 0 2px 10px; }
code { font-size: 8pt; }
</style></head><body>
EOF
  npx --yes marked@15.0.12 -i REPORT.md
  echo '</body></html>'
} > "$HTML"
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf=docs/report/REPORT.pdf "$HTML" 2>/dev/null
rm "$HTML"
echo "docs/report/REPORT.pdf: $(grep -a -o "/Count [0-9]*" docs/report/REPORT.pdf | head -1 | cut -d" " -f2) ページ"
