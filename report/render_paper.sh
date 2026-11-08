#!/bin/zsh
# Rebuild the Word paper from a saved Claude Docs read, export it to PDF through Word, and report pages.
# Usage: report/render_paper.sh <saved docs read .txt>
set -e
cd "/Users/alibelal/Swarm alignment"
python3 - "$1" <<'EOF'
import json, sys
raw = open(sys.argv[1]).read()
obj, _ = json.JSONDecoder().raw_decode(raw[raw.find('{'):])
open('report/paper_doc.xml', 'w').write(obj['data']['xml']); print('doc rev', obj['data']['rev'])
EOF
OUT="report/Shared randomness makes large LLM swarms coordinate like small ones.docx"
.venv/bin/python -m swarm.paper_docx "/Users/alibelal/Downloads/Copy of Apart Research hackathon submission template.docx" report/paper_doc.xml "$OUT"
cp "$OUT" report/render_check.docx && rm -f report/paper_check.pdf
osascript <<'EOS'
tell application "Microsoft Word"
  set d to open file name "Macintosh HD:Users:alibelal:Swarm alignment:report:render_check.docx"
  delay 2
  set d to document "render_check.docx"
  save as d file name "/Users/alibelal/Swarm alignment/report/paper_check.pdf" file format format PDF
  close d saving no
end tell
EOS
rm -f report/render_check.docx
